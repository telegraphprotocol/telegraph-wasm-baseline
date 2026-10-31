// Package remote asks a WASM module's author-hosted executor to score a batch
// of questions over HTTP, using the same inputs and outputs as the local
// wazero runtimes in pkg/wasm/runtime. The node uses it for the remote
// preliminary check, before any binary is downloaded.
package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// DefaultTimeout bounds a call made with a context that carries no deadline
// of its own (e.g. context.Background()). A context WITH a deadline is the
// only bound, longer or shorter than this: roundTrip applies DefaultTimeout to
// the context, never as http.Client.Timeout, which would cap every call at
// 30s even when the caller's own budget allows minutes.
const DefaultTimeout = 30 * time.Second

// maxResponseBytes caps how much of a response body roundTrip will read. A
// batch reply is a few hundred bytes; this is purely a defensive limit against
// an untrusted server sending an oversized response.
const maxResponseBytes = 64 << 10 // 64 KiB

// MaxBatchItems is the most questions one batch request may carry. The remote
// check asks 100 comparator cases (REMOTE_COMPARATOR_SAMPLE, default 100, so the
// default sits exactly at this limit) or 30 wide texts (REMOTE_PUBLIC_PAIRS,
// default 15 pairs, two texts each) per request. Both settings are capped so a
// request never goes over it (applyRemoteSampleEnv in
// pkg/scoring/remote_sample_config.go). Raising either default above this
// needs this limit raised too.
const MaxBatchItems = 100

// isBlockedAddr reports whether ip is loopback, private, link-local,
// unspecified, or in the shared address space - never a valid destination for
// an outbound call to an untrusted, on-chain-supplied URL.
func isBlockedAddr(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsUnspecified() || sharedAddressSpace.Contains(ip)
}

// sharedAddressSpace is 100.64.0.0/10 (RFC 6598): carrier-grade NAT, and the
// range Tailscale and some cloud networks give internal hosts. IsPrivate does
// not cover it.
var sharedAddressSpace = &net.IPNet{IP: net.IPv4(100, 64, 0, 0), Mask: net.CIDRMask(10, 32)}

// dialerControl is a net.Dialer.Control hook: it runs after DNS resolution
// (address is already the resolved IP:port) and before the OS connects, so
// it blocks both a directly-supplied internal IP and a hostname that
// resolves to one - including via DNS rebinding, since the check is against
// the resolved address, never the hostname string.
func dialerControl(network, address string, _ syscall.RawConn) error {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("remote executor: invalid address %q: %w", address, err)
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return fmt.Errorf("remote executor: could not parse resolved address %q", host)
	}
	if isBlockedAddr(ip) {
		return fmt.Errorf("remote executor: refusing to connect to blocked address %s", ip)
	}
	return nil
}

// noRedirect stops the client from following a redirect - an untrusted
// server's 3xx response must not silently retarget a POST (and its body) at
// a different, possibly internal, host.
func noRedirect(*http.Request, []*http.Request) error {
	return http.ErrUseLastResponse
}

// RemoteExecutor runs a WASM module's scoring logic over HTTP against the
// author's hosted executor at URL.
type RemoteExecutor struct {
	URL    string
	Client *http.Client
}

// allowPrivateAddresses reports whether the SSRF guard should be skipped,
// via WASM_REMOTE_ALLOW_PRIVATE=true - a dev/e2e-only escape hatch so a
// local test executor (which necessarily lives at a loopback/private
// address) can be reached through this same constructor. Never set this in
// production: it disables the one protection standing between this client
// and an untrusted, on-chain-supplied URL pointed at internal
// infrastructure.
func allowPrivateAddresses() bool {
	return os.Getenv("WASM_REMOTE_ALLOW_PRIVATE") == "true"
}

// NewRemoteExecutor builds a RemoteExecutor with a default HTTP client
// (no client-level timeout, see DefaultTimeout), the SSRF guard (blocks loopback/private/
// link-local/unspecified addresses, post-DNS-resolution), and redirects
// disabled. This guard is wired in ONLY here - a RemoteExecutor built via a
// bare struct literal (as tests targeting a local httptest server do) does
// not get it. See allowPrivateAddresses for the dev-only escape hatch.
func NewRemoteExecutor(url string) *RemoteExecutor {
	transport := &http.Transport{}
	if !allowPrivateAddresses() {
		dialer := &net.Dialer{Control: dialerControl}
		transport.DialContext = dialer.DialContext
	}
	return &RemoteExecutor{
		URL: url,
		Client: &http.Client{
			Transport:     transport,
			CheckRedirect: noRedirect,
		},
	}
}

// RemoteRejectedError means the remote executor responded but rejected the
// candidate or hit a scoring error - a verdict from the module, not a
// transport failure. Temporary statuses (see isTransientStatus) are never
// a RemoteRejectedError: they say nothing about the module.
type RemoteRejectedError struct {
	StatusCode int
	Message    string
}

func (e *RemoteRejectedError) Error() string {
	return fmt.Sprintf("remote executor rejected (status %d): %s", e.StatusCode, e.Message)
}

// errorResponse is the body an executor sends with a non-2xx status.
type errorResponse struct {
	Error string `json:"error"`
}

func (r *RemoteExecutor) client() *http.Client {
	if r.Client != nil {
		return r.Client
	}
	return &http.Client{}
}

// isTransientStatus reports whether status is a temporary server or proxy
// condition (408, 429, any 5xx) rather than the module's verdict. The
// executor protocol uses 400/422 for a module error, so these must not be
// recorded as the module's answer: one 503 from a load balancer would
// otherwise reject a correct comparator permanently.
func isTransientStatus(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusTooManyRequests || status >= 500
}

// roundTrip marshals body, POSTs it to r.URL and returns the status and the
// (size-capped) response body. Transport failures are returned wrapped with
// fmt.Errorf. It is the one place a request to the author's server is made, so
// the URL scheme check, the context deadline default and the response cap apply
// to every call.
func (r *RemoteExecutor) roundTrip(ctx context.Context, body any) (int, []byte, error) {
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, DefaultTimeout)
		defer cancel()
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return 0, nil, fmt.Errorf("remote executor: marshal request: %w", err)
	}

	u, err := url.Parse(r.URL)
	if err != nil {
		return 0, nil, fmt.Errorf("remote executor: invalid URL %q: %w", r.URL, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return 0, nil, fmt.Errorf("remote executor: unsupported URL scheme %q, only http/https allowed", u.Scheme)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.URL, bytes.NewReader(payload))
	if err != nil {
		return 0, nil, fmt.Errorf("remote executor: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client().Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("remote executor: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return 0, nil, fmt.Errorf("remote executor: read response: %w", err)
	}
	return resp.StatusCode, respBody, nil
}

// BatchResult is the executor's answer to one question of a batch: a score, or
// the module's own rejection of that input (Error non-empty).
type BatchResult struct {
	Score int32
	Error string
}

type batchItemResponse struct {
	Score json.Number `json:"score"`
	Error string      `json:"error"`
}

type batchResponse struct {
	Results []batchItemResponse `json:"results"`
	// The identity echo of a foundation-era batch (see BatchIdentity).
	FoundationHash string `json:"foundation_hash,omitempty"`
	AdapterHash    string `json:"adapter_hash,omitempty"`
	TokenizerHash  string `json:"tokenizer_hash,omitempty"`
	InputRoot      string `json:"input_root,omitempty"`
}

// BatchIdentity is what a model score is a function of besides the tokens:
// the Foundation Model, the Evaluator adapter and the tokenizer. Sent with a
// batch request and echoed back by the executor; a missing or different echo
// is a verdict on the candidate (the executor does not run the registered
// adapter on this node's foundation), never a pass.
type BatchIdentity struct {
	FoundationHash string `json:"foundation_hash"`
	AdapterHash    string `json:"adapter_hash"`
	TokenizerHash  string `json:"tokenizer_hash"`
	// InputRoot binds the batch's token inputs (BatchInputRoot). Sent with
	// the request; an executor MAY echo it, and an echo must match. Not part
	// of Equal: an executor that predates it still passes on F/A/T.
	InputRoot string `json:"input_root,omitempty"`
}

// BatchInputRoot is the input root of a batch of token sequences:
// runtime.InputRoot over "<index>:<runtime.InputHash(ids)>" per item.
func BatchInputRoot(batch [][]int64) string {
	leaves := make([]string, len(batch))
	for i, ids := range batch {
		leaves[i] = fmt.Sprintf("%d:%s", i, runtime.InputHash(ids))
	}
	return runtime.InputRoot(leaves)
}

// Equal compares identities case-insensitively (hashes are hex).
func (b BatchIdentity) Equal(o BatchIdentity) bool {
	return strings.EqualFold(b.FoundationHash, o.FoundationHash) &&
		strings.EqualFold(b.AdapterHash, o.AdapterHash) &&
		strings.EqualFold(b.TokenizerHash, o.TokenizerHash)
}

func (b BatchIdentity) empty() bool {
	return b.FoundationHash == "" && b.AdapterHash == "" && b.TokenizerHash == ""
}

// postBatch sends n questions in ONE request, {"batch": [...]}, and returns one
// BatchResult per question, in order. The runnable URL must implement this
// request: the node never falls back to one call per question.
//
//   - A transient status (isTransientStatus) or a transport failure is a plain
//     error: no verdict, the caller retries.
//   - Any other non-2xx, or a 2xx reply with no "results" array, means the
//     endpoint does not implement the batch request. That is *RemoteRejectedError
//     (a verdict on the candidate; retrying cannot change it).
//   - A reply whose results do not line up with the questions, or hold an item
//     with neither a score nor an error, is malformed: a plain error.
func (r *RemoteExecutor) postBatch(ctx context.Context, items any, n int) ([]BatchResult, error) {
	raw, _, err := r.postBatchItems(ctx, items, n, nil)
	if err != nil || raw == nil {
		return nil, err
	}
	out := make([]BatchResult, n)
	for i, it := range raw {
		if it.Error != "" {
			out[i] = BatchResult{Error: it.Error}
			continue
		}
		s, err := scoreToInt32(it.Score)
		if err != nil {
			return nil, err
		}
		out[i] = BatchResult{Score: s}
	}
	return out, nil
}

// postBatchItems is postBatch up to, not including, reading the scores: one
// request, the status rules, the "results" array check, the count check, and
// "every item has a score or an error". The caller decides what a score means
// (basis points for a comparator, f32 bits for a float adapter).
//
// With a non-nil ident the request also carries foundation_hash, adapter_hash
// and tokenizer_hash, and a reply that echoes them must echo all three
// unchanged: a different echo is a *RemoteRejectedError. echoed reports
// whether the reply carried them.
func (r *RemoteExecutor) postBatchItems(ctx context.Context, items any, n int, ident *BatchIdentity) ([]batchItemResponse, bool, error) {
	if n == 0 {
		return nil, false, nil
	}
	if n > MaxBatchItems {
		return nil, false, fmt.Errorf("remote executor: batch of %d questions exceeds the maximum of %d", n, MaxBatchItems)
	}
	body := map[string]any{"batch": items}
	if ident != nil {
		body["foundation_hash"] = ident.FoundationHash
		body["adapter_hash"] = ident.AdapterHash
		body["tokenizer_hash"] = ident.TokenizerHash
		if ident.InputRoot != "" {
			body["input_root"] = ident.InputRoot
		}
	}
	status, respBody, err := r.roundTrip(ctx, body)
	if err != nil {
		return nil, false, err
	}

	if status < 200 || status >= 300 {
		var decoded errorResponse
		_ = json.Unmarshal(respBody, &decoded) // best-effort - message may be empty
		if isTransientStatus(status) {
			return nil, false, fmt.Errorf("remote executor: temporary failure (status %d): %s", status, decoded.Error)
		}
		return nil, false, &RemoteRejectedError{StatusCode: status, Message: "batch request refused: " + decoded.Error}
	}

	var resp batchResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, false, fmt.Errorf("remote executor: decode batch response: %w", err)
	}
	if resp.Results == nil {
		return nil, false, &RemoteRejectedError{StatusCode: status,
			Message: `the reply has no "results" array: the endpoint does not implement the batch request`}
	}
	if len(resp.Results) != n {
		return nil, false, fmt.Errorf("remote executor: batch reply has %d results for %d questions", len(resp.Results), n)
	}

	for i, it := range resp.Results {
		if it.Error == "" && it.Score == "" {
			return nil, false, fmt.Errorf("remote executor: batch result %d has neither a score nor an error", i)
		}
	}
	echoed := false
	if ident != nil {
		got := BatchIdentity{FoundationHash: resp.FoundationHash, AdapterHash: resp.AdapterHash, TokenizerHash: resp.TokenizerHash}
		if !got.empty() {
			if !got.Equal(*ident) {
				return nil, false, &RemoteRejectedError{StatusCode: status, Message: fmt.Sprintf(
					"identity echo mismatch: the executor echoed foundation_hash %q adapter_hash %q tokenizer_hash %q, this node sent %q %q %q",
					got.FoundationHash, got.AdapterHash, got.TokenizerHash, ident.FoundationHash, ident.AdapterHash, ident.TokenizerHash)}
			}
			echoed = true
		}
		if resp.InputRoot != "" && !strings.EqualFold(resp.InputRoot, ident.InputRoot) {
			return nil, false, &RemoteRejectedError{StatusCode: status, Message: fmt.Sprintf(
				"identity echo mismatch: the executor echoed input_root %q, this node sent %q", resp.InputRoot, ident.InputRoot)}
		}
	}
	return resp.Results, echoed, nil
}

func scoreToInt32(score json.Number) (int32, error) {
	i, err := score.Int64()
	if err != nil {
		return 0, fmt.Errorf("remote executor: non-integer score %q: %w", score, err)
	}
	if i < 0 || i > runtime.ScoreBasisPoints {
		return 0, fmt.Errorf("remote executor: score %d out of domain [0, %d]", i, runtime.ScoreBasisPoints)
	}
	return int32(i), nil
}

type deterministicRequest struct {
	GroundTruthValue int64    `json:"ground_truth_value"`
	MinerValue       int64    `json:"miner_value"`
	Spec             [2]int64 `json:"spec"`
}

// DeterministicCase is one question of a comparator batch.
type DeterministicCase struct {
	MinerValue int64
	TruthValue int64
	Spec       [2]int64
}

// ScoreDeterministicBatch asks the comparator endpoint all cases in one request.
// Mirrors a loop of (*runtime.DeterministicRuntime).ScoreDeterministic.
func (r *RemoteExecutor) ScoreDeterministicBatch(ctx context.Context, cases []DeterministicCase) ([]BatchResult, error) {
	items := make([]deterministicRequest, len(cases))
	for i, c := range cases {
		items[i] = deterministicRequest{GroundTruthValue: c.TruthValue, MinerValue: c.MinerValue, Spec: c.Spec}
	}
	return r.postBatch(ctx, items, len(items))
}

type tokensRequest struct {
	Tokens []int64 `json:"tokens"`
}

// ModelBatchResult is the executor's answer to one token sequence of a float
// adapter batch: the model's exact f32 score, or the module's own rejection of
// that input (Error non-empty).
type ModelBatchResult struct {
	Score runtime.ModelScore
	Error string
}

// ScoreBatch asks a float adapter's endpoint all token sequences in one
// request. Mirrors a loop of (*runtime.LoRARuntime).ScoreTokens: each result's
// "score" is the f32 bit pattern of the model's output as an unsigned integer
// (the HEX line's value, e.g. 1104535916 for 0x41d5e16c), never a decimal,
// which would be rounded. NaN and +-Inf are refused (a malformed reply: a plain
// error). There is no remote hybrid call: the comparator and the fusion run on
// the validator (scoring.FuseHybrid), not the executor. The batch envelope,
// status handling and limits are postBatch's.
func (r *RemoteExecutor) ScoreBatch(ctx context.Context, tokens [][]int64) ([]ModelBatchResult, error) {
	out, _, err := r.ScoreBatchWithIdentity(ctx, tokens, nil)
	return out, err
}

// ScoreBatchWithIdentity is ScoreBatch with the score identity in the request:
//
//	{"batch":[{"tokens":[...]}, ...], "foundation_hash":F, "adapter_hash":A, "tokenizer_hash":T}
//
// and the executor echoing the three fields at the top level of its reply,
// next to "results". A present but different echo is a *RemoteRejectedError (a
// verdict). echoed reports whether the reply carried them: the caller decides
// what a missing echo means (one serving a delta adapter must echo them).
// ident nil sends the request without the three fields and checks nothing.
func (r *RemoteExecutor) ScoreBatchWithIdentity(ctx context.Context, tokens [][]int64, ident *BatchIdentity) ([]ModelBatchResult, bool, error) {
	items := make([]tokensRequest, len(tokens))
	for i, t := range tokens {
		items[i] = tokensRequest{Tokens: t}
	}
	raw, echoed, err := r.postBatchItems(ctx, items, len(items), ident)
	if err != nil || raw == nil {
		return nil, echoed, err
	}
	out := make([]ModelBatchResult, len(raw))
	for i, it := range raw {
		if it.Error != "" {
			out[i] = ModelBatchResult{Error: it.Error}
			continue
		}
		s, err := scoreToModelScore(it.Score)
		if err != nil {
			return nil, echoed, fmt.Errorf("remote executor: batch result %d: %w", i, err)
		}
		out[i] = ModelBatchResult{Score: s}
	}
	return out, echoed, nil
}

func scoreToModelScore(score json.Number) (runtime.ModelScore, error) {
	bits, err := strconv.ParseUint(score.String(), 10, 32)
	if err != nil {
		return runtime.ModelScore{}, fmt.Errorf("score %q is not f32 bits (an unsigned 32-bit integer): %w", score, err)
	}
	s, err := runtime.NewModelScoreFromBits(uint32(bits))
	if err != nil {
		return runtime.ModelScore{}, err
	}
	return s, nil
}
