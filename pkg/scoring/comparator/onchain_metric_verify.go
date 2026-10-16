package comparator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ONCHAIN_METRIC_VERIFY: a wallet's ETH balance at a PINNED block height,
// expressed in whole Gwei.
//
// This is the only one of the wired intents whose ground truth is exact rather
// than merely stable. An FX fix is a published opinion and an OSRM duration is
// one engine's model, so both tolerate disagreement; a balance at a fixed block
// is canonical consensus state, and every archive node on Earth returns the
// identical integer for it. That is why the band is 5 bps — the tightest of any
// deterministic intent — and it is the reason the pin below is not a detail but
// the whole design.
//
// Verified live rather than assumed: eth.merkle.io and the registered miner both
// returned 0x4254812b06de0cf3 wei for 0xd8dA…6045 at block 23,000,000, and
// eth.drpc.org matched merkle on the beacon deposit contract at the same height.

// ───────────────────────────────── pinned keys ─────────────────────────────────

// BalanceKey is one pinned (address, block) pair.
//
// The block is stored as a decimal uint64 and rendered to hex at the call, because
// JSON-RPC takes a hex quantity while the miner's registered `block` parameter
// accepts either. Keeping the canonical form decimal means the two renderings
// come from one number rather than from two hand-written strings that can drift.
type BalanceKey struct {
	Key     string // short stable ID, used as the question key
	Name    string // human description for logs and the audit trail
	Address string // 0x-prefixed, EIP-55 mixed case as published
	Block   uint64 // PINNED height; never a tag
}

// BlockHex renders the pinned height as the JSON-RPC quantity, which is
// minimal-width hex with no leading zeroes.
func (b BalanceKey) BlockHex() string { return fmt.Sprintf("0x%x", b.Block) }

// PinnedBlock is the height every ONCHAIN_METRIC_VERIFY key is read at.
//
// Pinning is the single most important decision in this port, and it is forced
// rather than preferred. The registered miner's `block` parameter DEFAULTS to
// "latest", and "latest" is not a reproducible key: Ethereum produces a block
// every ~12s, and a funded address moves between them. Two validators scoring
// the same epoch a few seconds apart would read two different balances, compute
// two different truths, and disagree about every miner in the round — and at a
// 5 bps band a single ordinary transfer is far more than enough to flip a score
// from 10000 to 0. The reference host onchain_metric_price.mjs makes the same
// judgement structurally, by *refusing* to accept blockTag == "latest" at all.
//
// So the tolerance does not absorb chain-tip drift, and it was never meant to:
// the spec's tolerance_meaning says the 5 bps exists only to absorb the sub-Gwei
// truncation of the wei->Gwei rescale, "not real disagreement". Pinning is what
// makes that claim true. Everything else here — exact integer parsing, archive
// RPC selection — is downstream of this line.
//
// 23,000,000 specifically: it is final beyond any reorg depth (a reorg past a
// few blocks has never happened on mainnet), it predates this port by months so
// it cannot be tuned to flatter a miner, and it was confirmed servable by two
// independent keyless archive providers. It is a constant rather than a rolling
// "tip minus N" because a rolling pin reintroduces exactly the disagreement the
// pin removes unless every validator also agrees on when "now" is.
const PinnedBlock uint64 = 23_000_000

// BalanceKeys are the (address, block) pairs ONCHAIN_METRIC_VERIFY is scored on.
//
// Three addresses rather than one, for the same reason ROUTE_ETA asks three
// routes: the miner carries a registered DEFAULT address, so a single question
// cannot distinguish a miner that honours the address it is given from one that
// ignores it and replays its default. Asking three catches that — a miner
// replaying a baked-in address scores 0 on the two it was not built around.
//
// They also span four orders of magnitude on purpose (≈4.8 Gether, ≈30.7
// Tgwei... through ≈6.5e16 Gwei), which is what exercises the integer-exactness
// problem documented on parseETHBalance. A key set clustered at one magnitude
// would let a float-based parser look correct.
var BalanceKeys = []BalanceKey{
	{
		Key: "VITALIK", Name: "vitalik.eth balance", Block: PinnedBlock,
		Address: "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
	},
	{
		// The beacon chain deposit contract: the largest single ETH holder, and
		// the key that proves the parser is exact. Its Gwei value is ~6.5e16,
		// above 2^53, so it is NOT representable at 1 Gwei resolution in a
		// float64 — see parseETHBalance.
		Key: "BEACON", Name: "beacon deposit contract balance", Block: PinnedBlock,
		Address: "0x00000000219ab540356cBB839Cbe05303d7705Fa",
	},
	{
		Key: "EXCHANGE", Name: "exchange cold wallet balance", Block: PinnedBlock,
		Address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
	},
}

// BalanceKeyIDs is the fixed question set, in the order rounds are run.
var BalanceKeyIDs = balanceKeyIDs()

func balanceKeyIDs() []string {
	keys := make([]string, 0, len(BalanceKeys))
	for _, k := range BalanceKeys {
		keys = append(keys, k.Key)
	}
	return keys
}

// BalanceKeyFor resolves a question key to its pinned address and block.
func BalanceKeyFor(key string) (BalanceKey, bool) {
	for _, k := range BalanceKeys {
		if k.Key == key {
			return k, true
		}
	}
	return BalanceKey{}, false
}

// onchainQuestion is the question put to every miner for one pinned key.
//
// It states the block height explicitly, and as a decimal number, because the
// height is the part that must survive translation into the miner's own call.
// A question that said only "the current balance" would be answered at the tip
// — which is precisely the non-reproducible read the pin exists to prevent.
func onchainQuestion(key string) string {
	k, ok := BalanceKeyFor(key)
	if !ok {
		return fmt.Sprintf("What is the on-chain balance for %s?", key)
	}
	return fmt.Sprintf(
		"What is the Ethereum balance in Gwei of address %s at block %d (%s)?",
		k.Address, k.Block, k.Name)
}

// describeOnchain renders a key for logs and the stored audit trail. The block
// is included because a score is only auditable against the height it was
// computed at.
func describeOnchain(key string) string {
	if k, ok := BalanceKeyFor(key); ok {
		return fmt.Sprintf("%s@%d", k.Key, k.Block)
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// onchainParsers holds one entry per registered ONCHAIN_METRIC_VERIFY miner,
// keyed by the slug in miner_registry.
//
// Exactly one: ocm-eth-balance (registration 2761). The other ten ocm-* rows in
// the registry — ocm-blockchair, ocm-llama-*, ocm-mempool-*, ocm-stablecoins and
// friends — are all deregistered, and none of them serve a per-address balance
// at a pinned block anyway; they report chain-wide aggregates (TVL, fee
// estimates, difficulty) that have no pinned-block reading at all. They are
// omitted rather than given a failing parser, for the usual reason: a miner that
// cannot be asked the question is excluded from the round, not scored zero.
//
// A one-entry map is not much of an argument for the declarative response
// mappings described in the package doc, but this parser does add a sixth verb
// to the vocabulary those mappings would need — an exact big-integer field with
// a rescale — beyond the invert / decimal_comma / string_value / cross_via /
// series_key that FX_NOW established.
var onchainParsers = map[string]parserFunc{
	"ocm-eth-balance": parseETHBalance, // {"value_wei":4779587126345731315,"value_gwei":4779587126.345731,...}
}

// parseETHBalance reads a balance out of the ocm-eth-balance response and
// returns it as whole Gwei.
//
// It reads value_wei, NOT the value_gwei that the miner's registered
// label_field names and that the build spec's value_path (".value_gwei") points
// at. That is a deliberate departure from both, and the reason is that
// value_gwei is a JSON float:
//
//	"value_wei": 65463218703536528312319773,
//	"value_gwei": 6.546321870353653e+16
//
// A float has no place in this path. The comparator is pure integer arithmetic,
// the spec's parse_rule is "wei / 1e9, truncated to an integer (Gwei) -> i64,
// and NEVER float", and the reference host does that division in BigInt. Above
// 2^53 Gwei (~9M ETH) a float64 cannot even represent adjacent Gwei — measured
// at the beacon deposit contract's balance the ULP is 8 Gwei, so the value one
// Gwei above it decodes back to the same number. value_wei is an exact JSON
// integer and is the only field that survives the trip.
//
// The truncating division matches the reference host exactly (BigInt `/`), and
// truncation rather than rounding is what the spec says. Note this is the one
// place where ScaleValue's rounding would be WRONG, which is why the rescale
// happens here and the registered Scale is 1: rounding up at the Gwei boundary
// would put the host half a Gwei above every honest miner that truncated.
//
// The returned float64 is the shared pass's waypoint, not a loss of precision
// we chose: values below 2^53 Gwei round-trip exactly, and the bound is checked
// rather than assumed so an implausible balance fails loudly instead of being
// silently quantised.
func parseETHBalance(key, body string) (float64, error) {
	var resp struct {
		Address  string          `json:"address"`
		Block    string          `json:"block"`
		ValueWei json.RawMessage `json:"value_wei"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding balance response: %w", err)
	}
	if len(resp.ValueWei) == 0 {
		return 0, fmt.Errorf("no value_wei in response")
	}

	// The miner echoes the block it actually read. Checking it is what turns a
	// miner that silently ignored the pin and answered at the tip into an
	// excluded non-participant rather than a plausible-looking wrong answer —
	// at 5 bps an unpinned read is indistinguishable from a lie.
	if k, ok := BalanceKeyFor(key); ok && resp.Block != "" {
		if !sameBlock(resp.Block, k) {
			return 0, fmt.Errorf("response is for block %q, not the pinned %s", resp.Block, k.BlockHex())
		}
		if resp.Address != "" && !strings.EqualFold(resp.Address, k.Address) {
			return 0, fmt.Errorf("response is for address %q, not the pinned %s", resp.Address, k.Address)
		}
	}

	gwei, err := weiToGwei(resp.ValueWei)
	if err != nil {
		return 0, err
	}
	return gweiToFloat(gwei)
}

// sameBlock reports whether an echoed block field names the pinned height.
// Both spellings are accepted because the miner normalises a decimal request
// into hex on the way out, so the form it echoes is not the form it was sent.
func sameBlock(echoed string, k BalanceKey) bool {
	e := strings.ToLower(strings.TrimSpace(echoed))
	if e == k.BlockHex() {
		return true
	}
	return e == fmt.Sprintf("%d", k.Block)
}

// weiToGwei converts a raw JSON integer of wei into whole Gwei.
//
// json.RawMessage rather than a decoded number: encoding/json decodes an
// unadorned JSON number into a float64, which for a 26-digit wei value has
// already lost the low digits before any of this code runs. big.Int parses the
// literal digits.
func weiToGwei(raw json.RawMessage) (*big.Int, error) {
	text := strings.TrimSpace(string(raw))
	// Tolerate the value arriving as a JSON string; some RPC proxies quote big
	// integers precisely to keep JavaScript clients from mangling them.
	if s, err := strconv.Unquote(text); err == nil {
		text = strings.TrimSpace(s)
	}
	// A hex quantity is the JSON-RPC wire form and shows up if a miner passes
	// the RPC result through untouched.
	wei, ok := new(big.Int).SetString(strings.TrimPrefix(text, "0x"), hexOrDecBase(text))
	if !ok {
		return nil, fmt.Errorf("value_wei %q is not an integer", text)
	}
	if wei.Sign() < 0 {
		return nil, fmt.Errorf("value_wei %q is negative", text)
	}
	// Quo truncates towards zero, matching the reference host's BigInt division
	// and the spec's "truncated to an integer (Gwei)".
	return new(big.Int).Quo(wei, big.NewInt(weiPerGwei)), nil
}

func hexOrDecBase(text string) int {
	if strings.HasPrefix(text, "0x") || strings.HasPrefix(text, "0X") {
		return 16
	}
	return 10
}

// weiPerGwei is the rescale from the spec's parse_rule.
const weiPerGwei = 1_000_000_000

// maxExactGwei is 2^53: the largest Gwei value below which EVERY integer is
// exactly representable as a float64.
//
// The shared scoring pass moves values through a float64 (ParseMinerValue ->
// ScaleValue), so this is the threshold below which this intent's integers are
// unconditionally safe on that hop. ~9.007e15 Gwei is about 9 million ETH.
//
// It is a threshold, NOT a ceiling, and the distinction cost a test failure
// worth recording. Above 2^53 floats become sparse rather than unusable: they
// still represent every 2nd, then 4th, then 8th integer. The beacon deposit
// contract's balance — 65,463,218,703,536,528 Gwei, about 7x this bound — is
// itself exactly representable because it happens to be a multiple of 8, and
// rejecting it on magnitude alone would have excluded the largest ETH holder on
// mainnet from the intent for no reason. So the check below is a per-value
// round-trip, and this constant only marks where that check starts to matter.
const maxExactGwei = int64(1) << 53

// gweiToFloat hands a Gwei integer to the shared pass's float64 waypoint,
// refusing to do so where the conversion would not be exact.
//
// Failing loudly is the right outcome: a quantised balance would still score
// plausibly (8 Gwei of error on 6.5e16 is ~1e-9 bps, deep inside the 5 bps
// band), so the wrongness would never surface as a bad score. It would surface
// as two validators on different builds disagreeing, which is the failure this
// whole path exists to prevent.
func gweiToFloat(gwei *big.Int) (float64, error) {
	if !gwei.IsInt64() {
		return 0, fmt.Errorf("balance %s Gwei overflows the i64 comparator domain", gwei)
	}
	g := gwei.Int64()
	// The round-trip is the real test. Below maxExactGwei it cannot fail; above
	// it, it fails only for the values that genuinely land between two
	// representable floats, which is the set we actually need to reject.
	f := float64(g)
	if int64(f) != g {
		return 0, fmt.Errorf(
			"balance %d Gwei does not round-trip through float64 (nearest is %d); "+
				"the intent's values must reach the comparator as exact integers",
			g, int64(f))
	}
	return f, nil
}

// ──────────────────────────────── key coverage ─────────────────────────────────

// onchainCoverage records which pinned keys each miner can serve.
//
// ocm-eth-balance takes both address and block as request parameters — the
// values baked into its registration are `default`s in its input_schema, not
// fixed values — so it can serve every key. That was probed rather than
// assumed: asking it for each of the three addresses at block 23,000,000
// returned three different balances, all matching an independent RPC, rather
// than three copies of its default address's balance.
//
// The map is kept rather than collapsed so that a miner later registered
// against a single address or a non-archive node can be given real coverage
// without reshaping the scoring pass.
var onchainCoverage = map[string]func(key string) bool{
	"ocm-eth-balance": anyQuote,
}

// ───────────────────────────────── ground truth ────────────────────────────────

// EthRPCSource reads the reference balance from an Ethereum JSON-RPC node.
//
// It deliberately does NOT read the miner's own /truth/eth-balance proxy, even
// though that endpoint would return the same number. The registered miner is
// served from omni-chat; if ground truth came from the same host, the
// comparator could never fail anyone — it would be comparing a host against
// itself and reporting the tautology as a perfect score. The truth here is
// fetched from a provider with no relationship to the miner, which is what
// makes a divergence meaningful.
//
// Read the resulting ranking with the honest caveat attached: with one active
// miner there is no ranking, only a verification. A perfect score says the
// miner reported canonical chain state correctly at the pinned block — which is
// a real and checkable claim, and exactly what this intent is for — but it is
// not a comparative quality signal.
type EthRPCSource struct {
	// Endpoints are tried in order. More than one because the pinned block is
	// months old and therefore an ARCHIVE read: several otherwise-fine keyless
	// providers serve only recent state and reject it. publicnode, the default
	// in the reference host, is one of them — it answers the chain tip happily
	// and returns "Archive requests require a personal token" for block
	// 23,000,000 — so following the reference host's default verbatim would
	// have produced a truth source that failed on every scored key.
	Endpoints []string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewEthRPCSource builds an EthRPCSource with defaults suitable for epoch
// scoring.
//
// Both defaults were verified to serve archive state at the pinned block and to
// return bit-identical results for it. That agreement is the property being
// relied on and it is not a coincidence: an eth_getBalance at a fixed height is
// canonical consensus state, so any two honest nodes MUST agree. A divergence
// between these two would be a node bug, not a tolerance to widen.
func NewEthRPCSource() *EthRPCSource {
	return &EthRPCSource{
		Endpoints: []string{
			"https://eth.merkle.io",
			"https://eth.drpc.org",
		},
		Client: &http.Client{Timeout: 15 * time.Second},
	}
}

// FetchReference resolves the pinned balance for one key, already scaled to
// whole Gwei.
//
// The Pin argument is accepted and ignored, and unusually for this interface
// that is a statement rather than a shrug: this intent's pin is not a date
// chosen per epoch, it is the block height baked into the key itself. A caller
// cannot move it without changing the scored key set, which is the point —
// PinForEpoch returns "latest" for FX_NOW, and honouring that here would
// reintroduce the chain-tip race the whole design removes.
func (s *EthRPCSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	k, ok := BalanceKeyFor(key)
	if !ok {
		return Reference{}, fmt.Errorf("no pinned balance key %q", key)
	}

	endpoints := s.Endpoints
	if len(endpoints) == 0 {
		endpoints = NewEthRPCSource().Endpoints
	}

	var lastErr error
	for _, ep := range endpoints {
		wei, err := s.getBalance(ctx, ep, k)
		if err != nil {
			lastErr = fmt.Errorf("%s: %w", ep, err)
			continue
		}

		// The reference goes through the SAME wei->Gwei truncation as a miner's
		// value. Truth and miner cannot diverge through two subtly different
		// readings of one quantity — the failure that would show up as every
		// miner being uniformly slightly wrong rather than as a bug here.
		gwei := new(big.Int).Quo(wei, big.NewInt(weiPerGwei))
		f, err := gweiToFloat(gwei)
		if err != nil {
			return Reference{}, fmt.Errorf("reading the reference balance: %w", err)
		}
		scaled, err := ScaleValue("ONCHAIN_METRIC_VERIFY", f)
		if err != nil {
			return Reference{}, fmt.Errorf("scaling the reference balance: %w", err)
		}

		return Reference{
			RateBps: scaled,
			RawRate: f,
			Pair:    k.Name,
			// The pinned height, not today's date. A wall-clock date would be
			// actively misleading in the audit trail: re-running this next year
			// reads the same block and must produce the same row.
			Date:   fmt.Sprintf("block:%d", k.Block),
			Source: fmt.Sprintf("eth_getBalance(%s)@%s via %s", k.Address, k.BlockHex(), ep),
		}, nil
	}
	return Reference{}, fmt.Errorf("no RPC endpoint served the pinned balance: %w", lastErr)
}

// getBalance performs one eth_getBalance call at the pinned height.
func (s *EthRPCSource) getBalance(ctx context.Context, endpoint string, k BalanceKey) (*big.Int, error) {
	payload, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "eth_getBalance",
		// The block is ALWAYS a hex height here. The reference host refuses
		// "latest" outright and this path has no way to express it.
		"params": []string{k.Address, k.BlockHex()},
	})
	if err != nil {
		return nil, fmt.Errorf("building the RPC request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(payload)))
	if err != nil {
		return nil, fmt.Errorf("building the RPC request: %w", err)
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "application/json")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling eth_getBalance: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("eth_getBalance returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("reading the RPC response: %w", err)
	}
	return parseRPCBalance(body)
}

// parseRPCBalance reads the wei quantity out of a JSON-RPC envelope.
//
// The error member is checked explicitly because a JSON-RPC node reports
// failures with HTTP 200 and an `error` object — an archive-state refusal
// arrives that way — so trusting the status line would turn a refusal into an
// empty result and, downstream, into a zero balance.
func parseRPCBalance(body []byte) (*big.Int, error) {
	var env struct {
		Result string `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, fmt.Errorf("decoding the RPC response: %w", err)
	}
	if env.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", env.Error.Code, env.Error.Message)
	}
	if env.Result == "" {
		return nil, fmt.Errorf("RPC response carried no result")
	}
	wei, ok := new(big.Int).SetString(strings.TrimPrefix(strings.ToLower(env.Result), "0x"), 16)
	if !ok {
		return nil, fmt.Errorf("eth_getBalance result %q is not a hex quantity", env.Result)
	}
	if wei.Sign() < 0 {
		return nil, fmt.Errorf("eth_getBalance result %q is negative", env.Result)
	}
	return wei, nil
}

// Register this intent's ground-truth fetcher so a spec row can name it.
// Lives here rather than in the registry file so the factory sits beside the
// parsing code that understands this upstream's quirks.
//
// Parameters are unused today: the pinned block and addresses live in the key set.
func init() {
	RegisterTruthFactory("eth_rpc", func(map[string]any) (TruthSource, error) {
		return NewEthRPCSource(), nil
	})
}
