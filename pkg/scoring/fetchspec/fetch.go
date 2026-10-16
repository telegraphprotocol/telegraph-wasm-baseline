package fetchspec

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// maxBody caps a response read. Without it a misbehaving upstream can hand the
// node an unbounded stream — and one of our registered "miners" was a 21MB
// blocklist that passed validation precisely because nobody looked at size.
const maxBody = 8 << 20

// DefaultTimeout bounds one fetch. An epoch cannot wait on a hung upstream:
// a truth fetch that never returns stalls scoring for every miner on that
// intent, not just the slow one.
const DefaultTimeout = 30 * time.Second

// Fetcher performs one request and returns the raw body.
//
// Separate from Extract so that the same JSON-reading logic serves every
// transport: a ZIP of CSV is converted to JSON here, and everything downstream
// is identical whether the bytes came from a REST call or a contract read.
type Fetcher interface {
	Fetch(ctx context.Context, spec Spec, params map[string]string) ([]byte, error)
}

// fetchers is the registry of transports a spec may name.
//
// Code, not data, and that boundary is the design. Which URL to call is
// configuration; how CAISO signals an error is not. OASIS answers a malformed
// timestamp with HTTP 200 and a valid ZIP whose single entry is an error
// document rather than a price list — no status code, no exception, just no
// price. That is knowledge with a test behind it.
//
// Four entries cover all nine of our truth sources and every miner we have
// seen. Adding one is the rare case and should be reviewed.
var fetchers = map[string]Fetcher{}

// Register adds a transport. Called from each fetcher's own file.
func Register(name string, f Fetcher) { fetchers[name] = f }

// Fetch resolves a spec against the network and returns the extracted number.
//
// params supply {placeholder} substitutions in the URL and body, and are also
// what the Echo check compares the response against.
//
// The HTTP client comes from the context via WithClient rather than an
// argument, so that a caller cannot pass one here and have it silently ignored
// by a fetcher that builds its own.
func Fetch(ctx context.Context, spec Spec, params map[string]string) (float64, error) {
	if err := spec.Validate(); err != nil {
		return 0, err
	}
	f, ok := fetchers[spec.FetcherOrDefault()]
	if !ok {
		return 0, fmt.Errorf("fetchspec: fetcher %q is named but not registered", spec.FetcherOrDefault())
	}

	resolved := spec
	resolved.URL = substitute(spec.URL, params, true)
	resolved.Body = substitute(spec.Body, params, false)

	// The field path is substituted too, because some upstreams key the answer
	// by the thing being asked about rather than returning it at a fixed path.
	// Frankfurter answers USD->EUR as {"rates":{"EUR":0.877}}, so the path is
	// rates.{quote} and varies per question key.
	//
	// Not URL-escaped: this is a lookup path into a decoded document, never
	// part of a request, so percent-encoding it would make "rates.EUR" fail to
	// match the key "EUR". A field path that still holds a placeholder is
	// caught below rather than looked up literally — an unsubstituted path
	// misses and reports "field not found", which reads like an upstream
	// change rather than a spec with a missing parameter.
	resolved.Field = substitute(spec.Field, params, false)

	if left := unresolved(resolved.URL); left != "" {
		// A placeholder that survives substitution would be sent literally and
		// the upstream would answer for the wrong thing, or 404. Naming it is
		// far cheaper than debugging the response.
		return 0, fmt.Errorf("fetchspec: url still contains %s after substitution; no such parameter was supplied", left)
	}
	if left := unresolved(resolved.Field); left != "" {
		return 0, fmt.Errorf("fetchspec: field path still contains %s after substitution; no such parameter was supplied", left)
	}

	body, err := f.Fetch(ctx, resolved, params)
	if err != nil {
		return 0, err
	}
	return Extract(resolved, body)
}

// substitute replaces {name} with the parameter value.
//
// escape is true for URLs so a parameter containing & or / cannot inject extra
// query arguments — a miner-supplied value reaches this path.
func substitute(s string, params map[string]string, escape bool) string {
	if s == "" || len(params) == 0 {
		return s
	}
	for k, v := range params {
		if escape {
			v = escapeParam(v)
		}
		s = strings.ReplaceAll(s, "{"+k+"}", v)
	}
	return s
}

// escapeParam makes a parameter safe to splice into a URL.
//
// url.PathEscape is NOT sufficient on its own: it leaves & ? and = untouched,
// because they are legal inside a path segment. Spliced into a URL that
// already has a query string, a miner-supplied value containing "&admin=true"
// would append a real query argument. Parameters reach this path from miner
// registration data, so that is an injection, not a hypothetical.
//
// The three separators are then restored because several upstreams reject
// their encoded forms: OSRM needs "lon,lat;lon,lat" and CAISO OASIS needs a
// literal colon in its timestamps.
func escapeParam(v string) string {
	v = url.PathEscape(v)
	for from, to := range map[string]string{
		"&": "%26",
		"?": "%3F",
		"=": "%3D",
	} {
		v = strings.ReplaceAll(v, from, to)
	}
	for from, to := range map[string]string{
		"%2C": ",",
		"%3B": ";",
		"%3A": ":",
	} {
		v = strings.ReplaceAll(v, from, to)
	}
	return v
}

// unresolved returns the first {placeholder} left in s, or "".
func unresolved(s string) string {
	i := strings.Index(s, "{")
	if i < 0 {
		return ""
	}
	j := strings.Index(s[i:], "}")
	if j < 0 {
		return ""
	}
	return s[i : i+j+1]
}

// httpJSON is the ordinary case: call a URL, get JSON back.
//
// Seven of our nine truth sources and every miner we have probed are this.
type httpJSON struct{}

func (httpJSON) Fetch(ctx context.Context, spec Spec, _ map[string]string) ([]byte, error) {
	var bodyReader io.Reader
	if spec.Body != "" {
		bodyReader = strings.NewReader(spec.Body)
	}
	req, err := http.NewRequestWithContext(ctx, spec.MethodOrDefault(), spec.URL, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("fetchspec: building the request: %w", err)
	}

	// A sensible default that several upstreams need, overridable per spec.
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "TelegraphNode/1.0")
	if spec.Body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range spec.Headers {
		req.Header.Set(k, v)
	}

	resp, err := clientFor(ctx).Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetchspec: fetching %s: %w", redact(spec.URL), err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		// Distinguished from other failures because the caller can retry it,
		// and because several upstreams throttle on burst rather than volume.
		return nil, fmt.Errorf("fetchspec: %s rate-limited the request (429)", redact(spec.URL))
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetchspec: %s returned %s", redact(spec.URL), resp.Status)
	}

	b, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		return nil, fmt.Errorf("fetchspec: reading the response: %w", err)
	}
	if len(b) == 0 {
		return nil, fmt.Errorf("fetchspec: %s returned an empty body with a 200 status", redact(spec.URL))
	}
	return b, nil
}

type ctxKey struct{}

// WithClient pins the HTTP client used by fetches on this context, so tests
// can serve fixtures without reaching the network.
func WithClient(ctx context.Context, c *http.Client) context.Context {
	return context.WithValue(ctx, ctxKey{}, c)
}

func clientFor(ctx context.Context) *http.Client {
	if c, ok := ctx.Value(ctxKey{}).(*http.Client); ok && c != nil {
		return c
	}
	return &http.Client{Timeout: DefaultTimeout}
}

// redact strips the query string from a URL before it reaches a log or an
// error. Query strings carry API keys for the paid sources.
func redact(u string) string {
	if i := strings.Index(u, "?"); i > 0 {
		return u[:i] + "?..."
	}
	return u
}

func init() {
	Register(HTTPJSON, httpJSON{})
}
