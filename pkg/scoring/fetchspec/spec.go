// Package fetchspec turns a declaration of WHERE a number lives into that
// number, for both halves of the deterministic scoring path.
//
// The two halves are the same shape. Ground truth is "call
// api.geckoterminal.com, read data.attributes.reserve_in_usd"; a miner answer
// is "call omni-chat/truth/liquidity-gecko, read liquidity_usd_cents". Until
// now both were hand-written Go — nine truth fetchers and twenty-seven
// per-miner parsers — which does not scale past the fifty-one miners we have,
// let alone the thirteen hundred registered.
//
// So the grammar is shared and the AUTHORS are not. An intent truth spec is
// written by the operator and stored in intent_specs; a miner spec is written
// by the miner and stored in miner_registry.yaml_raw. Keeping them in separate
// rows is not tidiness — if a miner could supply the truth spec it would be
// marking its own exam, which is exactly the oracle problem that makes
// nineteen of our fifty-one miners' perfect scores meaningless.
//
// What is NOT expressible here is deliberate. A Spec says which fetcher to
// use, not how that fetcher behaves: CAISO answering a malformed timestamp
// with HTTP 200 and a ZIP containing an error document is knowledge with a
// test behind it, not configuration. A new instance of a known format is a
// row; a genuinely new format is a reviewed Go change.
package fetchspec

import (
	"fmt"
	"sort"
	"strings"
)

// Spec declares where a number is and what to do to it.
//
// The zero value is not usable: Validate rejects it. That is deliberate — a
// spec that silently defaulted would produce a plausible number from the wrong
// place, which is the failure mode this package exists to remove.
type Spec struct {
	// Fetcher names the transport. Empty means HTTPJSON, which covers seven of
	// our nine truth sources and every miner we have seen.
	Fetcher string `yaml:"fetcher" json:"fetcher"`

	// URL to call. {placeholders} are substituted from the question key
	// parameters before the call.
	URL string `yaml:"url" json:"url"`

	// Method defaults to GET. POST carries Body.
	Method string `yaml:"method" json:"method"`
	Body   string `yaml:"body" json:"body"`

	// Headers sent with the request. Some upstreams need one to answer at all:
	// Cloudflare DNS returns HTTP 400 without Accept: application/dns-json,
	// and digitraffic refuses a request that does not accept gzip.
	Headers map[string]string `yaml:"headers" json:"headers"`

	// Field is the dotted path to the value: "data.attributes.reserve_in_usd",
	// or "1.0.value" where a segment indexes an array.
	Field string `yaml:"field" json:"field"`

	// Verbs are applied in order after the field is read.
	Verbs []Verb `yaml:"verbs" json:"verbs"`

	// Asserts are checked BEFORE the value is accepted. A failed assert means
	// the response is not an answer to the question asked, so it must be
	// rejected rather than scored — an Open Prices record in USD compared
	// against a EUR truth is a silent wrong answer, not a bad miner.
	Asserts []Assert `yaml:"asserts" json:"asserts"`

	// Echo names request parameters the response must echo back unchanged.
	//
	// This is the guard for the sharpest failure we have found. Three of our
	// miners carry a REGISTERED DEFAULT for their key parameter, so a request
	// that loses it still returns a well-formed, entirely valid answer — to a
	// different question. The GRID default hour is 2999 bps from the pinned
	// one, so an honest miner would be scored as a liar for our own bug.
	Echo map[string]string `yaml:"echo" json:"echo"`

	// Params carries fetcher-specific settings (a CSV column, a contract
	// method, a timestamp layout). Kept opaque so a new fetcher needs no
	// schema change.
	Params map[string]any `yaml:"params" json:"params"`
}

// Verb is one transform applied to the raw field value.
type Verb struct {
	// Op is one of the six operations below. They are not invented: each came
	// from a real miner or truth source during the nine ports.
	//
	//	string_value   the number arrives quoted ("106046544.1906")
	//	invert         1/x, for a rate quoted the other way round
	//	divide_by      rescale, e.g. a feed reporting at 8 decimals into 6
	//	multiply_by    the same in the other direction
	//	array_index    descend into a list before reading the rest of the path
	//	decimal_comma  "1,23" means 1.23
	Op string `yaml:"op" json:"op"`

	// Arg is the operand for divide_by / multiply_by / array_index.
	Arg float64 `yaml:"arg" json:"arg"`
}

// Assert requires a field to equal an expected value.
type Assert struct {
	Field  string `yaml:"field" json:"field"`
	Equals string `yaml:"equals" json:"equals"`
}

// Known fetcher names.
const (
	HTTPJSON = "http_json" // GET/POST returning JSON
	ZipCSV   = "zip_csv"   // a ZIP containing a CSV (CAISO OASIS)
	EthCall  = "eth_call"  // an EVM contract read (Chainlink)
	JSONRPC  = "json_rpc"  // JSON-RPC over HTTP (Ethereum nodes)
)

// Known verb operations.
const (
	OpStringValue  = "string_value"
	OpInvert       = "invert"
	OpDivideBy     = "divide_by"
	OpMultiplyBy   = "multiply_by"
	OpArrayIndex   = "array_index"
	OpDecimalComma = "decimal_comma"

	// OpEqualsZero maps 0 to 1 and everything else to 0.
	//
	// For a pass_fail intent whose upstream reports SUCCESS as zero. DNS is
	// the case that forces it: a DoH resolver answers with an rcode, where 0
	// (NOERROR) means the record exists — so the raw field is inverted
	// relative to the flag the comparator wants, and no arithmetic verb
	// expresses it. OpInvert is 1/x, which is undefined at exactly the value
	// that matters here.
	OpEqualsZero = "equals_zero"
)

var knownFetchers = map[string]bool{
	HTTPJSON: true, ZipCSV: true, EthCall: true, JSONRPC: true,
}

var knownOps = map[string]bool{
	OpStringValue: true, OpInvert: true, OpDivideBy: true,
	OpMultiplyBy: true, OpArrayIndex: true, OpDecimalComma: true,
	OpEqualsZero: true,
}

// Validate reports whether the spec is usable, naming what is wrong.
//
// Called at registration rather than at scoring time on purpose. A spec that
// is wrong at scoring time has already been accepted and the miner it belongs
// to is already being dispatched to, so the error then looks like a bad miner
// rather than a bad declaration.
func (s Spec) Validate() error {
	if strings.TrimSpace(s.URL) == "" {
		return fmt.Errorf("fetchspec: url is empty; there is nothing to call")
	}
	if strings.TrimSpace(s.Field) == "" {
		return fmt.Errorf("fetchspec: field is empty; a response with no named value cannot be scored")
	}
	f := s.FetcherOrDefault()
	if !knownFetchers[f] {
		return fmt.Errorf("fetchspec: unknown fetcher %q (known: %s)", f, knownList(knownFetchers))
	}
	for i, v := range s.Verbs {
		if !knownOps[v.Op] {
			return fmt.Errorf("fetchspec: verb %d has unknown op %q (known: %s)", i, v.Op, knownList(knownOps))
		}
		switch v.Op {
		case OpDivideBy, OpMultiplyBy:
			if v.Arg == 0 {
				// Dividing by zero is an obvious failure; multiplying by zero
				// is a silent one that collapses every value to the same
				// number and makes every miner look identical.
				return fmt.Errorf("fetchspec: verb %d (%s) has arg 0", i, v.Op)
			}
		case OpArrayIndex:
			if v.Arg < 0 {
				return fmt.Errorf("fetchspec: verb %d (array_index) has negative arg %g", i, v.Arg)
			}
		}
	}
	for i, a := range s.Asserts {
		if strings.TrimSpace(a.Field) == "" {
			return fmt.Errorf("fetchspec: assert %d has no field", i)
		}
	}
	return nil
}

// FetcherOrDefault returns the fetcher name, defaulting to HTTPJSON.
func (s Spec) FetcherOrDefault() string {
	if strings.TrimSpace(s.Fetcher) == "" {
		return HTTPJSON
	}
	return s.Fetcher
}

// MethodOrDefault returns the HTTP method, defaulting to GET.
func (s Spec) MethodOrDefault() string {
	if strings.TrimSpace(s.Method) == "" {
		return "GET"
	}
	return strings.ToUpper(s.Method)
}

func knownList(m map[string]bool) string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return strings.Join(out, ", ")
}
