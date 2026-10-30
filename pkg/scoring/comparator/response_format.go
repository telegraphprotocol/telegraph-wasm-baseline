package comparator

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/intentformat"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
)

// The per-intent response parser (the intent's response format + a small
// model) is the ONLY reader of a miner's answer (ParseMinerValue): no miner is
// named in code, so a new miner is read the same way as every other from its
// first answer. It needs a parser installed (PARSER_LLM_ENABLED=true) and a
// format with a scored field for the intent; without either, the intent's
// miners cannot be scored and are left out (never scored 0).
//
// The format comes from FormatFor: the node's loaded intent format registry
// (pkg/scoring/intentformat: intent_response_formats, then
// intent_specs.response_format, then the family's) once one is installed, and
// the spec's own ResponseFormat before that.
var responseParser struct {
	sync.RWMutex
	p *responseparser.Parser
}

// ErrNoParser: no parser model is installed, so a miner's answer cannot be
// read. The miner is left out of the question, not scored 0.
var ErrNoParser = errors.New("response parser not installed (PARSER_LLM_ENABLED): a miner's answer cannot be read")

// UseResponseParser installs the parser model; nil removes it.
func UseResponseParser(p *responseparser.Parser) {
	responseParser.Lock()
	defer responseParser.Unlock()
	responseParser.p = p
}

func activeResponseParser() *responseparser.Parser {
	responseParser.RLock()
	defer responseParser.RUnlock()
	return responseParser.p
}

// parseViaResponseFormat reads a miner's value with the parser model and the
// intent's format. Every failure is final for this question (not scored,
// never 0): no parser, no format, the model never answered (server down,
// timeout, body too large), or no usable value in the reply.
func parseViaResponseFormat(ctx context.Context, intentID, slug, key, body string) (float64, error) {
	p := activeResponseParser()
	if p == nil {
		return 0, ErrNoParser
	}
	format, _, ok := FormatFor(intentID)
	if !ok || !format.HasScored() {
		return 0, fmt.Errorf("intent %s has no response format with a scored field: miner %q's answer cannot be read", intentID, slug)
	}
	cfg, _ := ConfigFor(intentID)
	res, err := p.Parse(ctx, responseparser.Request{
		IntentID: intentID, Key: key, Body: body, Slug: slug,
		Format: format, KeyParams: cfg.KeyParams[key],
		Unit: MinerUnitFor(slug),
	})
	if err != nil {
		return 0, fmt.Errorf("parsing %s response via the %s response format: %w", slug, intentID, err)
	}
	return res.Value, nil
}

// ── miner-declared units ─────────────────────────────────────────────────────

// A number alone cannot say which way round an FX rate is quoted: HNB's
// 1,129800 is USD per EUR, Frankfurter's 0.8712 is EUR per USD. The intent's
// format lists the units it accepts (unit + unit_aliases); a miner that quotes
// in an alias says which one, in its registration YAML:
//
//	semantics:
//	  signal_mapping:
//	    unit: USD per {key}
//
// It is one line naming one of the intent's own units, not a mapping, and it
// is checked against the format at parse time (an unknown unit is not
// scored). Undeclared means the format's own unit.
var minerUnits struct {
	sync.RWMutex
	bySlug map[string]string
}

// minerYAML is the subset of a registration YAML (a miner's, or a truth
// source's) this package reads: the declared unit. Decoding only that means a
// miner adding an unrelated key cannot break scoring.
type minerYAML struct {
	Semantics struct {
		SignalMapping struct {
			Unit string `yaml:"unit"`
		} `yaml:"signal_mapping"`
	} `yaml:"semantics"`
}

// SetMinerUnit records the unit a miner quotes in; "" removes it.
func SetMinerUnit(slug, unit string) {
	minerUnits.Lock()
	defer minerUnits.Unlock()
	if minerUnits.bySlug == nil {
		minerUnits.bySlug = make(map[string]string)
	}
	if strings.TrimSpace(unit) == "" {
		delete(minerUnits.bySlug, slug)
		return
	}
	minerUnits.bySlug[slug] = strings.TrimSpace(unit)
}

// MinerUnitFor returns the unit a miner declared, or "".
func MinerUnitFor(slug string) string {
	minerUnits.RLock()
	defer minerUnits.RUnlock()
	return minerUnits.bySlug[slug]
}

// LoadMinerUnits reads semantics.signal_mapping.unit from each miner's
// registration YAML. Per-slug errors are returned rather than failing the
// batch.
func LoadMinerUnits(rawBySlug map[string]string) map[string]error {
	errs := make(map[string]error)
	for slug, raw := range rawBySlug {
		if strings.TrimSpace(raw) == "" {
			continue
		}
		var doc minerYAML
		if err := yaml.Unmarshal([]byte(raw), &doc); err != nil {
			errs[slug] = fmt.Errorf("registration YAML is not valid YAML: %w", err)
			continue
		}
		SetMinerUnit(slug, doc.Semantics.SignalMapping.Unit)
	}
	return errs
}

// ── the intent's format ──────────────────────────────────────────────────────

// FormatFor returns the response format that applies to an intent and its
// version. Once the node has installed its format registry
// (intentformat.Use, at startup) that is the only source: the intent's own row
// in intent_response_formats, else its intent_specs.response_format, else its
// family's. Before that (tests, tools, a node that has not loaded it) the
// intent spec's own ResponseFormat is used, which is exactly the old
// behaviour.
func FormatFor(intentID string) (*responseparser.Format, int, bool) {
	if intentformat.Loaded() {
		r, ok := intentformat.For(intentID)
		if !ok {
			return nil, 0, false
		}
		return r.Format, r.Version, true
	}
	cfg, ok := ConfigFor(intentID)
	if !ok || cfg.ResponseFormat == nil {
		return nil, 0, false
	}
	return cfg.ResponseFormat, intentformat.LegacyVersion, true
}

// ── user requests ────────────────────────────────────────────────────────────

// Result shapes of a routed ask reply.
const (
	// ShapeIntent: "result" is the intent-shaped object - the same keys, in
	// the same order, whichever miner answered.
	ShapeIntent = "intent"
	// ShapeRaw: the shape could not be produced; "result" is the miner's raw
	// answer and "shape_error" says why.
	ShapeRaw = "raw"
)

// Reasons a routed ask reply carries the raw answer (shape_error).
const (
	ShapeErrParserOff   = "parser_off"         // PARSER_LLM_ENABLED unset, or PARSER_LLM_USER_ENABLED=false
	ShapeErrNoFormat    = "no_format"          // the intent (and its family) has no response format
	ShapeErrUnavailable = "parser_unavailable" // the parser model did not answer (down, timed out, answer too long)
	ShapeErrNoFields    = "no_fields_found"    // the model answered, but no field could be read from the answer
	ShapeErrUnreadable  = "answer_unreadable"  // the miner's answer could not be rendered as text for the model
)

// UserShape is a routed ask's answer as the user receives it.
type UserShape struct {
	// Result is the intent-shaped object (responseparser.OrderedObject) when
	// Shape is ShapeIntent, else the miner's raw answer, untouched.
	Result any
	Shape  string
	// Error is set only when Shape is ShapeRaw.
	Error string
	// Version is the format's version (0 when there is no format).
	Version int
	// FieldStatus is each field's status (ok | missing | rejected |
	// unavailable) in the format's order; nil when Shape is ShapeRaw.
	FieldStatus responseparser.OrderedObject
	// Format is the intent whose format was used (the intent itself, or the
	// family it inherits from). "" when there is none.
	Format string
}

// RawShape is the reply for an answer that is not shaped.
func RawShape(raw any, reason string, version int) UserShape {
	return UserShape{Result: raw, Shape: ShapeRaw, Error: reason, Version: version}
}

// ShapeForUser reads the answer a user got into its intent's format: every
// field, by the parser model (one call per field, verbatim-guarded). raw is
// the miner's answer as the handler holds it and is returned untouched when
// no shape can be produced; body is that answer as the text the model reads
// and the guard checks - the same string for both.
//
// It never changes what is persisted or signed (the handlers commit to the
// raw answer before calling this), the HTTP status or settlement.
func ShapeForUser(ctx context.Context, intentID, slug, query, body string, raw any) UserShape {
	p := activeResponseParser()
	if p == nil || !p.UserEnabled() {
		return RawShape(raw, ShapeErrParserOff, 0)
	}
	format, version, ok := FormatFor(intentID)
	if strings.TrimSpace(intentID) == "" || !ok {
		return RawShape(raw, ShapeErrNoFormat, 0)
	}
	servedBy := intentID
	if r, ok := intentformat.For(intentID); ok {
		servedBy = r.ServedBy
	}
	shaped, err := p.ParseAll(ctx, responseparser.ShapeRequest{
		IntentID: intentID, Body: body, Format: format, User: true, Query: query,
		Unit: MinerUnitFor(slug), Slug: slug,
	})
	if err != nil {
		return RawShape(raw, ShapeErrNoFormat, version)
	}
	switch {
	case shaped.Count(responseparser.FieldUnavailable) == len(shaped.Fields):
		return RawShape(raw, ShapeErrUnavailable, version)
	case shaped.Count(responseparser.FieldOK) == 0:
		return RawShape(raw, ShapeErrNoFields, version)
	}
	return UserShape{
		Result:      shaped.Object(),
		Shape:       ShapeIntent,
		Version:     version,
		FieldStatus: shaped.Statuses(),
		Format:      servedBy,
	}
}

// AnswerText is a miner's answer as the text the parser model reads AND the
// verbatim guard checks - one string for both, so the guard checks exactly
// what the model saw. A decoded value is re-encoded as JSON, without HTML
// escaping so "<", ">" and "&" read as written; a string is used as is.
func AnswerText(data any) (string, bool) {
	switch v := data.(type) {
	case string:
		return v, true
	case []byte:
		return string(v), true
	case json.RawMessage:
		return string(v), true
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(data); err != nil {
		return "", false
	}
	return strings.TrimSuffix(buf.String(), "\n"), true
}

// ChainShape is a job's answer as its contract receives it: the processed
// answer ShapeForUser returns, in the format's fixed on-chain layout.
type ChainShape struct {
	Values  responseparser.OnChain
	Version int
	// Format is the intent whose format was used (the intent itself, or the
	// family it inherits from).
	Format string
}

// ShapeForChain reads a job's answer into its intent's format with the same
// parser model and calls as ShapeForUser, and lays it out for the contract
// (responseparser.EncodeOnChain): processed intelligence, never the raw
// answer (Ahmed, 2026-10-22).
//
// ok is false when there is nothing to deliver this way: the intent has no
// format, or its format cannot go on-chain (a number without decimals, an
// on-chain list or object). The job then keeps its miner's YAML mapping. With
// ok true, err says this answer cannot be delivered: a field was not read
// (the job tries its next miner instead of sending invented values), or the
// parser is off or did not answer, which wraps responseparser.ErrUnavailable:
// the node's failure, never a reason to fall back to a miner's own mapping,
// whose layout the contract does not know (Ahmed Q8).
func ShapeForChain(ctx context.Context, intentID, slug, query, body string) (ChainShape, bool, error) {
	if strings.TrimSpace(intentID) == "" {
		return ChainShape{}, false, nil
	}
	format, version, ok := FormatFor(intentID)
	if !ok || format.OnChainReady() != nil {
		return ChainShape{}, false, nil
	}
	servedBy := intentID
	if r, ok := intentformat.For(intentID); ok {
		servedBy = r.ServedBy
	}
	cs := ChainShape{Version: version, Format: servedBy}
	p := activeResponseParser()
	if p == nil || !p.UserEnabled() {
		return cs, true, fmt.Errorf("%w: the response parser is off (PARSER_LLM_ENABLED)", responseparser.ErrUnavailable)
	}
	if n := len(body); n > p.MaxBodyChars() {
		// The miner's answer, not the node: too long to read; the next miner.
		return cs, true, fmt.Errorf("the answer is %d chars, over PARSER_LLM_MAX_BODY_CHARS=%d", n, p.MaxBodyChars())
	}
	// No user waits on a job's answer, so its fields get the scoring timeout
	// (PARSER_LLM_TIMEOUT_SECONDS), not the API user's 10 s: a field the parser
	// did not read in time refuses the answer and, for every miner, fails the job.
	format = format.OnChainOnly()
	shaped, err := p.ParseAll(ctx, responseparser.ShapeRequest{
		IntentID: intentID, Body: body, Format: format, User: true, Query: query,
		Unit: MinerUnitFor(slug), Slug: slug, Timeout: p.ScoringTimeout(),
	})
	if err != nil {
		return cs, true, err
	}
	cs.Values, err = responseparser.EncodeOnChain(format, shaped)
	if err != nil {
		for i, fl := range format.Fields {
			if !fl.OffChain && i < len(shaped.Fields) && shaped.Fields[i].Status == responseparser.FieldUnavailable {
				return cs, true, fmt.Errorf("%w: %v", responseparser.ErrUnavailable, err)
			}
		}
	}
	return cs, true, err
}

// ── adapter input ────────────────────────────────────────────────────────────

// AdapterInput is the text a judgment adapter (or a hybrid's model half) scores
// for one miner answer. By DEFAULT it is the raw answer, unchanged: the
// adapters were trained on raw miner answers, and feeding them a reshaped
// object would change every judgment score.
//
// Only an intent whose format sets "adapter_input": "shaped" gets the shaped
// object instead, rendered as compact JSON text - and only when the parser is
// installed and at least one field was read; otherwise the raw answer.
// question fills {query} in the user-style field descriptions (an eval-pool
// question has no question key).
func AdapterInput(ctx context.Context, intentID, slug, question, body string) string {
	format, _, ok := FormatFor(intentID)
	if !ok || !format.ShapedAdapterInput() {
		return body
	}
	p := activeResponseParser()
	if p == nil {
		return body
	}
	shaped, err := p.ParseAll(ctx, responseparser.ShapeRequest{
		IntentID: intentID, Body: body, Format: format, User: true, Query: question,
		Unit: MinerUnitFor(slug), Slug: slug, Timeout: p.ScoringTimeout(),
	})
	if err != nil || shaped.Count(responseparser.FieldOK) == 0 {
		return body
	}
	b, err := shaped.Object().MarshalJSON()
	if err != nil {
		return body
	}
	return string(b)
}
