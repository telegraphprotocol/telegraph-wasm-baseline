package comparator

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v3"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/intentformat"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
)

// The per-intent response parser (the intent's response format + a small
// model) is the first of the three parse paths in ParseMinerValue:
//
//  1. the intent has a format with a scored field AND a parser is installed
//     (PARSER_LLM_ENABLED=true)                    -> the parser model
//  2. else the miner declared a mapping in its YAML -> ParseWithMapping
//  3. else a hand-written parser for the slug       -> parsersByIntent
//
// Paths 2 and 3 stay until the parser has shown parity on live responses
// (run with PARSER_LLM_SHADOW=true first). With no parser installed the
// first path does not exist and scoring is exactly what it was.
//
// The format comes from FormatFor: the node's loaded intent format registry
// (pkg/scoring/intentformat: intent_response_formats, then
// intent_specs.response_format, then the family's) once one is installed, and
// the spec's own ResponseFormat before that.
var responseParser struct {
	sync.RWMutex
	p *responseparser.Parser
}

// UseResponseParser installs the parser model; nil removes it (path off).
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

// parseViaResponseFormat runs the parser model for an intent with a format.
// handled=false means this path does not apply (no parser, no format) or the
// model never answered (server down, timeout, body too large) - the caller
// then uses the older paths, so an outage degrades to today's behaviour rather
// than to scoring nobody. Every other failure is final: not scored.
func parseViaResponseFormat(ctx context.Context, intentID, slug, key, body string) (v float64, handled bool, err error) {
	p := activeResponseParser()
	if p == nil {
		return 0, false, nil
	}
	format, _, ok := FormatFor(intentID)
	if !ok || !format.HasScored() {
		return 0, false, nil
	}
	cfg, _ := ConfigFor(intentID)
	res, err := p.Parse(ctx, responseparser.Request{
		IntentID: intentID, Key: key, Body: body, Slug: slug,
		Format: format, KeyParams: cfg.KeyParams[key],
		Unit: MinerUnitFor(slug),
	})
	if p.Shadow() {
		// Shadow mode: the older paths decide; the parser's answer is only
		// logged next to theirs, which is how parity is measured live before
		// an intent is switched over.
		legacy, legacyErr := parseLegacy(intentID, slug, key, body)
		ev := log.Info().Str("intent", intentID).Str("miner", slug).Str("key", key).
			Str("reply", res.Reply).Str("copied", res.Copied)
		if err == nil {
			ev = ev.Float64("parser_value", res.Value)
		} else {
			ev = ev.Str("parser_error", err.Error())
		}
		if legacyErr == nil {
			ev = ev.Float64("legacy_value", legacy)
		} else {
			ev = ev.Str("legacy_error", legacyErr.Error())
		}
		ev.Bool("agree", err == nil && legacyErr == nil && sameValue(res.Value, legacy)).
			Msg("[response_parser] shadow comparison")
		return 0, false, nil
	}
	if err != nil {
		if responseparser.IsUnavailable(err) {
			log.Warn().Err(err).Str("intent", intentID).Str("miner", slug).Str("key", key).
				Msg("[response_parser] model unavailable — falling back to the declared mapping / hardcoded parser")
			return 0, false, nil
		}
		return 0, true, fmt.Errorf("parsing %s response via the %s response format: %w", slug, intentID, err)
	}
	return res.Value, true, nil
}

func sameValue(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	m := b
	if m < 0 {
		m = -m
	}
	if m < 1 {
		m = 1
	}
	return d <= 1e-6*m
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
// registration YAML. Only the unit is read: this deliberately does NOT
// activate the miner's declared mapping (LoadMinerMappings does that), so
// turning the parser on cannot change which mapping path other miners take.
// Per-slug errors are returned rather than failing the batch.
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

// ── adapter input ────────────────────────────────────────────────────────────

// AdapterInput is the text a judgment adapter (or a hybrid's wide half) scores
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
