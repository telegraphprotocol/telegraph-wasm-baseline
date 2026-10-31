package responseparser

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
)

// ParseAll reads a miner's WHOLE answer into the intent's format: every field,
// in the format's order. This is what makes an intent's answer the same shape
// whichever miner produced it - the routed ask reply a user receives is this
// object, not the miner's raw response.
//
// It is the one-value path (Parse) run once per field, not one call asking the
// model for the whole object: the experiments that chose the model found it
// consistent only when it copies ONE value, so each field gets its own call
// with the tuned prompt, and code does the verbatim guard and the conversion
// exactly as on the scoring path. The calls for one answer run with bounded
// concurrency (MaxFieldConcurrency, under the LLM client's own lane limit)
// and share the parser's reply cache.
//
// A field that cannot be read does not fail the answer; it is reported with
// its status (missing / rejected / unavailable) and a null value, so the
// object's shape never changes.

// Field statuses in a Shaped answer.
const (
	FieldOK          = "ok"          // read, guarded, converted, in range
	FieldMissing     = "missing"     // the model says the value is not in the answer
	FieldRejected    = "rejected"    // a reply exists but is unusable (see Reason)
	FieldUnavailable = "unavailable" // the model never answered (down, timed out, answer too long)
)

// MaxFieldConcurrency bounds the parallel model calls for ONE answer's fields.
// The LLM client's fast lane bounds the node as a whole; this keeps one
// answer from taking the whole lane.
const MaxFieldConcurrency = 4

// FieldStatusOf classifies a per-field error.
func FieldStatusOf(err error) string {
	switch {
	case err == nil:
		return FieldOK
	case errors.Is(err, ErrNotFound):
		return FieldMissing
	case errors.Is(err, ErrUnavailable):
		return FieldUnavailable
	default:
		return FieldRejected
	}
}

// ShapeRequest is one answer to read into its intent's format.
type ShapeRequest struct {
	IntentID string
	Body     string
	Format   *Format

	// User selects the descriptions shown to the model. true: the user view
	// (Field.ForUser: user_description, or the description plus the query),
	// with Query filling {query}. false: the scoring view, with {key} and
	// {<param>} filled from Key / KeyParams exactly as Parse does.
	User      bool
	Query     string
	Key       string
	KeyParams map[string]string

	// Unit is the unit the miner declares it quotes the SCORED field in
	// ("" = the field's own). Other fields are always read in their own unit.
	Unit string

	// Timeout bounds the whole answer (all fields). 0 = the parser's user
	// timeout when User, else its scoring timeout.
	Timeout time.Duration

	// Slug is for logs only.
	Slug string
}

// FieldValue is one field of a shaped answer.
type FieldValue struct {
	Name   string
	Type   string
	Unit   string
	Status string
	// Reason says why a field is rejected / unavailable (logs and audit only).
	Reason string
	// Value is nil unless Status is ok: float64 (number), int64 (integer),
	// bool, string (string / enum / date / datetime), []string (list) or
	// map[string]any (object).
	Value any
	// Copied is the text the model copied, verbatim from the answer.
	Copied string
	Cached bool
}

// Shaped is a whole answer read into the intent's format, fields in order.
type Shaped struct {
	Fields []FieldValue
}

// Count returns how many fields have the given status.
func (s Shaped) Count(status string) int {
	n := 0
	for _, f := range s.Fields {
		if f.Status == status {
			n++
		}
	}
	return n
}

// Object is the answer as the ordered {field: value} object users receive.
// Fields that were not read are present with a null value.
func (s Shaped) Object() OrderedObject {
	out := make(OrderedObject, 0, len(s.Fields))
	for _, f := range s.Fields {
		out = append(out, KV{Key: f.Name, Value: f.Value})
	}
	return out
}

// Statuses is the per-field status object, in the format's order.
func (s Shaped) Statuses() OrderedObject {
	out := make(OrderedObject, 0, len(s.Fields))
	for _, f := range s.Fields {
		out = append(out, KV{Key: f.Name, Value: f.Status})
	}
	return out
}

// KV is one key of an OrderedObject.
type KV struct {
	Key   string
	Value any
}

// OrderedObject is a JSON object whose keys keep their order. The intent's
// format fixes the order, and every miner's answer is written in it.
type OrderedObject []KV

// MarshalJSON writes the object in order, without HTML escaping, so "<", ">"
// and "&" in a copied value read as written.
func (o OrderedObject) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, kv := range o {
		if i > 0 {
			buf.WriteByte(',')
		}
		k, err := marshalNoEscape(kv.Key)
		if err != nil {
			return nil, err
		}
		buf.Write(k)
		buf.WriteByte(':')
		v, err := marshalNoEscape(kv.Value)
		if err != nil {
			return nil, fmt.Errorf("field %q: %w", kv.Key, err)
		}
		buf.Write(v)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// Get returns the value under key.
func (o OrderedObject) Get(key string) (any, bool) {
	for _, kv := range o {
		if kv.Key == key {
			return kv.Value, true
		}
	}
	return nil, false
}

// Keys returns the keys in order.
func (o OrderedObject) Keys() []string {
	out := make([]string, len(o))
	for i, kv := range o {
		out[i] = kv.Key
	}
	return out
}

func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

// ParseAll reads every field of req.Format out of req.Body. The only error is
// a request that cannot be shaped at all (no format); every per-field outcome
// is in the result.
func (p *Parser) ParseAll(ctx context.Context, req ShapeRequest) (Shaped, error) {
	if req.Format == nil || len(req.Format.Fields) == 0 {
		return Shaped{}, fmt.Errorf("intent %s has no response format", req.IntentID)
	}
	out := Shaped{Fields: make([]FieldValue, len(req.Format.Fields))}
	for i, fl := range req.Format.Fields {
		out.Fields[i] = FieldValue{Name: fl.Name, Type: fl.Type, Unit: fl.Unit}
	}

	body := strings.TrimSpace(req.Body)
	if body == "" || body == "null" {
		// Nothing to read; no reason to ask the model.
		for i := range out.Fields {
			out.Fields[i].Status, out.Fields[i].Reason = FieldMissing, "empty answer"
		}
		p.logShape(req, out)
		return out, nil
	}

	timeout := req.Timeout
	if timeout <= 0 {
		if req.User {
			timeout = p.UserTimeout()
		} else {
			timeout = p.ScoringTimeout()
		}
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	scoredIdx := req.Format.ScoredIndex()
	params := req.Format.ParamsFor(req.Key, req.KeyParams)

	limit := MaxFieldConcurrency
	if n := len(req.Format.Fields); n < limit {
		limit = n
	}
	sem := make(chan struct{}, limit)
	var wg sync.WaitGroup
	for i := range req.Format.Fields {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				out.Fields[i].Status = FieldUnavailable
				out.Fields[i].Reason = fmt.Sprintf("%v: %v", ErrUnavailable, ctx.Err())
				return
			}
			defer func() { <-sem }()
			unit := ""
			if i == scoredIdx {
				unit = req.Unit
			}
			fv := p.parseField(ctx, req, req.Format.Fields[i], unit, params)
			out.Fields[i] = fv
		}(i)
	}
	wg.Wait()
	p.logShape(req, out)
	return out, nil
}

// parseField reads one field: build the view the model sees, ask (or hit the
// cache), then guard and convert in code.
func (p *Parser) parseField(ctx context.Context, req ShapeRequest, fl Field, declaredUnit string, params map[string]string) FieldValue {
	fv := FieldValue{Name: fl.Name, Type: fl.Type, Unit: fl.Unit}
	fail := func(err error) FieldValue {
		fv.Status, fv.Reason, fv.Value = FieldStatusOf(err), err.Error(), nil
		return fv
	}

	var view Field
	var k string
	conv := fl // the field conversion runs against: the filled scoring field, or the raw one
	if req.User {
		view = fl.ForUser(req.Query, declaredUnit)
		// Same key shape as the user path always used, so the scored field's
		// reply is shared with any earlier read of it.
		k = cacheKey(req.IntentID, "\x01user", view.Name+"|unit="+view.Unit+"|desc="+view.Description, req.Body)
	} else {
		filled := fl.Fill(req.Key, params)
		unit := FillText(declaredUnit, req.Key, params)
		if m := placeholderRE.FindString(filled.Description + " " + filled.Unit); m != "" {
			return fail(fmt.Errorf("%w: placeholder %s is not filled for key %q", ErrUnavailable, m, req.Key))
		}
		view = filled.PromptField(unit)
		conv, declaredUnit = filled, unit
		// The scoring path's own key shape (Parse), so the scored field is
		// shared with the comparator's read of the same response.
		k = cacheKey(req.IntentID, req.Key, view.Name+"|unit="+view.Unit, req.Body)
	}

	if fl.Processed {
		return p.processField(ctx, req, view, conv, declaredUnit)
	}

	reply, cached, err := p.reply(ctx, k, view, req.Body)
	fv.Cached = cached
	if err != nil {
		return fail(err)
	}
	copied, ok := ParseReply(reply)
	if !ok {
		return fail(ErrBadReply)
	}
	if copied.Kind == CopiedNone {
		return fail(ErrNotFound)
	}
	fv.Copied = copiedText(copied)
	v, err := ConvertValue(conv, copied, req.Body, declaredUnit)
	if err != nil {
		return fail(err)
	}
	fv.Status, fv.Value = FieldOK, v
	return fv
}

// processField reads a processed field (Field.Processed): the model works the
// value out of the answer for the request and cites the values it used. Code
// checks each cited value is in the answer (cited), then converts and
// range-checks the value like any other. It checks the evidence exists, not
// the model's arithmetic.
func (p *Parser) processField(ctx context.Context, req ShapeRequest, view, conv Field, unit string) FieldValue {
	fv := FieldValue{Name: conv.Name, Type: conv.Type, Unit: conv.Unit}
	fail := func(err error) FieldValue {
		fv.Status, fv.Reason, fv.Value = FieldStatusOf(err), err.Error(), nil
		return fv
	}
	request := req.Query
	if strings.TrimSpace(request) == "" {
		request = req.Key
	}
	// The request is part of the prompt, so it is part of the key: the same
	// answer read for another request is another reply.
	k := cacheKey(req.IntentID, "\x02processed\x00"+request,
		view.Name+"|unit="+view.Unit+"|desc="+view.Description, req.Body)
	reply, cached, err := p.ask(ctx, k, req.Body, ProcessSystemPrompt,
		ProcessUserMessage(view, req.Body, request), ProcessMaxTokens)
	fv.Cached = cached
	if err != nil {
		return fail(err)
	}
	c, from, ok := ParseProcessedReply(reply)
	if !ok {
		return fail(ErrBadReply)
	}
	if c.Kind == CopiedNone {
		return fail(ErrNotFound)
	}
	fv.Copied = copiedText(c)
	if c.Kind != CopiedText {
		return fail(fmt.Errorf("%w: %s field got a %s", ErrWrongShape, conv.Type, kindName(c.Kind)))
	}
	if len(from) == 0 && conv.Type != TypeString && conv.Type != TypeEnum {
		// "true" or "0" alone matches almost any JSON answer: a worked-out
		// number or yes/no must name what it rests on.
		return fail(fmt.Errorf("%w: a worked-out %s cites no values", ErrNotVerbatim, conv.Type))
	}
	if err := cited(c.Text, from, req.Body, request); err != nil {
		return fail(err)
	}
	v, err := ConvertProcessed(conv, c.Text, unit)
	if err != nil {
		return fail(err)
	}
	fv.Status, fv.Value = FieldOK, v
	return fv
}

// cited checks a processed value's evidence: every value it cites must be in
// the answer or the request (a threshold the request sets), and at least one
// of them (or the value itself, when it cites none) must be in the answer but
// not in the request. The request alone is never evidence, the caller writes
// it, and an answer that only echoes it (an error body repeating the
// question) backs nothing (reviews F, #230).
func cited(value string, from []string, body, request string) error {
	ownWords := func(s string) bool { return Verbatim(s, body) && !Verbatim(s, request) }
	if len(from) == 0 {
		if ownWords(value) {
			return nil
		}
		return fmt.Errorf("%w: %q is not in the answer's own words and cites no values", ErrNotVerbatim, value)
	}
	backed := false
	for _, f := range from {
		if !Verbatim(f, body) && !Verbatim(f, request) {
			return fmt.Errorf("%w: cited value %q is not in the answer", ErrNotVerbatim, f)
		}
		backed = backed || ownWords(f)
	}
	if !backed {
		return fmt.Errorf("%w: every cited value only repeats the request", ErrNotVerbatim)
	}
	return nil
}

// ConvertProcessed turns a processed value into the field's typed value: the
// conversions ConvertValue makes for one value (an enum's allowed values),
// without its verbatim check (cited guards a worked-out value).
func ConvertProcessed(fl Field, text, unit string) (any, error) {
	if !processable(fl.Type) {
		return nil, fmt.Errorf("%w: a %s field cannot be processed", ErrWrongShape, fl.Type)
	}
	if fl.Type == TypeBool {
		// The model replies true or false (ProcessSystemPrompt); true_when is
		// for copied text, and anything else is not an answer.
		switch strings.ToLower(strings.TrimSpace(text)) {
		case "true":
			return true, nil
		case "false":
			return false, nil
		}
		return nil, fmt.Errorf("%w: %q is not true or false", ErrConvert, text)
	}
	return convertOne(fl, text, unit)
}

// ConvertValue turns what the model copied into the field's typed value. It is
// where the verbatim guard runs for every field type, and where all
// conversion lives (the model never converts; a processed value is its own
// working, converted by ConvertProcessed):
//
//	number   : Convert (unit alias or inverse, then range) -> float64
//	integer  : a whole number, range-checked -> int64
//	bool     : true iff the copied text is one of true_when -> bool
//	enum     : the allowed value the copied text names (case-insensitive) -> string
//	string, date, datetime : the copied text -> string
//	list     : every item verbatim -> []string (a single copied value is a one-item list)
//	object   : every value verbatim -> map[string]any (null kept)
func ConvertValue(fl Field, c Copied, body, unit string) (any, error) {
	switch fl.Type {
	case TypeList:
		var items []string
		switch c.Kind {
		case CopiedList:
			items = c.List
		case CopiedText:
			items = []string{c.Text}
		default:
			return nil, fmt.Errorf("%w: %s field got an object", ErrWrongShape, fl.Type)
		}
		out := make([]string, 0, len(items))
		for _, it := range items {
			if strings.TrimSpace(it) == "" {
				continue
			}
			if !Verbatim(it, body) {
				return nil, fmt.Errorf("%w: list item %q", ErrNotVerbatim, it)
			}
			out = append(out, it)
		}
		if len(out) == 0 {
			return nil, ErrNotFound
		}
		return out, nil

	case TypeObject:
		if c.Kind != CopiedObject {
			return nil, fmt.Errorf("%w: object field got a %s", ErrWrongShape, kindName(c.Kind))
		}
		out := make(map[string]any, len(c.Object))
		for key, val := range c.Object {
			if val == nil {
				out[key] = nil
				continue
			}
			if !Verbatim(*val, body) {
				return nil, fmt.Errorf("%w: object value %s=%q", ErrNotVerbatim, key, *val)
			}
			out[key] = *val
		}
		return out, nil
	}

	// Every other type is one copied value.
	if c.Kind != CopiedText {
		return nil, fmt.Errorf("%w: %s field got a %s", ErrWrongShape, fl.Type, kindName(c.Kind))
	}
	text := c.Text
	if !Verbatim(text, body) {
		return nil, fmt.Errorf("%w: %q", ErrNotVerbatim, text)
	}
	return convertOne(fl, text, unit)
}

// convertOne converts one value by the field's type.
func convertOne(fl Field, text, unit string) (any, error) {
	switch fl.Type {
	case TypeNumber:
		return Convert(fl, text, unit)
	case TypeBool:
		x, err := Convert(fl, text, unit)
		return x == 1, err
	case TypeInteger:
		x, ok := ToNumber(text)
		if !ok {
			return nil, fmt.Errorf("%w: %q is not a number", ErrConvert, text)
		}
		if x != math.Trunc(x) || math.Abs(x) > 1<<53 {
			return nil, fmt.Errorf("%w: %q is not a whole number", ErrConvert, text)
		}
		if len(fl.Range) == 2 && (x < fl.Range[0] || x > fl.Range[1]) {
			return nil, fmt.Errorf("%w: %v is outside the plausible range [%v, %v]", ErrOutOfRange, x, fl.Range[0], fl.Range[1])
		}
		return int64(x), nil
	case TypeEnum:
		want := normEnum(text)
		for _, a := range fl.Allowed {
			if normEnum(a) == want {
				return a, nil
			}
		}
		return nil, fmt.Errorf("%w: %q (allowed: %s)", ErrNotAllowed, text, strings.Join(fl.Allowed, ", "))
	default: // string, date, datetime
		return text, nil
	}
}

// normEnum compares enum values loosely: case, surrounding space and trailing
// punctuation ("Approve." / "SAFE:") do not matter; the words do.
func normEnum(s string) string {
	return strings.ToLower(strings.Trim(strings.TrimSpace(s), ".,:;!\"'"))
}

func kindName(k CopiedKind) string {
	switch k {
	case CopiedText:
		return "single value"
	case CopiedList:
		return "list"
	case CopiedObject:
		return "object"
	default:
		return "null"
	}
}

// copiedText renders what the model copied, for the audit trail.
func copiedText(c Copied) string {
	switch c.Kind {
	case CopiedText:
		return c.Text
	case CopiedList:
		b, _ := json.Marshal(c.List)
		return string(b)
	case CopiedObject:
		b, _ := json.Marshal(c.Object)
		return string(b)
	default:
		return ""
	}
}

// ScoringTimeout is the bound on a scoring-path parse (PARSER_LLM_TIMEOUT_SECONDS).
func (p *Parser) ScoringTimeout() time.Duration {
	if p == nil || p.cfg.Timeout <= 0 {
		return DefaultTimeout
	}
	return p.cfg.Timeout
}

// logShape records one line per shaped answer with each field's status, and a
// warning per rejected field with the reason (the reply is in the per-field
// cache entry; the copied text is logged).
func (p *Parser) logShape(req ShapeRequest, s Shaped) {
	path := "scoring"
	if req.User {
		path = "user"
	}
	statuses := make([]string, 0, len(s.Fields))
	for _, f := range s.Fields {
		statuses = append(statuses, f.Name+"="+f.Status)
		if f.Status == FieldRejected {
			log.Warn().Str("path", path).Str("intent", req.IntentID).Str("miner", req.Slug).
				Str("field", f.Name).Str("copied", truncate(f.Copied, 200)).Str("reason", truncate(f.Reason, 300)).
				Msg("[response_parser] field rejected")
		}
	}
	log.Info().Str("path", path).Str("intent", req.IntentID).Str("miner", req.Slug).Str("model", p.cfg.Model).
		Int("ok", s.Count(FieldOK)).Int("fields", len(s.Fields)).Strs("status", statuses).
		Msg("[response_parser] answer shaped")
}
