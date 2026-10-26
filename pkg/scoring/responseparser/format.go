// Package responseparser reads a miner's raw response into its intent's
// response format - a shape defined ONCE PER INTENT (intent_response_formats,
// or for older comparator intents intent_specs.response_format) - using a
// small language model, instead of a mapping written per miner.
//
// Two reads use it: Parse reads the one SCORED field for the comparator, and
// ParseAll reads every field into the ordered object a routed ask returns to
// its user, so an intent's answer has the same shape whichever miner gave it.
// ParseAll is Parse run once per field, never one call for the whole object.
//
// The split of work is the whole design and must not drift:
//
//	model : finds the one value a field describes and COPIES it exactly as
//	        written -> {"value": "<text>"} or {"value": null}
//	code  : everything else - the verbatim guard, number parsing, unit change
//	        (multiplier or inverse), bool mapping, range check
//
// Asking the model to convert as well made it inconsistent, so it never does.
// The prompt, the reply shape and the conversion are a port of
// intent_training_experiments/scripts/llm_pick_miner_responses.py, where the
// model (Qwen3-4B-Instruct-2507, greedy) and the prompt were chosen.
package responseparser

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Field types. Every type can be shaped for a user; only number and bool can
// be the scored field, because the comparator scores one number.
const (
	TypeNumber   = "number"
	TypeInteger  = "integer"
	TypeString   = "string"
	TypeEnum     = "enum"
	TypeBool     = "bool"
	TypeDate     = "date"
	TypeDatetime = "datetime"
	TypeList     = "list"
	TypeObject   = "object"
)

var knownTypes = map[string]bool{
	TypeNumber: true, TypeInteger: true, TypeString: true, TypeEnum: true, TypeBool: true,
	TypeDate: true, TypeDatetime: true, TypeList: true, TypeObject: true,
}

// UnitInverse is the unit_aliases value meaning "the value is quoted the other
// way round": the scored value is 1 / copied. FX miners quote either direction.
const UnitInverse = "inverse"

// Adapter input modes (Format.AdapterInput): what text a judgment / hybrid
// adapter scores. Raw is the default and the only mode the adapters were
// trained on.
const (
	AdapterInputRaw    = "raw"
	AdapterInputShaped = "shaped"
)

// MaxFields bounds a format's size. Every field is one model call per answer,
// so a large format multiplies the parser's cost and the user's wait.
const MaxFields = 8

// Format is an intent's response format: the shape of the answer every miner's
// response is read into, for users (the routed ask reply) and for scoring (the
// comparator reads the scored field). Stored in intent_response_formats.format
// and, for older comparator intents, intent_specs.response_format.
type Format struct {
	Fields []Field `json:"fields"`

	// AdapterInput chooses the text a judgment / hybrid adapter scores:
	// "" or "raw" (default) = the miner's raw answer text, exactly what the
	// adapters were trained on; "shaped" = the intent-shaped object rendered
	// as JSON text. Opt-in per format; changing it changes every judgment
	// score for the intent.
	AdapterInput string `json:"adapter_input,omitempty"`

	// Vars holds per-key description values that the truth fetcher does not
	// need (a pool's display name, a country's name, a route's end points), so
	// a description can say what the key stands for in the same words the
	// experiments used. Keyed by question key; merged over the key's
	// truth_params.key_params, Vars winning.
	Vars map[string]map[string]string `json:"vars,omitempty"`
}

// ParamsFor merges the key's truth_params.key_params with the format's own
// Vars for that key (Vars win). The result fills {<param>} placeholders.
func (f *Format) ParamsFor(key string, keyParams map[string]string) map[string]string {
	if f == nil || len(f.Vars[key]) == 0 {
		return keyParams
	}
	out := make(map[string]string, len(keyParams)+len(f.Vars[key]))
	for k, v := range keyParams {
		out[k] = v
	}
	for k, v := range f.Vars[key] {
		out[k] = v
	}
	return out
}

// Field is one value an intent's answer must yield.
type Field struct {
	Name string `json:"name"`

	// Description says which value is wanted. {key} is filled from the
	// question key, exactly like question_template; {<param>} is filled from
	// the key's truth_params.key_params entry when one exists.
	Description string `json:"description"`
	Type        string `json:"type"`

	// Unit is the unit the comparator wants the value in (before the intent's
	// scale). UnitAliases lists other units a miner may quote, each with how to
	// get from it to Unit: a multiplier, or "inverse".
	Unit        string                `json:"unit,omitempty"`
	UnitAliases map[string]UnitFactor `json:"unit_aliases,omitempty"`

	// Range is the plausible [min, max] in Unit. A value outside it is not
	// scored: it is the cheap guard against a value off by 100x.
	Range []float64 `json:"range,omitempty"`

	// TrueWhen lists the raw texts that mean true (bool fields only).
	TrueWhen []string `json:"true_when,omitempty"`

	// Allowed lists the permitted values of an enum field (shown to the model).
	Allowed []string `json:"allowed,omitempty"`

	// UserDescription replaces Description when the value is read out of a
	// USER request's answer (POST /engine/v1/ask, WS ask) rather than an
	// epoch question. A user request has no question key, so {key} and
	// {<param>} cannot be filled; the template's only placeholder is {query},
	// the user's own query text. Optional: absent, DefaultUserDescription is
	// used. Never shown on the scoring path, which keeps Description.
	UserDescription string `json:"user_description,omitempty"`

	// Scored marks the one field the comparator scores (number or bool).
	// At most one field may set it. A deterministic or hybrid intent must
	// have one; a judgment intent normally has none. A legacy
	// intent_specs.response_format with no field marked keeps its old rule:
	// the first field is the scored one (ParseFormat sets the flag on it).
	Scored bool `json:"scored,omitempty"`
}

// QueryPlaceholder is the one placeholder a user_description may use: the
// text of the user's query.
const QueryPlaceholder = "{query}"

// UnitFactor is how to convert from an alias unit to the field's unit: either
// a multiplier or the inverse.
type UnitFactor struct {
	Multiplier float64
	Inverse    bool
}

// UnmarshalJSON accepts a JSON number (multiplier) or the string "inverse".
func (u *UnitFactor) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		if strings.EqualFold(strings.TrimSpace(s), UnitInverse) {
			*u = UnitFactor{Inverse: true}
			return nil
		}
		return fmt.Errorf("unit alias value %q: want a number or %q", s, UnitInverse)
	}
	var f float64
	if err := json.Unmarshal(b, &f); err != nil {
		return fmt.Errorf("unit alias value %s: want a number or %q", b, UnitInverse)
	}
	if f == 0 {
		return fmt.Errorf("unit alias multiplier 0 would turn every value into 0")
	}
	*u = UnitFactor{Multiplier: f}
	return nil
}

// MarshalJSON writes the factor back in the column's own form.
func (u UnitFactor) MarshalJSON() ([]byte, error) {
	if u.Inverse {
		return json.Marshal(UnitInverse)
	}
	return json.Marshal(u.Multiplier)
}

// ParseFormat decodes and validates a LEGACY intent_specs.response_format
// column. A NULL column (empty or "null") returns nil, nil: the intent has no
// format yet.
//
// The legacy rule is kept so nothing that already works changes: when no
// field is marked "scored", the FIRST field is the scored one and must be a
// number or bool. ParseFormat sets Scored on it, so the decoded format reads
// the same as one that marks it explicitly.
//
// Validation is strict on purpose. A malformed format would not fail loudly at
// scoring time; it would quietly score every miner wrong or not at all.
func ParseFormat(raw []byte) (*Format, error) {
	f, err := decodeFormat(raw)
	if f == nil || err != nil {
		return f, err
	}
	if f.ScoredIndex() < 0 {
		// The comparator scores ONE number, and under the legacy rule it is
		// the first field's.
		if t := f.Fields[0].Type; t != TypeNumber && t != TypeBool {
			return nil, fmt.Errorf("response_format first field %q is %q; only number and bool can be scored",
				f.Fields[0].Name, t)
		}
		f.Fields[0].Scored = true
	}
	return f, nil
}

// ParseIntentFormat decodes and validates an intent_response_formats.format
// value. Unlike ParseFormat there is no implicit scored field: a field is
// scored only when it says "scored": true. Whether the intent NEEDS one
// depends on its class and is checked by the loader (pkg/scoring/intentformat).
// An empty or "null" value is an error here: a row with no format is a bad row.
func ParseIntentFormat(raw []byte) (*Format, error) {
	f, err := decodeFormat(raw)
	if err != nil {
		return nil, err
	}
	if f == nil {
		return nil, fmt.Errorf("format is empty")
	}
	return f, nil
}

// decodeFormat is the validation both entry points share.
func decodeFormat(raw []byte) (*Format, error) {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return nil, nil
	}
	var f Format
	dec := json.NewDecoder(strings.NewReader(s))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("response_format is not a valid format object: %w", err)
	}
	if err := f.Validate(); err != nil {
		return nil, err
	}
	return &f, nil
}

// Validate checks the rules every format must meet, whatever its intent's
// class: at least one and at most MaxFields fields, unique names, known
// types, at most one scored field (number or bool), allowed only on enum (and
// required there), range only on number / integer, unit aliases only on a
// number with a unit, true_when on every bool, and user_description using
// only {query}.
func (f *Format) Validate() error {
	if len(f.Fields) == 0 {
		return fmt.Errorf("response_format declares no fields")
	}
	if len(f.Fields) > MaxFields {
		return fmt.Errorf("response_format declares %d fields; at most %d (each field is one model call per answer)",
			len(f.Fields), MaxFields)
	}
	switch f.AdapterInput {
	case "", AdapterInputRaw, AdapterInputShaped:
	default:
		return fmt.Errorf("response_format adapter_input %q: want %q or %q", f.AdapterInput, AdapterInputRaw, AdapterInputShaped)
	}
	seen := make(map[string]bool, len(f.Fields))
	scored := 0
	for i, fl := range f.Fields {
		if strings.TrimSpace(fl.Name) == "" {
			return fmt.Errorf("response_format field %d has no name", i)
		}
		if seen[fl.Name] {
			return fmt.Errorf("response_format field name %q is used twice", fl.Name)
		}
		seen[fl.Name] = true
		if strings.TrimSpace(fl.Description) == "" {
			return fmt.Errorf("response_format field %q has no description; the model has nothing to look for", fl.Name)
		}
		if !knownTypes[fl.Type] {
			return fmt.Errorf("response_format field %q has unknown type %q", fl.Name, fl.Type)
		}
		if len(fl.Range) != 0 {
			if fl.Type != TypeNumber && fl.Type != TypeInteger {
				return fmt.Errorf("response_format field %q is %q; range is only allowed on number and integer fields", fl.Name, fl.Type)
			}
			if len(fl.Range) != 2 || fl.Range[0] > fl.Range[1] {
				return fmt.Errorf("response_format field %q range must be [min, max], got %v", fl.Name, fl.Range)
			}
		}
		if len(fl.Allowed) > 0 && fl.Type != TypeEnum {
			return fmt.Errorf("response_format field %q is %q; allowed is only allowed on enum fields", fl.Name, fl.Type)
		}
		if fl.Type == TypeEnum && len(fl.Allowed) == 0 {
			return fmt.Errorf("response_format enum field %q has no allowed values", fl.Name)
		}
		if fl.Type == TypeBool && len(fl.TrueWhen) == 0 {
			return fmt.Errorf("response_format bool field %q has no true_when; every value would read as false", fl.Name)
		}
		if len(fl.TrueWhen) > 0 && fl.Type != TypeBool {
			return fmt.Errorf("response_format field %q is %q; true_when is only allowed on bool fields", fl.Name, fl.Type)
		}
		if len(fl.UnitAliases) > 0 {
			if strings.TrimSpace(fl.Unit) == "" {
				return fmt.Errorf("response_format field %q has unit_aliases but no unit to convert to", fl.Name)
			}
			if fl.Type != TypeNumber {
				return fmt.Errorf("response_format field %q is %q; unit_aliases are only allowed on number fields", fl.Name, fl.Type)
			}
		}
		if fl.Scored {
			scored++
			if fl.Type != TypeNumber && fl.Type != TypeBool {
				return fmt.Errorf("response_format scored field %q is %q; only number and bool can be scored", fl.Name, fl.Type)
			}
		}
		if ud := fl.UserDescription; ud != "" {
			if !strings.Contains(ud, QueryPlaceholder) {
				return fmt.Errorf("response_format field %q user_description must contain %s; "+
					"without the query the model is not told which value the user asked for", fl.Name, QueryPlaceholder)
			}
			// A user request has no question key, so {key} / {<param>} could
			// never be filled and would reach the model as literal braces.
			if m := placeholderRE.FindString(strings.ReplaceAll(ud, QueryPlaceholder, "")); m != "" {
				return fmt.Errorf("response_format field %q user_description uses %s; only %s is available "+
					"on a user request (there is no question key)", fl.Name, m, QueryPlaceholder)
			}
		}
	}
	if scored > 1 {
		return fmt.Errorf("response_format marks %d fields scored; at most one can be", scored)
	}
	return nil
}

// ScoredIndex is the index of the scored field, or -1 when no field is
// scored (a judgment intent's format).
func (f *Format) ScoredIndex() int {
	if f == nil {
		return -1
	}
	for i, fl := range f.Fields {
		if fl.Scored {
			return i
		}
	}
	return -1
}

// HasScored reports whether the format has a field the comparator can score.
func (f *Format) HasScored() bool { return f.ScoredIndex() >= 0 }

// Scored is the field the comparator scores. Callers check HasScored first;
// for a format with no scored field it returns the first field, which the
// comparator never reads (parse paths refuse a format without one).
func (f *Format) Scored() Field {
	if i := f.ScoredIndex(); i >= 0 {
		return f.Fields[i]
	}
	return f.Fields[0]
}

// ShapedAdapterInput reports whether the format opts in to scoring the
// intent-shaped object rather than the raw answer (adapter_input "shaped").
func (f *Format) ShapedAdapterInput() bool { return f != nil && f.AdapterInput == AdapterInputShaped }

// Fill returns a copy of the field with {key} and {<param>} placeholders
// replaced in the description, the unit and the alias names.
//
// {key} is filled first, exactly as question_template is; params come from the
// key's truth_params.key_params entry so a description can name what the key
// stands for (a pool address, a CVE, a year) rather than only its short id.
func (fl Field) Fill(key string, params map[string]string) Field {
	fill := func(s string) string { return FillText(s, key, params) }
	out := fl
	out.Description = fill(fl.Description)
	out.Unit = fill(fl.Unit)
	if len(fl.UnitAliases) > 0 {
		out.UnitAliases = make(map[string]UnitFactor, len(fl.UnitAliases))
		for u, f := range fl.UnitAliases {
			out.UnitAliases[fill(u)] = f
		}
	}
	return out
}

// FillText replaces {key} and then each {<param>} in s. Params are applied in
// sorted order so the result never depends on map order: every validator must
// build the identical prompt.
func FillText(s, key string, params map[string]string) string {
	s = strings.ReplaceAll(s, "{key}", key)
	names := make([]string, 0, len(params))
	for k := range params {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		s = strings.ReplaceAll(s, "{"+k+"}", params[k])
	}
	return s
}

// PromptField is the field as the model sees it, matching the experiments
// script (llm_pick_miner_responses.py), which showed exactly ONE unit: the unit
// this miner's response is in. That is the miner's declared unit when it has
// one (an alias such as "USD per EUR (inverse; ...)"), otherwise the field's
// own unit. Aliases are conversion data for code, never shown to the model.
func (fl Field) PromptField(declaredUnit string) Field {
	out := fl
	if strings.TrimSpace(declaredUnit) != "" {
		out.Unit = declaredUnit
	}
	out.UnitAliases = nil
	return out
}

// Units lists the units shown to the model: the field's own unit first, then
// the aliases sorted by name. The order matters - the prompt tells the model
// to prefer the first listed unit - so it is fixed rather than map order.
// The parser calls it on PromptField, so in practice it is the one unit.
func (fl Field) Units() []string {
	var out []string
	if strings.TrimSpace(fl.Unit) != "" {
		out = append(out, fl.Unit)
	}
	aliases := make([]string, 0, len(fl.UnitAliases))
	for u := range fl.UnitAliases {
		aliases = append(aliases, u)
	}
	sort.Strings(aliases)
	return append(out, aliases...)
}

// DefaultUserDescription is the user-request description used when a field
// has no user_description. It is built from the field as the model sees it
// (after PromptField), so the unit it names is the unit listed in "units":
//
//	Value of "<name>" (<type>, in <unit>) that answers this request: {query}
//
// ", in <unit>" is left out for a field with no unit (a bool).
func DefaultUserDescription(fl Field) string {
	kind := fl.Type
	if u := strings.TrimSpace(fl.Unit); u != "" {
		kind += ", in " + u
	}
	return fmt.Sprintf("Value of %q (%s) that answers this request: %s", fl.Name, kind, QueryPlaceholder)
}

// ForUser is the field as the model sees it on a USER request: exactly the
// scoring view (PromptField: the miner's declared unit or the field's own,
// no aliases; the same name, type and true_when) with only the description
// swapped for the user's view, with {query} filled. Nothing else is filled: a
// user request has no question key. The user's view is, in order:
//
//   - the field's user_description, when it has one;
//   - the field's own description followed by the query, when the description
//     has no placeholder to fill (most whole-answer fields: "Summary of the
//     answer in one or two sentences") - it already says what is wanted, the
//     query says which item;
//   - DefaultUserDescription otherwise (a description written for a question
//     key, e.g. "Current USD/{key} rate", which a user request cannot fill).
func (fl Field) ForUser(query, declaredUnit string) Field {
	out := fl.PromptField(declaredUnit)
	tmpl := fl.UserDescription
	if strings.TrimSpace(tmpl) == "" {
		if placeholderRE.MatchString(fl.Description) {
			tmpl = DefaultUserDescription(out)
		} else {
			tmpl = strings.TrimRight(strings.TrimSpace(fl.Description), ".") + ". For this request: " + QueryPlaceholder
		}
	}
	// One pass: a query that itself contains "{query}" is not expanded again.
	out.Description = strings.ReplaceAll(tmpl, QueryPlaceholder, query)
	return out
}
