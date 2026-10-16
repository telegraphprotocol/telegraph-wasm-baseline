package fetchspec

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Extract reads the spec's field out of a JSON body, applies the verbs, and
// returns the number.
//
// Every failure is an ERROR rather than a zero. That distinction is the whole
// point: a miner that cannot be parsed must be excluded from the round, not
// scored zero. Scoring zero says "this miner is wrong"; excluding says "we
// could not read it", and only one of those is usually true.
func Extract(spec Spec, body []byte) (float64, error) {
	var doc any
	if err := json.Unmarshal(body, &doc); err != nil {
		// Worth naming, because several upstreams answer with HTML and a 200
		// status when they are throttling. Reporting "not JSON" rather than
		// "field missing" points at the real cause.
		return 0, fmt.Errorf("response is not JSON (upstreams often serve throttling notices as HTML with a 200 status): %w", err)
	}

	for _, a := range spec.Asserts {
		got, err := lookup(doc, a.Field)
		if err != nil {
			return 0, fmt.Errorf("assert %s: %w", a.Field, err)
		}
		if s := asString(got); !strings.EqualFold(s, a.Equals) {
			return 0, fmt.Errorf("assert failed: %s is %q, want %q — this response answers a different question",
				a.Field, s, a.Equals)
		}
	}

	for param, want := range spec.Echo {
		got, err := lookup(doc, param)
		if err != nil {
			return 0, fmt.Errorf("echo check %s: %w", param, err)
		}
		if s := asString(got); !strings.EqualFold(s, want) {
			return 0, fmt.Errorf("echo mismatch: response says %s=%q but we asked for %q — "+
				"the request lost a parameter and this is a valid answer to the wrong question",
				param, s, want)
		}
	}

	// array_index verbs run BEFORE the field path, because they select which
	// element the rest of the path applies to.
	cur := doc
	var rest []Verb
	for _, v := range spec.Verbs {
		if v.Op == OpArrayIndex {
			arr, ok := cur.([]any)
			if !ok {
				return 0, fmt.Errorf("array_index applied to a %T, not a list", cur)
			}
			i := int(v.Arg)
			if i >= len(arr) {
				return 0, fmt.Errorf("array_index %d is past the end of a %d-element list", i, len(arr))
			}
			cur = arr[i]
			continue
		}
		rest = append(rest, v)
	}

	raw, err := lookup(cur, spec.Field)
	if err != nil {
		return 0, err
	}

	val, err := toNumber(raw, rest)
	if err != nil {
		return 0, fmt.Errorf("field %s: %w", spec.Field, err)
	}

	for _, v := range rest {
		switch v.Op {
		case OpInvert:
			if val == 0 {
				return 0, fmt.Errorf("cannot invert zero")
			}
			val = 1 / val
		case OpDivideBy:
			val /= v.Arg
		case OpMultiplyBy:
			val *= v.Arg
		case OpEqualsZero:
			// Deliberately not a comparison against v.Arg: this exists for
			// rcode-style fields where zero is the success value, and giving
			// it a configurable target would invite using it as a general
			// equality test, which the comparator's pass_fail already is.
			if val == 0 {
				val = 1
			} else {
				val = 0
			}
		}
	}

	if math.IsNaN(val) || math.IsInf(val, 0) {
		return 0, fmt.Errorf("field %s produced a non-finite value", spec.Field)
	}
	return val, nil
}

// lookup walks a dotted path. A numeric segment indexes a list.
//
// Dotted rather than JSONPath deliberately: every path we have needed across
// nine truth sources and fifty-one miners is a plain descent, and a full query
// language would let a spec express things we cannot verify at registration.
func lookup(doc any, path string) (any, error) {
	cur := doc
	for _, seg := range strings.Split(path, ".") {
		if seg == "" {
			continue
		}
		switch node := cur.(type) {
		case map[string]any:
			v, ok := node[seg]
			if !ok {
				return nil, fmt.Errorf("no field %q (present: %s)", seg, keysOf(node))
			}
			cur = v
		case []any:
			i, err := strconv.Atoi(seg)
			if err != nil {
				return nil, fmt.Errorf("path segment %q is not a list index, but the value is a list", seg)
			}
			if i < 0 || i >= len(node) {
				return nil, fmt.Errorf("index %d is outside a %d-element list", i, len(node))
			}
			cur = node[i]
		default:
			return nil, fmt.Errorf("cannot descend into %T at %q", cur, seg)
		}
	}
	return cur, nil
}

// toNumber converts a JSON value to a float, honouring the string-shaped verbs.
//
// A JSON null is rejected rather than read as zero. The World Bank uses null
// for "no figure published", and a zero there is an in-domain percentage that
// would score a miner as claiming zero unemployment.
func toNumber(v any, verbs []Verb) (float64, error) {
	switch n := v.(type) {
	case float64:
		return n, nil
	case string:
		s := strings.TrimSpace(n)
		for _, vb := range verbs {
			if vb.Op == OpDecimalComma {
				s = strings.ReplaceAll(s, ".", "")
				s = strings.ReplaceAll(s, ",", ".")
			}
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, fmt.Errorf("%q is not a number", n)
		}
		return f, nil
	case bool:
		// pass_fail intents legitimately answer with a boolean.
		if n {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, fmt.Errorf("value is null — a published figure is absent, which is not the same as zero")
	default:
		return 0, fmt.Errorf("value is a %T, not a number", v)
	}
}

func asString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case float64:
		return strconv.FormatFloat(s, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(s)
	default:
		return fmt.Sprintf("%v", v)
	}
}

// keysOf lists a map's keys for an error message, so a wrong field name says
// what WAS there rather than only what was not.
func keysOf(m map[string]any) string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
		if len(out) == 8 {
			out = append(out, "...")
			break
		}
	}
	return strings.Join(out, ", ")
}
