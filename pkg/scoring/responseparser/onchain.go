package responseparser

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf8"
)

// A job's answer reaches a contract as the same processed answer a routed ask
// returns (ParseAll), laid out in a fixed, bounded way the format alone
// decides: its fields in order, each in the array of its type. Whichever miner
// answered and however long its response, an intent's answer always has the
// same slots (Ahmed, 2026-10-22: processed intelligence, never raw output).
//
//	number            -> integers: round(value × 10^decimals), an int256
//	integer           -> integers
//	bool              -> bools
//	string, enum,
//	date, datetime    -> strings, as copied (at most MaxOnChainString bytes)
//
// A field marked off_chain is left out (the API answer keeps it). list and
// object have no fixed size, so a format with either on-chain cannot be
// delivered.

// MaxDecimals bounds Field.Decimals.
const MaxDecimals = 18

// MaxOnChainString bounds a text value in a job's answer, in bytes.
const MaxOnChainString = 256

// OnChain is a processed answer in its on-chain layout: the OnChainData a
// job's callback receives (it has no addresses).
type OnChain struct {
	Integers []*big.Int
	Strings  []string
	Bools    []bool
}

// OnChainReady reports why f cannot be delivered on-chain, or nil when it can:
// at least one field goes on-chain, every number field that does needs its
// decimals, and none of them may be a list or object.
func (f *Format) OnChainReady() error {
	on := 0
	for _, fl := range f.Fields {
		if fl.OffChain {
			continue
		}
		on++
		switch fl.Type {
		case TypeNumber:
			if fl.Decimals == nil {
				return fmt.Errorf("number field %q has no decimals", fl.Name)
			}
		case TypeList, TypeObject:
			return fmt.Errorf("field %q is a %s, which has no fixed size", fl.Name, fl.Type)
		}
	}
	if on == 0 {
		return fmt.Errorf("every field is off_chain")
	}
	return nil
}

// OnChainOnly is f with only the fields that go on-chain: a job's answer is
// read for those alone, sparing a parser call per off_chain field.
func (f *Format) OnChainOnly() *Format {
	c := *f
	c.Fields = make([]Field, 0, len(f.Fields))
	for _, fl := range f.Fields {
		if !fl.OffChain {
			c.Fields = append(c.Fields, fl)
		}
	}
	return &c
}

// EncodeOnChain lays a processed answer out for a contract. Every field must
// have been read: a missing or rejected value is never sent as an invented 0
// or empty string; the answer is refused instead.
func EncodeOnChain(f *Format, s Shaped) (OnChain, error) {
	if err := f.OnChainReady(); err != nil {
		return OnChain{}, err
	}
	if len(s.Fields) != len(f.Fields) {
		return OnChain{}, fmt.Errorf("answer has %d fields, format %d", len(s.Fields), len(f.Fields))
	}
	out := OnChain{Integers: []*big.Int{}, Strings: []string{}, Bools: []bool{}}
	for i, fl := range f.Fields {
		if fl.OffChain {
			continue
		}
		v := s.Fields[i]
		if v.Status != FieldOK {
			return OnChain{}, fmt.Errorf("field %q was not read (%s)", fl.Name, v.Status)
		}
		switch fl.Type {
		case TypeNumber:
			x, ok := v.Value.(float64)
			if !ok {
				return OnChain{}, fmt.Errorf("field %q: number is %T", fl.Name, v.Value)
			}
			n, err := fixedPoint(x, *fl.Decimals)
			if err != nil {
				return OnChain{}, fmt.Errorf("field %q: %w", fl.Name, err)
			}
			out.Integers = append(out.Integers, n)
		case TypeInteger:
			x, ok := v.Value.(int64)
			if !ok {
				return OnChain{}, fmt.Errorf("field %q: integer is %T", fl.Name, v.Value)
			}
			out.Integers = append(out.Integers, big.NewInt(x))
		case TypeBool:
			b, ok := v.Value.(bool)
			if !ok {
				return OnChain{}, fmt.Errorf("field %q: bool is %T", fl.Name, v.Value)
			}
			out.Bools = append(out.Bools, b)
		default: // string, enum, date, datetime
			t, ok := v.Value.(string)
			if !ok {
				return OnChain{}, fmt.Errorf("field %q: text is %T", fl.Name, v.Value)
			}
			if len(t) > MaxOnChainString {
				if !fl.Processed {
					// The miner's own text: cut, it could say something else.
					return OnChain{}, fmt.Errorf("field %q is %d bytes; at most %d on-chain", fl.Name, len(t), MaxOnChainString)
				}
				// The node's own sentence (a brief): bounded by cutting it at a
				// character boundary rather than failing the job over its length.
				t = cutUTF8(t, MaxOnChainString)
			}
			out.Strings = append(out.Strings, t)
		}
	}
	return out, nil
}

// cutUTF8 is s cut to at most n bytes, at a character boundary.
func cutUTF8(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// fixedPoint is round(x × 10^decimals), half away from zero, of the decimal
// x stands for: the shortest decimal that reads back as x (what the miner
// wrote, as far as a float64 holds it), not its binary value. 1.005 at 2
// decimals is 101, where the binary 1.00499999... rounded to 100.
func fixedPoint(x float64, decimals int) (*big.Int, error) {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return nil, fmt.Errorf("%v is not a number", x)
	}
	r, ok := new(big.Rat).SetString(strconv.FormatFloat(x, 'g', -1, 64))
	if !ok {
		return nil, fmt.Errorf("%v is not a number", x)
	}
	r.Mul(r, new(big.Rat).SetInt(new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(decimals)), nil)))
	// Half away from zero: |r| + 1/2, truncated, with r's sign.
	num, den := new(big.Int).Abs(r.Num()), r.Denom()
	n := new(big.Int).Quo(new(big.Int).Add(new(big.Int).Mul(num, big.NewInt(2)), den), new(big.Int).Mul(den, big.NewInt(2)))
	if r.Sign() < 0 {
		n.Neg(n)
	}
	if n.BitLen() > 255 {
		return nil, fmt.Errorf("%v × 10^%d does not fit an int256", x, decimals)
	}
	return n, nil
}

// DecodeOnChainInputs reads a job's params into the intent's inputs, by the
// same layout rule as an answer: the inputs in order, each from the array of
// its type (number and integer from integers, bool from bools, the rest from
// strings). A number is the int256 / 10^decimals. A request missing a value, or
// with one outside its range or allowed values, is refused: no miner is asked
// a question the caller did not ask.
func DecodeOnChainInputs(f *Format, integers []*big.Int, strs []string, bools []bool) (OrderedObject, error) {
	out := make(OrderedObject, 0, len(f.Inputs))
	ni, ns, nb := 0, 0, 0
	for _, in := range f.Inputs {
		var v any
		switch in.Type {
		case TypeNumber, TypeInteger:
			if ni >= len(integers) || integers[ni] == nil {
				return nil, fmt.Errorf("input %q: the request has %d integers", in.Name, len(integers))
			}
			n := signed(integers[ni])
			ni++
			var x float64
			if in.Type == TypeInteger {
				if !n.IsInt64() {
					return nil, fmt.Errorf("input %q: %s does not fit an integer", in.Name, n)
				}
				v, x = n.Int64(), float64(n.Int64())
			} else {
				x, _ = new(big.Rat).SetFrac(n, new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(*in.Decimals)), nil)).Float64()
				v = x
			}
			if len(in.Range) == 2 && (x < in.Range[0] || x > in.Range[1]) {
				return nil, fmt.Errorf("input %q: %v is outside [%v, %v]", in.Name, x, in.Range[0], in.Range[1])
			}
		case TypeBool:
			if nb >= len(bools) {
				return nil, fmt.Errorf("input %q: the request has %d bools", in.Name, len(bools))
			}
			v = bools[nb]
			nb++
		default: // string, enum, date, datetime
			if ns >= len(strs) {
				return nil, fmt.Errorf("input %q: the request has %d strings", in.Name, len(strs))
			}
			t := strs[ns]
			ns++
			if len(t) > MaxOnChainString {
				return nil, fmt.Errorf("input %q is %d bytes; at most %d", in.Name, len(t), MaxOnChainString)
			}
			if in.Type == TypeEnum && !allowedValue(in.Allowed, t) {
				return nil, fmt.Errorf("input %q: %q is not one of %s", in.Name, t, strings.Join(in.Allowed, ", "))
			}
			v = t
		}
		out = append(out, KV{Key: in.Name, Value: v})
	}
	return out, nil
}

// InputsQuestion is a request's inputs as the question the request builder
// reads (the same model a routed ask uses to build a miner's API request):
// each input named, described and valued, in order.
func InputsQuestion(intentID string, f *Format, in OrderedObject) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s request.", intentID)
	for _, input := range f.Inputs {
		v, _ := in.Get(input.Name)
		val := fmt.Sprint(v)
		if x, ok := v.(float64); ok {
			val = strconv.FormatFloat(x, 'f', -1, 64)
		}
		if input.Unit != "" {
			val += " " + input.Unit
		}
		fmt.Fprintf(&b, " %s (%s): %s.", input.Name, input.Description, val)
	}
	return b.String()
}

func allowedValue(allowed []string, v string) bool {
	for _, a := range allowed {
		if normEnum(a) == normEnum(v) {
			return true
		}
	}
	return false
}

// signed reads a uint256 word as an int256 (two's complement).
func signed(w *big.Int) *big.Int {
	if w.Bit(255) == 1 {
		return new(big.Int).Sub(w, new(big.Int).Lsh(big.NewInt(1), 256))
	}
	return w
}
