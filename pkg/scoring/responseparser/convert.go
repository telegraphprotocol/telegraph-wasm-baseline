package responseparser

import (
	"fmt"
	"strings"
)

// Convert turns the copied text into the number the comparator scores, in the
// field's unit (the intent's scale is applied later, by the caller, exactly as
// for the other parse paths).
//
// This is where ALL conversion lives; the model never converts (a processed
// value is the model's own working, still converted here).
//
//	number : parse the text (decimal comma aware), then the unit: the field's
//	         own unit as is, an alias by its multiplier, or "inverse" as 1/x;
//	         then the range check.
//	bool   : 1 if the copied text is one of true_when (case-insensitive), else 0.
//
// unit names the unit the miner quotes in; "" means the field's own unit. It
// comes from the miner's registration (semantics.signal_mapping.unit), because
// a number alone cannot say which way round an FX rate is quoted.
func Convert(fl Field, copied, unit string) (float64, error) {
	switch fl.Type {
	case TypeBool:
		t := strings.ToUpper(strings.TrimSpace(copied))
		for _, w := range fl.TrueWhen {
			if t == strings.ToUpper(strings.TrimSpace(w)) {
				return 1, nil
			}
		}
		return 0, nil

	case TypeNumber:
		x, ok := ToNumber(copied)
		if !ok {
			return 0, fmt.Errorf("%w: %q is not a number", ErrConvert, copied)
		}
		factor, err := fl.factorFor(unit)
		if err != nil {
			return 0, err
		}
		switch {
		case factor.Inverse:
			if x == 0 {
				return 0, fmt.Errorf("%w: cannot invert 0", ErrConvert)
			}
			x = 1 / x
		case factor.Multiplier != 0:
			x *= factor.Multiplier
		}
		if len(fl.Range) == 2 && (x < fl.Range[0] || x > fl.Range[1]) {
			return 0, fmt.Errorf("%w: %v %s is outside the plausible range [%v, %v]",
				ErrOutOfRange, x, fl.Unit, fl.Range[0], fl.Range[1])
		}
		return x, nil

	default:
		return 0, fmt.Errorf("%w: field type %q cannot be scored (only number and bool)", ErrConvert, fl.Type)
	}
}

// factorFor resolves the unit the miner quotes in to a conversion factor. The
// field's own unit, or no declared unit, is the identity.
func (fl Field) factorFor(unit string) (UnitFactor, error) {
	u := strings.TrimSpace(unit)
	if u == "" || strings.EqualFold(u, strings.TrimSpace(fl.Unit)) {
		return UnitFactor{Multiplier: 1}, nil
	}
	for name, f := range fl.UnitAliases {
		if strings.EqualFold(u, strings.TrimSpace(name)) {
			return f, nil
		}
	}
	return UnitFactor{}, fmt.Errorf("%w: miner declares unit %q, which is not one of %q",
		ErrUnknownUnit, unit, fl.Units())
}
