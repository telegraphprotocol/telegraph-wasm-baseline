package runtime

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
)

// ModelScore is the exact output of a float adapter: the IEEE-754 binary32
// bit pattern the guest printed on its HEX line. It is never NaN or ±Inf (the
// constructors refuse them), and it is never rounded: every method below is an
// exact function of the 32 bits, computed with integer arithmetic only, so two
// validators holding the same bits derive the same order key, the same basis
// points and the same text on any CPU.
type ModelScore struct {
	bits uint32
}

// NewModelScoreFromBits validates bits as a finite f32 and wraps it.
func NewModelScoreFromBits(bits uint32) (ModelScore, error) {
	exp := (bits >> 23) & 0xff
	if exp == 0xff {
		if bits&0x7fffff != 0 {
			return ModelScore{}, fmt.Errorf("wasm/runtime: adapter score is NaN (0x%08x)", bits)
		}
		return ModelScore{}, fmt.Errorf("wasm/runtime: adapter score is infinite (0x%08x)", bits)
	}
	return ModelScore{bits: bits}, nil
}

// NewModelScore wraps a finite float32. Used by tests and stubs; the real path
// is NewModelScoreFromBits on the parsed HEX line.
func NewModelScore(f float32) (ModelScore, error) {
	return NewModelScoreFromBits(math.Float32bits(f))
}

// Bits is the f32 bit pattern.
func (s ModelScore) Bits() uint32 { return s.bits }

// Float32 is the score as a float32 (exact: it IS the bits).
func (s ModelScore) Float32() float32 { return math.Float32frombits(s.bits) }

// Hex is the score as the guest prints it: "0x" + 8 lowercase hex digits.
func (s ModelScore) Hex() string { return fmt.Sprintf("0x%08x", s.bits) }

// String is Hex plus the decimal value, for logs.
func (s ModelScore) String() string {
	return s.Hex() + " (" + strconv.FormatFloat(float64(s.Float32()), 'g', -1, 32) + ")"
}

// OrderKey maps the score to an int64 whose order is exactly the IEEE order
// of the finite f32 values: a < b as floats iff a.OrderKey() < b.OrderKey(),
// and equal floats (including -0 and +0) get equal keys. This is what the
// scorers sort and compare on, so ranking never depends on a float operation.
//
// The standard sign-magnitude to two's-complement fold: a non-negative float's
// bits already order like integers; a negative one's magnitude orders
// backwards, so it is negated.
func (s ModelScore) OrderKey() int64 {
	mag := int64(s.bits & 0x7fffffff)
	if s.bits&0x80000000 != 0 {
		return -mag // -0 folds onto +0
	}
	return mag
}

// MinOrderKey sorts below every finite score's OrderKey. Scorers use it for a
// miner that has no score at all (did not answer, or could not be scored).
const MinOrderKey int64 = math.MinInt64

// Q32Code is the score as a signed fixed-point integer with 32 fractional
// bits: trunc(value * 2^32), saturated to the int64 range. It is computed from
// the bits alone (mantissa shifted by the exponent), so it involves no float
// arithmetic. The truncation is toward zero on the magnitude, then the sign is
// reapplied.
//
// Why this representation: the judgment-to-basis-points boundary conversion
// (a sigmoid over a 2^-32 fixed-point value, see fixedSigmoid) and the hybrid
// fusion that consumes it are integer-only and pinned by golden vectors; this
// feeds the float model's output into exactly that conversion.
func (s ModelScore) Q32Code() int64 {
	exp := int((s.bits >> 23) & 0xff)
	frac := uint64(s.bits & 0x7fffff)
	var mant uint64
	var e int // value = mant * 2^e
	if exp == 0 {
		mant, e = frac, -149 // subnormal
	} else {
		mant, e = frac|0x800000, exp-150
	}
	shift := e + FixedFracBits // code = mant * 2^shift
	var mag uint64
	switch {
	case mant == 0:
		mag = 0
	case shift >= 0:
		// mant < 2^24; anything past 2^63 saturates.
		if shift > 39 {
			mag = math.MaxInt64
		} else {
			mag = mant << uint(shift)
			if mag > math.MaxInt64 {
				mag = math.MaxInt64
			}
		}
	case shift <= -64:
		mag = 0
	default:
		mag = mant >> uint(-shift)
	}
	if s.bits&0x80000000 != 0 {
		return -int64(mag)
	}
	return int64(mag)
}

// BasisPoints maps the score into the [0, ScoreBasisPoints] judgment domain:
// a sigmoid over Q32Code, scaled to basis points, with the same integer
// arithmetic the adapter score domain has always used. A reward head emits an
// unbounded logit (the real CODE_REVIEW adapter returns about +26.7 on its
// reference input), so the sigmoid is what keeps negative-logit miners apart
// instead of clamping them all to 0. It is monotone, so it never reorders two
// scores — though it can tie them, which is why ranking uses OrderKey.
func (s ModelScore) BasisPoints() int32 {
	return JudgmentToBasisPoints(s.Q32Code())
}

// FixedFracBits is the number of fractional bits in the fixed-point judgment
// code BasisPoints converts from (value = code / 2^FixedFracBits). The golden
// vectors in pkg/scoring/testdata/hybrid_fusion pin it.
const FixedFracBits = 32

// JudgmentToBasisPoints maps a fixed-point judgment code (value * 2^32) to
// basis points: clamp(sigmoid(code) * 10000 >> 32, 0, 10000).
//
// Every step is exact integer arithmetic. A float sigmoid here would put a
// platform-dependent rounding step between the model and the leaderboard, and
// two validators disagreeing by one basis point would fork the epoch. The
// helpers below are pinned against the reference implementation's golden
// vectors (judg_in/judg_out in pkg/scoring/testdata/hybrid_fusion).
func JudgmentToBasisPoints(code int64) int32 {
	return unitToBasisPoints(fixedSigmoid(code))
}

// unitToBasisPoints maps a fixed-point value already in [0,1] to basis points:
//
//	clamp(code * SCORE_ONE >> FixedFracBits, 0, SCORE_ONE)
func unitToBasisPoints(code int64) int32 {
	prod := new(big.Int).Mul(big.NewInt(code), big.NewInt(int64(ScoreBasisPoints)))
	prod.Rsh(prod, FixedFracBits)

	if prod.Sign() < 0 {
		return 0
	}
	if prod.Cmp(big.NewInt(int64(ScoreBasisPoints))) > 0 {
		return ScoreBasisPoints
	}
	return int32(prod.Int64())
}

// The fixed-point helpers below are the integer sigmoid the score domain is
// defined by. Ported exactly — the shift direction and the truncation
// convention are part of the contract, not implementation details:
//
//   - fixedDiv truncates the MAGNITUDE toward zero and reapplies the sign.
//
// big.Int rather than a two-limb int128: this runs once per score, against a
// model that takes seconds, so exactness is worth more than the cycles.

const fixedOne = int64(1) << FixedFracBits

// fixedDiv is a/b in fixed point, truncating the magnitude toward zero.
func fixedDiv(a, b int64) int64 {
	if b == 0 {
		return 0
	}
	num := new(big.Int).Abs(big.NewInt(a))
	num.Lsh(num, FixedFracBits)
	num.Quo(num, new(big.Int).Abs(big.NewInt(b)))
	if (a < 0) != (b < 0) {
		num.Neg(num)
	}
	return num.Int64()
}

// fixedExpDyadic is exp(x) = (1 + x/2^k)^(2^k) by k repeated squarings, k=14.
func fixedExpDyadic(x int64) int64 {
	const k = 14
	r := big.NewInt(fixedOne + (x >> k))
	for i := 0; i < k; i++ {
		r.Mul(r, r)
		r.Rsh(r, FixedFracBits)
	}
	if !r.IsInt64() {
		// Unreachable for the clamped input fixedSafeExp allows (exp(10) is
		// about 22026), but a wrapped i64 here would come back as a negative
		// "probability" and invert the ranking rather than failing.
		return math.MaxInt64
	}
	return r.Int64()
}

// fixedSafeExp is exp(x) clamped to [-10, 10].
func fixedSafeExp(x int64) int64 {
	maxX := int64(10) * fixedOne
	if x > maxX {
		x = maxX
	}
	if x < -maxX {
		return 0
	}
	if x >= 0 {
		return fixedExpDyadic(x)
	}
	pos := fixedExpDyadic(-x)
	if pos == 0 {
		return 0
	}
	return fixedDiv(fixedOne, pos)
}

// fixedSigmoid is 1/(1+e^-x) in fixed point, saturating past |x| > 10.
//
// Monotone, which is what makes it safe as a ranking transform: it changes the
// numbers a miner sees but never the order they finish in — except in the
// saturated tails, where |logit| > 10 maps to 0 or 10000 bps. Ranking uses
// ModelScore.OrderKey, so even those tails stay ordered.
func fixedSigmoid(x int64) int64 {
	ten := int64(10) * fixedOne
	if x > ten {
		return fixedOne
	}
	if x < -ten {
		return 0
	}
	return fixedDiv(fixedOne, fixedOne+fixedSafeExp(-x))
}
