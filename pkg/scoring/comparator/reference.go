package comparator

import (
	"math"
	"math/big"
)

// ReferenceScoreOne is the comparator's perfect score: 1.0 in basis points.
const ReferenceScoreOne int64 = 10_000

// ReferenceScore is the Go statement of the reference comparator — the
// expected output of imw_score_deterministic for one (spec, truth, miner)
// triple, in basis points 0..10000.
//
// It is the answer key for the comparator promotion gate: a comparator WASM
// registered for a deterministic intent is promoted only if it reproduces this
// function exactly. It ports comparator_core and linear_decay from
// telegraph-wasm-baseline's wide_runtime/wide_wasm/src/hybrid.rs line for line, including the
// parts that look like they could be "simplified":
//
//   - The code is truncated to int32 before it is matched, and an unknown code
//     falls back to Exact, as Comparator::from_i32 does.
//   - Multiplications saturate rather than wrap (saturating_mul).
//   - RelTol computes diff*10000 in 128 bits, so a wei- or satoshi-scale value
//     cannot overflow into a small error and a near-perfect score.
//
// A unit test pins this function to the shipped comparator modules, so the
// answer key cannot drift from the binary it stands in for.
func ReferenceScore(code, tolerance, truth, miner int64) int64 {
	diff := absWrap(miner - truth)
	switch referenceKind(code) {
	case CmpAbsTol:
		return referenceLinearDecay(diff, tolerance)
	case CmpRelTol:
		denom := absWrap(truth)
		if denom < 1 {
			denom = 1
		}
		q := new(big.Int).Mul(big.NewInt(diff), big.NewInt(ReferenceScoreOne))
		q.Quo(q, big.NewInt(denom)) // truncates toward zero, as Rust i128 division does
		errBps := int64(math.MaxInt64)
		if q.IsInt64() {
			errBps = q.Int64()
		}
		return referenceLinearDecay(errBps, tolerance)
	default: // CmpExact and CmpPassFail
		if miner == truth {
			return ReferenceScoreOne
		}
		return 0
	}
}

// referenceKind mirrors Comparator::from_i32(s[0] as i32): the i64 code is
// truncated to i32, and anything that is not AbsTol, RelTol or PassFail is
// treated as Exact.
func referenceKind(code int64) int64 {
	switch int32(code) {
	case int32(CmpAbsTol):
		return CmpAbsTol
	case int32(CmpRelTol):
		return CmpRelTol
	case int32(CmpPassFail):
		return CmpPassFail
	default:
		return CmpExact
	}
}

// referenceLinearDecay mirrors linear_decay: error 0 scores ReferenceScoreOne,
// error >= 2*tol scores 0, linear in between. tol <= 0 means exact match.
func referenceLinearDecay(errVal, tol int64) int64 {
	if tol <= 0 {
		if errVal == 0 {
			return ReferenceScoreOne
		}
		return 0
	}
	window := saturatingMul(tol, 2)
	if errVal >= window {
		return 0
	}
	return saturatingMul(ReferenceScoreOne, window-errVal) / window
}

// absWrap is i64::abs in a release build, where overflow checks are off:
// math.MinInt64 has no positive counterpart and comes back unchanged.
func absWrap(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

// saturatingMul is i64::saturating_mul: the exact product, or the int64 bound
// it would have overflowed past.
func saturatingMul(a, b int64) int64 {
	p := new(big.Int).Mul(big.NewInt(a), big.NewInt(b))
	if p.IsInt64() {
		return p.Int64()
	}
	if p.Sign() < 0 {
		return math.MinInt64
	}
	return math.MaxInt64
}
