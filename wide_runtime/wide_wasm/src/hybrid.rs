//! Hybrid-intent scoring for the wide Q32.32 runtime: integer comparator + fusion gate.
//!
//! A hybrid intent's score is not one judgment, it is two halves fused. The **comparator**
//! half checks a fact the node fetched independently (the miner's claimed value vs a
//! ground-truth value, passed IN — the wasm never fetches). The **judgment** half is the
//! adapter's opinion of the prose. The fusion gate decides how they combine, and its whole
//! purpose is that a miner who gets the FACT wrong cannot be rescued by fluent writing.
//!
//! Ported from `int8_runtime/imw_wasm/src/hybrid.rs`. The logic is unchanged — it is already
//! pure integer, so it carries over to Q32.32 without touching the arithmetic. What differs
//! is only where the judgment comes from: int8 ran a model baked into the module, the wide
//! path runs the Q32.32 interpreter over a separate `.imwq` artifact.
//!
//! Everything here is integer, so it passes the static-float-opcode scan and is bit-identical
//! across CPUs — the whole point.
//!
//! Score domain: a fixed-point [0,1] where 1.0 == `SCORE_ONE` (10000, i.e. score basis
//! points). The model's judgment code is mapped into this domain by the caller; the
//! comparator and fusion produce and consume it. One integer domain lets the two halves fuse
//! without any float.
//!
//! All per-intent constants (tolerance, fusion weights/thresholds) are passed in from the
//! host's spec buffer, so a validator cannot silently change a tolerance.

#![allow(dead_code)]

/// Fixed-point [0,1] score domain: 1.0 == 10000 (score basis points).
pub const SCORE_ONE: i64 = 10_000;

/// Comparator kinds (mirror the IntentComparatorSpec `comparator` enum).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Comparator {
    /// Exact integer equality -> full score if equal, else 0.
    Exact = 0,
    /// Absolute tolerance: |miner - truth| <= tol -> linearly scored to 0 at 2*tol.
    AbsTol = 1,
    /// Relative tolerance in basis points: |miner - truth| * 10000 / |truth| <= tol_bps.
    RelTol = 2,
    /// Pass/fail: miner must equal truth (e.g. a boolean/flag) -> SCORE_ONE or 0.
    PassFail = 3,
}

impl Comparator {
    pub fn from_i32(v: i32) -> Comparator {
        match v {
            1 => Comparator::AbsTol,
            2 => Comparator::RelTol,
            3 => Comparator::PassFail,
            _ => Comparator::Exact,
        }
    }
}

/// Compute the comparator-core score in [0, SCORE_ONE] from the miner's claimed value vs the
/// passed-in ground truth. Pure integer.
///
/// For the tolerant modes the score decays linearly: full score at zero error, zero score at
/// `2 * tolerance` of error (a simple, deterministic, monotone shape — no float, no division
/// surprises beyond one integer divide with a guarded denominator).
pub fn comparator_core(cmp: Comparator, miner: i64, truth: i64, tolerance: i64) -> i64 {
    let diff = (miner - truth).abs();
    match cmp {
        Comparator::Exact => {
            if miner == truth { SCORE_ONE } else { 0 }
        }
        Comparator::PassFail => {
            if miner == truth { SCORE_ONE } else { 0 }
        }
        Comparator::AbsTol => {
            // error band measured in absolute units; window = 2*tol (0 error -> full, 2*tol -> 0).
            linear_decay(diff, tolerance)
        }
        Comparator::RelTol => {
            // relative error in basis points: err_bps = diff * 10000 / |truth|.
            // The intermediate diff*10000 is done in i128 so it CANNOT overflow at any scale:
            // max |i64| * 10000 ~ 9.2e22 << i128::MAX (~1.7e38). Using i64 with saturating_mul
            // here was a silent-fraud trap — for wei/satoshi/1e18-token magnitudes it saturated
            // to i64::MAX, then dividing by an equally-huge truth yielded a SMALL err_bps -> a
            // wildly wrong value scored near-perfect. i128 removes any per-intent rescaling.
            let denom = (truth.abs().max(1)) as i128;
            let err_bps_i128 = (diff as i128) * 10_000 / denom;
            // err_bps is bounded by 10000 * (diff/truth); clamp into i64 for linear_decay. A
            // truly enormous ratio saturates high -> score 0 (fail closed), the safe direction.
            let err_bps = err_bps_i128.min(i64::MAX as i128) as i64;
            linear_decay(err_bps, tolerance)
        }
    }
}

/// Linear decay: `error` 0 -> SCORE_ONE, `error` == 2*tol -> 0, clamped. `tol <= 0` means exact.
///
/// Both divisions here have a positive denominator (`window > 0` is guaranteed by the `tol <= 0`
/// early return), so truncate-toward-zero and floor-toward-negative-infinity agree and this
/// matches the Python golden's `_tdiv` convention without needing it.
fn linear_decay(error: i64, tol: i64) -> i64 {
    if tol <= 0 {
        return if error == 0 { SCORE_ONE } else { 0 };
    }
    let window = tol.saturating_mul(2);
    if error >= window {
        return 0;
    }
    // SCORE_ONE * (window - error) / window  — integer, denom = window > 0.
    SCORE_ONE.saturating_mul(window - error) / window
}

/// Fusion kinds (mirror `fusion_mode` in the spec).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Fusion {
    /// Wrong fact caps the score low; correct fact unlocks the model's quality grade. (default)
    Gate = 0,
    /// Weighted combine of the two signals.
    Blend = 1,
}

impl Fusion {
    pub fn from_i32(v: i32) -> Fusion {
        match v {
            1 => Fusion::Blend,
            _ => Fusion::Gate,
        }
    }
}

/// The per-intent baked fusion constants. Field order matches the host's spec buffer.
pub struct FusionConst {
    // gate
    /// comparator_core threshold to "unlock" the model grade.
    pub pass: i64,
    /// multiplier applied to comparator_core when gated shut (in SCORE_ONE units).
    pub low: i64,
    /// minimum score once unlocked.
    pub floor: i64,
    /// range above floor the model grade can add (floor+span should be <= SCORE_ONE).
    pub span: i64,
    // blend
    /// comparator weight.
    pub w_c: i64,
    /// judgment weight.
    pub w_j: i64,
}

/// Fuse comparator_core + adapter_judgment (both in [0, SCORE_ONE]) -> final [0, SCORE_ONE].
///
/// `gate`: if comparator_core < PASS, final = comparator_core * LOW / SCORE_ONE (a low,
/// fact-driven score — a wrong fact cannot be masked by a good model score). Else the model's
/// judgment is mapped into [FLOOR, FLOOR+SPAN]:
/// final = FLOOR + SPAN * adapter_judgment / SCORE_ONE.
///
/// `blend`: final = (W_C * comparator_core + W_J * adapter_judgment) / (W_C + W_J).
///
/// Every division here divides a non-negative numerator (both inputs are clamped into
/// [0, SCORE_ONE] first, and the constants are non-negative by construction) by a positive
/// denominator, so truncation direction is not observable — but the clamp on entry is what
/// guarantees that, so do not remove it in the name of saving two comparisons.
pub fn fuse(mode: Fusion, comparator_core: i64, adapter_judgment: i64, k: &FusionConst) -> i64 {
    let c = comparator_core.clamp(0, SCORE_ONE);
    let j = adapter_judgment.clamp(0, SCORE_ONE);
    let out = match mode {
        Fusion::Gate => {
            if c < k.pass {
                // gated shut: a wrong/insufficient fact drives a low, fact-only score.
                c.saturating_mul(k.low) / SCORE_ONE
            } else {
                // unlocked: model grade maps into [floor, floor+span].
                k.floor + k.span.saturating_mul(j) / SCORE_ONE
            }
        }
        Fusion::Blend => {
            let wsum = (k.w_c + k.w_j).max(1);
            (k.w_c.saturating_mul(c) + k.w_j.saturating_mul(j)) / wsum
        }
    };
    out.clamp(0, SCORE_ONE)
}
