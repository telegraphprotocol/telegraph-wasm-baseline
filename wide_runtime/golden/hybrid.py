"""Q32.32 hybrid scoring golden reference — comparator core + fusion gate.

The Python spec that `wide_wasm/src/hybrid.rs` must match bit-for-bit, in the same role
`fixed.py` and `ops.py` play for the forward pass. `wide_wasm/tests/hybrid_golden.rs` asserts
that against `hybrid_fixture.json`, generated from here by `tests/gen_hybrid_fixture.py`.

A fusion gate that is wrong in the third decimal is not detectable by eye, and a hybrid score
looks entirely plausible when the comparator and the judgment have been combined slightly
wrong — so the parity has to be pinned, not inspected.

Division convention: every divide below has a POSITIVE denominator and a non-negative
numerator (the inputs are clamped into [0, SCORE_ONE] and the tolerance window is guarded
`> 0`), so Python's floor-toward-negative-infinity `//` and Rust's truncate-toward-zero `/`
agree here. `_tdiv` is used anyway, explicitly, so the property is stated rather than relied
on by accident — if a future constant is allowed to go negative the two runtimes will not
silently diverge.
"""

from __future__ import annotations

# Fixed-point [0,1] score domain: 1.0 == 10000 (score basis points).
SCORE_ONE = 10_000

# Comparator codes — MUST match hybrid.rs Comparator and the Go side's Spec.Code.
EXACT = 0
ABS_TOL = 1
REL_TOL = 2
PASS_FAIL = 3

# Fusion modes — MUST match hybrid.rs Fusion and the Go side's Fusion.Mode.
GATE = 0
BLEND = 1

_I64_MAX = (1 << 63) - 1
_I64_MIN = -(1 << 63)


def _tdiv(a: int, b: int) -> int:
    """Truncate-toward-zero integer division — matches Rust `/` and C.

    Python `//` floors toward -inf and would diverge from the Rust runtime on any negative
    numerator. See the module docstring: the current call sites cannot produce one, but the
    convention is what keeps that a fact rather than a coincidence.
    """
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _sat_mul(a: int, b: int) -> int:
    """i64 saturating multiply — mirrors Rust `saturating_mul`.

    The Rust side saturates rather than wrapping so an absurd tolerance or weight drives the
    score to a bound instead of wrapping to the opposite sign, which would turn a failing
    miner into a winning one.
    """
    v = a * b
    if v > _I64_MAX:
        return _I64_MAX
    if v < _I64_MIN:
        return _I64_MIN
    return v


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


def linear_decay(error: int, tol: int) -> int:
    """`error` 0 -> SCORE_ONE, `error` == 2*tol -> 0, clamped. `tol <= 0` means exact."""
    if tol <= 0:
        return SCORE_ONE if error == 0 else 0
    window = _sat_mul(tol, 2)
    if error >= window:
        return 0
    return _tdiv(_sat_mul(SCORE_ONE, window - error), window)


def comparator_core(cmp: int, miner: int, truth: int, tolerance: int) -> int:
    """The fact-check half, in [0, SCORE_ONE]. Pure integer."""
    diff = abs(miner - truth)
    if cmp in (EXACT, PASS_FAIL):
        return SCORE_ONE if miner == truth else 0
    if cmp == ABS_TOL:
        return linear_decay(diff, tolerance)
    if cmp == REL_TOL:
        # relative error in basis points. The intermediate is unbounded in Python and i128 in
        # Rust, so neither overflows at wei/1e18-token magnitudes — the int64 version of this
        # saturated and scored a wildly wrong value near-perfect.
        denom = max(abs(truth), 1)
        err_bps = min(_tdiv(diff * 10_000, denom), _I64_MAX)
        return linear_decay(err_bps, tolerance)
    raise ValueError(f"unknown comparator code {cmp}")


def fuse(mode: int, core: int, judgment: int, *, pass_: int, low: int, floor: int, span: int,
         w_c: int = 0, w_j: int = 0) -> int:
    """Combine the two halves into the final [0, SCORE_ONE] score.

    Under the gate (mode 0, the only one wired), a miner below `pass_` on the fact check is
    capped at a low fact-only score and cannot be rescued by prose; above it, the adapter's
    judgment maps into [floor, floor+span].
    """
    c = _clamp(core, 0, SCORE_ONE)
    j = _clamp(judgment, 0, SCORE_ONE)
    if mode == GATE:
        if c < pass_:
            out = _tdiv(_sat_mul(c, low), SCORE_ONE)
        else:
            out = floor + _tdiv(_sat_mul(span, j), SCORE_ONE)
    elif mode == BLEND:
        wsum = max(w_c + w_j, 1)
        out = _tdiv(_sat_mul(w_c, c) + _sat_mul(w_j, j), wsum)
    else:
        raise ValueError(f"unknown fusion mode {mode}")
    return _clamp(out, 0, SCORE_ONE)


def judgment_to_score_domain(code: int) -> int:
    """Raw Q32.32 score code -> [0, SCORE_ONE] judgment.

    Sigmoid, then scale to basis points. This must agree with `judgment_to_score_domain` in
    `wide_wasm/src/lib.rs` AND with `WideToScoreDomain` in `pkg/wasm/runtime/wide.go`: the
    same adapter has to grade the same whether the caller went through the hybrid export or
    scored bare and let the host convert.

    The sigmoid is not cosmetic. The head is a sequence-classification reward head
    (num_labels=1) emitting an unbounded LOGIT — the real CODE_REVIEW adapter returns about
    -2.32 — and a plain divide-and-clamp puts every negative-logit miner at 0, where they all
    tie at the bottom.
    """
    from . import fixed as F
    unit = F.fp_sigmoid(code)
    bps = (unit * SCORE_ONE) >> F.FP_PRECISION
    return _clamp(bps, 0, SCORE_ONE)


def spec_buffer(comparator: int, tolerance: int, fusion_mode: int, pass_: int, low: int,
                floor: int, span: int, w_c: int = 0, w_j: int = 0) -> list[int]:
    """The exact ten-int buffer `imwq_score_hybrid` reads, in order.

    Positional, no length argument: a short buffer is read past its end and a reordered one
    fuses against the wrong thresholds, and neither fails loudly. Mirrors
    `comparator.Fusion.Buffer()` in pkg/scoring/comparator/specs_db.go.
    """
    return [comparator, tolerance, fusion_mode, pass_, low, floor, span, w_c, w_j, 0]
