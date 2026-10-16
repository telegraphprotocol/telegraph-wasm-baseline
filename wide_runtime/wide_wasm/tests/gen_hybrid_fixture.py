"""Emit the comparator + fusion parity fixture, flat keys, from the Python golden.

Run:
    python wide_runtime/wide_wasm/tests/gen_hybrid_fixture.py

Same flat-key shape as gen_fixture.py so the Rust test needs only the trivial array/scalar
reader (no serde dep). Each case is a parallel array: cmp[i], miner[i], truth[i], tol[i] ->
core[i]; and mode[i], fcore[i], judg[i], pass/low/floor/span/w_c/w_j[i] -> fused[i].

The cases are chosen to cover the places the two runtimes could silently disagree:
  - negative values through RelTol (Python `//` vs Rust `/` on the same numbers)
  - the exact decay boundary (error == 2*tol, which must be 0, not 1)
  - wei-magnitude values that overflowed the old i64 RelTol intermediate
  - a comparator score straddling the gate's `pass` threshold in both directions
"""
from __future__ import annotations

import json
from pathlib import Path

from wide_runtime.golden import hybrid as H

fx: dict[str, object] = {}

# --- comparator_core cases -------------------------------------------------- #
# (comparator, miner, truth, tolerance)
cmp_cases = [
    (H.EXACT, 100, 100, 0),            # equal -> full
    (H.EXACT, 100, 101, 0),            # off by one -> zero, no partial credit
    (H.PASS_FAIL, 1, 1, 0),
    (H.PASS_FAIL, 0, 1, 0),
    (H.ABS_TOL, 100, 100, 50),         # zero error inside a window -> full
    (H.ABS_TOL, 125, 100, 50),         # half the window
    (H.ABS_TOL, 200, 100, 50),         # exactly 2*tol -> must be 0, not 1
    (H.ABS_TOL, 201, 100, 50),         # past the window
    (H.ABS_TOL, 75, 100, 50),          # negative direction, same magnitude
    (H.ABS_TOL, 100, 100, 0),          # tol<=0 degenerates to exact
    (H.ABS_TOL, 101, 100, 0),
    (H.REL_TOL, 10_000, 10_000, 100),  # 0 bps error
    (H.REL_TOL, 10_050, 10_000, 100),  # 50 bps error, half the window
    (H.REL_TOL, 10_200, 10_000, 100),  # 200 bps == 2*tol -> 0
    (H.REL_TOL, 9_950, 10_000, 100),   # negative direction
    (H.REL_TOL, -10_050, -10_000, 100),  # both negative — truth.abs() in the denominator
    (H.REL_TOL, 100, -100, 100),       # sign flip: 20000 bps error, far past the window
    # wei magnitudes: the i64 intermediate saturated here and scored a wrong value ~perfect.
    (H.REL_TOL, 1_000_000_000_000_000_050, 1_000_000_000_000_000_000, 100),
    (H.REL_TOL, 1_100_000_000_000_000_000, 1_000_000_000_000_000_000, 100),
    (H.REL_TOL, 100, 0, 100),          # zero truth -> denominator guarded to 1
]
fx["cmp_n"] = len(cmp_cases)
fx["cmp_kind"] = [c[0] for c in cmp_cases]
fx["cmp_miner"] = [c[1] for c in cmp_cases]
fx["cmp_truth"] = [c[2] for c in cmp_cases]
fx["cmp_tol"] = [c[3] for c in cmp_cases]
fx["cmp_out"] = [H.comparator_core(*c) for c in cmp_cases]

# --- fuse cases ------------------------------------------------------------- #
# (mode, core, judgment, pass, low, floor, span, w_c, w_j)
fuse_cases = [
    (H.GATE, 10_000, 10_000, 8_000, 2_000, 3_000, 7_000, 0, 0),  # perfect fact + perfect prose
    (H.GATE, 10_000, 0, 8_000, 2_000, 3_000, 7_000, 0, 0),       # perfect fact, worthless prose
    (H.GATE, 10_000, 5_000, 8_000, 2_000, 3_000, 7_000, 0, 0),   # mid judgment
    (H.GATE, 8_000, 5_000, 8_000, 2_000, 3_000, 7_000, 0, 0),    # exactly at pass -> unlocked
    (H.GATE, 7_999, 5_000, 8_000, 2_000, 3_000, 7_000, 0, 0),    # one below pass -> gated shut
    (H.GATE, 0, 10_000, 8_000, 2_000, 3_000, 7_000, 0, 0),       # wrong fact, perfect prose
    (H.GATE, 4_000, 10_000, 8_000, 2_000, 3_000, 7_000, 0, 0),   # half-right fact, gated
    (H.GATE, 12_000, 12_000, 8_000, 2_000, 3_000, 7_000, 0, 0),  # out-of-domain inputs clamped
    (H.GATE, -500, 5_000, 8_000, 2_000, 3_000, 7_000, 0, 0),     # negative core clamped to 0
    (H.GATE, 10_000, 3_333, 8_000, 2_000, 3_000, 7_000, 0, 0),   # truncation-visible judgment
    (H.BLEND, 10_000, 0, 8_000, 2_000, 3_000, 7_000, 1, 1),
    (H.BLEND, 10_000, 0, 8_000, 2_000, 3_000, 7_000, 3, 1),
    (H.BLEND, 3_333, 6_667, 8_000, 2_000, 3_000, 7_000, 1, 2),
    (H.BLEND, 5_000, 5_000, 8_000, 2_000, 3_000, 7_000, 0, 0),   # zero weights -> wsum guard
]
fx["fuse_n"] = len(fuse_cases)
fx["fuse_mode"] = [c[0] for c in fuse_cases]
fx["fuse_core"] = [c[1] for c in fuse_cases]
fx["fuse_judg"] = [c[2] for c in fuse_cases]
fx["fuse_pass"] = [c[3] for c in fuse_cases]
fx["fuse_low"] = [c[4] for c in fuse_cases]
fx["fuse_floor"] = [c[5] for c in fuse_cases]
fx["fuse_span"] = [c[6] for c in fuse_cases]
fx["fuse_wc"] = [c[7] for c in fuse_cases]
fx["fuse_wj"] = [c[8] for c in fuse_cases]
fx["fuse_out"] = [
    H.fuse(m, core, judg, pass_=p, low=lo, floor=fl, span=sp, w_c=wc, w_j=wj)
    for (m, core, judg, p, lo, fl, sp, wc, wj) in fuse_cases
]

# --- judgment_to_score_domain cases ----------------------------------------- #
# Raw Q32.32 logits, including the real CODE_REVIEW adapter's observed -9984098080 (~ -2.32)
# and both saturation tails.
FP_ONE = 1 << 32
judg_cases = [
    0,                       # logit 0 -> 0.5 -> 5000 bps
    FP_ONE,                  # +1.0
    -FP_ONE,                 # -1.0
    -9_984_098_080,          # the real CODE_REVIEW adapter's output, ~ -2.32
    3 * FP_ONE,
    -3 * FP_ONE,
    20 * FP_ONE,             # past the +10 saturation
    -20 * FP_ONE,            # past the -10 saturation
    1,                       # one ULP above zero
    -1,
]
fx["judg_n"] = len(judg_cases)
fx["judg_in"] = judg_cases
fx["judg_out"] = [H.judgment_to_score_domain(c) for c in judg_cases]

Path(__file__).with_name("hybrid_fixture.json").write_text(
    json.dumps(fx, separators=(",", ":")), encoding="utf-8")
print("wrote hybrid_fixture.json —",
      f"{fx['cmp_n']} comparator, {fx['fuse_n']} fusion, {fx['judg_n']} judgment cases")
