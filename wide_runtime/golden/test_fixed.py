"""Unit tests for the Q32.32 fixed-point core (golden/fixed.py) — Stage 1 verification.

Each primitive is checked against its float value to the accuracy int-llm documents. Run:
    python -m wide_runtime.golden.test_fixed
Exit 0 = all pass.
"""

from __future__ import annotations

import math

from . import fixed as F


def _relerr(got_q: int, want: float) -> float:
    got = F.to_float(got_q)
    denom = abs(want) if abs(want) > 1e-9 else 1.0
    return abs(got - want) / denom


def _check(name: str, got_q: int, want: float, tol: float, results: list) -> None:
    e = _relerr(got_q, want)
    ok = e <= tol
    results.append(ok)
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:28s} got={F.to_float(got_q):+.6g} want={want:+.6g} relerr={e:.2e} tol={tol:.0e}")


def main() -> int:
    r: list[bool] = []
    print("Q32.32 fixed-point core — vs float\n")

    # round-trip (1 ULP at 2^-48 ~ 3.5e-15 absolute; use a rel tol that covers small values)
    for v in [0.0, 1.0, -1.0, 3.14159, -2.71828, 123.5, -0.0011827]:
        _check(f"roundtrip {v}", F.fp_from_float(v), v, 1e-7, r)

    print()
    # mul / div
    _check("mul 3*0.25", F.fp_mul(F.fp_from_float(3), F.fp_from_float(0.25)), 0.75, 1e-12, r)
    _check("mul -1.5*4", F.fp_mul(F.fp_from_float(-1.5), F.fp_from_float(4)), -6.0, 1e-12, r)
    _check("div 1/4", F.fp_div(F.fp_from_float(1), F.fp_from_float(4)), 0.25, 1e-12, r)
    _check("div -7/2", F.fp_div(F.fp_from_float(-7), F.fp_from_float(2)), -3.5, 1e-12, r)

    print()
    # sqrt / inv_sqrt
    for v in [1.0, 2.0, 4.0, 100.0, 0.25, 1234.5]:
        _check(f"sqrt {v}", F.fp_sqrt(F.fp_from_float(v)), math.sqrt(v), 1e-6, r)
    for v in [1.0, 2.0, 4.0, 100.0, 0.25, 9.0]:
        _check(f"inv_sqrt {v}", F.fp_inv_sqrt(F.fp_from_float(v)), 1.0 / math.sqrt(v), 1e-6, r)

    print()
    # exp: int-llm's dyadic (1+x/2^14)^(2^14) has a real approximation error that grows with
    # |x| (~1e-3 at |x|=5). This is the documented accuracy of fp_exp, not a port bug.
    for v in [0.0, 1.0, -1.0, 2.5, -3.0, 5.0, -5.0]:
        _check(f"exp {v}", F.fp_exp(F.fp_from_float(v)), math.exp(v), 5e-3, r)
    # safe_exp clamps
    _check("safe_exp 20->clamp10", F.fp_safe_exp(F.fp_from_float(20)), math.exp(10), 5e-3, r)
    _check("safe_exp -20->0", F.fp_safe_exp(F.fp_from_float(-20)), 0.0, 1e-9, r)

    print()
    # sigmoid / silu (inherit exp's ~1e-3 error)
    for v in [0.0, 1.0, -1.0, 3.0, -3.0]:
        _check(f"sigmoid {v}", F.fp_sigmoid(F.fp_from_float(v)), 1 / (1 + math.exp(-v)), 1e-3, r)
    for v in [0.0, 1.0, -1.0, 3.0, -3.0]:
        _check(f"silu {v}", F.fp_silu(F.fp_from_float(v)), v / (1 + math.exp(-v)), 1e-3, r)

    print()
    # gelu (tanh approx)
    def gelu(z):
        return 0.5 * z * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))
    for v in [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]:
        _check(f"gelu {v}", F.fp_gelu_tanh(F.fp_from_float(v)), gelu(v), 2e-3, r)

    print("\n" + "=" * 50)
    ok = all(r)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}  ({sum(r)}/{len(r)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
