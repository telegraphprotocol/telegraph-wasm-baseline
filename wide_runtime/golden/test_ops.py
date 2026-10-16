"""Wide-op tests vs float — Stage 2 verification. Run:
    python -m wide_runtime.golden.test_ops
Because activations stay Q32.32 (not int8), these should match float TIGHTLY (~1e-3),
unlike the int8 ops which only tracked loosely.
"""
from __future__ import annotations
import numpy as np
from . import fixed as F
from . import ops as O

rng = np.random.default_rng(0)
Q = np.vectorize(F.fp_from_float, otypes=[object])
D = np.vectorize(F.to_float, otypes=[float])


def relerr(gotq, want):
    got = D(gotq).astype(float)
    return float(np.max(np.abs(got - want)) / (np.max(np.abs(want)) + 1e-9))


def main() -> int:
    r = []
    def chk(name, e, tol):
        ok = e <= tol; r.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} relerr={e:.2e} tol={tol:.0e}")

    # matmul
    x = rng.normal(0, 0.5, (4, 32)); w = rng.normal(0, 0.3, (16, 32))
    got = O.qmatmul(Q(x), Q(w))
    chk("qmatmul", relerr(got, x @ w.T), 1e-3)

    # rmsnorm
    x = rng.normal(0, 0.8, (3, 64)); g = rng.normal(1, 0.1, 64); eps = 1e-5
    ms = np.mean(x*x, axis=1, keepdims=True); ref = x/np.sqrt(ms+eps)*g
    got = O.rmsnorm(Q(x), Q(g), F.fp_from_float(eps))
    chk("rmsnorm", relerr(got, ref), 5e-3)

    # layernorm
    mu = x.mean(1, keepdims=True); var = ((x-mu)**2).mean(1, keepdims=True)
    b = rng.normal(0, 0.05, 64); ref = (x-mu)/np.sqrt(var+eps)*g + b
    got = O.layernorm(Q(x), Q(g), Q(b), F.fp_from_float(eps))
    chk("layernorm", relerr(got, ref), 5e-3)

    # gelu / silu
    x = rng.normal(0, 1.5, (5, 20))
    import math
    gelu = lambda z: 0.5*z*(1+np.tanh(math.sqrt(2/math.pi)*(z+0.044715*z**3)))
    chk("gelu", relerr(O.act_gelu(Q(x)), gelu(x)), 3e-3)
    silu = lambda z: z/(1+np.exp(-np.clip(z,-30,30)))
    chk("silu", relerr(O.act_silu(Q(x)), silu(x)), 3e-3)

    # softmax attention (1 head)
    hd = 16; seq = 6
    q = rng.normal(0,0.4,(seq,hd)); k = rng.normal(0,0.4,(seq,hd)); v = rng.normal(0,0.4,(seq,hd))
    scale = 1/math.sqrt(hd)
    sc = (q@k.T)*scale; sc = sc - sc.max(1,keepdims=True)
    wts = np.exp(sc); wts /= wts.sum(1,keepdims=True); ref = wts@v
    got = O.softmax_attn(Q(q), Q(k), Q(v), F.fp_from_float(scale), causal=False)
    chk("softmax_attn", relerr(got, ref), 1e-2)

    print("\n" + "="*40)
    ok = all(r); print(f"OVERALL: {'PASS' if ok else 'FAIL'} ({sum(r)}/{len(r)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
