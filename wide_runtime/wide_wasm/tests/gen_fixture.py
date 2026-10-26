"""Emit a Q32.32 op-parity fixture with FLAT, uniquely-prefixed keys (no nested objects, so
the Rust test needs only a trivial array/scalar reader). Run:
    python wide_runtime/wide_wasm/tests/gen_fixture.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from wide_runtime.golden import fixed as F
from wide_runtime.golden import ops as O

rng = np.random.default_rng(1234)
Q = lambda a: np.array([F.fp_from_float(float(v)) for v in np.asarray(a).reshape(-1)],
                       dtype=object).reshape(np.asarray(a).shape)
flat = lambda o: [int(v) for v in np.asarray(o).reshape(-1)]
fx = {}

# qmatmul
sx, sw = (4, 32), (16, 32)
xq = Q(rng.normal(0, 0.5, sx)); wq = Q(rng.normal(0, 0.3, sw))
fx.update(qm_seq=sx[0], qm_in=sx[1], qm_out=sw[0],
          qm_x=flat(xq), qm_w=flat(wq), qm_o=flat(O.qmatmul(xq, wq)))

# rmsnorm
seq, dim = 3, 64
xq = Q(rng.normal(0, 0.8, (seq, dim))); gq = Q(rng.normal(1, 0.1, dim)); eps = F.fp_from_float(1e-5)
fx.update(rm_seq=seq, rm_dim=dim, rm_eps=eps, rm_x=flat(xq), rm_g=flat(gq),
          rm_o=flat(O.rmsnorm(xq, gq, eps)))

# layernorm
xq = Q(rng.normal(0.2, 0.8, (seq, dim))); gq = Q(rng.normal(1, 0.1, dim)); bq = Q(rng.normal(0, 0.05, dim))
fx.update(ln_seq=seq, ln_dim=dim, ln_eps=eps, ln_x=flat(xq), ln_g=flat(gq), ln_b=flat(bq),
          ln_o=flat(O.layernorm(xq, gq, bq, eps)))

# attention
aseq, hd = 6, 16
qq = Q(rng.normal(0, 0.4, (aseq, hd))); kq = Q(rng.normal(0, 0.4, (aseq, hd))); vq = Q(rng.normal(0, 0.4, (aseq, hd)))
scale = F.fp_from_float(1.0 / math.sqrt(hd))
fx.update(at_seq=aseq, at_hd=hd, at_scale=scale, at_q=flat(qq), at_k=flat(kq), at_v=flat(vq),
          at_o=flat(O.softmax_attn(qq, kq, vq, scale, causal=False)))

# activations
xq = Q(rng.normal(0, 1.5, 40))
fx.update(ac_x=flat(xq), ac_gelu=flat(O.act_gelu(xq)), ac_silu=flat(O.act_silu(xq)))

Path(__file__).with_name("fixture.json").write_text(
    json.dumps(fx, separators=(",", ":")), encoding="utf-8")   # compact: no space after ':'
print("wrote fixture.json — flat keys:", len(fx))
