"""Emit a small synthetic 1-layer encoder .imwq + its Python-golden score, so the Rust
interpreter can be checked end-to-end for bit-identity. Run:
    python wide_runtime/wide_wasm/tests/gen_model_fixture.py
Writes model_fixture.imwq and model_fixture.txt (tokens + expected i64 score).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from wide_runtime.format import WideWriter, WideReader, WideOp, Op, FLAG_MEAN_POOL, FLAG_CAUSAL
from wide_runtime.interp import WideInterp
from wide_runtime.golden import fixed as F

rng = np.random.default_rng(7)
Q = lambda a: np.array([F.fp_from_float(float(v)) for v in np.asarray(a).reshape(-1)],
                       dtype=object).reshape(np.asarray(a).shape)

H = 16; V = 20; heads = 2; hd = H // heads; inter = 32
w = WideWriter()
emb = rng.normal(0, 0.3, (V, H))
ln1g = np.ones(H) + rng.normal(0, 0.05, H); ln1b = rng.normal(0, 0.02, H)
wq = rng.normal(0, 0.3, (H, H)); wk = rng.normal(0, 0.3, (H, H)); wv = rng.normal(0, 0.3, (H, H))
wo = rng.normal(0, 0.3, (H, H))
ln2g = np.ones(H) + rng.normal(0, 0.05, H); ln2b = rng.normal(0, 0.02, H)
w1 = rng.normal(0, 0.3, (inter, H)); w2 = rng.normal(0, 0.3, (H, inter))
lnfg = np.ones(H) + rng.normal(0, 0.05, H); lnfb = rng.normal(0, 0.02, H)
sw = rng.normal(0, 0.3, (1, H))
eps = F.fp_from_float(1e-5)

ei = w.add_tensor("emb", Q(emb))
g1 = w.add_tensor("ln1g", Q(ln1g)); b1 = w.add_tensor("ln1b", Q(ln1b))
qi = w.add_tensor("wq", Q(wq)); ki = w.add_tensor("wk", Q(wk)); vi = w.add_tensor("wv", Q(wv))
oi = w.add_tensor("wo", Q(wo))
g2 = w.add_tensor("ln2g", Q(ln2g)); b2 = w.add_tensor("ln2b", Q(ln2b))
i1 = w.add_tensor("w1", Q(w1)); i2 = w.add_tensor("w2", Q(w2))
gf = w.add_tensor("lnfg", Q(lnfg)); bf = w.add_tensor("lnfb", Q(lnfb))
si = w.add_tensor("score", Q(sw))

# op-list: EMBED -> LN1 -> (Q,K,V) -> ATTN -> proj -> ADD -> LN2 -> W1 -> GELU -> W2 -> ADD -> LNf -> POOL -> SCORE
w.add_op(WideOp(op=Op.EMBED, in0=0, out=0, weight_idx=ei))            # slot0 = h
w.add_op(WideOp(op=Op.LAYERNORM, in0=0, out=1, weight_idx=g1, bias_idx=b1, arg0=eps))
w.add_op(WideOp(op=Op.QMATMUL, in0=1, out=2, weight_idx=qi))
w.add_op(WideOp(op=Op.QMATMUL, in0=1, out=3, weight_idx=ki))
w.add_op(WideOp(op=Op.QMATMUL, in0=1, out=4, weight_idx=vi))
w.add_op(WideOp(op=Op.ATTN, in0=2, in1=3, out=5, arg0=hd, arg1=heads, arg2=heads, arg3=4))
w.add_op(WideOp(op=Op.QMATMUL, in0=5, out=6, weight_idx=oi))
w.add_op(WideOp(op=Op.ADD, in0=0, in1=6, out=7))
w.add_op(WideOp(op=Op.LAYERNORM, in0=7, out=8, weight_idx=g2, bias_idx=b2, arg0=eps))
w.add_op(WideOp(op=Op.QMATMUL, in0=8, out=9, weight_idx=i1))
w.add_op(WideOp(op=Op.ACT_GELU, in0=9, out=10))
w.add_op(WideOp(op=Op.QMATMUL, in0=10, out=11, weight_idx=i2))
w.add_op(WideOp(op=Op.ADD, in0=7, in1=11, out=12))
w.add_op(WideOp(op=Op.LAYERNORM, in0=12, out=13, weight_idx=gf, bias_idx=bf, arg0=eps))
w.add_op(WideOp(op=Op.POOL, in0=13, out=14, flags=FLAG_MEAN_POOL))
w.add_op(WideOp(op=Op.SCORE, in0=14, out=15, weight_idx=si))
w.set_config(arch="encoder", hidden_size=H, num_hidden_layers=1, num_attention_heads=heads,
             head_dim=hd, intermediate_size=inter, max_seq_len=64, rope=False)

here = Path(__file__).parent
h = w.write(str(here / "model_fixture.imwq"))
tokens = [1, 5, 9, 3, 7]
m = WideReader(w.to_bytes())
# raw integer score (before float boundary): re-run and grab slot 15's int
it = WideInterp(m); it.run(np.array(tokens))
raw = int(np.asarray(it.slots[15]).reshape(-1)[0])
(here / "model_fixture.txt").write_text(f"{','.join(map(str,tokens))}\n{raw}\n", encoding="utf-8")
print("wrote model_fixture.imwq (sha", h[:12], ") tokens", tokens, "raw_score_q", raw)
