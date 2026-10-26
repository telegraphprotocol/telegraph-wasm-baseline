"""Emit a small synthetic 1-layer encoder .imwq that USES RoPE, plus its Python-golden score.

The sibling gen_model_fixture.py builds a model with `rope=False`, so nothing in the golden
suite ever executes the ROPE opcode. The real ModernBERT adapters DO use it — the converted
VULNERABILITY_TRIAGE artifact contains 44 ROPE ops — so the one op that most separates the
real workload from the tested one had no parity coverage at all.

That gap hid a live bug: in `interp.rs::build_rope` the table-fill loop is annotated
`#[cfg(not(target_arch = "wasm32"))]`, so on the wasm32 target the cos/sin tables stay all
zeros. `ops::rope` then computes `x1*0 - x2*0 = 0` for every rotated position, zeroing Q and
K in every attention layer. Nothing errors; the model returns a confident number computed
with attention destroyed. Native builds fill the table and are correct, which is exactly why
`cargo test` (native) stayed green.

Run:
    python wide_runtime/wide_wasm/tests/gen_rope_model_fixture.py

Writes rope_model_fixture.imwq and rope_model_fixture.txt (tokens + expected i64 score).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wide_runtime.format import WideWriter, WideReader, WideOp, Op, FLAG_MEAN_POOL
from wide_runtime.interp import WideInterp
from wide_runtime.golden import fixed as F
from wide_runtime.golden import ops as O

rng = np.random.default_rng(11)
Q = lambda a: np.array([F.fp_from_float(float(v)) for v in np.asarray(a).reshape(-1)],  # noqa: E731
                       dtype=object).reshape(np.asarray(a).shape)

H = 16
V = 20
heads = 2
hd = H // heads
inter = 32

w = WideWriter()
emb = rng.normal(0, 0.3, (V, H))
ln1g = np.ones(H) + rng.normal(0, 0.05, H)
ln1b = rng.normal(0, 0.02, H)
wq = rng.normal(0, 0.3, (H, H))
wk = rng.normal(0, 0.3, (H, H))
wv = rng.normal(0, 0.3, (H, H))
wo = rng.normal(0, 0.3, (H, H))
ln2g = np.ones(H) + rng.normal(0, 0.05, H)
ln2b = rng.normal(0, 0.02, H)
w1 = rng.normal(0, 0.3, (inter, H))
w2 = rng.normal(0, 0.3, (H, inter))
lnfg = np.ones(H) + rng.normal(0, 0.05, H)
lnfb = rng.normal(0, 0.02, H)
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

# Baked RoPE cos/sin, exactly as the converter now emits them. Without these the wasm32
# build has no way to build the table (it has no floats) and silently rotates by zero.
MAX_SEQ = 64
THETA = 160_000.0
cos_t, sin_t = O.build_rope_table(MAX_SEQ, hd, THETA)
rc = w.add_tensor("rope.cos", cos_t)
rs = w.add_tensor("rope.sin", sin_t)

# EMBED -> LN1 -> (Q,K,V) -> ROPE(q) -> ROPE(k) -> ATTN -> proj -> ADD -> LN2 ->
# W1 -> GELU -> W2 -> ADD -> LNf -> POOL -> SCORE
w.add_op(WideOp(op=Op.EMBED, in0=0, out=0, weight_idx=ei))
w.add_op(WideOp(op=Op.LAYERNORM, in0=0, out=1, weight_idx=g1, bias_idx=b1, arg0=eps))
w.add_op(WideOp(op=Op.QMATMUL, in0=1, out=2, weight_idx=qi))
w.add_op(WideOp(op=Op.QMATMUL, in0=1, out=3, weight_idx=ki))
w.add_op(WideOp(op=Op.QMATMUL, in0=1, out=4, weight_idx=vi))
w.add_op(WideOp(op=Op.ROPE, in0=2, out=5, arg2=hd, arg3=heads, weight_idx=rc, aux_idx=rs))
w.add_op(WideOp(op=Op.ROPE, in0=3, out=6, arg2=hd, arg3=heads, weight_idx=rc, aux_idx=rs))
w.add_op(WideOp(op=Op.ATTN, in0=5, in1=6, out=7, arg0=hd, arg1=heads, arg2=heads, arg3=4))
w.add_op(WideOp(op=Op.QMATMUL, in0=7, out=8, weight_idx=oi))
w.add_op(WideOp(op=Op.ADD, in0=0, in1=8, out=9))
w.add_op(WideOp(op=Op.LAYERNORM, in0=9, out=10, weight_idx=g2, bias_idx=b2, arg0=eps))
w.add_op(WideOp(op=Op.QMATMUL, in0=10, out=11, weight_idx=i1))
w.add_op(WideOp(op=Op.ACT_GELU, in0=11, out=12))
w.add_op(WideOp(op=Op.QMATMUL, in0=12, out=13, weight_idx=i2))
w.add_op(WideOp(op=Op.ADD, in0=9, in1=13, out=14))
w.add_op(WideOp(op=Op.LAYERNORM, in0=14, out=15, weight_idx=gf, bias_idx=bf, arg0=eps))
w.add_op(WideOp(op=Op.POOL, in0=15, out=16, flags=FLAG_MEAN_POOL))
w.add_op(WideOp(op=Op.SCORE, in0=16, out=17, weight_idx=si))

# rope=True and a real theta, matching what the converter emits for ModernBERT.
w.set_config(arch="encoder", hidden_size=H, num_hidden_layers=1, num_attention_heads=heads,
             head_dim=hd, intermediate_size=inter, max_seq_len=MAX_SEQ,
             rope_theta_milli=int(THETA * 1000), rope=True)

here = Path(__file__).parent
h = w.write(str(here / "rope_model_fixture.imwq"))
tokens = [2, 11, 4, 17, 8]
m = WideReader(w.to_bytes())
it = WideInterp(m)
it.run(np.array(tokens))
raw = int(np.asarray(it.slots[17]).reshape(-1)[0])
(here / "rope_model_fixture.txt").write_text(
    f"{','.join(map(str, tokens))}\n{raw}\n", encoding="utf-8")
print("wrote rope_model_fixture.imwq (sha", h[:12], ") tokens", tokens, "raw_score_q", raw)

# --- and the SAME model with the tables NOT baked ------------------------------------- #
# Byte-for-byte identical except that the two ROPE ops carry weight_idx/aux_idx = -1, which
# is what every artifact written before this fix looks like. The Rust test loads this one to
# prove two things: that the unbaked path reaches a DIFFERENT answer than the baked one (so
# the parity test is testing something rather than passing either way), and that the wasm32
# build refuses it instead of scoring with attention destroyed.
w.m.ops[5].weight_idx = -1
w.m.ops[5].aux_idx = -1
w.m.ops[6].weight_idx = -1
w.m.ops[6].aux_idx = -1
h2 = w.write(str(here / "rope_model_fixture_unbaked.imwq"))

# What the unbaked artifact scored when the table was all zeros — i.e. exactly what the
# wasm32 build silently returned before the fix. Computed by handing the golden a zeroed
# table, so the Rust test can assert the two differ rather than assume they must.
zeros = np.zeros((MAX_SEQ, hd), dtype=object)
m2 = WideReader(w.to_bytes())
it2 = WideInterp(m2)
it2._rope = (zeros, zeros)
it2.run(np.array(tokens))
raw_zero = int(np.asarray(it2.slots[17]).reshape(-1)[0])
(here / "rope_model_fixture_unbaked.txt").write_text(
    f"{','.join(map(str, tokens))}\n{raw_zero}\n", encoding="utf-8")
print("wrote rope_model_fixture_unbaked.imwq (sha", h2[:12], ") zero-table raw_score_q", raw_zero)
if raw == raw_zero:
    raise SystemExit("FIXTURE USELESS: the zeroed RoPE table reaches the same score as the "
                     "baked one, so a parity test on it would pass either way")
print("baked vs zero-table differ:", raw, "!=", raw_zero)
