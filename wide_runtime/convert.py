"""Convert a merged float HF encoder (ModernBERT) into a wide Q32.32 .imwq artifact.

Far simpler than the int8 converter: NO calibration, NO {mul,shift} scales, NO SmoothQuant,
NO activation LUTs. Every weight becomes Q32.32 (fp_from_float); the op-list mirrors the int8
encoder exactly (EMBED → [LN → ATTN → proj → ADD → LN → GeGLU-FFN → ADD]×L → final-LN →
POOL(mean) → head.dense → GELU → head.norm → SCORE). Activations stay Q32.32 throughout.

The HF loaders / tolerant weight-name resolution (load_encoder, first_present, get_param)
are in hf_helpers.py, extracted from the int8 pipeline so this tree stands alone.
"""

from __future__ import annotations

import numpy as np

from .format import WideWriter, WideOp, Op, FLAG_MEAN_POOL, FLAG_CAUSAL
from .golden import fixed as F
from .golden import ops as O

# HF plumbing (loaders + tolerant name resolution), extracted from the int8 pipeline
from .hf_helpers import load_encoder, get_param, first_present  # noqa: F401


def _q(arr: np.ndarray) -> np.ndarray:
    """Float ndarray -> Q32.32 object array (round-to-nearest, boundary conversion)."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    q = np.array([F.fp_from_float(float(v)) for v in flat], dtype=object)
    return q.reshape(arr.shape)


class _B:
    """Tiny builder: allocates slots, adds Q32.32 weight tensors, appends ops."""
    def __init__(self):
        self.w = WideWriter()
        self._slot = 0

    def slot(self):
        s = self._slot; self._slot += 1; return s

    def wt(self, name, arr):
        return self.w.add_tensor(name, _q(arr))

    def op(self, **kw):
        self.w.add_op(WideOp(**kw))


def convert_encoder_from_sd(sd: dict, cfg: dict, out_path: str) -> dict:
    H = cfg["hidden_size"]; n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]; head_dim = H // n_heads
    inter = cfg["intermediate_size"]
    eps_q = F.fp_from_float(cfg.get("layer_norm_eps", 1e-5) or 1e-5)
    b = _B()
    prefix = _find_layer_prefix(sd)

    emb_w = first_present(sd, ["model.embeddings.tok_embeddings.weight",
                               "bert.embeddings.word_embeddings.weight",
                               "embeddings.word_embeddings.weight"])
    emb_idx = b.wt("emb", emb_w)
    tok = b.slot()
    b.op(op=Op.EMBED, in0=tok, out=tok, weight_idx=emb_idx)
    h = tok

    pos_w = first_present(sd, ["bert.embeddings.position_embeddings.weight",
                               "embeddings.position_embeddings.weight"], required=False)
    rope = pos_w is None
    if pos_w is not None:
        pos_idx = b.wt("pos", pos_w)
        o = b.slot(); b.op(op=Op.POS_ADD, in0=h, out=o, weight_idx=pos_idx); h = o

    # embedding norm (ModernBERT: embeddings.norm)
    en = first_present(sd, ["model.embeddings.norm.weight", "bert.embeddings.LayerNorm.weight"],
                       required=False)
    if en is not None:
        h = _ln(b, sd, ["model.embeddings.norm", "bert.embeddings.LayerNorm"], h, eps_q, H)

    # Bake the RoPE cos/sin tables into the artifact as ordinary Q32.32 tensors.
    #
    # Not an optimisation — a correctness fix. The table is built from math.cos/math.sin, and
    # a runtime that computes it at load time needs floats to do so. On wasm32 that is not
    # available (the module is integer-only by construction, which is the determinism
    # guarantee), and the Rust interpreter's fallback was `#[cfg(not(target_arch = "wasm32"))]`
    # — so on the actual deployment target the table stayed all zeros, `rope()` computed
    # x*0 - x*0 = 0, and Q and K were zeroed in every attention layer. Nothing errored; the
    # model returned a confident score with attention destroyed, and only on wasm, which is
    # why the native `cargo test` suite stayed green.
    #
    # Baking it here computes it ONCE, at conversion, where float is already allowed (the same
    # boundary that converts every weight) and where both runtimes then read identical
    # integers. ~8 MB for ModernBERT-base's 8192 x 64 — negligible against a 1.2 GB artifact.
    rope_cos_idx = rope_sin_idx = -1
    if rope:
        max_seq = int(cfg.get("max_position_embeddings", 8192))
        theta = float(cfg.get("rope_theta", 10000.0))
        cos_t, sin_t = O.build_rope_table(max_seq, head_dim, theta)
        rope_cos_idx = b.w.add_tensor("rope.cos", cos_t)
        rope_sin_idx = b.w.add_tensor("rope.sin", sin_t)

    for li in range(n_layers):
        h = _layer(b, sd, prefix, li, h, H, n_heads, head_dim, eps_q, rope,
                   rope_cos_idx, rope_sin_idx)

    fn = first_present(sd, ["model.final_norm.weight", "final_norm.weight"], required=False)
    if fn is not None and fn.ndim == 1:
        h = _ln(b, sd, ["model.final_norm", "final_norm"], h, eps_q, H)

    use_mean = str(cfg.get("classifier_pooling", "mean")).lower() == "mean"
    pooled = b.slot(); b.op(op=Op.POOL, in0=h, out=pooled, flags=(FLAG_MEAN_POOL if use_mean else 0))

    head_dense = first_present(sd, ["head.dense.weight"], required=False)
    if head_dense is not None:
        hd_i = b.wt("head.dense", head_dense)
        hb = first_present(sd, ["head.dense.bias"], required=False)
        hb_i = b.wt("head.dense.bias", hb) if hb is not None else -1
        dense = b.slot(); b.op(op=Op.QMATMUL, in0=pooled, out=dense, weight_idx=hd_i, bias_idx=hb_i)
        acted = b.slot(); b.op(op=Op.ACT_GELU, in0=dense, out=acted)
        hn = first_present(sd, ["head.norm.weight"], required=False)
        if hn is not None:
            acted = _ln(b, sd, ["head.norm"], acted, eps_q, H, in_slot=acted)
        pooled = acted

    score_w = first_present(sd, ["classifier.weight", "score.weight"])
    score_b = first_present(sd, ["classifier.bias", "score.bias"], required=False)
    sc_i = b.wt("score", score_w)
    sb_i = b.wt("score.bias", score_b) if score_b is not None else -1
    out = b.slot(); b.op(op=Op.SCORE, in0=pooled, out=out, weight_idx=sc_i, bias_idx=sb_i)

    b.w.set_config(arch="encoder", hidden_size=H, num_hidden_layers=n_layers,
                   num_attention_heads=n_heads, head_dim=head_dim, intermediate_size=inter,
                   max_seq_len=cfg.get("max_position_embeddings", 8192),
                   rope_theta_milli=int(cfg.get("rope_theta", 0) * 1000), rope=rope)
    h_hash = b.w.write(out_path)
    return {"hash": h_hash, "arch": "encoder", "layers": n_layers, "hidden": H, "path": out_path}


def _ln(b, sd, prefixes, slot, eps_q, H, in_slot=None):
    """Emit a LAYERNORM op. prefixes are candidate module prefixes (weight/bias)."""
    g = None
    for p in prefixes:
        g = sd.get(p + ".weight")
        if g is not None:
            beta = sd.get(p + ".bias"); break
    gi = b.wt(prefixes[0] + ".g", g)
    bi = b.wt(prefixes[0] + ".b", beta) if beta is not None else -1
    src = in_slot if in_slot is not None else slot
    out = b.slot(); b.op(op=Op.LAYERNORM, in0=src, out=out, weight_idx=gi, bias_idx=bi, arg0=eps_q)
    return out


def _layer(b, sd, prefix, li, h, H, n_heads, head_dim, eps_q, rope,
           rope_cos_idx=-1, rope_sin_idx=-1):
    lp = f"{prefix}.{li}"
    site = lambda k: [f"{lp}.{k}"]
    width = n_heads * head_dim

    # attn norm
    an = get_param(sd, site("attn_norm.weight") + site("attention.output.LayerNorm.weight"),
                   required=False)
    if an is not None:
        gi = b.wt(f"l{li}.an_g", an)
        anb = get_param(sd, site("attn_norm.bias") + site("attention.output.LayerNorm.bias"),
                        required=False)
        bi = b.wt(f"l{li}.an_b", anb) if anb is not None else -1
        xin = b.slot(); b.op(op=Op.LAYERNORM, in0=h, out=xin, weight_idx=gi, bias_idx=bi, arg0=eps_q)
    else:
        xin = h

    wqkv = get_param(sd, site("attn.Wqkv.weight"), required=False)
    if wqkv is not None:
        wq, wk, wv = np.split(wqkv, 3, axis=0)
    else:
        wq = get_param(sd, site("attention.self.query.weight"))
        wk = get_param(sd, site("attention.self.key.weight"))
        wv = get_param(sd, site("attention.self.value.weight"))
    qi = b.wt(f"l{li}.wq", wq); ki = b.wt(f"l{li}.wk", wk); vi = b.wt(f"l{li}.wv", wv)
    q = b.slot(); b.op(op=Op.QMATMUL, in0=xin, out=q, weight_idx=qi)
    k = b.slot(); b.op(op=Op.QMATMUL, in0=xin, out=k, weight_idx=ki)
    v = b.slot(); b.op(op=Op.QMATMUL, in0=xin, out=v, weight_idx=vi)
    if rope:
        # weight_idx/aux_idx carry the BAKED cos/sin tensors. A runtime that finds them uses
        # them; one that finds -1 falls back to computing the table, which only works where
        # floats are available (i.e. not on wasm32 — see the note in convert_encoder_from_sd).
        qr = b.slot(); b.op(op=Op.ROPE, in0=q, out=qr, arg2=head_dim, arg3=n_heads,
                            weight_idx=rope_cos_idx, aux_idx=rope_sin_idx); q = qr
        kr = b.slot(); b.op(op=Op.ROPE, in0=k, out=kr, arg2=head_dim, arg3=n_heads,
                            weight_idx=rope_cos_idx, aux_idx=rope_sin_idx); k = kr
    ctx = b.slot()
    b.op(op=Op.ATTN, in0=q, in1=k, out=ctx, arg0=head_dim, arg1=n_heads, arg2=n_heads, arg3=v,
         flags=0)
    wo = get_param(sd, site("attn.Wo.weight") + site("attention.output.dense.weight"))
    oi = b.wt(f"l{li}.wo", wo)
    proj = b.slot(); b.op(op=Op.QMATMUL, in0=ctx, out=proj, weight_idx=oi)
    h2 = b.slot(); b.op(op=Op.ADD, in0=h, in1=proj, out=h2); h = h2

    # mlp norm
    mn = get_param(sd, site("mlp_norm.weight") + site("output.LayerNorm.weight"), required=False)
    if mn is not None:
        gi = b.wt(f"l{li}.mn_g", mn)
        mnb = get_param(sd, site("mlp_norm.bias") + site("output.LayerNorm.bias"), required=False)
        bi = b.wt(f"l{li}.mn_b", mnb) if mnb is not None else -1
        xin = b.slot(); b.op(op=Op.LAYERNORM, in0=h, out=xin, weight_idx=gi, bias_idx=bi, arg0=eps_q)
    else:
        xin = h

    w1 = get_param(sd, site("mlp.Wi.weight") + site("intermediate.dense.weight"))
    w2 = get_param(sd, site("mlp.Wo.weight") + site("output.dense.weight"))
    if w1.shape[0] == 2 * (w2.shape[1]):
        # ModernBERT GeGLU: Wi -> [input, gate]; out = Wo( gelu(input) * gate )
        w_in, w_gate = np.split(w1, 2, axis=0)
        gi = b.wt(f"l{li}.mlp_in", w_in); ui = b.wt(f"l{li}.mlp_gate", w_gate)
        # SWIGLU op but with GELU activation on the gate branch: emit as gate=gelu(x@w_in),
        # up=x@w_gate, prod. Reuse SWIGLU wiring with a GELU flag via aux; simplest: inline.
        inp = b.slot(); b.op(op=Op.QMATMUL, in0=xin, out=inp, weight_idx=gi)
        gate = b.slot(); b.op(op=Op.ACT_GELU, in0=inp, out=gate)
        up = b.slot(); b.op(op=Op.QMATMUL, in0=xin, out=up, weight_idx=ui)
        # elementwise product via a dedicated ADD-like? use SWIGLU semantics: reuse a MUL op.
        prod = b.slot(); b.op(op=Op.SWIGLU, in0=gate, in1=up, out=prod, flags=1)  # flags=1: already-activated product
        d2 = b.wt(f"l{li}.w2", w2)
        ff = b.slot(); b.op(op=Op.QMATMUL, in0=prod, out=ff, weight_idx=d2)
    else:
        i1 = b.wt(f"l{li}.w1", w1)
        pre = b.slot(); b.op(op=Op.QMATMUL, in0=xin, out=pre, weight_idx=i1)
        act = b.slot(); b.op(op=Op.ACT_GELU, in0=pre, out=act)
        d2 = b.wt(f"l{li}.w2", w2)
        ff = b.slot(); b.op(op=Op.QMATMUL, in0=act, out=ff, weight_idx=d2)
    h3 = b.slot(); b.op(op=Op.ADD, in0=h, in1=ff, out=h3)
    return h3


def _find_layer_prefix(sd: dict) -> str:
    for cand in ("model.layers", "bert.encoder.layer", "encoder.layer", "layers"):
        if any(k.startswith(cand + ".0.") for k in sd):
            return cand
    raise ValueError("could not find layer prefix in state dict")
