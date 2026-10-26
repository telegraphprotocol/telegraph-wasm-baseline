"""Wide Q32.32 transformer ops — the golden reference for the wide runtime.

Every activation and weight is a Q32.32 fixed-point value (see fixed.py). Unlike the int8
path, there is NO requantisation between layers and NO {mul,shift}: a slot just holds Q32.32
ints, layer to layer. That is the whole fix — the reward signal is never crushed to 256
levels.

Arithmetic uses NumPy **object arrays of Python ints** so every intermediate (the int128-wide
products int-llm accumulates) is exact and arbitrary-precision. That makes this the bit-exact
oracle the Rust runtime (which uses a real int128 / two-limb backend) must match. Object
arrays are slow, but this is the reference, not the fast path.

Each op mirrors int-llm's arithmetic:
  qmatmul     — accumulate raw products, one >>32 at the end (llama_int matmul_vec)
  rmsnorm     — int-sum-of-squares -> fp_inv_sqrt -> x*inv*gamma (llama_int rmsnorm)
  layernorm   — mean + var -> fp_inv_sqrt -> affine
  softmax_attn— per head: QK^T (>>32), *scale, max-subtract, fp_safe_exp, fp_div normalize, @V
  act (gelu/silu) — computed elementwise via fixed.py (NO lut)
  rope        — precomputed Q32.32 cos/sin table, rotate-half
  residual add— plain Q32.32 int add (same scale — no reconciliation needed)
"""

from __future__ import annotations

import numpy as np

from . import fixed as F

FP = F.FP_PRECISION          # 32
ONE = F.FP_ONE


def _obj(a) -> np.ndarray:
    """View/convert to an object array of Python ints (exact, arbitrary precision)."""
    return np.asarray(a, dtype=object)


def _tdiv(a: int, b: int) -> int:
    """Truncate-toward-zero integer division — matches Rust `/` and C, so the wide Python
    golden and the Rust runtime divide identically (Python `//` floors toward -inf and would
    diverge on negative values, e.g. the LayerNorm mean)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


# --------------------------------------------------------------------------- #
#  Linear — wide × wide, accumulate raw products, single >>32
# --------------------------------------------------------------------------- #

def qmatmul(x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """y[s,o] = ( Σ_k x[s,k]*w[o,k] ) >> 32, all Q32.32. x:[seq,in] w:[out,in] -> [seq,out].

    Accumulates the raw integer products (no per-multiply shift), then one >>32 — exactly
    int-llm's matmul_vec. With object dtype the accumulation is exact regardless of width.
    Optional bias is Q32.32, added AFTER the shift (it is already in the output domain).
    """
    x = _obj(x); w = _obj(w)
    acc = x @ w.T                                  # exact integer accumulation (object)
    y = _shr(acc, FP)                              # single arithmetic shift back to Q32.32
    if bias is not None:
        y = y + _obj(bias)[None, :]
    return y


def _shr(a: np.ndarray, k: int) -> np.ndarray:
    """Arithmetic right shift on an object array (floor toward -inf, like Python >> and C ASR)."""
    return np.vectorize(lambda v: int(v) >> k, otypes=[object])(a)


# --------------------------------------------------------------------------- #
#  Elementwise activations — computed, no LUT
# --------------------------------------------------------------------------- #

def act_gelu(x: np.ndarray) -> np.ndarray:
    return np.vectorize(F.fp_gelu_tanh, otypes=[object])(_obj(x))


def act_silu(x: np.ndarray) -> np.ndarray:
    return np.vectorize(F.fp_silu, otypes=[object])(_obj(x))


def residual_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Q32.32 + Q32.32 -> Q32.32. Same scale, so a plain integer add (no reconciliation)."""
    return _obj(a) + _obj(b)


# --------------------------------------------------------------------------- #
#  Reductions — norms
# --------------------------------------------------------------------------- #

def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps_q: int) -> np.ndarray:
    """RMSNorm over the last dim, all Q32.32 (int-llm rmsnorm).

    mean_sq = (Σ x^2 >>32) / dim ; inv = fp_inv_sqrt(mean_sq+eps) ; out = x*inv*gamma.
    """
    x = _obj(x); gamma = _obj(gamma)
    seq, dim = x.shape
    out = np.empty((seq, dim), dtype=object)
    for s in range(seq):
        row = x[s]
        sum_sq = 0
        for v in row:
            sum_sq += int(v) * int(v)
        mean_sq = _tdiv(sum_sq >> FP, dim)
        inv = F.fp_inv_sqrt(mean_sq + eps_q)
        for j in range(dim):
            out[s, j] = F.fp_mul(F.fp_mul(int(row[j]), inv), int(gamma[j]))
    return out


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps_q: int) -> np.ndarray:
    """LayerNorm over the last dim, all Q32.32: subtract mean, /std, affine."""
    x = _obj(x); gamma = _obj(gamma)
    beta = _obj(beta) if beta is not None else None
    seq, dim = x.shape
    out = np.empty((seq, dim), dtype=object)
    for s in range(seq):
        row = [int(v) for v in x[s]]
        mean = _tdiv(sum(row), dim)
        centered = [v - mean for v in row]
        var_sum = 0
        for c in centered:
            var_sum += (c * c) >> FP
        var = _tdiv(var_sum, dim)
        inv = F.fp_inv_sqrt(var + eps_q)
        for j in range(dim):
            normed = F.fp_mul(centered[j], inv)
            val = F.fp_mul(normed, int(gamma[j]))
            out[s, j] = val + int(beta[j]) if beta is not None else val
    return out


# --------------------------------------------------------------------------- #
#  Attention — per head, all Q32.32
# --------------------------------------------------------------------------- #

def softmax_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                 scale_q: int, causal: bool = False) -> np.ndarray:
    """Single-head scaled-dot-product attention, all Q32.32. q,k,v: [seq, head_dim].

    scale_q = 1/sqrt(head_dim) as a Q32.32 value. Scores = (q·k >>32) * scale; softmax with
    max-subtract + fp_safe_exp + fp_div normalize (int-llm softmax); context = Σ w*v.
    """
    q = _obj(q); k = _obj(k); v = _obj(v)
    seq, hd = q.shape
    out = np.empty((seq, hd), dtype=object)
    NEG = -(F.fp_from_int(30))
    for i in range(seq):
        # scores over all j
        scores = [0] * seq
        for j in range(seq):
            if causal and j > i:
                scores[j] = NEG
                continue
            dot = 0
            for d in range(hd):
                dot += int(q[i, d]) * int(k[j, d])
            scores[j] = F.fp_mul(dot >> FP, scale_q)
        # softmax: max-subtract, exp, normalize
        m = max(scores)
        exps = [F.fp_safe_exp(s - m) for s in scores]
        denom = sum(exps)
        if denom <= 0:
            denom = 1
        weights = [F.fp_div(e, denom) for e in exps]
        # context = Σ_j w_j * v_j
        for d in range(hd):
            acc = 0
            for j in range(seq):
                acc += F.fp_mul(weights[j], int(v[j, d]))
            out[i, d] = acc
    return out


# --------------------------------------------------------------------------- #
#  RoPE — precomputed Q32.32 cos/sin table, rotate-half
# --------------------------------------------------------------------------- #

def build_rope_table(max_seq: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute [max_seq, head_dim] cos and sin tables in Q32.32 (boundary float, done once)."""
    import math
    half = head_dim // 2
    cos = np.empty((max_seq, head_dim), dtype=object)
    sin = np.empty((max_seq, head_dim), dtype=object)
    for pos in range(max_seq):
        for i in range(half):
            freq = 1.0 / (theta ** (2.0 * i / head_dim))
            ang = pos * freq
            c = F.fp_from_float(math.cos(ang)); s = F.fp_from_float(math.sin(ang))
            cos[pos, i] = c; cos[pos, i + half] = c
            sin[pos, i] = s; sin[pos, i + half] = s
    return cos, sin


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """Apply rotate-half RoPE to [seq, n_heads*head_dim], all Q32.32."""
    x = _obj(x)
    seq = x.shape[0]
    half = head_dim // 2
    out = np.empty_like(x)
    for s in range(seq):
        for h in range(n_heads):
            base = h * head_dim
            for i in range(half):
                x1 = int(x[s, base + i]); x2 = int(x[s, base + half + i])
                c = int(cos[s, i]); sn = int(sin[s, i])
                out[s, base + i] = F.fp_mul(x1, c) - F.fp_mul(x2, sn)
                out[s, base + half + i] = F.fp_mul(x2, c) + F.fp_mul(x1, sn)
    return out
