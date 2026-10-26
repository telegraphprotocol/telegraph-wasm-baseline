"""Wide Q32.32 op-list interpreter — the runtime for .imwq artifacts.

Reads a WideReader, runs the op-list keeping every slot in Q32.32 (object array of ints),
returns the final scalar score as a float (the one boundary conversion). No scales, no
requant — activations stay wide the whole way, which is the fix for the int8 ranking collapse.

Slot convention mirrors int8_runtime's interpreter so the converters stay parallel:
EMBED seeds slot 0 from token ids; SCORE writes the final [1,1]; attention uses q=in0,
k=in1, v=arg3-as-slot, arg0=head_dim, arg1=n_heads, arg2=n_kv_heads.
"""

from __future__ import annotations

import numpy as np

from . import format as W
from .format import Op
from .golden import ops as O
from .golden import fixed as F


class WideInterp:
    def __init__(self, model: W.WideReader):
        self.m = model
        self.cfg = model.config
        self.slots: dict[int, np.ndarray] = {}
        # RoPE table built lazily from config
        self._rope = None

    def _get(self, sid): return self.slots[sid]
    def _set(self, sid, val): self.slots[sid] = np.asarray(val, dtype=object)

    def run(self, token_ids) -> float:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        self.slots = {}
        final_slot = None
        final_scale = 1.0
        for op in self.m.ops:
            oc = op.op
            if oc == Op.END:
                break
            elif oc == Op.EMBED:
                emb = self.m.tensor(op.weight_idx)
                self._set(op.out, emb[token_ids])
            elif oc == Op.QMATMUL:
                x = self._get(op.in0)
                w = self.m.tensor(op.weight_idx)
                bias = self.m.tensor(op.bias_idx) if op.bias_idx >= 0 else None
                self._set(op.out, O.qmatmul(x, w, bias))
            elif oc == Op.ADD:
                self._set(op.out, O.residual_add(self._get(op.in0), self._get(op.in1)))
            elif oc == Op.RMSNORM:
                g = self.m.tensor(op.weight_idx)
                self._set(op.out, O.rmsnorm(self._get(op.in0), g, op.arg0))
            elif oc == Op.LAYERNORM:
                g = self.m.tensor(op.weight_idx)
                b = self.m.tensor(op.bias_idx) if op.bias_idx >= 0 else None
                self._set(op.out, O.layernorm(self._get(op.in0), g, b, op.arg0))
            elif oc == Op.ACT_GELU:
                self._set(op.out, O.act_gelu(self._get(op.in0)))
            elif oc == Op.ACT_SILU:
                self._set(op.out, O.act_silu(self._get(op.in0)))
            elif oc == Op.ROPE:
                self._op_rope(op)
            elif oc == Op.POS_ADD:
                pos = self.m.tensor(op.weight_idx)
                x = self._get(op.in0)
                self._set(op.out, O.residual_add(x, pos[:x.shape[0]]))
            elif oc == Op.ATTN:
                self._op_attn(op)
            elif oc == Op.SWIGLU:
                self._op_swiglu(op)
            elif oc == Op.POOL:
                x = self._get(op.in0)
                if op.flags & W.FLAG_MEAN_POOL:
                    dim = x.shape[1]
                    # _tdiv, NOT `//`. Python's `//` floors toward -inf; the Rust runtime
                    # divides an i128 with `/`, which truncates toward zero. They differ by
                    # exactly 1 on every column whose sum is negative and not a multiple of
                    # the row count — which is most of them on a real model. The synthetic
                    # non-RoPE fixture happened to have no such column, so the divergence
                    # went unnoticed until a RoPE fixture was added.
                    pooled = np.array([O._tdiv(sum(int(x[s, j]) for s in range(x.shape[0])),
                                               x.shape[0])
                                       for j in range(dim)], dtype=object)
                    self._set(op.out, pooled[None, :])
                else:
                    self._set(op.out, x[0][None, :])
            elif oc == Op.LAST_TOKEN:
                x = self._get(op.in0)
                pad = op.arg0
                nonpad = np.where(token_ids != pad)[0]
                p = int(nonpad[-1]) if nonpad.size else x.shape[0] - 1
                self._set(op.out, x[p][None, :])
            elif oc == Op.SCORE:
                x = self._get(op.in0)
                w = self.m.tensor(op.weight_idx)
                bias = self.m.tensor(op.bias_idx) if op.bias_idx >= 0 else None
                y = O.qmatmul(x, w, bias)              # [1,1] Q32.32 — wide, no clamp
                final_slot = op.out
                self._set(op.out, y)
            else:
                raise ValueError(f"wide interp: unimplemented opcode {oc}")
        if final_slot is None:
            raise ValueError("wide interp: no SCORE op")
        raw = int(np.asarray(self._get(final_slot)).reshape(-1)[0])
        return F.to_float(raw) * final_scale         # ONE boundary float conversion

    def _op_rope(self, op):
        head_dim = op.arg2; n_heads = op.arg3
        # Baked tables (weight_idx = cos, aux_idx = sin) take precedence, so this golden
        # reads the SAME integers the Rust runtime does rather than recomputing them from
        # float trig and hoping the two agree. The recompute path is kept only for
        # artifacts written before the converter baked them.
        if op.weight_idx >= 0 and op.aux_idx >= 0:
            cos = self.m.tensor(op.weight_idx); sin = self.m.tensor(op.aux_idx)
        else:
            if self._rope is None:
                theta = self.cfg.get("rope_theta_milli", 10000000) / 1000.0
                max_seq = self.cfg.get("max_seq_len", 8192)
                self._rope = O.build_rope_table(max_seq, head_dim, theta)
            cos, sin = self._rope
        self._set(op.out, O.rope(self._get(op.in0), cos, sin, n_heads, head_dim))

    def _op_attn(self, op):
        head_dim = op.arg0; n_heads = op.arg1; n_kv = op.arg2; v_slot = op.arg3
        q = self._get(op.in0); k = self._get(op.in1); v = self._get(v_slot)
        seq = q.shape[0]
        causal = bool(op.flags & W.FLAG_CAUSAL)
        # integer 1/sqrt(head_dim) — MUST match the Rust runtime (which has no float), so use
        # fp_inv_sqrt here too rather than fp_from_float(1/sqrt(.)).
        scale = F.fp_inv_sqrt(F.fp_from_int(head_dim))
        q = q.reshape(seq, n_heads, head_dim)
        k = k.reshape(seq, n_kv, head_dim)
        v = v.reshape(seq, n_kv, head_dim)
        group = n_heads // n_kv
        out = np.empty((seq, n_heads, head_dim), dtype=object)
        for h in range(n_heads):
            kv = h // group
            ctx = O.softmax_attn(q[:, h], k[:, kv], v[:, kv], scale, causal=causal)
            out[:, h] = ctx
        self._set(op.out, out.reshape(seq, n_heads * head_dim))

    def _op_swiglu(self, op):
        if op.flags & 1:
            # encoder GeGLU: in0 = already-activated gate, in1 = up. Just multiply.
            gate = self._get(op.in0); up = self._get(op.in1)
            self._set(op.out, np.vectorize(F.fp_mul, otypes=[object])(gate, up))
            return
        # decoder SwiGLU: in0 = normed x; weight_idx = gate proj, aux_idx = up proj.
        x = self._get(op.in0)
        gate = O.act_silu(O.qmatmul(x, self.m.tensor(op.weight_idx)))
        up = O.qmatmul(x, self.m.tensor(op.aux_idx))
        self._set(op.out, np.vectorize(F.fp_mul, otypes=[object])(gate, up))


def run_wide(model: W.WideReader, token_ids) -> float:
    return WideInterp(model).run(token_ids)
