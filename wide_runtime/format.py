"""Wide .imwq format — a self-contained Q32.32 artifact for the wide runtime.

Deliberately simpler than int8_runtime's .imw: because activations stay wide, there are NO
per-layer {mul,shift} scale pools, NO activation LUTs, NO calibration ranges. An artifact is
just: header + config + Q32.32 weight tensors + a flat op-list. This keeps the wide runtime
fully separate from the int8 pipeline.

All little-endian. Weights are int64 Q32.32. The op-list reuses int8_runtime's OpCode integer
values so the two converters stay parallel; ops carry tensor refs + a few int args (no scale
offsets). JSON is used for the header/config/op-list (readable, tiny); tensor blobs are raw
int64 appended after a marker. Determinism of the artifact = canonical JSON (sorted keys) +
raw little-endian int64.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, asdict

import numpy as np

MAGIC = b"IMWQ"
VERSION = 1


# opcode integers — MUST match int8_runtime/format/spec.py OpCode
class Op:
    END = 0; EMBED = 1; QMATMUL = 2; ADD = 3; LAYERNORM = 4; RMSNORM = 5
    ACT_GELU = 6; ROPE = 7; POS_ADD = 8; ATTN = 9; SWIGLU = 10
    POOL = 11; LAST_TOKEN = 12; SCORE = 13; ACT_SILU = 15   # 14=SCALE_HEAD reserved

FLAG_CAUSAL = 1 << 0
FLAG_MEAN_POOL = 1 << 2


@dataclass
class WideOp:
    op: int
    in0: int = -1
    in1: int = -1
    out: int = -1
    weight_idx: int = -1
    bias_idx: int = -1
    aux_idx: int = -1
    flags: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0
    arg3: int = 0


@dataclass
class WideTensor:
    name: str
    shape: tuple
    # data stored separately as raw int64; index into the blob list
    blob_index: int = -1


@dataclass
class WideModel:
    config: dict = field(default_factory=dict)
    tensors: list = field(default_factory=list)     # list[WideTensor]
    ops: list = field(default_factory=list)          # list[WideOp]
    _blobs: list = field(default_factory=list)       # list[np.ndarray int64], parallel to tensors


class WideWriter:
    def __init__(self):
        self.m = WideModel()

    def set_config(self, **kw):
        self.m.config.update(kw)

    def add_tensor(self, name: str, data_q: np.ndarray) -> int:
        """Register a Q32.32 tensor (int64 array). Returns its tensor index."""
        arr = np.asarray(data_q, dtype=object)     # exact ints
        idx = len(self.m.tensors)
        self.m.tensors.append(WideTensor(name=name, shape=tuple(arr.shape), blob_index=idx))
        # store as int64 little-endian; values are Q32.32 and fit int64 by construction
        self.m._blobs.append(np.asarray([int(v) for v in arr.reshape(-1)], dtype=np.int64))
        return idx

    def add_op(self, op: WideOp) -> int:
        self.m.ops.append(op); return len(self.m.ops) - 1

    def to_bytes(self) -> bytes:
        meta = {
            "version": VERSION,
            "config": self.m.config,
            "tensors": [{"name": t.name, "shape": list(t.shape)} for t in self.m.tensors],
            "ops": [asdict(o) for o in self.m.ops],
        }
        mjson = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        out = bytearray()
        out += MAGIC
        out += struct.pack("<I", len(mjson))
        out += mjson
        # --- binary op-list + config for the no_std Rust reader (JSON above is for Python) ---
        out += self._pack_binary()
        # blobs: count, then per-blob (ndim, dims, raw int64 little-endian)
        out += struct.pack("<I", len(self.m._blobs))
        for t, b in zip(self.m.tensors, self.m._blobs):
            shape = t.shape
            out += struct.pack("<I", len(shape))
            for d in shape:
                out += struct.pack("<Q", int(d))
            out += struct.pack("<Q", b.size)
            out += b.astype("<i8").tobytes()
        return bytes(out)

    def _pack_binary(self) -> bytes:
        """Fixed-width binary op-list + a small config block the Rust reader parses directly.

        Layout: MAGIC2 'BOPS' | num_ops:u32 | ops (each 60B — see the byte count below) |
        config: arch:i32, hidden:i32, layers:i32, heads:i32, kv_heads:i32, head_dim:i32,
                inter:i32, max_seq:i32, rope_theta_milli:i32, rope:i32
        Op field order matches WideOp: op,in0,in1,out,weight_idx,bias_idx,aux_idx,flags,
        arg0,arg1,arg2,arg3 (arg0 can hold eps_q which is a large i64 -> we widen it: see note).
        NOTE: eps_q is Q32.32 (large). We store arg0/arg2/arg3 as i64 (norm eps, dequant), the
        rest as i32. So each op is: 2xu16(op,flags)+ 5xi32(in0,in1,out,weight,bias)+aux i32 +
        4xi64(arg0..arg3) = 4+20+4+32 = 60 bytes.
        """
        c = self.m.config
        out = bytearray()
        out += b"BOPS"
        out += struct.pack("<I", len(self.m.ops))
        for o in self.m.ops:
            out += struct.pack("<HH", o.op & 0xFFFF, o.flags & 0xFFFF)
            out += struct.pack("<iiiii", o.in0, o.in1, o.out, o.weight_idx, o.bias_idx)
            out += struct.pack("<i", o.aux_idx)
            out += struct.pack("<qqqq", o.arg0, o.arg1, o.arg2, o.arg3)
        archmap = {"encoder": 1, "decoder": 2}
        out += struct.pack("<iiiiiiiiii",
                           archmap.get(c.get("arch"), 1),
                           int(c.get("hidden_size", 0)), int(c.get("num_hidden_layers", 0)),
                           int(c.get("num_attention_heads", 0)), int(c.get("num_kv_heads",
                               c.get("num_attention_heads", 0))),
                           int(c.get("head_dim", 0)), int(c.get("intermediate_size", 0)),
                           int(c.get("max_seq_len", 8192)),
                           int(c.get("rope_theta_milli", 0)), int(bool(c.get("rope", False))))
        return bytes(out)

    def write(self, path: str) -> str:
        blob = self.to_bytes()
        with open(path, "wb") as f:
            f.write(blob)
        return hashlib.sha256(blob).hexdigest()


class WideReader:
    def __init__(self, blob: bytes):
        self.sha256 = hashlib.sha256(blob).hexdigest()
        if blob[:4] != MAGIC:
            raise ValueError("not an IMWQ file")
        off = 4
        (mlen,) = struct.unpack_from("<I", blob, off); off += 4
        meta = json.loads(blob[off:off + mlen]); off += mlen
        self.config = meta["config"]
        self.ops = [WideOp(**o) for o in meta["ops"]]
        self._tensor_meta = meta["tensors"]
        # skip the binary op-list/config block (Rust-only): 'BOPS' | num_ops | ops(60B) | cfg(40B)
        assert blob[off:off + 4] == b"BOPS", "missing BOPS block"
        off += 4
        (n_ops,) = struct.unpack_from("<I", blob, off); off += 4
        off += n_ops * 60          # each op record is 60 bytes (see _pack_binary)
        off += 40                  # config: 10 x i32
        (nb,) = struct.unpack_from("<I", blob, off); off += 4
        self._tensors = []
        for _ in range(nb):
            (ndim,) = struct.unpack_from("<I", blob, off); off += 4
            dims = []
            for _d in range(ndim):
                (dv,) = struct.unpack_from("<Q", blob, off); off += 8
                dims.append(dv)
            (n,) = struct.unpack_from("<Q", blob, off); off += 8
            arr = np.frombuffer(blob, dtype="<i8", count=n, offset=off).astype(object)
            off += n * 8
            shape = tuple(dims)
            self._tensors.append(arr.reshape(shape) if shape else arr)

    def tensor(self, idx: int) -> np.ndarray:
        return self._tensors[idx]


def load_wide(path: str) -> WideReader:
    with open(path, "rb") as f:
        return WideReader(f.read())
