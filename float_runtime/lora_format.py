"""The LoRA delta adapter package (abi "lora-v1"): a standard safetensors file.

The container is plain safetensors (what PEFT already writes): an 8-byte little-endian
header length, a JSON header, then the raw tensor bytes. Every field the package needs
beyond its tensors lives in the safetensors `__metadata__` string map:

    format_version   "1"
    abi              "lora-v1"
    class            "judgment" | "hybrid"
    intent           the intent (family prefix) the adapter serves
    foundation_hash  sha256 of the Foundation Model ONNX it was exported for
    foundation_version
    tokenizer_hash   sha256 of the node's pinned tokenizer.json
    rank             "8"            (<= R_MAX)
    alpha            "32"           (lora_alpha; the engine feeds lora.scale = alpha / rank)
    target_modules   '["Wi","Wo","Wqkv"]'   (JSON array string)
    max_tokens       "512"
    training         optional JSON string of the training record (standard section 9)
    x-*              optional extension keys, ignored by the node

Tensors (dtype F32, pre-transposed so the graph is x.A.B with no Transpose), for a
foundation with L layers and hidden size H, rank r <= R_MAX:

    lora.<l>.<target>.A   [in, r]     target in TARGETS, l in 0..L-1
    lora.<l>.<target>.B   [r, out]
    head.weight           [H, 1]
    head.bias             [1]

The node's parser is Telegraph pkg/wasm/runtime/lora_adapter.go; the engine re-checks in
the guest. Keep the three in step. The file's identity is its keccak256 (wasm_hash).
"""

from __future__ import annotations

import json

FORMAT_VERSION = 1
ABI = "lora-v1"
FILE_EXT = ".safetensors"
R_MAX = 16
MAX_PACKAGE_BYTES = 64 << 20
NODE_MAX_LEN = 512
# The node's pinned tokenizer.json digest (Telegraph tokenizer.ModernBERTSHA256).
NODE_TOKENIZER_SHA256 = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"

# Target linears per layer, as the graph names them. PEFT's target_modules ["Wqkv","Wo","Wi"]
# matches by suffix, so "Wo" is BOTH attn.Wo and mlp.Wo.
TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
TARGET_MODULES = ("Wqkv", "Wo", "Wi")
METADATA_KEYS = ("format_version", "abi", "class", "intent", "foundation_hash", "foundation_version",
                 "tokenizer_hash", "rank", "alpha", "target_modules", "max_tokens")


def target_dims(cfg) -> dict[str, tuple[int, int]]:
    """(in, out) of each target linear for a ModernBERT config."""
    h, i = cfg.hidden_size, cfg.intermediate_size
    return {"attn.Wqkv": (h, 3 * h), "attn.Wo": (h, h), "mlp.Wi": (h, 2 * i), "mlp.Wo": (i, h)}


def tensor_names(num_layers: int) -> list[str]:
    """Every tensor a package must carry."""
    names = [f"lora.{l}.{t}.{m}" for l in range(num_layers) for t in TARGETS for m in "AB"]
    return names + ["head.weight", "head.bias"]


def metadata(*, cls: str, intent: str, foundation_hash: str, foundation_version: str, rank: int,
             alpha: float, target_modules, max_tokens: int, training: dict | None = None,
             tokenizer_hash: str = NODE_TOKENIZER_SHA256) -> dict[str, str]:
    """The __metadata__ map (every value a string, as safetensors requires)."""
    m = {
        "format_version": str(FORMAT_VERSION),
        "abi": ABI,
        "class": cls,
        "intent": intent,
        "foundation_hash": foundation_hash,
        "foundation_version": foundation_version,
        "tokenizer_hash": tokenizer_hash,
        "rank": str(int(rank)),
        "alpha": str(int(alpha)) if float(alpha) == int(alpha) else repr(float(alpha)),
        "target_modules": json.dumps(sorted(target_modules), separators=(",", ":")),
        "max_tokens": str(int(max_tokens)),
    }
    if training is not None:
        m["training"] = json.dumps(training, separators=(",", ":"))
    return m


def write_package(path, meta: dict[str, str], tensors: dict) -> bytes:
    """Write {name: float32 numpy array} with meta as a safetensors file; returns its bytes."""
    import numpy as np
    from safetensors.numpy import save, save_file
    clean = {}
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if not np.isfinite(a).all():
            raise ValueError(f"tensor {name} has a non-finite value")
        clean[name] = a
    for k, v in meta.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"metadata {k!r} must map a string to a string")
    data = save(clean, metadata=meta)
    if len(data) > MAX_PACKAGE_BYTES:
        raise ValueError(f"package is {len(data)} bytes, above the {MAX_PACKAGE_BYTES} cap")
    if path is not None:
        save_file(clean, str(path), metadata=meta)
    return data


def read_package(path) -> tuple[dict[str, str], dict]:
    """(metadata, {name: numpy array}) of a package file."""
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), framework="np") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return meta, out
