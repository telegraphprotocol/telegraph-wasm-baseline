#!/usr/bin/env python3
"""
Export MiniLM-L6-v2 weights + vocab into the MLM2 binary format consumed by
src/embed.rs (real_weights build) and tokenizer.rs's build.rs.

Requires (on your machine — this script was NOT run or verified in the
sandbox that generated it, since that environment has no network access):

    pip install torch transformers numpy

Usage (from the repo root):
    python3 scripts/export_minilm_weights.py

Writes:
    weights/minilm_l6_v2_q8.bin   (embedded via include_bytes!)
    vocab.txt                     (read by build.rs)

Then self-verifies: reloads the just-written binary, runs a pure-NumPy
forward pass that mirrors embed.rs's Rust algorithm EXACTLY (same multi-head
attention, same LayerNorm placement, same GELU, same INT8 dequantization),
and compares its output against the real (non-quantized) HF model's output
on a few sentence pairs. This catches format/architecture mismatches BEFORE
you spend time compiling to WASM — if this check fails, the .bin file is
wrong; don't bother building.
"""

import struct
import sys

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN = 128  # MUST match tokenizer::MAX_SEQ_LEN in the Rust crate

OUT_WEIGHTS = "weights/minilm_l6_v2_q8.bin"
OUT_VOCAB = "vocab.txt"


def load_model():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        sys.exit(
            "Missing deps. Run: pip install torch transformers numpy"
        )
    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model, torch


# ── Quantization ──────────────────────────────────────────────────────────

def quantize_i8(arr: np.ndarray):
    """Per-tensor symmetric INT8 quantization. Returns (scale, int8 array)."""
    arr = arr.astype(np.float32)
    amax = float(np.abs(arr).max())
    scale = amax / 127.0 if amax > 0 else 1e-8
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return scale, q


def dequantize_i8(scale: float, q: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale


# ── Binary writers (must match embed.rs's byte-for-byte reading order) ────

def w_u32(f, v: int):
    f.write(struct.pack("<I", v))


def w_f32(f, v: float):
    f.write(struct.pack("<f", v))


def w_f32_vec(f, arr: np.ndarray):
    f.write(arr.astype("<f4").tobytes())


def w_qtable(f, arr: np.ndarray):
    """[f32 scale][int8 bytes], row-major, exactly as stored (no transpose —
    PyTorch nn.Linear.weight is already [out_features, in_features], which
    matches the [out_dim][in_dim] row-major layout embed.rs's matmul_row_bias
    expects. Embedding tables ([rows, hidden]) are already the right shape
    for direct row lookup too."""
    scale, q = quantize_i8(arr)
    w_f32(f, scale)
    f.write(q.tobytes())


# ── Main export ─────────────────────────────────────────────────────────

def export(tok, model, torch):
    import os

    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)

    sd = model.state_dict()
    cfg = model.config

    hidden_size = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    intermediate_size = cfg.intermediate_size
    num_layers = cfg.num_hidden_layers
    vocab_size = cfg.vocab_size

    print(
        f"layers={num_layers} hidden={hidden_size} heads={num_heads} "
        f"intermediate={intermediate_size} vocab={vocab_size}"
    )

    def np_(key):
        return sd[key].detach().numpy().astype(np.float32)

    with open(OUT_WEIGHTS, "wb") as f:
        f.write(b"MLM2")
        w_u32(f, num_layers)
        w_u32(f, hidden_size)
        w_u32(f, num_heads)
        w_u32(f, intermediate_size)
        w_u32(f, vocab_size)
        w_u32(f, MAX_SEQ_LEN)

        # ── Embedding layer ──
        word_emb = np_("embeddings.word_embeddings.weight")  # [vocab, hidden]
        assert word_emb.shape == (vocab_size, hidden_size)
        w_qtable(f, word_emb)

        pos_emb_full = np_("embeddings.position_embeddings.weight")
        pos_emb = pos_emb_full[:MAX_SEQ_LEN]  # slice down to what Rust ever indexes
        assert pos_emb.shape == (MAX_SEQ_LEN, hidden_size), (
            f"position_embeddings has only {pos_emb_full.shape[0]} rows, "
            f"need >= MAX_SEQ_LEN={MAX_SEQ_LEN}"
        )
        w_qtable(f, pos_emb)

        type_emb = np_("embeddings.token_type_embeddings.weight")[:1]  # row 0 only
        w_qtable(f, type_emb)

        w_f32_vec(f, np_("embeddings.LayerNorm.weight"))
        w_f32_vec(f, np_("embeddings.LayerNorm.bias"))

        # ── Per-layer ──
        for i in range(num_layers):
            p = f"encoder.layer.{i}."

            w_qtable(f, np_(p + "attention.self.query.weight"))
            w_f32_vec(f, np_(p + "attention.self.query.bias"))
            w_qtable(f, np_(p + "attention.self.key.weight"))
            w_f32_vec(f, np_(p + "attention.self.key.bias"))
            w_qtable(f, np_(p + "attention.self.value.weight"))
            w_f32_vec(f, np_(p + "attention.self.value.bias"))

            w_qtable(f, np_(p + "attention.output.dense.weight"))
            w_f32_vec(f, np_(p + "attention.output.dense.bias"))
            w_f32_vec(f, np_(p + "attention.output.LayerNorm.weight"))
            w_f32_vec(f, np_(p + "attention.output.LayerNorm.bias"))

            w_qtable(f, np_(p + "intermediate.dense.weight"))
            w_f32_vec(f, np_(p + "intermediate.dense.bias"))
            w_qtable(f, np_(p + "output.dense.weight"))
            w_f32_vec(f, np_(p + "output.dense.bias"))
            w_f32_vec(f, np_(p + "output.LayerNorm.weight"))
            w_f32_vec(f, np_(p + "output.LayerNorm.bias"))

            print(f"  layer {i} written")

    size = os.path.getsize(OUT_WEIGHTS)
    print(f"Wrote {OUT_WEIGHTS} ({size / 1024:.1f} KB)")

    # ── vocab.txt: line N = token with id N (standard BERT vocab.txt format,
    # required by build.rs) ──
    ids = list(range(vocab_size))
    tokens = tok.convert_ids_to_tokens(ids)
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"Wrote {OUT_VOCAB} ({len(tokens)} tokens)")

    return {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
    }


# ── Self-verification: pure-NumPy reimplementation of embed.rs's exact math,
# reading back the just-written quantized file, compared against the real
# (unquantized) HF model. ──

def numpy_forward_from_binfile(path, input_ids, attention_mask):
    with open(path, "rb") as f:
        buf = f.read()

    c = 0

    def ru32():
        nonlocal c
        v = struct.unpack_from("<I", buf, c)[0]
        c += 4
        return v

    def rf32():
        nonlocal c
        v = struct.unpack_from("<f", buf, c)[0]
        c += 4
        return v

    def r_f32vec(n):
        nonlocal c
        v = np.frombuffer(buf, dtype="<f4", count=n, offset=c).astype(np.float32)
        c += 4 * n
        return v

    def r_qtable(rows, cols):
        nonlocal c
        scale = rf32()
        n = rows * cols
        q = np.frombuffer(buf, dtype=np.int8, count=n, offset=c).astype(np.float32)
        c += n
        return (q * scale).reshape(rows, cols)

    assert buf[0:4] == b"MLM2"
    c = 4
    num_layers = ru32()
    hidden_size = ru32()
    num_heads = ru32()
    intermediate_size = ru32()
    vocab_size = ru32()
    num_positions = ru32()
    head_dim = hidden_size // num_heads

    word_emb = r_qtable(vocab_size, hidden_size)
    pos_emb = r_qtable(num_positions, hidden_size)
    type_emb = r_qtable(1, hidden_size)
    emb_ln_g = r_f32vec(hidden_size)
    emb_ln_b = r_f32vec(hidden_size)

    def layer_norm(x, g, b, eps=1e-12):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * g + b

    def gelu(x):
        c_ = 0.7978846
        return 0.5 * x * (1.0 + np.tanh(c_ * (x + 0.044715 * x**3)))

    seq_len = len(input_ids)
    hidden = (
        word_emb[input_ids]
        + pos_emb[np.arange(seq_len) % num_positions]
        + type_emb[0]
    )
    hidden = layer_norm(hidden, emb_ln_g, emb_ln_b)

    mask = np.array(attention_mask, dtype=np.float32)

    for _ in range(num_layers):
        q_w = r_qtable(hidden_size, hidden_size); q_b = r_f32vec(hidden_size)
        k_w = r_qtable(hidden_size, hidden_size); k_b = r_f32vec(hidden_size)
        v_w = r_qtable(hidden_size, hidden_size); v_b = r_f32vec(hidden_size)
        out_w = r_qtable(hidden_size, hidden_size); out_b = r_f32vec(hidden_size)
        attn_ln_g = r_f32vec(hidden_size); attn_ln_b = r_f32vec(hidden_size)
        ffn1_w = r_qtable(intermediate_size, hidden_size); ffn1_b = r_f32vec(intermediate_size)
        ffn2_w = r_qtable(hidden_size, intermediate_size); ffn2_b = r_f32vec(hidden_size)
        out_ln_g = r_f32vec(hidden_size); out_ln_b = r_f32vec(hidden_size)

        q = hidden @ q_w.T + q_b
        k = hidden @ k_w.T + k_b
        v = hidden @ v_w.T + v_b

        q = q.reshape(seq_len, num_heads, head_dim)
        k = k.reshape(seq_len, num_heads, head_dim)
        v = v.reshape(seq_len, num_heads, head_dim)

        scores = np.einsum("ihd,jhd->hij", q, k) / np.sqrt(head_dim)
        scores = scores + (1.0 - mask)[None, None, :] * -1e9
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)

        context = np.einsum("hij,jhd->ihd", probs, v).reshape(seq_len, hidden_size)

        attn_out = context @ out_w.T + out_b
        hidden = layer_norm(attn_out + hidden, attn_ln_g, attn_ln_b)

        mid = gelu(hidden @ ffn1_w.T + ffn1_b)
        ffn_out = mid @ ffn2_w.T + ffn2_b
        hidden = layer_norm(ffn_out + hidden, out_ln_g, out_ln_b)

    pooled = (hidden * mask[:, None]).sum(axis=0) / mask.sum()
    return pooled / np.linalg.norm(pooled)


def self_verify(tok, model, torch):
    print("\n── Self-verification ──")
    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Bananas are a tropical fruit."),
        ("The cat sat on the mat.", "A cat was sitting on a mat."),
    ]

    for a, b in pairs:
        enc_a = tok(a, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        enc_b = tok(b, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")

        with torch.no_grad():
            hf_a = model(**enc_a).last_hidden_state[0].numpy()
            hf_b = model(**enc_b).last_hidden_state[0].numpy()
        mask_a = enc_a["attention_mask"][0].numpy().astype(np.float32)
        mask_b = enc_b["attention_mask"][0].numpy().astype(np.float32)
        hf_pool_a = (hf_a * mask_a[:, None]).sum(0) / mask_a.sum()
        hf_pool_b = (hf_b * mask_b[:, None]).sum(0) / mask_b.sum()
        hf_pool_a /= np.linalg.norm(hf_pool_a)
        hf_pool_b /= np.linalg.norm(hf_pool_b)
        hf_cos = float(hf_pool_a @ hf_pool_b)

        q_a = numpy_forward_from_binfile(
            OUT_WEIGHTS, enc_a["input_ids"][0].tolist(), enc_a["attention_mask"][0].tolist()
        )
        q_b = numpy_forward_from_binfile(
            OUT_WEIGHTS, enc_b["input_ids"][0].tolist(), enc_b["attention_mask"][0].tolist()
        )
        q_cos = float(q_a @ q_b)

        print(f'  "{a[:40]}" vs "{b[:40]}"')
        print(f"    real (unquantized) model cosine: {hf_cos:.4f}")
        print(f"    quantized MLM2 graph cosine:      {q_cos:.4f}")

    print(
        "\nIf the quantized-graph cosines track the real-model cosines in "
        "relative order (related pair scores higher than unrelated pair, "
        "roughly similar magnitude) — the MLM2 export and embed.rs's algorithm "
        "agree. If they diverge sharply, something in embed.rs's Rust "
        "reimplementation doesn't match this script's forward pass — re-check "
        "the two side by side before building to WASM."
    )


if __name__ == "__main__":
    tok, model, torch = load_model()
    export(tok, model, torch)
    self_verify(tok, model, torch)
