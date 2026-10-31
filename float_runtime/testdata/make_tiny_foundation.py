#!/usr/bin/env python3
"""Tiny stub Foundation Model and safetensors delta fixtures for the Go unit tests (no real model, no torch).

Builds, with onnx.helper and the safetensors library only, a few-KB ONNX graph with the SAME LoRA-input layout the real
foundation has (input_ids/attention_mask, lora.scale, head.weight/head.bias and
lora.<l>.<target>.A/B for attn.Wqkv, attn.Wo, mlp.Wi, mlp.Wo per layer) over a toy
2-layer network: embedding -> per layer [Wqkv -> slice -> Wo -> Wi -> slice -> relu ->
mlp.Wo] -> mean pool -> head. Every target linear adds its LoRA branch x.A.B*scale.

Writes to --out:
    tiny_foundation.onnx          the stub foundation (FOUNDATION_MODEL_PATH in tests)
    tiny_foundation.json          its sha256 and spec
    tiny_judgment.safetensors     a rank-2 adapter, class judgment, intent CODE_REVIEW
    tiny_hybrid.safetensors       a rank-2 adapter, class hybrid, intent VULNERABILITY_TRIAGE
    tiny_legacy.onnx              the merged equivalent of tiny_judgment (for a size comparison)

    python -m float_runtime.testdata.make_tiny_foundation --out float_runtime/testdata
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from float_runtime.lora_format import NODE_TOKENIZER_SHA256, R_MAX, TARGETS, metadata, tensor_names, write_package

VOCAB, H, I, L, SEED = 64, 8, 12, 2, 7
DIMS = {"attn.Wqkv": (H, 3 * H), "attn.Wo": (H, H), "mlp.Wi": (H, 2 * I), "mlp.Wo": (I, H)}


def build_graph(rng, merged: dict | None = None):
    """The toy graph. merged: {target name -> (A[in,r], B[r,out], scale)} folds the LoRA into
    the base weights and declares NO lora inputs (the legacy, merged form)."""
    inits, nodes, inputs = [], [], []
    emb = rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.5
    inits.append(numpy_helper.from_array(emb, "emb"))
    inits.append(numpy_helper.from_array(np.array([0], np.int64), "zero"))
    inits.append(numpy_helper.from_array(np.array([H], np.int64), "h_end"))
    inits.append(numpy_helper.from_array(np.array([I], np.int64), "i_end"))
    inits.append(numpy_helper.from_array(np.array([2], np.int64), "ax2"))
    inputs.append(helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]))
    inputs.append(helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]))
    if merged is None:
        inputs.append(helper.make_tensor_value_info("lora.scale", TensorProto.FLOAT, [1]))
        inputs.append(helper.make_tensor_value_info("head.weight", TensorProto.FLOAT, [H, 1]))
        inputs.append(helper.make_tensor_value_info("head.bias", TensorProto.FLOAT, [1]))
    nodes.append(helper.make_node("Gather", ["emb", "input_ids"], ["x0"], axis=0))
    x = "x0"
    weights = {}
    for l in range(L):
        for t in TARGETS:
            fi, fo = DIMS[t]
            w = (rng.standard_normal((fi, fo)).astype(np.float32) / np.sqrt(fi)).astype(np.float32)
            name = f"lora.{l}.{t}"
            weights[name] = w
            if merged is not None and name in merged:
                a, b, s = merged[name]
                w = (w + (a @ b) * s).astype(np.float32)
            inits.append(numpy_helper.from_array(w, f"W.{name}"))
            nodes.append(helper.make_node("MatMul", [x, f"W.{name}"], [f"{name}.base"]))
            if merged is None:
                inputs.append(helper.make_tensor_value_info(f"{name}.A", TensorProto.FLOAT, [fi, R_MAX]))
                inputs.append(helper.make_tensor_value_info(f"{name}.B", TensorProto.FLOAT, [R_MAX, fo]))
                nodes.append(helper.make_node("MatMul", [x, f"{name}.A"], [f"{name}.xa"]))
                nodes.append(helper.make_node("MatMul", [f"{name}.xa", f"{name}.B"], [f"{name}.xab"]))
                nodes.append(helper.make_node("Mul", [f"{name}.xab", "lora.scale"], [f"{name}.delta"]))
                nodes.append(helper.make_node("Add", [f"{name}.base", f"{name}.delta"], [f"{name}.y"]))
                y = f"{name}.y"
            else:
                y = f"{name}.base"
            if t == "attn.Wqkv":
                nodes.append(helper.make_node("Slice", [y, "zero", "h_end", "ax2"], [f"{name}.q"]))
                y = f"{name}.q"
            elif t == "mlp.Wi":
                nodes.append(helper.make_node("Slice", [y, "zero", "i_end", "ax2"], [f"{name}.g"]))
                nodes.append(helper.make_node("Relu", [f"{name}.g"], [f"{name}.r"]))
                y = f"{name}.r"
            x = y
    nodes.append(helper.make_node("ReduceMean", [x], ["pooled"], axes=[1], keepdims=0))
    if merged is None:
        nodes.append(helper.make_node("MatMul", ["pooled", "head.weight"], ["hw"]))
        nodes.append(helper.make_node("Add", ["hw", "head.bias"], ["logits"]))
    else:
        inits.append(numpy_helper.from_array(merged["head.weight"], "head.w"))
        inits.append(numpy_helper.from_array(merged["head.bias"], "head.b"))
        nodes.append(helper.make_node("MatMul", ["pooled", "head.w"], ["hw"]))
        nodes.append(helper.make_node("Add", ["hw", "head.b"], ["logits"]))
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])
    graph = helper.make_graph(nodes, "tiny_foundation", inputs, [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="make_tiny_foundation")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def adapter_tensors(rng, rank: int):
    out = {}
    for l in range(L):
        for t in TARGETS:
            fi, fo = DIMS[t]
            out[f"lora.{l}.{t}.A"] = (rng.standard_normal((fi, rank)) * 0.3).astype(np.float32)
            out[f"lora.{l}.{t}.B"] = (rng.standard_normal((rank, fo)) * 0.3).astype(np.float32)
    out["head.weight"] = (rng.standard_normal((H, 1)) * 0.5).astype(np.float32)
    out["head.bias"] = (rng.standard_normal((1,)) * 0.1).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    model = build_graph(rng)
    fpath = out / "tiny_foundation.onnx"
    onnx.save(model, str(fpath))
    digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
    spec = {"foundation_version": "tiny-stub-v1", "foundation_hash": digest, "onnx": "tiny_foundation.onnx",
            "bytes": fpath.stat().st_size, "r_max": R_MAX, "num_layers": L, "hidden_size": H,
            "tokenizer_hash": NODE_TOKENIZER_SHA256, "targets": {t: list(DIMS[t]) for t in TARGETS}}
    (out / "tiny_foundation.json").write_text(json.dumps(spec, indent=2))
    rank, alpha = 2, 4.0
    for fname, cls, intent, seed in (("tiny_judgment.safetensors", "judgment", "CODE_REVIEW", 11),
                                     ("tiny_hybrid.safetensors", "hybrid", "VULNERABILITY_TRIAGE", 12)):
        t = adapter_tensors(np.random.default_rng(seed), rank)
        meta = metadata(cls=cls, intent=intent, foundation_hash=digest, foundation_version="tiny-stub-v1",
                        rank=rank, alpha=alpha, target_modules=["Wi", "Wo", "Wqkv"], max_tokens=512)
        data = write_package(out / fname, meta, {n: t[n] for n in tensor_names(L)})
        print(f"{fname}: {len(data)} bytes")
        if cls == "judgment":
            merged = {f"lora.{l}.{tt}": (t[f"lora.{l}.{tt}.A"], t[f"lora.{l}.{tt}.B"], alpha / rank) for l in range(L) for tt in TARGETS}
            merged["head.weight"], merged["head.bias"] = t["head.weight"], t["head.bias"]
            legacy = build_graph(np.random.default_rng(SEED), merged)
            onnx.save(legacy, str(out / "tiny_legacy.onnx"))
    print(f"tiny_foundation.onnx: {fpath.stat().st_size} bytes sha256 {digest}; tiny_legacy.onnx: {(out / 'tiny_legacy.onnx').stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
