#!/usr/bin/env python3
"""Export the shared Foundation Model: ModernBERT-base as ONE float32 ONNX graph whose LoRA
branches and reward head are graph INPUTS. Run once by the Telegraph team, not per adapter.

    python -m float_runtime.export_foundation --out foundation/ [--check-adapter <PEFT dir|HF spec>]

For every target linear (attn.Wqkv, attn.Wo, mlp.Wi, mlp.Wo in each layer) the graph computes

    y = x.W^T + ((x.A).B) * s

with A [in, R_MAX] and B [R_MAX, out] inputs (an adapter of rank r <= R_MAX is zero-padded,
which adds exact zeros), s = alpha / r the input `lora.scale`. The reward head is
`logits = pooled . head.weight + head.bias`; the pooling, head.dense and head.norm are frozen
checkpoint weights and stay in the graph. The base weights stay initializers, so the graph
is never mutated: an Evaluator adapter is only the set of tensors fed at score time.

Inputs:
    input_ids       int64 [batch, seq]     attention_mask  int64 [batch, seq]
    lora.scale      f32 [1]
    head.weight     f32 [hidden, 1]        head.bias       f32 [1]
    lora.<layer>.<target>.A   f32 [in, R_MAX]    lora.<layer>.<target>.B   f32 [R_MAX, out]
Output: logits f32 [batch, 1].

Writes <out>/foundation_lora_fp32.onnx and <out>/foundation.json (sha256, version, shape
table) and prints the sha256 that goes in FOUNDATION_MODEL_HASH. How the file reaches
validators is decided outside this kit.

--check-adapter runs the graph with onnxruntime on a real PEFT adapter and compares it with
the adapter merged into a PyTorch copy of the base (in memory, the reference only; no
artifact is written from it) at several lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from float_runtime.lora_format import (NODE_TOKENIZER_SHA256, R_MAX, TARGET_MODULES, TARGETS,
                                       target_dims)
from float_runtime.train_adapter import DEFAULT_BASE, TrainError, check_tokenizer, render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("export_foundation")

OPSET = 17
ONNX_NAME = "foundation_lora_fp32.onnx"
DEFAULT_VERSION = "modernbert-base-lora16-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_names(num_layers: int) -> list[str]:
    names = ["input_ids", "attention_mask", "lora.scale", "head.weight", "head.bias"]
    for i in range(num_layers):
        for t in TARGETS:
            names += [f"lora.{i}.{t}.A", f"lora.{i}.{t}.B"]
    return names


def build_graph_module(base: str):
    """(torch module whose forward takes the inputs above, config)."""
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModelForSequenceClassification

    cfg = AutoConfig.from_pretrained(base)
    if (getattr(cfg, "model_type", "") or "").lower() != "modernbert":
        raise TrainError(f"{base} is a '{cfg.model_type}' model: the foundation is ModernBERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, torch_dtype=torch.float32, attn_implementation="sdpa").float().eval()

    class Feed:
        lora: dict = {}
        scale = None
        w = None
        b = None

    feed = Feed()

    class LoRALinear(nn.Module):
        def __init__(self, inner, key):
            super().__init__()
            self.inner, self.key = inner, key

        def forward(self, x):
            a, b = feed.lora[self.key]
            return self.inner(x) + torch.matmul(torch.matmul(x, a), b) * feed.scale

    class RewardHead(nn.Module):
        def forward(self, x):
            return torch.matmul(x, feed.w) + feed.b

    for i, layer in enumerate(model.model.layers):
        for t in TARGETS:
            parent_name, attr = t.split(".")
            parent = getattr(layer, parent_name)
            setattr(parent, attr, LoRALinear(getattr(parent, attr), f"{i}.{t}"))
    model.classifier = RewardHead()
    keys = [f"{i}.{t}" for i in range(cfg.num_hidden_layers) for t in TARGETS]

    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, lora_scale, head_weight, head_bias, *lora):
            feed.scale, feed.w, feed.b = lora_scale, head_weight, head_bias
            feed.lora = {k: (lora[2 * j], lora[2 * j + 1]) for j, k in enumerate(keys)}
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    return Graph().eval(), cfg


def trace_feed(cfg) -> list:
    """Inputs for tracing: small random LoRA tensors (so no branch is folded away), unit scale."""
    import torch
    g = torch.Generator().manual_seed(0)
    dims = target_dims(cfg)
    out = [torch.ones(1), torch.randn(cfg.hidden_size, 1, generator=g) * 0.02, torch.zeros(1)]
    for _ in range(cfg.num_hidden_layers):
        for t in TARGETS:
            i, o = dims[t]
            out += [torch.randn(i, R_MAX, generator=g) * 0.02, torch.randn(R_MAX, o, generator=g) * 0.02]
    return out


def peft_feed(cfg, adapter_dir: Path) -> tuple[list, dict]:
    """The graph's adapter inputs from a PEFT adapter directory (padded to R_MAX)."""
    import numpy as np
    import torch
    from float_runtime.export_adapter import read_peft
    tensors, meta = read_peft(adapter_dir, cfg)
    dims = target_dims(cfg)
    r = meta["rank"]
    out = [torch.tensor([meta["alpha"] / r], dtype=torch.float32),
           torch.from_numpy(tensors["head.weight"].copy()),
           torch.from_numpy(tensors["head.bias"].copy())]
    for i in range(cfg.num_hidden_layers):
        for t in TARGETS:
            fi, fo = dims[t]
            a = np.zeros((fi, R_MAX), np.float32)
            b = np.zeros((R_MAX, fo), np.float32)
            a[:, :r] = tensors[f"lora.{i}.{t}.A"]
            b[:r, :] = tensors[f"lora.{i}.{t}.B"]
            out += [torch.from_numpy(a), torch.from_numpy(b)]
    return out, meta


def export(graph, cfg, ids, path: Path) -> None:
    import torch
    t = torch.tensor([ids], dtype=torch.long)
    names = input_names(cfg.num_hidden_layers)
    with torch.no_grad():
        torch.onnx.export(
            graph, (t, torch.ones_like(t), *trace_feed(cfg)), str(path), dynamo=False,
            input_names=names, output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
            opset_version=OPSET, do_constant_folding=True)


def check_graph(path: Path, cfg) -> None:
    import onnx
    onnx.checker.check_model(str(path))
    m = onnx.load(str(path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    got = [i.name for i in m.graph.input if i.name not in inits]
    want = input_names(cfg.num_hidden_layers)
    if sorted(got) != sorted(want):
        missing = [n for n in want if n not in got]
        raise TrainError(f"graph inputs differ from the LoRA layout (missing {missing[:4]}, "
                         f"got {len(got)} want {len(want)}): an input was folded away")
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer):
        raise TrainError("weights stored outside the .onnx file")


def truncate_tokens(ids: list[int], n: int) -> list[int]:
    """The node's runtime.TruncateTokens: keep the first n-1 ids and the closing [SEP]."""
    if n <= 0 or len(ids) <= n:
        return list(ids)
    return list(ids[: n - 1]) + [ids[-1]]


def load_reward_head(model, path: Path) -> None:
    """Copy an external score head (state dict with weight [1, H], optional bias) into the
    model's classifier. A shape mismatch is an error, never a silent partial load."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or "weight" not in sd:
        raise TrainError(f"{path}: expected a state dict with a 'weight' key")
    head = getattr(model, "classifier", None) or getattr(model, "score", None)
    if head is None:
        raise TrainError("model has no classifier/score head to load the reward head into")
    if not hasattr(head, "weight"):
        head = next(m for m in head.modules() if isinstance(m, torch.nn.Linear))
    w = sd["weight"].float()
    if tuple(head.weight.shape) != tuple(w.shape):
        raise TrainError(f"reward head shape {tuple(w.shape)} != model head {tuple(head.weight.shape)}")
    with torch.no_grad():
        head.weight.copy_(w)
        if "bias" in sd and getattr(head, "bias", None) is not None:
            head.bias.copy_(sd["bias"].float())
    log.info("loaded external reward head %s", path)


def load_merged(base: str, adapter_dir: Path, head: Path | None):
    """The adapter merged into a PyTorch copy of the base (in memory, for --check-adapter
    only): the reference the split graph is compared with. No artifact is ever written from it."""
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForSequenceClassification

    cfg = AutoConfig.from_pretrained(base)
    if (getattr(cfg, "model_type", "") or "").lower() != "modernbert":
        raise TrainError(f"{base} is a '{cfg.model_type}' model: the node runs ModernBERT only")
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, torch_dtype=torch.float32, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    if head is not None:
        load_reward_head(model, head)
    model = model.float().eval()
    if model.config.num_labels != 1:
        raise TrainError(f"the model has {model.config.num_labels} labels; a scoring adapter has one")
    return model


def torch_score(model, ids: list[int]) -> float:
    import torch
    t = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=t, attention_mask=torch.ones_like(t)).logits
    return float(out.reshape(-1)[0].item())


def check_adapter(path: Path, base: str, cfg, adapter: str, tok, lengths) -> list[dict]:
    """onnxruntime on the split graph vs the merged PyTorch model."""
    import numpy as np
    import onnxruntime as ort
    from float_runtime.export_adapter import resolve_adapter
    adapter_dir, head = resolve_adapter(adapter)
    merged = load_merged(base, adapter_dir, head)
    feed, _ = peft_feed(cfg, adapter_dir)
    names = input_names(cfg.num_hidden_layers)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    long_ids = tok(render("Triage this report.", "The parser reads past the buffer end. " * 80),
                   truncation=True, max_length=512)["input_ids"]
    rows = []
    for n in lengths:
        x = truncate_tokens(long_ids, n)
        arr = np.array([x], dtype=np.int64)
        inp = {"input_ids": arr, "attention_mask": np.ones_like(arr)}
        inp.update({k: v.numpy() for k, v in zip(names[2:], feed)})
        got = float(sess.run(None, inp)[0].reshape(-1)[0])
        ref = torch_score(merged, x)
        rows.append({"tokens": len(x), "merged_torch": ref, "split_onnx": got, "abs_diff": abs(got - ref)})
        log.info("check %3d tokens: merged %+.6f split %+.6f diff %.2e", len(x), ref, got, abs(got - ref))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default=DEFAULT_VERSION, help="FOUNDATION_MODEL_VERSION")
    ap.add_argument("--check-adapter", help="PEFT adapter dir or HF spec to check the graph with")
    ap.add_argument("--check-lengths", default="2,9,33,64,129")
    a = ap.parse_args(argv)
    try:
        from transformers import AutoTokenizer
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(a.base)
        check_tokenizer(tok)
        graph, cfg = build_graph_module(a.base)
        ids = [int(i) for i in tok(render("Review this change and say whether it is safe.",
                                          "It retries the charge without an idempotency key. " * 12),
                                   truncation=True, max_length=512)["input_ids"]]
        path = out / ONNX_NAME
        log.info("exporting %s (traced at %d tokens, R_MAX %d)", path, len(ids), R_MAX)
        export(graph, cfg, ids, path)
        check_graph(path, cfg)
        digest = sha256_file(path)
        dims = target_dims(cfg)
        spec = {
            "foundation_version": a.version,
            "foundation_hash": digest,
            "onnx": ONNX_NAME,
            "bytes": path.stat().st_size,
            "base_model": a.base,
            "tokenizer_hash": NODE_TOKENIZER_SHA256,
            "r_max": R_MAX,
            "num_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "target_modules": list(TARGET_MODULES),
            "targets": {t: list(dims[t]) for t in TARGETS},
            "opset": OPSET,
        }
        if a.check_adapter:
            del graph
            spec["check"] = check_adapter(path, a.base, cfg, a.check_adapter, tok,
                                          [int(x) for x in a.check_lengths.split(",")])
        (out / "foundation.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    except TrainError as e:
        log.error("%s", e)
        return 2
    print(f"FOUNDATION_MODEL_PATH={(out / ONNX_NAME).as_posix()}")
    print(f"FOUNDATION_MODEL_HASH={digest}")
    print(f"FOUNDATION_MODEL_VERSION={a.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
