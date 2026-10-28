#!/usr/bin/env python3
"""Merge an existing Telegraph LoRA checkpoint and export ABI-v2 float ONNX.

This is conversion only: it never trains or changes checkpoint weights. The
output contract is input_ids/attention_mask int64 [batch, sequence] -> logits
float32 [batch, 1], with a dynamic sequence length up to the node's 512-token
limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def export_adapter(base_model: str, adapter: str, out_path: str,
                   reward_head: str | None = None) -> dict:
    import onnx
    import torch
    # Conversion is CPU-only. PEFT otherwise follows a macOS MPS default device,
    # which cannot load the BF16 classifier stored by these checkpoints.
    torch.set_default_device("cpu")
    # Intel macOS only provides torch 2.2 wheels for Python 3.12. That release
    # rejects torch.compile on 3.12 while ModernBERT decorates one helper at
    # import time. Export is deliberately eager, so make that decorator a
    # no-op in this one unsupported toolchain combination.
    if sys.version_info >= (3, 12) and tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 4):
        def eager_compile(model=None, **_kwargs):
            return (lambda fn: fn) if model is None else model
        torch.compile = eager_compile
    from peft import PeftModel
    # PEFT 0.21 picks MPS for adapter tensors whenever it is available. The
    # archived adapters store their classifier in BF16, a dtype MPS cannot
    # accept. Widen only the conversion copy before PEFT loads it.
    from peft.utils import save_and_load
    _safe_load_file = save_and_load.safe_load_file
    def _load_adapter_fp32(path, device="cpu"):
        state = _safe_load_file(path, device="cpu")
        return {key: (value.float() if value.dtype == torch.bfloat16 else value)
                for key, value in state.items()}
    save_and_load.safe_load_file = _load_adapter_fp32
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    from .hf_helpers import _load_external_reward_head, _resolve_adapter
    from .train_adapter import check_tokenizer

    local_adapter, auto_head = _resolve_adapter(adapter)
    reward_head = reward_head or auto_head
    cfg = AutoConfig.from_pretrained(base_model)
    if (getattr(cfg, "model_type", "") or "").lower() != "modernbert":
        raise ValueError(f"{base_model} is {cfg.model_type}, expected ModernBERT")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    check_tokenizer(tokenizer)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        device_map="cpu",
    )
    base = base.to("cpu")
    merged = PeftModel.from_pretrained(base, local_adapter, device="cpu", autocast_adapter_dtype=False).merge_and_unload().float().eval()
    if reward_head:
        _load_external_reward_head(merged, reward_head, torch)
    merged.config._attn_implementation = "eager"

    class LogitsOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = LogitsOnly(merged).eval()
    dummy_ids = torch.tensor([[50281, 4118, 17526, 27, 187, 18658, 436, 13814, 1818, 50282]], dtype=torch.long)
    dummy_mask = torch.ones_like(dummy_ids)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = wrapped(dummy_ids, dummy_mask).detach().cpu().float().reshape(-1)[0]
        torch.onnx.export(
            wrapped,
            (dummy_ids, dummy_mask),
            str(out),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    model = onnx.load(str(out), load_external_data=False)
    onnx.checker.check_model(model)
    result = {
        "base_model": base_model,
        "adapter": adapter,
        "resolved_adapter": local_adapter,
        "onnx": str(out),
        "bytes": out.stat().st_size,
        "reference_f32": float(reference),
        "reference_hex": f"0x{reference.view(torch.int32).item() & 0xffffffff:08x}",
    }
    (out.parent / (out.name + ".export.json")).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--adapter", required=True, help="local dir or namespace/repo/subfolder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reward-head")
    args = ap.parse_args(argv)
    print(json.dumps(export_adapter(args.base, args.adapter, args.out, args.reward_head)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
