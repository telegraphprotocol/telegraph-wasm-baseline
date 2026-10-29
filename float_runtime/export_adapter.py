#!/usr/bin/env python3
"""Write an Evaluator adapter as a safetensors delta package (abi "lora-v1") for the shared
Foundation Model. Step 2 of the kit:

    1. train      python -m float_runtime.train_adapter --intent X --data pairs.jsonl --out runs/X
    2. export     python -m float_runtime.export_adapter --run runs/X --intent X --class judgment \\
                      --foundation foundation/foundation.json
    3. package    go run ./cmd/adapterbuild lora --intent X --package runs/X/X.safetensors --out runs/X
    4. register   registerWasm(keccak256, url, "X", feeAddress) with the packaged file

The package is a plain safetensors file (lora_format.py): only the LoRA A/B matrices of
every target linear and the reward head, never the base weights (it never calls
merge_and_unload), with the package fields in the safetensors `__metadata__` map. The
metadata names the foundation (sha256 + version) and the tokenizer it was trained against;
a node with a different foundation or tokenizer refuses it (foundation_mismatch /
tokenizer_mismatch).

Tensors (pre-transposed, rank r <= R_MAX, zero-padded by the engine at load):
    lora.<l>.<attn.Wqkv|attn.Wo|mlp.Wi|mlp.Wo>.A   f32 [in, r]
    lora.<l>.<...>.B                              f32 [r, out]
    head.weight  f32 [hidden, 1]    head.bias  f32 [1]

One lora.scale (alpha / r) for the whole adapter: PEFT rank_pattern, alpha_pattern, rsLoRA
and DoRA are refused. PEFT's target_modules ["Wqkv","Wo","Wi"] matches mlp.Wo as well as
attn.Wo (suffix match), so each layer has four targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from float_runtime.lora_format import (FILE_EXT, NODE_MAX_LEN, NODE_TOKENIZER_SHA256, R_MAX,
                                       TARGET_MODULES, TARGETS, metadata, target_dims, tensor_names,
                                       write_package)
from float_runtime.train_adapter import TrainError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("export_adapter")


def resolve_adapter(adapter: str) -> tuple[Path, Path | None]:
    """(local adapter dir, external reward_head.pt or None).

    `adapter` is a local directory holding adapter_config.json, or a Hugging Face spec
    "namespace/repo[/subfolder]" (downloaded with snapshot_download). An external
    reward_head.pt next to the adapter is picked up: it is not part of the LoRA, and skipping
    it would score through an untrained classifier."""
    p = Path(adapter)
    if p.is_dir() and (p / "adapter_config.json").exists():
        local = p
    else:
        parts = adapter.replace("\\", "/").split("/")
        if len(parts) < 2 or p.exists():
            raise TrainError(f"{adapter!r} is neither an adapter directory (with adapter_config.json) "
                             "nor a 'namespace/repo[/subfolder]' Hugging Face spec")
        repo_id, sub = "/".join(parts[:2]), "/".join(parts[2:])
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo_id))
        local = root / sub if sub else root
        if not (local / "adapter_config.json").exists():
            raise TrainError(f"no adapter_config.json under {local} (Hugging Face {repo_id!r}, "
                             f"subfolder {sub!r})")
    rh = local / "reward_head.pt"
    return local, (rh if rh.exists() else None)


def read_peft(adapter_dir: Path, cfg) -> tuple[dict, dict]:
    """({package tensor name: numpy f32}, {rank, alpha, target_modules}) from a PEFT dir."""
    import numpy as np
    from safetensors.torch import load_file
    conf = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if conf.get("peft_type") != "LORA":
        raise TrainError(f"{adapter_dir}: peft_type {conf.get('peft_type')!r}, want LORA")
    for k in ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern"):
        if conf.get(k):
            raise TrainError(f"{k} is not supported by lora-v1 (one rank and one scale per adapter)")
    r, alpha = int(conf["r"]), float(conf["lora_alpha"])
    if not 1 <= r <= R_MAX:
        raise TrainError(f"rank {r} is outside 1..{R_MAX} (R_MAX of the foundation)")
    mods = sorted(set(conf.get("target_modules") or []))
    if any(m not in TARGET_MODULES for m in mods) or not mods:
        raise TrainError(f"target_modules {mods}: lora-v1 allows only {sorted(TARGET_MODULES)}")
    if conf.get("bias", "none") != "none":
        raise TrainError("LoRA bias training is not supported by lora-v1")
    # bf16 checkpoints widen to f32 exactly (PEFT does the same on load).
    raw = {k: v.float().numpy() for k, v in load_file(str(adapter_dir / "adapter_model.safetensors")).items()}
    dims = target_dims(cfg)
    out = {}
    for i in range(cfg.num_hidden_layers):
        for t in TARGETS:
            fi, fo = dims[t]
            pa = f"base_model.model.model.layers.{i}.{t}.lora_A.weight"
            pb = f"base_model.model.model.layers.{i}.{t}.lora_B.weight"
            if t.split(".")[1] not in mods:
                a, b = np.zeros((r, fi), np.float32), np.zeros((fo, r), np.float32)
            else:
                if pa not in raw or pb not in raw:
                    raise TrainError(f"adapter is missing {pa} / {pb}")
                a, b = raw.pop(pa).astype(np.float32), raw.pop(pb).astype(np.float32)
            if a.shape != (r, fi) or b.shape != (fo, r):
                raise TrainError(f"layer {i} {t}: shapes {a.shape} {b.shape}, want {(r, fi)} {(fo, r)}")
            out[f"lora.{i}.{t}.A"] = np.ascontiguousarray(a.T)  # [in, r]
            out[f"lora.{i}.{t}.B"] = np.ascontiguousarray(b.T)  # [r, out]
    w = next((raw.pop(k) for k in ("base_model.model.classifier.weight",
                                   "base_model.model.classifier.modules_to_save.default.weight",
                                   "base_model.model.score.weight")
              if k in raw), None)
    bias = next((raw.pop(k) for k in ("base_model.model.classifier.bias",
                                      "base_model.model.classifier.modules_to_save.default.bias",
                                      "base_model.model.score.bias")
                 if k in raw), None)
    rh = adapter_dir / "reward_head.pt"
    if rh.exists():
        import torch
        sd = torch.load(rh, map_location="cpu", weights_only=True)
        w = sd["weight"].float().numpy()
        bias = sd["bias"].float().numpy() if "bias" in sd else np.zeros(1, np.float32)
        log.info("reward head from %s", rh)
    if w is None:
        raise TrainError("no reward head: the adapter has no classifier weights and no reward_head.pt")
    if bias is None:
        bias = np.zeros(1, np.float32)
    if w.shape != (1, cfg.hidden_size) or bias.reshape(-1).shape != (1,):
        raise TrainError(f"reward head shapes {w.shape} {bias.shape}, want (1, {cfg.hidden_size}) (1,)")
    out["head.weight"] = np.ascontiguousarray(w.T.astype(np.float32))  # [hidden, 1]
    out["head.bias"] = bias.astype(np.float32).reshape(1)
    leftover = [k for k in raw if "lora_" in k or "classifier" in k or "score" in k]
    if leftover:
        raise TrainError(f"adapter has tensors lora-v1 cannot carry: {leftover[:4]}")
    return out, {"rank": r, "alpha": alpha, "target_modules": mods}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", help="train_adapter output dir (adapter/ and result.json)")
    ap.add_argument("--adapter", help="PEFT adapter dir or HF spec namespace/repo[/subfolder]")
    ap.add_argument("--intent", required=True, help="intent (family prefix) the adapter serves")
    ap.add_argument("--class", dest="cls", required=True, choices=["judgment", "hybrid"])
    ap.add_argument("--foundation", required=True, help="foundation.json from export_foundation")
    ap.add_argument("--max-tokens", type=int, default=NODE_MAX_LEN)
    ap.add_argument("--training-record", help="JSON training record to embed (standard section 9)")
    ap.add_argument("--out", help=f"output {FILE_EXT} path (default <run>/<intent>{FILE_EXT})")
    a = ap.parse_args(argv)
    try:
        from transformers import AutoConfig
        if not 1 <= a.max_tokens <= NODE_MAX_LEN:
            raise TrainError(f"--max-tokens must be in [1, {NODE_MAX_LEN}]")
        spec = json.loads(Path(a.foundation).read_text(encoding="utf-8"))
        adapter = a.adapter or (str(Path(a.run) / "adapter") if a.run else None)
        if not adapter:
            raise TrainError("give --run or --adapter")
        adapter_dir, _ = resolve_adapter(adapter)
        cfg = AutoConfig.from_pretrained(spec["base_model"])
        if spec["targets"] != {t: list(d) for t, d in target_dims(cfg).items()}:
            raise TrainError("foundation.json does not describe this base model's shapes")
        if spec.get("r_max") != R_MAX:
            raise TrainError(f"foundation R_MAX {spec.get('r_max')} != kit R_MAX {R_MAX}")
        if a.run and (Path(a.run) / "result.json").exists():
            res = json.loads((Path(a.run) / "result.json").read_text(encoding="utf-8"))
            if res.get("base_model") and res["base_model"] != spec["base_model"]:
                raise TrainError(f"the run trained on {res['base_model']}, the foundation is {spec['base_model']}")
            if res.get("tokenizer_sha256") and res["tokenizer_sha256"] != NODE_TOKENIZER_SHA256:
                raise TrainError(f"the run's tokenizer {res['tokenizer_sha256']} is not the node's {NODE_TOKENIZER_SHA256}")
        tensors, meta = read_peft(adapter_dir, cfg)
        training = None
        if a.training_record:
            training = json.loads(Path(a.training_record).read_text(encoding="utf-8"))
        tj = adapter_dir / "tokenizer.json"
        if tj.exists():
            got = hashlib.sha256(tj.read_bytes()).hexdigest()
            if got != NODE_TOKENIZER_SHA256:
                raise TrainError(f"{tj} has sha256 {got}, not the node's tokenizer {NODE_TOKENIZER_SHA256}")
        meta = metadata(cls=a.cls, intent=a.intent, foundation_hash=spec["foundation_hash"],
                        foundation_version=spec["foundation_version"], rank=meta["rank"], alpha=meta["alpha"],
                        target_modules=meta["target_modules"], max_tokens=a.max_tokens, training=training)
        names = tensor_names(cfg.num_hidden_layers)
        out = Path(a.out) if a.out else Path(a.run or ".") / f"{a.intent}{FILE_EXT}"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = write_package(out, meta, {n: tensors[n] for n in names})
    except TrainError as e:
        log.error("%s", e)
        return 2
    log.info("wrote %s (%d bytes, sha256 %s, rank %s, %d tensors)", out, len(data),
             hashlib.sha256(data).hexdigest(), meta["rank"], len(names))
    print(f"\nNext:\n  go run ./cmd/adapterbuild lora --intent {a.intent} --package {out.as_posix()} "
          f"--out {out.parent.as_posix()} --foundation {Path(a.foundation).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
