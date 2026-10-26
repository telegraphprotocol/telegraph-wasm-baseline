"""Merge a float LoRA adapter into its backbone and convert to a wide Q32.32 `.imwq` artifact.

The driver for the whole conversion: resolve the adapter (local dir or HF repo/subfolder),
merge the LoRA into the base in float with peft (which is exactly how it was applied at
training time), load an external reward head if one ships beside the adapter, extract the
merged state dict, and hand it to `convert.convert_encoder_from_sd`.

No calibration and no {mul,shift} scales, unlike the int8 pipeline: in Q32.32 every weight is
just `fp_from_float(w)` and activations stay in the same format end to end, so there is
nothing to calibrate.

Run:
    python -m wide_runtime.convert_adapter \
        --base answerdotai/ModernBERT-base \
        --adapter 316usman/vulnerability-triage-adapter/VULNERABILITY_TRIAGE/bb-v2-modernbert \
        --out local/builds/wide/VULNERABILITY_TRIAGE.imwq

The artifact is ~1.2 GB for ModernBERT-base (int64 weights, 8x the int8 build). It belongs
under `local/`, which is gitignored — it is rebuildable from this code, which is the thing
worth committing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def merge_to_wide(base_model: str, adapter: str, out_path: str, *,
                  reward_head: str | None = None, trust_remote_code: bool = False):
    """(base, adapter) -> a Q32.32 `.imwq` on disk. Returns (summary, merged_model, tokenizer).

    The merged model and tokenizer come back so a caller can score the same texts through
    float PyTorch and compare — the only way to tell "the runtime lost the ranking" from
    "the adapter never had one".
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
    from peft import PeftModel
    from .hf_helpers import _norm_cfg, _load_external_reward_head, _resolve_adapter
    from .convert import convert_encoder_from_sd

    local_adapter, auto_head = _resolve_adapter(adapter)
    reward_head = reward_head or auto_head

    hfcfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, trust_remote_code=trust_remote_code)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    merged = PeftModel.from_pretrained(base, local_adapter).merge_and_unload().eval()
    if reward_head is not None:
        _load_external_reward_head(merged, reward_head, torch)
    sd = {k: v.detach().cpu().float().numpy() for k, v in merged.state_dict().items()}
    cfg = _norm_cfg(hfcfg)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    summary = convert_encoder_from_sd(sd, cfg, out_path)
    return summary, merged, tok


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return _pearson(ra, rb)


def evaluate(out_path, merged, tok, texts, max_length=96):
    """Score `texts` through float(merged) and wide(.imwq); report correlation.

    This measures RUNTIME FIDELITY, not adapter quality: a rho near 1.0 says the Q32.32
    artifact reproduces the float model, which is true whether the float model ranks well or
    not. Judging the adapter needs texts with known-good and known-bad answers.
    """
    import torch
    from .format import load_wide
    from .interp import WideInterp

    m = load_wide(out_path)
    fs, ws = [], []
    merged.eval()
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            fs.append(float(merged(**enc).logits.squeeze().item()))
        ids = np.asarray(tok(t, truncation=True, max_length=max_length)["input_ids"],
                         dtype=np.int64)
        ws.append(WideInterp(m).run(ids))
    fs = np.array(fs); ws = np.array(ws)
    print(f"n={len(texts)}  Pearson r={_pearson(fs, ws):+.3f}  "
          f"Spearman rho={_spearman(fs, ws):+.3f}")
    print(f"  float range [{fs.min():+.2f},{fs.max():+.2f}]  "
          f"wide range [{ws.min():+.2f},{ws.max():+.2f}]")
    return _spearman(fs, ws)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reward-head", default=None)
    ap.add_argument("--max-length", type=int, default=96)
    a = ap.parse_args(argv)
    summary, merged, tok = merge_to_wide(a.base, a.adapter, a.out, reward_head=a.reward_head)
    print(f"wide artifact: {a.out}  sha256={summary['hash'][:16]}  layers={summary['layers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
