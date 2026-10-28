"""HuggingFace plumbing the wide converter needs — extracted from the int8 pipeline.

These five functions are the only things the wide pipeline used out of
`intent_training_experiments/pipelines/int8_runtime/convert/`, which is otherwise ~590 lines
of calibration, SmoothQuant and int8 requantisation that the wide path deliberately does not
have (Q32.32 needs no calibration and no {mul,shift} scales).

Extracted rather than imported so this repo has no dependency on the experiments tree:

  ``load_encoder``, ``get_param``, ``first_present``, ``_norm_cfg``  <- convert/hf_common.py
  ``_resolve_adapter``                                              <- convert/pipeline.py
  ``_load_external_reward_head``                                    <- convert/adapter.py

torch/transformers are imported lazily inside the functions that need them, so `format.py`,
`interp.py` and the golden tests all run without the heavy deps installed.
"""

from __future__ import annotations

import os
from pathlib import Path


# --------------------------------------------------------------------------- #
#  State-dict name resolution — HF families disagree about weight names
# --------------------------------------------------------------------------- #

def first_present(sd: dict, names: list[str], required: bool = True):
    """Return the first of `names` present in the state dict.

    Tolerant lookup, because the same tensor is called `model.embeddings.tok_embeddings.weight`
    in ModernBERT and `bert.embeddings.word_embeddings.weight` in BERT. Raises with the first
    keys it did see, so a rename shows up as a readable error rather than a KeyError on a name
    nobody recognises.
    """
    for n in names:
        if n in sd:
            return sd[n]
    if required:
        seen = sorted(list(sd.keys()))[:20]
        raise KeyError(f"none of {names} in state dict. First keys seen: {seen}")
    return None


def get_param(sd: dict, names: list[str], required: bool = True):
    return first_present(sd, names, required=required)


def _norm_cfg(hfcfg) -> dict:
    """Normalize HF config field names across families into a stable dict."""
    g = lambda *names, default=None: next(  # noqa: E731
        (getattr(hfcfg, n) for n in names if getattr(hfcfg, n, None) is not None), default)
    return {
        "hidden_size": g("hidden_size", "n_embd", "d_model"),
        "num_hidden_layers": g("num_hidden_layers", "n_layer", "num_layers"),
        "num_attention_heads": g("num_attention_heads", "n_head"),
        "num_key_value_heads": g("num_key_value_heads", "num_attention_heads", "n_head"),
        "intermediate_size": g("intermediate_size", "n_inner", default=None),
        "max_position_embeddings": g("max_position_embeddings", "n_positions", default=2048),
        "layer_norm_eps": g("layer_norm_eps", "rms_norm_eps", "norm_eps", default=1e-5),
        "rope_theta": g("rope_theta", "rotary_emb_base", default=10000.0),
        "vocab_size": g("vocab_size"),
        "classifier_pooling": g("classifier_pooling", default="mean"),
    }


def load_encoder(model_name: str, trust_remote_code: bool = False):
    """Load an encoder reward model (num_labels=1) + tokenizer + a normalized config dict."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
    hfcfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, trust_remote_code=trust_remote_code)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    cfg = _norm_cfg(hfcfg)
    return model.eval(), tok, cfg


# --------------------------------------------------------------------------- #
#  Adapter resolution + external reward head
# --------------------------------------------------------------------------- #

def _resolve_adapter(adapter: str) -> tuple[str, str | None]:
    """Return (local_adapter_dir, reward_head_path_or_None).

    ``adapter`` may be:
      - a local directory containing adapter_config.json (used as-is), or
      - a HF spec "repo_id" or "repo_id/sub/folder" — downloaded via snapshot_download,
        then the subfolder (if any) is descended into.
    An external ``reward_head.pt`` sitting next to the adapter is auto-detected. The head is
    NOT part of the LoRA, so a converter that skipped it would merge the adapter and then
    score through the base model's randomly-initialised classifier — plausible numbers, no
    trained judgment at all.
    """
    p = Path(adapter)
    if p.exists() and (p / "adapter_config.json").exists():
        local = str(p)
    else:
        # treat as HF: first path segment is the repo id, the rest is a subfolder
        parts = adapter.replace("\\", "/").split("/")
        if len(parts) < 2:
            raise ValueError(
                f"{adapter!r} is neither a local adapter dir nor a "
                "'namespace/repo[/subfolder]' HF spec")
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        from huggingface_hub import snapshot_download
        cache_root = Path(os.environ.get("TELEGRAPH_ADAPTER_CACHE", Path.home() / ".cache"))
        root = snapshot_download(
            repo_id,
            local_dir=str(cache_root / ("telegraph_adapters_" + repo_id.replace("/", "_"))))
        local = str(Path(root) / subfolder) if subfolder else root
        if not (Path(local) / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"no adapter_config.json under {local} (from HF {repo_id!r}, "
                f"subfolder {subfolder!r})")

    rh = Path(local) / "reward_head.pt"
    return local, (str(rh) if rh.exists() else None)


def _find_linear(module):
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return m
    raise AttributeError("no nn.Linear found in the head module")


def _load_external_reward_head(model, reward_head_path: str, torch) -> None:
    """Copy an external score head (weight [1,H], optional bias) into the model's head.

    Finds the classification/score Linear (transformers names it `score` for decoders,
    `classifier` for encoders) and overwrites its weight/bias. Fails loudly if shapes
    don't match — a silent mismatch would produce a wrong-but-plausible score.
    """
    sd = torch.load(reward_head_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or "weight" not in sd:
        raise ValueError(f"{reward_head_path}: expected a state dict with a 'weight' key")
    w = sd["weight"].float()
    head = getattr(model, "score", None) or getattr(model, "classifier", None)
    if head is None:
        raise AttributeError("model has no `score` or `classifier` head to load into")
    # some heads are a bare Linear, some a small module ending in a Linear — resolve it.
    linear = head if hasattr(head, "weight") else _find_linear(head)
    if tuple(linear.weight.shape) != tuple(w.shape):
        raise ValueError(f"reward head shape {tuple(w.shape)} != model head "
                         f"{tuple(linear.weight.shape)}")
    with torch.no_grad():
        linear.weight.copy_(w.to(linear.weight.dtype))
        if "bias" in sd and getattr(linear, "bias", None) is not None:
            linear.bias.copy_(sd["bias"].float().to(linear.bias.dtype))
