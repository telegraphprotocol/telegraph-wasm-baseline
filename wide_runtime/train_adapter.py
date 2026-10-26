#!/usr/bin/env python3
"""Train a scoring adapter for a judgment or hybrid intent.

The first step of the wide adapter pipeline. It fine-tunes a LoRA Bradley-Terry
reward model on ModernBERT-base from preference pairs (a prompt, a better answer,
a worse answer) and writes the adapter to disk. The other steps:

    1. train      python -m wide_runtime.train_adapter --intent X --data pairs.jsonl --out runs/X
    2. convert    python -m wide_runtime.convert_adapter --base answerdotai/ModernBERT-base \
                      --adapter runs/X/adapter --out runs/X/X.imwq
    3. package    go run ./cmd/adapterbuild --intent X --imwq runs/X/X.imwq --out runs/X
    4. register   registerWasm(keccak256, url, "X", feeAddress) with the .wasm from step 3

Run every step from the root of this repository.

Input
-----
A JSONL file, one pair per line, in the same shape the evaluation pool takes:

    {"prompt": "...", "chosen": "the better answer", "rejected": "the worse answer"}

`label_kind` may be present; rows whose kind is not `preference_pair` are skipped.
`margin` (float, default 0) is honoured. A Hugging Face dataset with
prompt/chosen/rejected columns works too (--hf-dataset), using its named splits.

For a hybrid intent, `prompt` is the question key the miner was asked and the
answers are whole miner response bodies, because that is what the node scores.

What the node fixes, and so this script fixes
---------------------------------------------
- Backbone: ModernBERT only. The node ships one tokenizer
  and the converter reads ModernBERT weights; another backbone cannot be scored.
  The base model's tokenizer is checked against the node's conformance vectors.
- Input text: "### Task:\\n<prompt>\\n\\n### Answer:\\n<answer>", the node's
  RenderForJudgment. Training on any other template teaches a different task.
- Length: the node scores the first 256 tokens (tokenizer.DefaultMaxLen). Training
  on longer inputs teaches the model to use text it will never see at scoring time,
  so --max-length defaults to 256.

Two gates
---------
- Dataset gate, before training: refuses a dataset a shortcut can solve (reused
  negatives, "pick the longer answer", train/eval leakage, a bag-of-words tell, or
  rejected == chosen). An adapter trained on such data passes its own evaluation
  and then ranks real answers at chance. --check-data runs only this gate.
- Evaluation gate, after training: pairwise accuracy on held-out pairs must reach
  --gate-min-acc AND beat the "prefer the longer answer" baseline by a margin.
  The exit code is 0 on PASS, 1 on FAIL, 2 on an error.

Passing here is necessary, not sufficient: the node decides with its own hidden
evaluation pool (see the scoring-adapter standard).

Requirements: torch, transformers, peft, numpy; `datasets` only for
--hf-dataset. A GPU is strongly recommended; on CPU training runs in fp32.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_adapter")

DEFAULT_BASE = "answerdotai/ModernBERT-base"
NODE_MAX_LEN = 256  # the node's tokenizer.DefaultMaxLen

# The node's tokenizer conformance vectors (Telegraph pkg/wasm/tokenizer/testdata).
TOKENIZER_VECTORS = Path(__file__).resolve().parent / "testdata" / "tokenizer_vectors.json"


class TrainError(RuntimeError):
    """A step failed with a message meant for the person running the script."""


# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class Hyperparams:
    max_length: int = NODE_MAX_LEN
    batch_size: int = 1
    grad_accum: int = 16           # effective batch = batch_size * grad_accum
    grad_checkpointing: bool = True
    # Small rank, dropout and weight decay: preference datasets are small and a
    # reward model memorises a tell long before it learns the task.
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.01
    eval_split: float = 0.15       # held-out share when no validation file is given
    seed: int = 42
    eval_batch: int = 4
    gate_min_acc: float = 0.65
    # "Prefer the longer answer" scored 100% on one shipped dataset, so accuracy only
    # counts once it clears that baseline by this much.
    gate_min_margin_over_baseline: float = 0.10
    # Early stopping keeps the best epoch and stops on rising validation loss or on a
    # training loss that collapses toward zero (memorisation).
    early_stopping: bool = True
    es_metric: str = "val_loss"    # val_loss (lower is better) | pairwise_accuracy
    es_patience: int = 1
    es_min_delta: float = 0.001
    loss_floor: float = 1e-4

    @classmethod
    def from_file(cls, path) -> "Hyperparams":
        """YAML or JSON with any subset of the fields; a top-level `hyperparameters`
        key is honoured."""
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise TrainError(f"{path} is YAML but PyYAML is not installed") from e
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw) if raw.strip() else {}
        if isinstance(data.get("hyperparameters"), dict):
            data = data["hyperparameters"]
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            log.warning("ignoring unknown config keys: %s", ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _pair(prompt, chosen, rejected, margin=0.0) -> dict:
    return {"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected),
            "margin": float(margin or 0.0)}


def read_pairs_jsonl(path) -> list[dict]:
    """Preference pairs from a JSONL file. Rows of another label kind, rows missing a
    field and malformed lines are skipped and counted rather than fatal."""
    pairs, skipped = [], Counter()
    text = Path(path).read_text(encoding="utf-8")
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped["malformed JSON"] += 1
            continue
        if not isinstance(row, dict):
            skipped["not an object"] += 1
            continue
        if row.get("label_kind", "preference_pair") not in ("preference_pair", ""):
            skipped[f"label_kind {row.get('label_kind')}"] += 1
            continue
        if not all(str(row.get(k) or "").strip() for k in ("prompt", "chosen", "rejected")):
            skipped["missing prompt/chosen/rejected"] += 1
            continue
        pairs.append(_pair(row["prompt"], row["chosen"], row["rejected"], row.get("margin")))
    for why, c in skipped.items():
        log.warning("%s: skipped %d rows (%s)", path, c, why)
    log.info("%s: %d pairs", path, len(pairs))
    return pairs


_PROMPT_KEYS = ("prompt", "question", "instruction", "input", "query", "context")
_CHOSEN_KEYS = ("chosen", "preferred", "better", "positive", "chosen_response")
_REJECTED_KEYS = ("rejected", "dispreferred", "worse", "negative", "rejected_response")


def _column(cols, candidates, explicit=None):
    if explicit:
        if explicit not in cols:
            raise TrainError(f"column '{explicit}' not in dataset columns {sorted(cols)}")
        return explicit
    return next((c for c in candidates if c in cols), None)


def load_hf_splits(repo_id, *, token=None, config=None, prompt_col=None,
                   chosen_col=None, rejected_col=None) -> dict[str, list[dict]]:
    """Every split of a Hugging Face preference dataset, as {split: [pair, ...]}."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise TrainError("--hf-dataset needs the 'datasets' package (pip install datasets)") from e
    dsd = load_dataset(repo_id, name=config, token=token)
    if not hasattr(dsd, "items"):
        dsd = {"train": dsd}
    out = {}
    for split, ds in dsd.items():
        cols = set(ds.column_names)
        p = _column(cols, _PROMPT_KEYS, prompt_col)
        c = _column(cols, _CHOSEN_KEYS, chosen_col)
        r = _column(cols, _REJECTED_KEYS, rejected_col)
        if not (p and c and r):
            raise TrainError(f"split '{split}' has no prompt/chosen/rejected columns "
                             f"(found {sorted(cols)}); name them with --prompt-col etc.")
        out[split] = [_pair(row[p], row[c], row[r], row.get("margin")) for row in ds]
        log.info("split '%s': %d pairs (prompt=%s chosen=%s rejected=%s)",
                 split, len(out[split]), p, c, r)
    return out


def split_pairs(pairs, eval_frac, seed):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(len(pairs) * eval_frac)) if len(pairs) > 1 else 0
    ev = set(idx[:n_eval])
    return [pairs[i] for i in idx if i not in ev], [pairs[i] for i in idx if i in ev]


def render(prompt: str, answer: str) -> str:
    """The node's RenderForJudgment (Telegraph pkg/wasm/tokenizer/render.go), character for
    character. ModernBERT has no chat template, so this is the only template."""
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


# ---------------------------------------------------------------------------
# dataset gate: refuse data a shortcut can solve
# ---------------------------------------------------------------------------
# Thresholds were set against seven real datasets: the one clean corpus passes
# comfortably and the six shortcut-bearing ones fail. Each is a hard failure.
MIN_UNIQUE_REJECTED = 0.80   # unique `rejected` strings as a share of rows
MAX_SINGLE_REJECTED = 0.02   # no single `rejected` string above this share
MAX_LENGTH_ACC = 0.65        # accuracy of "pick the longer (or shorter) answer"
MAX_LEAK = 0.10              # eval negatives that appear verbatim in train
MAX_IDENTICAL = 0.01         # rows where rejected == chosen (no signal at all)
MAX_SURFACE_ACC = 0.65       # bag-of-words probe with the prompt hidden (0.50 = no tell)

_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _surface_probe_acc(rows: list[dict], seed: int, eval_frac: float, max_feats: int = 4000) -> float:
    """Held-out accuracy of a bag-of-words logistic regression that tells chosen from
    rejected WITHOUT seeing the prompt. Anything it learns above chance is a tell the
    reward model can use instead of learning the task. Deliberately weak: it looks for
    tells so blatant that a trivial learner finds them."""
    if len(rows) < 20:
        return float("nan")
    tr, ev = split_pairs(rows, eval_frac, seed)
    if not tr or not ev:
        return float("nan")
    df = Counter()
    for r in tr:  # vocabulary from the train half only
        for side in ("chosen", "rejected"):
            df.update(set(_tokens(r[side])))
    vocab = {w: i for i, (w, _) in enumerate(df.most_common(max_feats))}
    if not vocab:
        return float("nan")
    d = len(vocab)

    def vec(texts: Iterable[str]) -> np.ndarray:
        # L2-normalised, so the probe cannot simply rediscover length
        texts = list(texts)
        m = np.zeros((len(texts), d), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _tokens(t):
                j = vocab.get(w)
                if j is not None:
                    m[i, j] += 1.0
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-8)

    def xy(pairs):
        x = vec([p["chosen"] for p in pairs] + [p["rejected"] for p in pairs])
        y = np.concatenate([np.ones(len(pairs)), np.zeros(len(pairs))]).astype(np.float32)
        return x, y

    xtr, ytr = xy(tr)
    xev, yev = xy(ev)
    rng = np.random.default_rng(seed)
    w = np.zeros(d, dtype=np.float32)
    b = np.float32(0.0)
    lr, l2, epochs, bs = 0.5, 1e-4, 60, 64
    for _ in range(epochs):
        order = rng.permutation(xtr.shape[0])
        for s in range(0, len(order), bs):
            sl = order[s:s + bs]
            p = 1.0 / (1.0 + np.exp(-(xtr[sl] @ w + b)))
            g = (p - ytr[sl]) / len(sl)
            w -= lr * (xtr[sl].T @ g + l2 * w)
            b -= lr * g.sum()
    return float((((xev @ w + b) > 0) == (yev > 0.5)).mean())


def gate_dataset(rows: list[dict], hp: Hyperparams) -> dict:
    """Every dataset-level check. Returns {ok, n, metrics, failures}."""
    n = len(rows)
    if n == 0:
        return {"ok": False, "n": 0, "metrics": {}, "failures": ["dataset has 0 usable pairs"]}
    failures = []
    counts = Counter(r["rejected"].strip() for r in rows)

    # 1. repetition: a few stock negatives reused across the corpus
    uniq = len(counts) / n
    if uniq < MIN_UNIQUE_REJECTED:
        failures.append(f"only {len(counts)} unique 'rejected' answers in {n} pairs "
                        f"({uniq:.1%} < {MIN_UNIQUE_REJECTED:.0%}): the model can memorise them")
    top_text, top_n = counts.most_common(1)[0]
    if top_n / n > MAX_SINGLE_REJECTED:
        failures.append(f"one 'rejected' answer is reused in {top_n}/{n} pairs "
                        f"({top_n / n:.1%} > {MAX_SINGLE_REJECTED:.0%}): {top_text[:90]!r}")
    identical = sum(1 for r in rows if r["chosen"].strip() == r["rejected"].strip())
    if identical / n > MAX_IDENTICAL:
        failures.append(f"{identical}/{n} pairs have 'rejected' identical to 'chosen': "
                        f"they carry no preference, drop them")

    # 2. length: ties count as half, as a coin-flip tie-break would score them
    longer = sum(1 for r in rows if len(r["chosen"]) > len(r["rejected"]))
    ties = sum(1 for r in rows if len(r["chosen"]) == len(r["rejected"]))
    len_acc = (longer + 0.5 * ties) / n
    len_shortcut = max(len_acc, 1.0 - len_acc)
    if len_shortcut > MAX_LENGTH_ACC:
        direction = "longer" if len_acc >= 0.5 else "shorter"
        failures.append(f"'pick the {direction} answer' scores {len_shortcut:.1%} "
                        f"(> {MAX_LENGTH_ACC:.0%}): match the length of the negatives to the positives")

    # 3. leakage against the split training will actually use
    tr, ev = split_pairs(rows, hp.eval_split, hp.seed)
    seen = {r["rejected"].strip() for r in tr}
    leak = (sum(1 for r in ev if r["rejected"].strip() in seen) / len(ev)) if ev else 0.0
    if leak > MAX_LEAK:
        failures.append(f"{leak:.1%} of held-out negatives also appear in training "
                        f"(> {MAX_LEAK:.0%}): the evaluation would measure memory, not judgement")

    # 4. surface tell, averaged over several shuffles (one seed is a ~3-point coin flip)
    vals = [a for a in (_surface_probe_acc(rows, s, hp.eval_split) for s in (42, 0, 1, 7, 123))
            if not math.isnan(a)]
    surface = sum(vals) / len(vals) if vals else float("nan")
    if not math.isnan(surface) and surface > MAX_SURFACE_ACC:
        failures.append(f"a bag-of-words probe tells chosen from rejected at {surface:.1%} "
                        f"(> {MAX_SURFACE_ACC:.0%}) without seeing the prompt: the negatives "
                        f"share a surface style the model will learn instead of the task")

    return {
        "ok": not failures,
        "n": n,
        "metrics": {
            "unique_rejected_frac": round(uniq, 4),
            "top_rejected_share": round(top_n / n, 4),
            "length_shortcut_acc": round(len_shortcut, 4),
            "eval_negative_leak": round(leak, 4),
            "surface_probe_acc": None if math.isnan(surface) else round(surface, 4),
            "mean_len_chosen": round(sum(len(r["chosen"]) for r in rows) / n, 1),
            "mean_len_rejected": round(sum(len(r["rejected"]) for r in rows) / n, 1),
        },
        "failures": failures,
    }


def format_gate(rep: dict) -> str:
    m = rep["metrics"]
    lines = [f"dataset gate: {'PASS' if rep['ok'] else 'FAIL'} ({rep['n']} pairs)"]
    if m:
        sp = m["surface_probe_acc"]
        lines += [
            f"  unique negatives    {m['unique_rejected_frac']:.1%}",
            f"  largest single neg  {m['top_rejected_share']:.1%}",
            f"  length shortcut     {m['length_shortcut_acc']:.1%}",
            f"  held-out leak       {m['eval_negative_leak']:.1%}",
            f"  surface probe       {'n/a' if sp is None else f'{sp:.1%}'}  (50% = no tell)",
            f"  mean length         chosen {m['mean_len_chosen']:.0f} / rejected "
            f"{m['mean_len_rejected']:.0f} characters",
        ]
    lines += [f"  x {f}" for f in rep["failures"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tokenizer check
# ---------------------------------------------------------------------------
def check_tokenizer(tok) -> None:
    """Refuse a base model whose tokenizer does not produce the node's token ids. The
    node tokenises with its own copy, so a mismatch trains on ids it never sends."""
    if not TOKENIZER_VECTORS.exists():
        log.warning("tokenizer vectors not found at %s: skipping the tokenizer check",
                    TOKENIZER_VECTORS)
        return
    vectors = json.loads(TOKENIZER_VECTORS.read_text(encoding="utf-8"))["vectors"]
    for v in vectors:
        got = tok(v["text"], add_special_tokens=v["add_special_tokens"])["input_ids"]
        if list(got) != list(v["ids"]):
            raise TrainError(
                f"the base model's tokenizer disagrees with the node's on {v['text'][:60]!r} "
                f"(got {list(got)[:12]}, node {v['ids'][:12]}): the node can only score "
                f"adapters trained on {DEFAULT_BASE}'s tokenizer")
    log.info("tokenizer matches the node (%d vectors)", len(vectors))


# ---------------------------------------------------------------------------
# model, training, evaluation
# ---------------------------------------------------------------------------
class PairCollator:
    def __init__(self, tok, max_length):
        self.tok, self.max_length = tok, max_length

    def _enc(self, texts):
        return self.tok(texts, truncation=True, max_length=self.max_length,
                        padding=True, return_tensors="pt")

    def __call__(self, batch):
        import torch
        c = self._enc([render(p["prompt"], p["chosen"]) for p in batch])
        r = self._enc([render(p["prompt"], p["rejected"]) for p in batch])
        return {
            "chosen_input_ids": c["input_ids"], "chosen_attention_mask": c["attention_mask"],
            "rejected_input_ids": r["input_ids"], "rejected_attention_mask": r["attention_mask"],
            "margin": torch.tensor([p["margin"] for p in batch], dtype=torch.float32),
        }


def build_model(base_model: str, hp: Hyperparams):
    """ModernBERT with a one-logit sequence-classification head and LoRA on the
    attention and MLP projections. The head is trained in full (modules_to_save),
    which is what convert_adapter merges back."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    cfg = AutoConfig.from_pretrained(base_model)
    if (getattr(cfg, "model_type", "") or "").lower() != "modernbert":
        raise TrainError(f"{base_model} is a '{cfg.model_type}' model: the node runs ModernBERT "
                         f"only (use {DEFAULT_BASE})")
    tok = AutoTokenizer.from_pretrained(base_model)
    check_tokenizer(tok)
    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, torch_dtype=dtype, attn_implementation="eager")
    if tok.pad_token_id is None:
        raise TrainError(f"{base_model}'s tokenizer has no pad token")
    model.config.pad_token_id = tok.pad_token_id
    if hp.grad_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=hp.lora_rank,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        target_modules=["Wqkv", "Wo", "Wi"],
        bias="none",
        modules_to_save=["score", "classifier"],
    )
    model = get_peft_model(model, lora)
    if hp.grad_checkpointing:
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("%s | trainable %s / %s parameters", base_model, f"{trainable:,}", f"{total:,}")
    return model, tok


def _reward(model, ids, mask):
    return model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)


def bt_loss(rc, rr, margin):
    import torch.nn.functional as F
    return -F.logsigmoid(rc - rr - margin).mean()


def pick_device():
    import torch
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        log.info("device: cuda (%s)", torch.cuda.get_device_name(0))
        return torch.device("cuda"), dtype
    log.warning("no GPU: training on CPU in fp32. This is slow; a GPU runtime is recommended.")
    return torch.device("cpu"), torch.float32


def evaluate(model, tok, pairs, hp: Hyperparams, train_negatives: Optional[set] = None) -> dict:
    """Pairwise accuracy and loss on held-out pairs, against the length baseline."""
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(pairs, batch_size=hp.eval_batch, shuffle=False,
                        collate_fn=PairCollator(tok, hp.max_length))
    diffs, loss_sum = [], 0.0
    with torch.no_grad():
        for b in loader:
            rc = _reward(model, b["chosen_input_ids"].to(device), b["chosen_attention_mask"].to(device))
            rr = _reward(model, b["rejected_input_ids"].to(device), b["rejected_attention_mask"].to(device))
            loss_sum += bt_loss(rc.float(), rr.float(), b["margin"].to(device)).item() * rc.size(0)
            diffs += (rc - rr).float().cpu().tolist()
    n = len(diffs)
    acc = sum(1 for d in diffs if d > 0) / n if n else 0.0

    wins = sum(1 for p in pairs if len(p["chosen"]) > len(p["rejected"]))
    ties = sum(1 for p in pairs if len(p["chosen"]) == len(p["rejected"]))
    len_acc = (wins + 0.5 * ties) / len(pairs) if pairs else 0.5
    length_baseline = max(len_acc, 1.0 - len_acc)
    beats_length = acc >= length_baseline + hp.gate_min_margin_over_baseline
    passed = acc >= hp.gate_min_acc and beats_length

    report = {
        "n_eval": n,
        "pairwise_accuracy": round(acc, 4),
        "val_loss": round(loss_sum / n, 6) if n else None,
        "length_baseline": round(length_baseline, 4),
        "beats_length_baseline": beats_length,
        "verdict": "PASS" if passed else "FAIL",
    }
    if train_negatives is not None and pairs:
        report["negative_memorisation"] = round(
            sum(1 for p in pairs if p["rejected"].strip() in train_negatives) / len(pairs), 4)
    if acc < hp.gate_min_acc:
        report["fail_reason"] = f"accuracy {acc:.4f} is below --gate-min-acc {hp.gate_min_acc}"
    elif not beats_length:
        report["fail_reason"] = (f"accuracy {acc:.4f} does not beat the length-only baseline "
                                 f"{length_baseline:.4f} by {hp.gate_min_margin_over_baseline}: "
                                 f"the model may be ranking by length, not quality")
    return report


def train(base_model, train_pairs, val_pairs, adapter_dir: Path, hp: Hyperparams):
    """Train, keeping the best epoch's adapter on disk. Returns (model, tok)."""
    import torch
    from torch.utils.data import DataLoader

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed_all(hp.seed)

    model, tok = build_model(base_model, hp)
    device, dtype = pick_device()
    model.to(device)
    train_negatives = {p["rejected"].strip() for p in train_pairs}

    loader = DataLoader(train_pairs, batch_size=hp.batch_size, shuffle=True,
                        collate_fn=PairCollator(tok, hp.max_length))
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
    steps = max(1, len(loader) // hp.grad_accum) * hp.epochs
    warm = int(steps * 0.06)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, s / max(1, warm)) if s < warm
        else max(0.0, (steps - s) / max(1, steps - warm)))
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def save():
        model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)

    lower_better = hp.es_metric == "val_loss"
    best, best_epoch, no_improve, epoch = (math.inf if lower_better else -math.inf), -1, 0, -1
    gstep = 0
    for epoch in range(hp.epochs):
        model.train()
        running = 0.0
        optim.zero_grad(set_to_none=True)
        for i, b in enumerate(loader):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                rc = _reward(model, b["chosen_input_ids"].to(device), b["chosen_attention_mask"].to(device))
                rr = _reward(model, b["rejected_input_ids"].to(device), b["rejected_attention_mask"].to(device))
                loss = bt_loss(rc, rr, b["margin"].to(device)) / hp.grad_accum
            (scaler.scale(loss).backward() if use_scaler else loss.backward())
            running += loss.item() * hp.grad_accum
            if (i + 1) % hp.grad_accum == 0:
                if use_scaler:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if use_scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 10 == 0:
                    log.info("epoch %d step %d/%d loss %.4f", epoch, gstep, steps, running / (i + 1))
        avg_loss = running / max(1, len(loader))
        log.info("epoch %d done: train loss %.4f", epoch, avg_loss)

        if not val_pairs:
            continue
        val = evaluate(model, tok, val_pairs, hp, train_negatives)
        metric = val["val_loss"] if lower_better else val["pairwise_accuracy"]
        log.info("epoch %d validation: accuracy %.4f, loss %.4f",
                 epoch, val["pairwise_accuracy"], val["val_loss"])
        improved = (metric < best - hp.es_min_delta) if lower_better else (metric > best + hp.es_min_delta)
        if improved or best_epoch < 0:
            best, best_epoch, no_improve = metric, epoch, 0
            save()
            log.info("epoch %d: new best %s %.4f, adapter saved", epoch, hp.es_metric, best)
        else:
            no_improve += 1
        if hp.early_stopping and avg_loss < hp.loss_floor:
            log.warning("stopping: train loss %.6f is below the floor (memorising); "
                        "keeping epoch %d", avg_loss, best_epoch)
            break
        if hp.early_stopping and no_improve > hp.es_patience:
            log.info("stopping: %d epochs without improvement; keeping epoch %d",
                     no_improve, best_epoch)
            break

    if not val_pairs or best_epoch < 0:
        save()
    elif best_epoch != epoch:
        # the in-memory model is a later, worse epoch: reload the saved best
        model.load_adapter(str(adapter_dir), adapter_name="default", is_trainable=False)
        model.set_adapter("default")
        model.to(device)
        log.info("restored the best epoch (%d)", best_epoch)
    return model, tok, train_negatives


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    hp = Hyperparams()
    p = argparse.ArgumentParser(
        description="Train a ModernBERT LoRA reward-model adapter for a judgment or hybrid "
                    "intent. Convert the result with wide_runtime.convert_adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--intent", required=True, help="intent the adapter scores, e.g. CODE_REVIEW")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="training pairs, JSONL (prompt/chosen/rejected per line)")
    src.add_argument("--hf-dataset", help="Hugging Face dataset id with prompt/chosen/rejected columns")
    p.add_argument("--val-data", help="validation pairs JSONL (default: hold out --eval-split of --data)")
    p.add_argument("--test-data", help="final-gate pairs JSONL (default: the validation pairs)")
    p.add_argument("--train-split", default="train", help="--hf-dataset split to train on")
    p.add_argument("--val-split", default="validation", help="--hf-dataset validation split")
    p.add_argument("--test-split", default="test", help="--hf-dataset final-gate split")
    p.add_argument("--hf-config", default=None, help="--hf-dataset config/subset name")
    p.add_argument("--prompt-col", default=None)
    p.add_argument("--chosen-col", default=None)
    p.add_argument("--rejected-col", default=None)
    p.add_argument("--hf-token", default=None, help="for private datasets (else env HF_TOKEN)")
    p.add_argument("--base-model", default=DEFAULT_BASE,
                   help="must be ModernBERT with the node's tokenizer")
    p.add_argument("--out", default=None, help="output directory (default runs/<INTENT>)")
    p.add_argument("--config", default=None,
                   help="YAML/JSON hyperparameters; flags below override it")
    p.add_argument("--check-data", action="store_true",
                   help="run the dataset gate only, then exit (no model is loaded)")
    p.add_argument("--skip-dataset-gate", action="store_true",
                   help="train even if the dataset gate fails (the result will usually rank at chance)")

    g = p.add_argument_group("hyperparameters")
    for name, typ, help_ in [
        ("max_length", int, "tokens per answer; the node scores 256"),
        ("batch_size", int, None), ("grad_accum", int, None),
        ("lora_rank", int, None), ("lora_alpha", int, None), ("lora_dropout", float, None),
        ("epochs", int, None), ("lr", float, None), ("weight_decay", float, None),
        ("eval_split", float, "held-out share when no --val-data"),
        ("seed", int, None), ("eval_batch", int, None),
        ("gate_min_acc", float, "evaluation gate: minimum pairwise accuracy"),
        ("gate_min_margin_over_baseline", float, "evaluation gate: margin over the length baseline"),
        ("es_patience", int, None), ("es_min_delta", float, None), ("loss_floor", float, None),
    ]:
        g.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None,
                       help=f"{help_ + ' ' if help_ else ''}(default {getattr(hp, name)})")
    g.add_argument("--es-metric", dest="es_metric", default=None,
                   choices=["val_loss", "pairwise_accuracy"], help=f"(default {hp.es_metric})")
    g.add_argument("--no-early-stopping", dest="early_stopping", action="store_false", default=None)
    g.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false",
                   default=None)
    return p


def resolve_hyperparams(args) -> Hyperparams:
    hp = Hyperparams.from_file(args.config) if args.config else Hyperparams()
    for f in fields(Hyperparams):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(hp, f.name, v)
    if hp.max_length > NODE_MAX_LEN:
        log.warning("--max-length %d is above the %d tokens the node scores: the model will "
                    "learn from text it never sees at scoring time", hp.max_length, NODE_MAX_LEN)
    return hp


def load_data(args, hp):
    """(train, validation, test) pair lists; validation/test may be None. With no
    validation set the caller holds one out of train AFTER the dataset gate, so the
    gate's leakage check measures the split training really uses."""
    if args.data:
        train_pairs = read_pairs_jsonl(args.data)
        val = read_pairs_jsonl(args.val_data) if args.val_data else None
        test = read_pairs_jsonl(args.test_data) if args.test_data else None
        return train_pairs, val, test
    token = args.hf_token or os.environ.get("HF_TOKEN")
    splits = load_hf_splits(args.hf_dataset, token=token, config=args.hf_config,
                            prompt_col=args.prompt_col, chosen_col=args.chosen_col,
                            rejected_col=args.rejected_col)
    if args.train_split not in splits:
        raise TrainError(f"split '{args.train_split}' not found; available: {sorted(splits)}")
    train_pairs = splits[args.train_split]
    val = splits.get(args.val_split) if args.val_split else None
    test = splits.get(args.test_split) if args.test_split else None
    return train_pairs, val, test


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        hp = resolve_hyperparams(args)
        train_pairs, val_pairs, test_pairs = load_data(args, hp)
        if not train_pairs:
            raise TrainError("no training pairs")

        rep = gate_dataset(train_pairs, hp)
        print(format_gate(rep))
        if args.check_data:
            return 0 if rep["ok"] else 1
        if not rep["ok"]:
            if not args.skip_dataset_gate:
                raise TrainError("the dataset gate failed (see above); fix the data, or pass "
                                 "--skip-dataset-gate to train anyway")
            log.warning("training despite a failed dataset gate (--skip-dataset-gate)")
        if val_pairs is None:
            train_pairs, val_pairs = split_pairs(train_pairs, hp.eval_split, hp.seed)
            log.info("held out %d of %d pairs for validation",
                     len(val_pairs), len(val_pairs) + len(train_pairs))

        out = Path(args.out or Path("runs") / args.intent)
        adapter_dir = out / "adapter"
        out.mkdir(parents=True, exist_ok=True)
        log.info("hyperparameters: %s", hp)

        import torch
        try:
            model, tok, train_negatives = train(args.base_model, train_pairs, val_pairs,
                                                adapter_dir, hp)
        except torch.cuda.OutOfMemoryError as e:
            raise TrainError("GPU out of memory: lower --max-length or raise --grad-accum "
                             f"with --batch-size 1 ({e})") from e

        gate_set, gate_name = (test_pairs, "test") if test_pairs else (val_pairs, "validation")
        report = evaluate(model, tok, gate_set, hp, train_negatives)
        result = {
            "intent": args.intent,
            "base_model": args.base_model,
            "adapter_dir": str(adapter_dir),
            "train_pairs": len(train_pairs),
            "gate_split": gate_name,
            "dataset_gate": rep,
            **report,
            "hyperparameters": asdict(hp),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        del model
        gc.collect()
    except TrainError as e:
        log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130

    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("dataset_gate", "hyperparameters")}, indent=2))
    if result["verdict"] != "PASS":
        log.error("evaluation gate FAILED: %s", result.get("fail_reason", ""))
        return 1
    imwq = out / f"{args.intent}.imwq"
    print("\nNext:\n"
          f"  python -m wide_runtime.convert_adapter "
          f"--base {args.base_model} --adapter {adapter_dir} --out {imwq}\n"
          f"  go run ./cmd/adapterbuild --intent {args.intent} --imwq {imwq} --out {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
