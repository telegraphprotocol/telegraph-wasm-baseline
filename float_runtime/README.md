# Evaluator adapters — technical reference

## The shared Foundation Model and safetensors delta packages

A Telegraph **Evaluator adapter** (judgment and hybrid intents) is a **safetensors delta package**
(`export_adapter.py`, metadata schema in `lora_format.py`): a standard safetensors file holding the
LoRA A/B matrices of every target linear (`attn.Wqkv`, `attn.Wo`, `mlp.Wi`, `mlp.Wo` in each of the
22 layers; PEFT's `target_modules=["Wqkv","Wo","Wi"]` matches both `Wo`s) and the one-logit reward
head (`head.weight [768,1]`, `head.bias [1]`), stored pre-transposed (`A [in, r]`, `B [r, out]`,
`r <= R_MAX = 16`, dtype F32), with the package fields in the safetensors `__metadata__` map:
`format_version "1"`, `abi "lora-v1"`, `class`, `intent`, `foundation_hash`, `foundation_version`,
`tokenizer_hash`, `rank`, `alpha`, `target_modules` (JSON array string), `max_tokens`, optional
`training` (JSON string), optional `x-*` extension keys. A real rank-8 package is **6,776,908 B**;
its identity is its keccak256 (the registered `wasm_hash`). Nodes recognise it by parsing the
header, never by a magic.

The base weights never travel with an adapter. They are the **Foundation Model**
(`export_foundation.py`): ModernBERT-base as ONE ONNX graph whose LoRA branches and reward
head are graph *inputs* (`x·W + ((x·A)·B)·s`, A/B zero-padded to `R_MAX`, `s = alpha / r`),
599,117,769 B, identified by its sha256. The **engine** (`engine/`, `build_engine.sh`; a WASI
reactor, ~2.6 MB, `telegraph_abi() = 2`: `tg_alloc`/`tg_free`, `load_foundation`,
`foundation_spec`, `load_adapter`, `score`, `unload_adapter`, `message_ptr`/`message_len`)
loads the graph once per validator and scores every adapter on it by feeding the package's
tensors as inputs; the graph is never mutated and no custom kernel exists. The node's side is
Telegraph `pkg/wasm/runtime/foundation.go` (host), `lora_adapter.go` (parser, re-checked in
the guest) and `lora_runtime.go`; `adapterbuild lora` checks a package with that parser.

Measured (spike, real VULNERABILITY_TRIAGE bb-v3, 64-token inputs, oxionnx under wasm): the
split graph is within **7.0e-5** of the same adapter merged into the base on 20 inputs (same
rank order), the same adapter twice in one instance gives identical bits, and two adapters
interleaved in one instance give the same bits as each in a fresh instance. Fixtures for the
Go tests (a few-KB stub foundation with the same layout) are made by
`testdata/make_tiny_foundation.py`.

**Earlier formats.** The full-model float `.wasm` (the merged model baked into one WASI
command with `oxionnx`, ~574 MB, scoring ABI version 2 `runtime: "onnx_f32"`) is no longer
accepted: nodes reject a registration of one at Stage 1 with `legacy_full_model: full-model
float adapters are no longer accepted; repackage the adapter as a safetensors delta for the
shared Foundation Model and re-register it`, and an adapter already Active in it is
superseded. Version 1, the Q32.32 integer format (`.imwq`, `imwq_*` exports), is retired the
same way. Neither can be converted: re-export the delta from the adapter's PEFT checkpoint and
`reward_head.pt` with `export_adapter.py`.

## Files

| Path | What |
|---|---|
| `train_adapter.py` | LoRA Bradley-Terry reward model on ModernBERT-base, behind a dataset gate and an evaluation gate; also writes `reward_head.pt` and the base/tokenizer sha256s |
| `export_adapter.py` | Writes the safetensors delta package (never merges) for a published foundation (`foundation.json`) |
| `lora_format.py` | The safetensors layout: constants, writer, reader |
| `export_foundation.py` | One-time: builds the Foundation Model ONNX with LoRA inputs and `foundation.json`; `--check-adapter` compares the split graph with the adapter merged into a PyTorch copy of the base (the reference only) |
| `engine/` + `build_engine.sh` | One-time: the foundation engine crate (WASI reactor) and its pinned build |
| `testdata/make_tiny_foundation.py` | The stub foundation and delta fixtures the Go tests use (`tiny_foundation.onnx/.json`, `tiny_judgment.safetensors`, `tiny_hybrid.safetensors`, `foundation_engine.wasm`) |
| `testdata/tokenizer_vectors.json` | The node's tokenizer conformance vectors; training and export refuse a tokenizer that disagrees |
| `../cmd/adapterbuild` | `adapterbuild lora`: checks a package with the node's own code, prints the keccak256 |

## The scoring contract

### Where the tokens come from

- Text is `"### Task:\n" + prompt + "\n\n### Answer:\n" + answer` (the node's
  `RenderForJudgment`; ModernBERT has no chat template). For a hybrid intent the prompt is the
  question key and the answer the miner's whole response body. `train_adapter.py` trains on
  exactly this rendering.
- Tokenizer: ModernBERT's `tokenizer.json` (sha256 `67bc76f8…dc456`), `[CLS]`/`[SEP]` added
  (50281/50282), capped at 512 tokens, then truncated to the adapter's `max_tokens` (default
  512) keeping the closing `[SEP]` in the last slot. A Hugging Face tokenizer called with
  `truncation=True, max_length=N` produces the same ids.
- Every text is sent at its own length, 1 to `max_tokens` tokens. There is no padding and no
  fixed length.

### One score

The node's host copies the token ids into the engine instance and calls
`score(handle, ids_ptr, n)`; the engine feeds `input_ids`, an all-ones `attention_mask`, the
adapter's tensors and `lora.scale` to the foundation graph and returns the IEEE-754 binary32
bits of the reward logit. The score fails — never becomes a number — on a trap, a non-zero
exit of the instance, a refused adapter (the engine re-checks shapes and finite values at
`load_adapter`), a call past `ADAPTER_SCORE_TIMEOUT` (default 5 min), or a NaN/±Inf value.
Scores are serialised per adapter; a stuck adapter cannot hold another intent's scoring.

### The foundation graph (`export_foundation.py`)

| | name | type | shape |
|---|---|---|---|
| input | `input_ids` | int64 | `[batch, seq]` |
| input | `attention_mask` | int64 | `[batch, seq]` |
| input | `lora.scale` | float32 | `[1]` |
| input | `head.weight`, `head.bias` | float32 | `[768, 1]`, `[1]` |
| input | `lora.<l>.<target>.A`, `.B` | float32 | `[in, 16]`, `[16, out]` per target per layer |
| output | `logits` | float32 | `[batch, 1]` — the first value is the score |

`seq` is a dynamic axis. Opset 17, TorchScript exporter (`dynamo=False`), all base weights
inside the one `.onnx` file as initializers. `foundation.json` records the sha256, version,
tokenizer hash, `r_max`, layer count, hidden size and the per-target shapes; `export_adapter`
and `adapterbuild lora` check a package against it.

### The package metadata

| Key | Rule |
|---|---|
| `format_version` | `"1"` |
| `abi` | `"lora-v1"` |
| `class` | `"judgment"` or `"hybrid"`, and it must be the intent's class. `adapterbuild lora` takes the class from the compiled intent catalog and refuses deterministic intents |
| `intent` | the intent (family prefix) the adapter serves; must equal `--intent` |
| `foundation_hash`, `foundation_version` | the foundation the package was exported for (`foundation_mismatch` elsewhere) |
| `tokenizer_hash` | the node's pinned tokenizer (`tokenizer_mismatch` otherwise) |
| `rank`, `alpha`, `target_modules` | plain PEFT LoRA: `rank <= 16`, one `alpha`, modules among `Wqkv`, `Wo`, `Wi` |
| `max_tokens` | 1–512. Set it (`export_adapter --max-tokens`) when you trained with a shorter `--max-length` |
| `training` | optional training record (family + children + metrics) as a JSON string, shape-checked only, never used for a decision (`--training-record`, standard §9) |
| `x-*` | extension keys, ignored by the node |

A package with a missing or unknown key, a wrong dtype, a wrong shape, a non-finite value, or
tensors that do not tile the data exactly is refused at Stage 1 (reasons in the Telegraph
`docs/troubleshooting.md`).

## Determinism

Every validator must get the same 32 bits for the same adapter and text, or the epoch forks.

- **WebAssembly fixes the arithmetic.** f32 operations and their order are defined by the
  engine module, with no CPU-specific dispatch. WebAssembly 2.0 has no fused multiply-add, and
  the node's runtime does not enable the relaxed-SIMD proposal (implementation-defined
  results). Fixed-width SIMD (`+simd128`, which `build_engine.sh` enables) is deterministic
  and allowed.
- **The host fixes everything else.** The engine sees no host files, an empty environment, a
  frozen wall clock (2022-01-01T00:00:00Z) and monotonic clock, `nanosleep` that returns at
  once, a fixed random stream (SHA-256 over a fixed seed; Rust's `HashMap` seeds come from it)
  and no sockets. Imports outside the node's allow-list are refused when the engine is
  checked at startup.
- **Zero padding adds exact zeros.** An adapter of rank `r < 16` is padded with zeros to
  `R_MAX`; the extra products are exactly zero, so the score is the rank-`r` score.
- **The node never computes with the float.** The score is kept as its bits: rankings and
  competition pairs compare a total-order integer key, and the displayed basis points come
  from an integer sigmoid (standard §3).
- **Measured.** Same package and tokens give the same bits on repeat, after a fresh instance,
  and interleaved with other adapters in one instance (spike and node tests). Cross-CPU bit
  identity is not claimed here.

## Limits and costs

| | |
|---|---|
| Package size | a node refuses a package above `ADAPTER_LORA_MAX_BYTES` (default **64 MiB**); a rank-8 ModernBERT-base delta is 6.8 MB |
| Foundation | `foundation_lora_fp32.onnx` ~599 MB, loaded once per validator (`FOUNDATION_MODEL_PATH`), engine ~2.6 MB |
| Memory | linear memory capped at 3 GiB per engine instance; the real foundation with adapters loaded peaks at ~1.7 GB |
| Time | a 64-token score takes about a second; a 512-token score several seconds, depending on the CPU. Each score is one inference |
| Export | `export_adapter` runs on a CPU in seconds; `export_foundation` needs the base checkpoint and a few GB of RAM, once |

## Checking an adapter by hand

```bash
# the node's own checks against foundation.json, plus a load and two scores on the foundation
go run ./cmd/adapterbuild lora --intent CODE_REVIEW --package runs/CODE_REVIEW/CODE_REVIEW.safetensors \
  --foundation foundation/foundation.json --foundation-onnx foundation/foundation_lora_fp32.onnx \
  --engine foundation_engine.wasm --input runs/CODE_REVIEW/sample_input.json

# from a Telegraph checkout: the package header as the node reads it
go run ./cmd/wasmprobe runs/CODE_REVIEW/CODE_REVIEW.safetensors
```
