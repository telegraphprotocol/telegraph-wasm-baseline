# Telegraph Evaluator Adapter Kit

> **Changed (2026-10-30): full-model float `.wasm` adapters are no longer accepted;
> repackage as a delta.** An Evaluator adapter is now a **safetensors delta package**
> (a few MB of LoRA deltas and a reward head) that validators run on one shared
> **Foundation Model**. Nodes reject a full-model `.wasm` (the ~600 MB model baked into a
> WASI command) with `legacy_full_model`, and an adapter already Active in that format is
> taken out of service when a validator upgrades. Re-export the delta from your original
> training run (the `adapter/` folder `train_adapter` wrote; the reward head is read from
> it or from `reward_head.pt`) with `export_adapter.py`, no retraining needed, and register
> it again. No checkpoint: retrain. The full-model tools (`export_onnx.py`, `bake_wasm.sh`,
> the `vm_float_wasm` crate) are removed; they remain in git history only.

Everything you need to build an **Evaluator adapter** for a Telegraph judgment or hybrid
intent: the model that reads a question and a miner's answer and scores how good the
answer is. Register a better one than the current adapter and it replaces it.

Full guide: [Train an Evaluator Adapter](https://docs.telegraphprotocol.com/docs/scoring/train-a-scoring-adapter).
Upgrading from a full-model adapter:
[Upgrading to the shared Foundation Model](https://docs.telegraphprotocol.com/docs/scoring/upgrade-shared-foundation).

## The workflow

| Step | Tool | Output |
|---|---|---|
| 1. Train | `float_runtime/train_adapter.py` | a PEFT LoRA adapter and `reward_head.pt` |
| 2. Export the delta | `float_runtime/export_adapter.py` (never merges the base weights) | `<INTENT>.safetensors` |
| 3. Check and package | `cmd/adapterbuild lora` (the node's own parser) | the keccak256 to register, a manifest |
| 4. Host the file | any `https://` or `ipfs://` URL | `wasmUrl` |
| 5. Run the executor | `cmd/adapterexec` (reference) | `wasmRunnableUrl` |
| 6. Register | `registerWasm(keccak256, wasmUrl, prefix, feeAddress, wasmRunnableUrl)` | a registration id |
| 7. The node checks | remote check, download, Stage 1, confirmation, hidden competition | Active, or a rejection reason |

From the root of this repository (on Windows, run `bash` steps from Git Bash):

```bash
pip install -r requirements.txt

# 1. train (pairs.jsonl: one {"prompt", "chosen", "rejected"} per line; GPU recommended)
python -m float_runtime.train_adapter --intent CODE_REVIEW --data pairs.jsonl --out runs/CODE_REVIEW

# 2. export the delta for the validators' Foundation Model (CPU, seconds)
python -m float_runtime.export_adapter --run runs/CODE_REVIEW --intent CODE_REVIEW \
  --class judgment --foundation foundation/foundation.json

# 3. check it the way a node will, and get the hash to register
go run ./cmd/adapterbuild lora --intent CODE_REVIEW \
  --package runs/CODE_REVIEW/CODE_REVIEW.safetensors --foundation foundation/foundation.json \
  [--foundation-onnx foundation/foundation_lora_fp32.onnx --engine float_runtime/engine/foundation_engine.wasm]
```

Check your data before spending GPU time:
`python -m float_runtime.train_adapter --intent CODE_REVIEW --data pairs.jsonl --check-data`.
Each step prints the next command.

### 1. Train

`train_adapter.py` fine-tunes `answerdotai/ModernBERT-base` with plain PEFT LoRA (rank 8,
alpha 32 by default) and a one-logit reward head on preference pairs, behind a dataset
gate and an evaluation gate (at least 65% held-out pairwise accuracy and 10 points over
"pick the longer answer"). It trains at 512 tokens by default, the most the node scores.
It writes the PEFT adapter, `reward_head.pt`, and `result.json` with the base model's and
the tokenizer's sha256.

The LoRA must be plain PEFT LoRA: `r <= 16`, `target_modules` among `Wqkv`, `Wo`, `Wi`,
one `lora_alpha` (no `rank_pattern`, `alpha_pattern`, rsLoRA or DoRA).

### 2. Export the delta

`export_adapter.py --run <dir> --intent <PREFIX> --class judgment|hybrid --foundation
foundation.json` writes `<INTENT>.safetensors`: the LoRA A/B matrices and the reward head,
never the base weights. It refuses a run trained on another base model or tokenizer.
If you trained shorter than 512 tokens, pass the same number as `--max-tokens`.
`--adapter` takes a PEFT adapter directory (or a Hugging Face spec) instead of `--run`, for
an adapter trained elsewhere.

### 3. Check and package

`adapterbuild lora` runs the node's Stage 1 parser on the package against
`foundation.json`, derives the class from the intent catalog (deterministic intents are
refused), and, with `--foundation-onnx` and `--engine`, loads the package on the foundation
through the node's runtime and scores an input twice (identical bits required). It prints
one JSON line on stdout (`keccak256`, `sha256`, `bytes`, `package_path`, `manifest_path`,
`score_hex` when scored, the manifest) and writes `<INTENT>.safetensors.manifest.json`.
Nothing is rewritten: the hash you register is the hash of the file `export_adapter` wrote.

### 4. Host the file

Upload `<INTENT>.safetensors` unchanged to an `https://` or `ipfs://` URL that serves the
exact bytes (no re-encoding or transcoding). Validators download it within 120 s, at most
`ADAPTER_LORA_MAX_BYTES` (64 MiB by default), and check its keccak256 against the registered
hash: a different file is rejected on the first attempt (`wasm hash mismatch`).

### 5. Run the runnable-URL executor

Every registration needs a `wasmRunnableUrl`: an endpoint that runs **your package on the
validators' Foundation Model**. A validator scores a sample there before it downloads
anything, and later re-scores the same texts itself, bit for bit. See
[The runnable-URL executor](#the-runnable-url-executor) below.

### 6. Register

```
registerWasm(bytes32 wasmHash, string wasmUrl, string intent, address feeAddress, string wasmRunnableUrl)
```

`wasmHash` is the package's keccak256, `intent` is the family prefix the adapter serves,
and `feeAddress` must be non-zero. Approve the adapter bond in MACHINA first: it is fixed
in USD (`entityBonds()`), pulled by `registerWasm`, and released in full on
deregistration. Re-registering an exact hash you already registered reverts with
`duplicate wasm hash`; a new package is a new hash.

### 7. What the node checks

In this order (each failure is a terminal rejection with the reason shown, unless noted):

1. **Runnable URL present**, else `wasm_runnable_url is required`.
2. **Remote check** at `wasmRunnableUrl`, before any download. It waits (Pending, uncharged)
   for the first confirmed epoch after the registration (`awaiting_seed_epoch`) and for
   public evaluation material for the intent (at least 10 usable pairs). Then it sends 15
   public pairs (up to 30 texts) in one batch request with `foundation_hash`, `adapter_hash`
   and `tokenizer_hash`. A different echo is rejected (`identity echo mismatch`); fewer
   than 70% of pairs ordered correctly is rejected; an unreachable or too slow endpoint
   (`WASM_REMOTE_EVAL_BUDGET`, 30 minutes) is retried at the next epoch rollover or within
   the hour, and rejected after 3 failed attempts.
3. **Download and hash**: the first MiB decides the kind. A full-model `.wasm` is refused
   at once (`legacy_full_model`); a package is read up to `ADAPTER_LORA_MAX_BYTES`; the
   keccak256 must equal the registered hash.
4. **Stage 1**: the package rules below against this validator's foundation and tokenizer.
   A validator with no Foundation Model refuses every package (`no_foundation`).
5. **Confirmation**: the validator scores the same public texts locally; every score must
   equal the executor's bits (`remote score mismatch`), and an executor that did not echo
   the identity is rejected (`its runnable URL did not echo the score identity`).
6. **Competition** on the hidden evaluation split: at least 70% (60% on each child of a
   family) and, with an incumbent, at least 2 points better than it on the same sample.
   With no incumbent, the first adapter that clears the floor becomes the first Canonical.

**Timing.** Minutes, not seconds. On a laptop CPU one 512-token text takes about 13-14 s on
the foundation, so the remote check (up to 30 texts) is up to about 7 minutes on your executor,
the confirmation the same again on the validator, and the hidden round (30 pairs, 60 texts)
about 14 minutes. Waiting for the seed epoch comes on top. A local end-to-end run took about
21 minutes from registration to Active, all of it scoring.

## What the package must contain

A standard safetensors file: `u64 LE header length | JSON header | raw tensor bytes`, the
header mapping each tensor to `{dtype, shape, data_offsets}`, and the package fields in its
`__metadata__` string map (every value a string). The node's parser is
`pkg/wasm/runtime/lora_adapter.go`; the kit writes it with `float_runtime/lora_format.py`.

| Metadata key | Rule | Rejection if wrong |
|---|---|---|
| `format_version` | `"1"` | `bad_version` |
| `abi` | `"lora-v1"` | `bad_abi` |
| `class` | `"judgment"` or `"hybrid"`, and the intent's class | `class_mismatch` |
| `intent` | the prefix the adapter serves, not empty | `bad_header` |
| `foundation_hash` | sha256 of the validators' foundation graph | `foundation_mismatch` |
| `foundation_version` | the foundation's version label | (informational) |
| `tokenizer_hash` | sha256 of ModernBERT's `tokenizer.json` (below) | `tokenizer_mismatch` |
| `rank` | 1 to `R_MAX` (16) | `rank_too_large` |
| `alpha` | a positive finite number | `bad_header` |
| `target_modules` | a JSON array string, each of `Wqkv`, `Wo`, `Wi` at most once | `bad_target_module` |
| `max_tokens` | 1 to 512 | `bad_max_tokens` |
| `training` | optional training record as a JSON string, shape-checked only | `training_record` |
| `x-*` | optional extension keys, ignored | any other key: `bad_header` |

A missing required key is `missing_metadata`; a file that is not safetensors is
`bad_container`.

Tensors, all dtype `F32` (`bad_dtype` otherwise), for the 22-layer ModernBERT-base
foundation (hidden size 768), rank `r`:

| Tensor | Shape |
|---|---|
| `lora.<l>.attn.Wqkv.A` / `.B` | `[768, r]` / `[r, 2304]` |
| `lora.<l>.attn.Wo.A` / `.B` | `[768, r]` / `[r, 768]` |
| `lora.<l>.mlp.Wi.A` / `.B` | `[768, r]` / `[r, 2304]` |
| `lora.<l>.mlp.Wo.A` / `.B` | `[1152, r]` / `[r, 768]` |
| `head.weight` | `[768, 1]` |
| `head.bias` | `[1]` |

`l` runs 0 to 21, so a package has exactly 178 tensors. B is stored pre-transposed. Every
tensor of every target is required whatever `target_modules` says (PEFT's `Wo` matches
both `attn.Wo` and `mlp.Wo`). Missing, repeated or misshapen tensors are `bad_shape`; the
tensors must tile the data contiguously with nothing left over (`bad_offsets`,
`trailing_bytes`, `truncated`); every value must be finite (`non_finite`).

**Size**: at most `ADAPTER_LORA_MAX_BYTES` (default 64 MiB, `too_large`). A rank-8
package is 6.8 MB.

## Current foundation and tokenizer

| | Value | Status |
|---|---|---|
| Tokenizer `tokenizer_hash` | `67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456` | pinned in the node (`answerdotai/ModernBERT-base` `tokenizer.json`) |
| Engine `FOUNDATION_ENGINE_WASM_HASH` | `77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de` | what `bash float_runtime/build_engine.sh` builds from this branch (2,686,528 B); also committed as `float_runtime/engine/foundation_engine.wasm` |
| Foundation `foundation_hash` | reference export: `0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6` (`modernbert-base-lora16-v1`, 599,117,769 B) | the hash validators run is published with the foundation (not yet) |

**How to obtain the foundation.** The engine is in this repository
(`float_runtime/engine/foundation_engine.wasm`) and the tokenizer is ModernBERT-base's public
`tokenizer.json`. The Foundation Model (`foundation_lora_fp32.onnx`) and its `foundation.json`
are **not published yet**: the Telegraph team will publish them, with the hash every
validator must run, at a public location announced with the deploy. Adapter authors need
only `foundation.json` (1-2 KB: the hash, version, tokenizer hash and LoRA layout);
executor operators need the `.onnx` and the engine too.
`python -m float_runtime.export_foundation --out foundation/` builds one locally from the
public base model (needs a few GB of RAM), for experiments only: a local export is not
guaranteed to have the published hash, and a package built against a different hash is
rejected (`foundation_mismatch`).

## The runnable-URL executor

`cmd/adapterexec` is the reference: it loads the foundation and your package with the node's
own runtime (the copy under `pkg/`) and serves the batch request validators send.

```bash
export FOUNDATION_MODEL_PATH=foundation/foundation_lora_fp32.onnx
export FOUNDATION_MODEL_HASH=<sha256 of it>
export FOUNDATION_ENGINE_WASM_PATH=float_runtime/engine/foundation_engine.wasm
export FOUNDATION_ENGINE_WASM_HASH=77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de
go run ./cmd/adapterexec --package runs/CODE_REVIEW/CODE_REVIEW.safetensors --listen 127.0.0.1:8080
```

It refuses to start when a file's sha256 differs from its pinned hash. Put it behind a TLS
reverse proxy and register the public `https://` URL: validators refuse loopback, private,
link-local and 100.64.0.0/10 addresses and do not follow redirects. It needs about 1.8 GB of
RAM and scores one text at a time.

**Request** (one `POST`, every question in it; at most 100 items, accept bodies up to 1 MiB):

```json
{
  "batch": [{"tokens": [50281, 4118, 17526, 27, 187, 18658, 436, 13814, 1818, 50282]}],
  "foundation_hash": "<sha256 of the validator's foundation graph>",
  "adapter_hash": "<the registered keccak256, lowercase hex, no 0x>",
  "tokenizer_hash": "<sha256 of the validator's tokenizer.json>",
  "input_root": "<sha256 binding the batch's token inputs>"
}
```

The validator renders and tokenises the text itself, so the executor needs no tokenizer.
Score each token list as given (truncated to the package's `max_tokens`).

**Reply** `200`, one result per item, in the same order, and the identity at the top level:

```json
{
  "results": [{"score": 3204838382}],
  "foundation_hash": "<the foundation you scored with>",
  "adapter_hash": "<the keccak256 of the package you scored with, lowercase hex, no 0x>",
  "tokenizer_hash": "<the tokenizer hash of that package>",
  "input_root": "<optional: the root of the tokens you received>"
}
```

- `score` is the f32 bit pattern of the reward logit as an unsigned 32-bit integer
  (`3204838382` is `0xbf05f3ee`), never a decimal. NaN and infinities are refused.
- `{"error": "..."}` for an item your model refuses; a refused public pair rejects the
  registration.
- Echo what you actually ran, not what you were sent. All three fields must match the
  request (hex, case-insensitive): a different value rejects the registration at once
  (`identity echo mismatch`), and a missing echo rejects it after the download (`its runnable
  URL did not echo the score identity`). `input_root` is optional, but a different one is the
  same mismatch.
- `4xx`, or a `200` without `results`, means "no batch support" and rejects at once.
  `408`, `429` and `5xx` are retried later.
- The reply is at most 64 KiB, and the whole check must finish within 30 minutes.

The scores must be the exact bits the validators' engine computes, so run the same
foundation, engine and package (as `adapterexec` does). A different runtime (for example
native ONNX Runtime) gives close but not identical bits, and the confirmation rejects it.

## Needs

- Python 3.10+ (training and export; a GPU for training),
- Go 1.25+ (`adapterbuild`, `adapterexec`),
- Rust via [rustup](https://rustup.rs) with the `wasm32-wasip1` target, only to rebuild the
  engine (`build_engine.sh`; the pinned 1.98.1 toolchain is installed automatically).

## What's in here

| Path | What |
|---|---|
| `float_runtime/train_adapter.py` | Trains a ModernBERT LoRA reward model from preference pairs, behind a dataset gate and an evaluation gate; writes the PEFT adapter and `reward_head.pt` |
| `float_runtime/export_adapter.py` | Writes the safetensors delta package for a foundation (never merges the base weights) |
| `float_runtime/lora_format.py` | The package layout: metadata schema, tensor names, writer, reader |
| `float_runtime/export_foundation.py` | One-time: builds the Foundation Model ONNX with LoRA inputs and `foundation.json` |
| `float_runtime/engine/`, `float_runtime/build_engine.sh` | The foundation engine (a WASI reactor around `oxionnx`) and its pinned build |
| `cmd/adapterbuild` | `adapterbuild lora`: checks a package with the node's own code and prints the keccak256 to register |
| `cmd/adapterexec` | The reference runnable-URL executor |
| `pkg/` | The Go code both commands use, including copies of the Telegraph node's parser, runtime and remote-check client, so a package is checked and served exactly as a node runs it |
| `float_runtime/README.md` | Technical reference: the package format, the engine ABI, the scoring contract, determinism, limits |
| `float_runtime/testdata/` | A few-KB stub foundation and tiny delta packages (`make_tiny_foundation.py`) for the tests; the tokenizer conformance vectors |
| `scripts/sync-node-packages.sh` | Re-copies the node packages under `pkg/` from a Telegraph checkout |
| `legacy/v1-minilm/` | The retired V1 MiniLM `rank_answer` module. Nodes no longer accept it. Kept for reference only |

## Same result as the node

The code under `pkg/wasm/runtime`, `pkg/wasm/wasmbin`, `pkg/wasm/remote`, `pkg/scoring` and
`pkg/engine/llm` is copied from the Telegraph node, unchanged apart from import paths. When
the node's copy changes, run `bash scripts/sync-node-packages.sh <path/to/Telegraph>`.

## License

MIT. See [LICENSE](LICENSE).
