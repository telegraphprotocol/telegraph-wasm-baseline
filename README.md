# Telegraph Scoring Adapter Kit

Everything you need to build an **Evaluator adapter** for a Telegraph judgment
or hybrid intent: the model that reads a question and a miner's answer and
scores how good the answer is. Register a better one than the current adapter
and it replaces it.

An adapter is a **safetensors delta package**: the LoRA deltas and reward head
of a ModernBERT reward model, a few MB, that validators run on their shared
**Foundation Model** (one immutable ONNX graph, loaded once per validator) in a
fixed sandbox, getting bit-identical scores for the same (foundation, adapter,
tokenizer, input). The earlier format — one ~574 MB `.wasm` with the whole
model baked in — is no longer accepted by nodes: an adapter already registered
in it must be re-exported as a delta from its PEFT checkpoint and reward head
and registered again.

Full guide: [Train a Scoring Adapter](https://docs.telegraphprotocol.com/docs/scoring/train-a-scoring-adapter)

## Quick start

From the root of this repository (on Windows, run the `bash` step from Git Bash):

```bash
pip install -r requirements.txt

# 1. train (pairs.jsonl: one {"prompt", "chosen", "rejected"} per line; GPU recommended)
python -m float_runtime.train_adapter --intent CODE_REVIEW --data pairs.jsonl --out runs/CODE_REVIEW

# 2. export the trained adapter as a safetensors delta package for the published Foundation
#    Model (foundation.json names its sha256, version, tokenizer and LoRA layout; CPU, seconds)
python -m float_runtime.export_adapter --run runs/CODE_REVIEW --intent CODE_REVIEW \
  --class judgment --foundation foundation/foundation.json

# 3. check the package with the node's own parser (and, given the foundation graph and
#    engine, load and score it twice through the node's runtime); prints its keccak256
go run ./cmd/adapterbuild lora --intent CODE_REVIEW --package runs/CODE_REVIEW/CODE_REVIEW.safetensors \
  --foundation foundation/foundation.json --out runs/CODE_REVIEW \
  [--foundation-onnx foundation/foundation_lora_fp32.onnx --engine foundation_engine.wasm]

# 4. register: host runs/CODE_REVIEW/CODE_REVIEW.safetensors at an https:// or ipfs:// URL and call
#    registerWasm(keccak256, url, "CODE_REVIEW", feeAddress)
```

See [Build a Scoring Module](https://docs.telegraphprotocol.com/docs/scoring/build-a-scoring-module)
for registering and how the competition works.

Check your data before spending GPU time:
`python -m float_runtime.train_adapter --intent CODE_REVIEW --data pairs.jsonl --check-data`

Each step prints the next command. `adapterbuild lora` prints one JSON line on
stdout (`keccak256`, `sha256`, `bytes`, `package_path`, `manifest_path`,
`score_hex` when scored, the full manifest) and writes
`runs/CODE_REVIEW/CODE_REVIEW.safetensors.manifest.json` next to the package. The
package is the registrable artifact: nothing is rewritten, so the hash you
register is the hash of the file `export_adapter` wrote.

## One-time setup: the Foundation Model

**The Foundation Model** is published by the Telegraph team, not built per
adapter: `python -m float_runtime.export_foundation --out foundation/` writes
`foundation_lora_fp32.onnx` (ModernBERT-base with the LoRA branches and reward
head as graph inputs, ~599 MB) and `foundation.json`, and
`bash float_runtime/build_engine.sh` builds the engine (`foundation_engine.wasm`,
~2.6 MB). Validators run them as `FOUNDATION_MODEL_PATH` /
`FOUNDATION_ENGINE_WASM_PATH` with their sha256s pinned. An adapter's package
names the foundation it was exported for; a validator on another foundation
rejects it (`foundation_mismatch`). How the files reach validators is decided
outside this kit. Adapter authors only need `foundation.json`.

**Needs:**

- Python 3.10+ (training and export; a GPU for training),
- Go 1.25+ (the package check),
- Rust via [rustup](https://rustup.rs) with the `wasm32-wasip1` target only to
  rebuild the engine (`build_engine.sh`; the pinned 1.98.1 toolchain is
  installed automatically).

**Sizes and costs:** a rank-8 ModernBERT-base delta package is **6.8 MB**
(nodes accept up to 64 MiB, `ADAPTER_LORA_MAX_BYTES`); the shared foundation
is ~599 MB and is loaded once per validator. A 64-token score takes about a
second on the foundation path (measured on a laptop, see the Telegraph
`scripts/bench/foundation_bench.sh` report).

**Training length:** the node scores up to 512 tokens of the rendered question
and answer, so `train_adapter` trains at 512 by default. If you train shorter,
pass the same number to `export_adapter --max-tokens` (the commands it prints
already do). `train_adapter` also writes `reward_head.pt` next to the PEFT
adapter and records the base model's sha256 and the tokenizer's sha256 in
`result.json`; `export_adapter` refuses a run trained on another base model or
tokenizer. The LoRA must be plain PEFT LoRA with `r <= 16`, `target_modules`
among `Wqkv`, `Wo`, `Wi`, one `lora_alpha` (no `rank_pattern`, `alpha_pattern`,
rsLoRA or DoRA).

## What's in here

| Path | What |
|---|---|
| `float_runtime/train_adapter.py` | Trains a ModernBERT LoRA reward model from preference pairs, behind a dataset gate and an evaluation gate; writes the PEFT adapter and `reward_head.pt` |
| `float_runtime/export_adapter.py` | Writes the safetensors delta package for a published foundation (never merges the base weights) |
| `float_runtime/lora_format.py` | The package layout: metadata schema, tensor names, writer, reader |
| `float_runtime/export_foundation.py` | One-time: builds the Foundation Model ONNX with LoRA inputs and `foundation.json` |
| `float_runtime/engine/`, `float_runtime/build_engine.sh` | One-time: the foundation engine (a WASI reactor around `oxionnx`) and its pinned build |
| `cmd/adapterbuild` | `adapterbuild lora`: checks a package with the node's own code and prints the keccak256 to register |
| `pkg/` | The Go code `adapterbuild` uses, including copies of the Telegraph node's own parser and runtime, so a package is checked exactly as a node will run it |
| `float_runtime/README.md` | Technical reference: the package format, the engine ABI, the scoring contract, determinism rules, limits |
| `float_runtime/testdata/` | A few-KB stub foundation and tiny delta packages (made by `make_tiny_foundation.py`) used by the tests; the tokenizer conformance vectors |
| `scripts/sync-node-packages.sh` | Re-copies the node packages under `pkg/` from a Telegraph checkout |
| `legacy/v1-minilm/` | The retired V1 MiniLM `rank_answer` module. Nodes no longer accept it. Kept for reference only |

## Same result as the node

A package's keccak256 is what you register, and validators check it against the
bytes they download. Before writing its manifest, `adapterbuild lora`:

- derives the adapter's class (judgment or hybrid) from the intent catalog and
  refuses deterministic intents;
- runs the node's Stage 1 parse on the package against `foundation.json`: the
  foundation and tokenizer hashes, the class, the rank (`<= 16`), the target
  modules, every tensor shape for the pinned foundation, the offsets, the size
  cap, finite values;
- checks an embedded training record against the children the family serves
  (`--require-training-record`);
- with `--foundation-onnx` and `--engine`, loads it on the foundation through
  the node's own runtime and scores an input twice in the node's fixed
  sandbox, requiring the same bits both times.

The code under `pkg/wasm/runtime`, `pkg/wasm/wasmbin`, `pkg/scoring` and
`pkg/engine/llm` is copied from the
[Telegraph node](https://github.com/telegraphprotocol/Telegraph), unchanged
apart from import paths. When the node's copy changes, run
`bash scripts/sync-node-packages.sh <path/to/Telegraph>`.

## License

MIT. See [LICENSE](LICENSE).
