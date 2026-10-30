# Telegraph scoring adapter kit

This repository builds a V2 scoring adapter for a Telegraph `judgment` or
`hybrid` intent. An adapter receives the canonical rendered question and a
miner answer from the node, then returns one model score. It is a candidate
until the node has structurally validated it, run its smoke score, and passed
it through the intent's evaluation material.

The current format is an ONNX FP32 ModernBERT reward model baked into a
restricted WASI WebAssembly command. It is identified by this metadata:

```json
{"abi_version":2,"runtime":"onnx_f32","class":"judgment","max_tokens":512}
```

The packager writes that metadata. Do not add or edit it by hand.

## Requirements

- Python 3.10 or newer
- Go 1.25 or newer
- Rust with the `wasm32-wasip1` target
- A ModernBERT base model and a compatible LoRA reward-model checkpoint

Training normally needs a GPU. Exporting, baking, and packaging can run on a
CPU, although a real model requires substantial memory.

```bash
python -m pip install -r requirements.txt
rustup target add wasm32-wasip1
```

## Build an adapter

Use the deployed intent ID, including its V2 suffix where applicable. The
example below uses `CODE_REVIEW_GENERAL`.

### 1. Train or obtain a compatible reward-model checkpoint

The training utility accepts JSONL preference pairs:

```json
{"prompt":"question","chosen":"better miner answer","rejected":"worse miner answer"}
```

```bash
python -m wide_runtime.train_adapter \
  --intent CODE_REVIEW_GENERAL \
  --data pairs.jsonl \
  --out runs/CODE_REVIEW_GENERAL
```

Before training, run the data checks alone if desired:

```bash
python -m wide_runtime.train_adapter \
  --intent CODE_REVIEW_GENERAL \
  --data pairs.jsonl \
  --check-data
```

### 2. Export the merged model to ONNX

`export_onnx` merges the LoRA checkpoint with ModernBERT and exports one
dynamic-length ONNX graph. It does not train or alter checkpoint weights.

```bash
python -m wide_runtime.export_onnx \
  --base answerdotai/ModernBERT-base \
  --adapter runs/CODE_REVIEW_GENERAL/adapter \
  --out runs/CODE_REVIEW_GENERAL/model.onnx
```

The export command also writes `model.onnx.export.json`, including its input
and output contract and a reference score. Keep it with the build records.

The output is one self-contained file of about 600 MB. On torch 2.9 or newer,
`torch.onnx.export` defaults to the dynamo exporter, which writes the weights to
a separate `model.onnx.data` file that the bake cannot embed; `export_onnx`
therefore uses the TorchScript exporter and refuses any model whose tensors are
stored as external data.

### 3. Bake the ONNX graph into a WASI module

```bash
mkdir -p runs/CODE_REVIEW_GENERAL
wide_runtime/vm_float_wasm/bake_wasm.sh \
  runs/CODE_REVIEW_GENERAL/model.onnx \
  runs/CODE_REVIEW_GENERAL/model.baked.wasm
```

The bake embeds only the named `.onnx` file. It refuses to build when a
`<model>.onnx.data` weights file sits beside it, or when the model is under
16 MiB (a graph without its weights); set `TELEGRAPH_ALLOW_SMALL_ONNX=1` only
for a deliberately tiny test model.

The guest has no network access and receives only the node-provided input
file. It accepts `input_ids` and `attention_mask` and emits one finite FP32
score. The node sets the token IDs, attention mask, command arguments, wall
clock, I/O limits, and memory limits; an adapter cannot choose them.

### 4. Package the artifact

```bash
go run ./cmd/adapterbuild \
  --intent CODE_REVIEW_GENERAL \
  --baked-wasm runs/CODE_REVIEW_GENERAL/model.baked.wasm \
  --out runs/CODE_REVIEW_GENERAL
```

This writes:

```text
runs/CODE_REVIEW_GENERAL/CODE_REVIEW_GENERAL.wasm
runs/CODE_REVIEW_GENERAL/CODE_REVIEW_GENERAL.wasm.manifest.json
```

`adapterbuild` refuses a module that is not a WASI command, already contains
adapter metadata, exports retired Q32.32 functions, or cannot compile under
wazero. It packages twice and requires byte-identical output. The resulting
manifest contains the SHA-256 hash, Keccak-256 hash, declared class, ABI,
runtime, token cap, and build provenance.

Use `--max-tokens N` only to lower the cap for this artifact; accepted values
are 1 through 512. The default is 512.

The intent class is resolved from the catalog. `--class judgment` or
`--class hybrid` is optional and only asserts the expected class; it cannot
override the node's live intent classification. The compiled-in catalog does
not list every live judgment or hybrid prefix: for an intent it does not list,
check the node (`GET /api/wasm?resolve=<INTENT>`) and pass its class with
`--class`. A class mismatch is a failed
build or registration, not a fallback.

## Verify before submission

Run the kit tests:

```bash
go test ./...
python -m wide_runtime.test_export_onnx
```

Check the package and manifest before publishing them:

```bash
shasum -a 256 runs/CODE_REVIEW_GENERAL/CODE_REVIEW_GENERAL.wasm
cat runs/CODE_REVIEW_GENERAL/CODE_REVIEW_GENERAL.wasm.manifest.json
```

The node performs additional checks when the candidate is submitted:

1. It verifies ABI v2, `onnx_f32`, the intent class, the token cap, module
   imports, exports, and size limits.
2. It runs a fixed smoke input. A module that traps, emits malformed output,
   or returns a non-finite result is rejected.
3. It evaluates a valid candidate against the intent's evaluation material.
   Lack of compatible material leaves the candidate pending; it is not a pass.

A successfully packaged file is therefore not automatically active or ranked.

## Publish and submit

`adapterbuild --publish` uploads a built artifact and its manifest to the
configured S3-compatible object store. It does not activate the adapter.

```bash
OBJECT_STORE_ENDPOINT=https://object-store.example \
OBJECT_STORE_BUCKET=telegraph-adapters \
OBJECT_STORE_ACCESS_KEY=... \
OBJECT_STORE_SECRET_KEY=... \
go run ./cmd/adapterbuild \
  --intent CODE_REVIEW_GENERAL \
  --artifact runs/CODE_REVIEW_GENERAL/CODE_REVIEW_GENERAL.wasm \
  --publish
```

Submit the resulting immutable URL and Keccak-256 hash through the Telegraph
V2 adapter-registration flow. Do not replace bytes at a URL that has already
been registered: validators verify downloaded bytes against the registered
hash. The publisher rejects replacing different bytes unless `--overwrite` is
explicitly provided.

## Determinism and compatibility

The FP32 model path is the supported V2 format. Packaging is reproducible for
the same baked input, but model-score bit identity across arbitrary CPUs and
runtimes is not established by this kit. Do not claim cross-machine
bit-identical inference without an end-to-end test against the target node
versions and CPU environments.

The retired Q32.32 `.imwq` tooling remains in `wide_runtime/wide_wasm` only as
historical reference. V2 model adapters must use the ONNX/WASI path above.

## Repository layout

| Path | Purpose |
| --- | --- |
| `wide_runtime/train_adapter.py` | Trains a ModernBERT LoRA reward model from preference pairs. |
| `wide_runtime/export_onnx.py` | Merges and exports the trained model to ONNX. |
| `wide_runtime/vm_float_wasm/` | Restricted WASI guest and bake script. |
| `cmd/adapterbuild` | Adds V2 metadata, verifies the module, writes the artifact and manifest, and optionally publishes. |
| `pkg/adapterbuild/` | Builder, catalog-class, manifest, and publish checks. |
| `pkg/wasm/runtime/` | Shared package metadata validation used by the kit. |
| `wide_runtime/wide_wasm/` | Retired Q32.32 implementation; not the V2 model path. |

## License

MIT. See [LICENSE](LICENSE).
