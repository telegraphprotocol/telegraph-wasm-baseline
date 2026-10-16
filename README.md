# Telegraph Scoring Adapter Kit

Everything you need to build a **scoring adapter** for a Telegraph judgment or
hybrid intent: the model that reads a question and a miner's answer and scores
how good the answer is. Register a better one than the current adapter and it
replaces it.

Full guide: [Train a Scoring Adapter](https://docs.telegraphprotocol.com/docs/scoring/train-a-scoring-adapter)

## Quick start

From the root of this repository:

```bash
pip install -r requirements.txt

# 1. train (pairs.jsonl: one {"prompt", "chosen", "rejected"} per line; GPU recommended)
python -m wide_runtime.train_adapter --intent CODE_REVIEW --data pairs.jsonl --out runs/CODE_REVIEW

# 2. convert to the node's Q32.32 fixed-point format (~1.2 GB)
python -m wide_runtime.convert_adapter --base answerdotai/ModernBERT-base \
  --adapter runs/CODE_REVIEW/adapter --out runs/CODE_REVIEW/CODE_REVIEW.imwq

# 3. package into the one .wasm you register; prints its keccak256
go run ./cmd/adapterbuild --intent CODE_REVIEW \
  --imwq runs/CODE_REVIEW/CODE_REVIEW.imwq --out runs/CODE_REVIEW
```

Then host `runs/CODE_REVIEW/CODE_REVIEW.wasm` at an `https://` or `ipfs://` URL
and call `registerWasm(keccak256, url, "CODE_REVIEW", feeAddress)`. See
[Build a Scoring Module](https://docs.telegraphprotocol.com/docs/scoring/build-a-scoring-module)
for registering and how the competition works.

Check your data before spending GPU time:
`python -m wide_runtime.train_adapter --intent CODE_REVIEW --data pairs.jsonl --check-data`

**Needs:** Python 3.10+, Go 1.25+, and a GPU for training.

## What's in here

| Path | What |
|---|---|
| `wide_runtime/train_adapter.py` | Trains a ModernBERT LoRA reward model from preference pairs, behind a dataset gate and an evaluation gate |
| `wide_runtime/convert_adapter.py`, `convert.py`, `format.py` | Converts the trained adapter to a Q32.32 `.imwq` model |
| `wide_runtime/golden/`, `interp.py` | The Python reference for the fixed-point math |
| `wide_runtime/wide_wasm/` | The Rust interpreter that runs the model inside WebAssembly |
| `wide_runtime/wide_wasm/hosts/` | The two pinned interpreter builds every package is made from, and their lock file |
| `cmd/adapterbuild` | Packages one `.imwq` into the registrable `.wasm` |
| `pkg/` | The Go code `adapterbuild` uses, including copies of the Telegraph node's own loader, so a package is checked exactly as a node will load it |
| `wide_runtime/README.md` | Technical reference: the `.imwq` format, the WASM ABI, how packages and hosts are built |
| `legacy/v1-minilm/` | The retired V1 MiniLM `rank_answer` module. Nodes no longer accept it. Kept for reference only |

## Same result as the node

A package's keccak256 is what you register, and validators check it against the
bytes they download. Packaging is a pure function of the pinned host, your
`.imwq` and the intent's class, so anyone with the same `.imwq` gets the same
hash. `adapterbuild` builds every package twice, requires identical bytes, and
loads the result with the node's own loader before writing it.

The code under `pkg/wasm/runtime` and `pkg/scoring` is copied from the
[Telegraph node](https://github.com/telegraphprotocol/Telegraph), unchanged apart
from import paths. When the node's copy changes, copy it here again.

## License

MIT. See [LICENSE](LICENSE).
