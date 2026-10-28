#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <model.onnx> <baked.wasm>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT_DIR="$(cd "$(dirname "$2")" && pwd)"
OUT="$OUT_DIR/$(basename "$2")"

[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
rustup target list --installed | grep -qx wasm32-wasip1 || rustup target add wasm32-wasip1

TELEGRAPH_ONNX_PATH="$MODEL" cargo build \
  --manifest-path "$SCRIPT_DIR/Cargo.toml" \
  --target wasm32-wasip1 \
  --release

cp "$SCRIPT_DIR/target/wasm32-wasip1/release/baked.wasm" "$OUT"
echo "$OUT"
