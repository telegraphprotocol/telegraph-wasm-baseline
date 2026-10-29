#!/usr/bin/env bash
# Build the shared Foundation Model engine (float_runtime/engine, a WASI reactor) to
# wasm32-wasip1 with the pinned toolchain, and print its sha256: that is
# FOUNDATION_ENGINE_WASM_HASH. Built once by the Telegraph team, not per adapter.
#
#   bash float_runtime/build_engine.sh [out.wasm]      (default: float_runtime/engine/foundation_engine.wasm)
#
# Pins: rust-toolchain.toml (1.98.1), Cargo.lock (--locked), +simd128, paths remapped.
# Repeat builds on one machine are byte-identical; builds on different OSes may differ in
# bytes (path separators in panic locations). FOUNDATION_ENGINE_WASM_HASH pins the one
# validators run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$HERE/engine"
OUT="${1:-$CRATE/foundation_engine.wasm}"
TARGET="wasm32-wasip1"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "build_engine: cargo not found" >&2; exit 1; }
is_windows() { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) return 0 ;; esac; return 1; }
native() { if is_windows && [[ "$1" =~ ^/([a-zA-Z])/(.*)$ ]]; then echo "${BASH_REMATCH[1]^^}:/${BASH_REMATCH[2]}"; else echo "$1"; fi; }
native_bs() { local p; p="$(native "$1")"; is_windows && p="${p//\//\\}"; echo "$p"; }
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
RUSTFLAGS_USED="-C target-feature=+simd128 --remap-path-prefix=$(native_bs "$CARGO_HOME_DIR")=/cargo --remap-path-prefix=$(native_bs "$CRATE")=/engine"
( cd "$CRATE" && RUSTFLAGS="$RUSTFLAGS_USED" CARGO_INCREMENTAL=0 \
    cargo build --locked --release --target "$TARGET" >&2 )
cp "$CRATE/target/$TARGET/release/foundation_engine.wasm" "$OUT"
SHA="$(sha256sum "$OUT" | cut -d' ' -f1)"
echo ">> built $OUT ($(wc -c <"$OUT" | tr -d ' ') bytes)" >&2
echo "FOUNDATION_ENGINE_WASM_PATH=$OUT"
echo "FOUNDATION_ENGINE_WASM_HASH=$SHA"
