#!/usr/bin/env bash
# Build the two pinned wide EMBED HOSTS reproducibly and (re)write hosts/ + hosts.lock.json.
#
#   ./build-hosts.sh           build twice, gate, install into hosts/, write hosts.lock.json
#   ./build-hosts.sh --check   build twice, gate, and require the result to equal the committed
#                              hosts/ + hosts.lock.json (writes nothing) — the "anyone can
#                              reproduce the pinned hosts" check
#
# Gates (any failure exits non-zero and installs nothing):
#   1. reproducibility — two independent `docker build --no-cache` runs must produce
#      byte-identical hosts (sha256);
#   2. determinism — the f32/f64 opcode scan (wide_runtime/build_gates.py) must be
#      clean on both hosts;
#   3. (--check only) the rebuilt hosts must equal the committed ones.
#
# Why pinned hosts matter: every per-intent self-contained package is this host + one .imwq, and
# the package's keccak256 is what registerWasm commits. cmd/adapterbuild refuses any host whose
# sha256 is not in hosts.lock.json, so changing these bytes is a deliberate, reviewed event —
# bump the toolchain/image/source, rerun this, commit hosts/ + the lock together.
#
# Needs: docker (BuildKit), sha256sum, and a Python 3 for the float gate (PYTHON=..., default
# python3 then python).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOSTS_DIR="$SCRIPT_DIR/hosts"
LOCK="$HOSTS_DIR/hosts.lock.json"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.hosts"
WORK="$SCRIPT_DIR/target/hostbuild"          # under target/: gitignored, outside the docker context
CLASSES=(judgment hybrid)

MODE=install
case "${1:-}" in
    "") ;;
    --check) MODE=check ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (use --check or nothing)" >&2; exit 2 ;;
esac

die() { echo "build-hosts: $*" >&2; exit 1; }

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then
            PYTHON="$c"; break
        fi
    done
fi
[ -n "$PYTHON" ] || die "no Python 3 found for the float gate — set PYTHON=/path/to/python"
command -v docker >/dev/null || die "docker not found"
command -v sha256sum >/dev/null || die "sha256sum not found"

# Docker on Windows (Git Bash / MSYS) needs a native path: /c/Work/x -> C:/Work/x. Done by hand
# because a `cygpath` from another toolchain (e.g. conda's) can sit first on PATH and mis-map.
native() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            if [[ "$1" =~ ^/([a-zA-Z])/(.*)$ ]]; then echo "${BASH_REMATCH[1]}:/${BASH_REMATCH[2]}"; return; fi ;;
    esac
    echo "$1"
}

# Single source of truth for the image and rustflags: the Dockerfile's ARG defaults.
IMAGE="$(sed -n 's/^ARG RUST_IMAGE=//p' "$DOCKERFILE" | head -1)"
RUSTFLAGS_PIN="$(sed -n 's/^ARG HOST_RUSTFLAGS="\(.*\)"$/\1/p' "$DOCKERFILE" | head -1)"
TOOLCHAIN="$(sed -n 's/^channel *= *"\(.*\)"$/\1/p' "$SCRIPT_DIR/rust-toolchain.toml" | head -1)"
[ -n "$IMAGE" ] && [ -n "$RUSTFLAGS_PIN" ] && [ -n "$TOOLCHAIN" ] || die "could not read IMAGE/RUSTFLAGS/toolchain pins"
case "$IMAGE" in *@sha256:*) ;; *) die "RUST_IMAGE in $DOCKERFILE is not pinned by digest: $IMAGE" ;; esac
case "$IMAGE" in rust:"$TOOLCHAIN"*) ;; *) die "image $IMAGE does not match rust-toolchain.toml channel $TOOLCHAIN" ;; esac

rm -rf "$WORK"; mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

for run in a b; do
    echo "── build $run/2: docker build --no-cache ($IMAGE)" >&2
    DOCKER_BUILDKIT=1 docker build --no-cache --progress=plain \
        -f "$(native "$DOCKERFILE")" \
        --output "type=local,dest=$(native "$WORK/$run")" \
        "$(native "$SCRIPT_DIR")" >"$WORK/build-$run.log" 2>&1 \
        || { tail -40 "$WORK/build-$run.log" >&2; die "docker build $run failed (log above)"; }
done

declare -A SHA SIZE
for class in "${CLASSES[@]}"; do
    f="wide_embed_$class.wasm"
    [ -s "$WORK/a/$f" ] && [ -s "$WORK/b/$f" ] || die "build did not produce $f"
    sa="$(sha256sum "$WORK/a/$f" | cut -d' ' -f1)"
    sb="$(sha256sum "$WORK/b/$f" | cut -d' ' -f1)"
    [ "$sa" = "$sb" ] || die "REPRODUCIBILITY GATE FAILED for $class: build a=$sa build b=$sb — nothing installed"
    SHA[$class]="$sa"
    SIZE[$class]="$(wc -c <"$WORK/a/$f" | tr -d ' ')"
    echo "   $class: $sa (${SIZE[$class]} B) — identical across both builds" >&2

    # The float gate (run from the repo root with relative paths so a Windows Python is happy).
    rel="${WORK#"$REPO_ROOT"/}/a/$f"
    ( cd "$REPO_ROOT" && "$PYTHON" -m wide_runtime.build_gates "$rel" ) >&2 \
        || die "FLOAT GATE FAILED for $class — nothing installed"
done

CARGO_LOCK_SHA="$(sha256sum "$SCRIPT_DIR/Cargo.lock" | cut -d' ' -f1)"
# Source digest: sha256 over "<sha256>  <path>" lines of every src file, sorted by path.
SRC_SHA="$(cd "$SCRIPT_DIR" && find src -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"

lock_json() {
    cat <<EOF
{
  "schema": 1,
  "builder": "wide_runtime/wide_wasm/build-hosts.sh",
  "toolchain": "$TOOLCHAIN",
  "image": "$IMAGE",
  "image_digest": "${IMAGE##*@}",
  "rustflags": "$RUSTFLAGS_PIN",
  "cargo_lock_sha256": "$CARGO_LOCK_SHA",
  "src_sha256": "$SRC_SHA",
  "judgment": {"file": "wide_embed_judgment.wasm", "features": "embedded", "sha256": "${SHA[judgment]}", "size": ${SIZE[judgment]}},
  "hybrid": {"file": "wide_embed_hybrid.wasm", "features": "embedded-hybrid", "sha256": "${SHA[hybrid]}", "size": ${SIZE[hybrid]}}
}
EOF
}

if [ "$MODE" = check ]; then
    [ -f "$LOCK" ] || die "--check: no committed $LOCK"
    for class in "${CLASSES[@]}"; do
        f="$HOSTS_DIR/wide_embed_$class.wasm"
        [ -f "$f" ] || die "--check: committed host $f missing"
        committed="$(sha256sum "$f" | cut -d' ' -f1)"
        [ "$committed" = "${SHA[$class]}" ] || die "--check: rebuilt $class host ${SHA[$class]} != committed $committed"
        grep -q "\"sha256\": \"${SHA[$class]}\"" "$LOCK" || die "--check: $LOCK does not record the $class sha256 ${SHA[$class]}"
    done
    echo "── --check PASSED: rebuilt hosts are byte-identical to the committed ones" >&2
    exit 0
fi

mkdir -p "$HOSTS_DIR"
for class in "${CLASSES[@]}"; do
    cp "$WORK/a/wide_embed_$class.wasm" "$HOSTS_DIR/wide_embed_$class.wasm"
done
lock_json >"$LOCK"
echo "── installed hosts into $HOSTS_DIR and wrote $LOCK" >&2
cat "$LOCK"
