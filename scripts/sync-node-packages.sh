#!/usr/bin/env bash
# Re-copy the Telegraph node packages adapterbuild uses into pkg/, identical to the node's
# committed files apart from the module path on import lines (telegraph/ ->
# github.com/telegraphprotocol/telegraph-wasm-baseline/), then verify that claim.
#
#   bash scripts/sync-node-packages.sh [path/to/Telegraph] [git-ref]   (defaults: ../Telegraph HEAD)
#
# Only IMPORT LINES are rewritten. A blanket s#"telegraph/#...# would also rewrite string
# literals such as the float host's random seed "telegraph/float-adapter/random/v1", and the
# kit would then verify packages under a different host than the node runs.
#
# The package list is what `go list -deps ./cmd/adapterbuild` reaches inside the module; if
# adapterbuild starts importing another node package, add it here.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE="${1:-$KIT/../Telegraph}"
REF="${2:-HEAD}"
MOD="github.com/telegraphprotocol/telegraph-wasm-baseline"
PKGS="pkg/scoring/family pkg/wasm/wasmbin pkg/wasm/runtime pkg/scoring/catalog pkg/scoring/fetchspec
pkg/engine/llm pkg/scoring/responseparser pkg/scoring/intentformat pkg/scoring/comparator"

git -C "$NODE" rev-parse --verify -q "$REF^{commit}" >/dev/null || { echo "no commit $REF in $NODE" >&2; exit 1; }
echo "syncing from $NODE @ $(git -C "$NODE" rev-parse --short "$REF")" >&2
for p in $PKGS; do
    mkdir -p "$KIT/$p"
    find "$KIT/$p" -maxdepth 1 -name '*.go' ! -name '*_test.go' -delete
    for f in $(git -C "$NODE" ls-tree --name-only "$REF" "$p/" | grep '\.go$' | grep -v '_test\.go$'); do
        git -C "$NODE" show "$REF:$f" |
            sed -E "s#^([[:space:]]*([A-Za-z0-9_]+ )?)\"telegraph/(pkg/[A-Za-z0-9_/]*)\"\$#\\1\"$MOD/\\3\"#" >"$KIT/$f"
    done
done

bad=0
for p in $PKGS; do
    for f in "$KIT/$p"/*.go; do
        case "$f" in *_test.go) continue ;; esac
        r="${f#"$KIT"/}"
        if ! diff -q <(sed -E "s#\"$MOD/(pkg/[A-Za-z0-9_/]*)\"\$#\"telegraph/\\1\"#" "$f") <(git -C "$NODE" show "$REF:$r") >/dev/null; then
            echo "DIFFERS: $r" >&2; bad=1
        fi
    done
done
[ "$bad" = 0 ] || { echo "sync check FAILED" >&2; exit 1; }
cd "$KIT" && go build ./... && echo "synced; go build ./... ok" >&2
