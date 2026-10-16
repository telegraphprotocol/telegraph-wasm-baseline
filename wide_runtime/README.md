# Wide Q32.32 deterministic inference pipeline

Converts a float HuggingFace encoder (ModernBERT) plus a LoRA adapter into a `.imwq`
artifact whose weights **and** activations are Q32.32 fixed-point int64, and the Rust/wasm
runtime that executes it.

## Why wide rather than int8

int8 is deterministic but it does not rank. Activations are requantised to 256 levels
between every layer, and the reward signal separating a good answer from a bad one is
smaller than the rounding error accumulated across ~22 layers — measured Spearman against
the float model was ≈0.001, indistinguishable from shuffling the miners. Calibration,
SmoothQuant and a wider score op were all tried; none recovered it.

Q32.32 is int64 with 32 integer bits (±2.1e9) and 32 fractional bits (~2.3e-10). Op-level
error against float is ~3e-10 rather than ~1%. It is still fully integer — no f32/f64 opcode
in the forward pass — so it keeps the property int8 was chosen for: every validator computes
bit-identical scores and the epoch cannot fork on floating-point rounding.

Q16.48 was tried first and overflowed. Real ModernBERT activations reach ~27,000, which does
not fit 15 integer bits; the products wrapped i64 and every score collapsed to zero.

## Layout

| Path | What |
|---|---|
| `convert.py` | float state-dict → Q32.32 weights + op-list → `.imwq` |
| `train_adapter.py` | train the LoRA reward-model adapter (ModernBERT, Bradley-Terry) from preference pairs, behind a dataset gate and an evaluation gate |
| `convert_adapter.py` | driver: resolve HF adapter, merge LoRA with peft, call the converter |
| `format.py` | the `.imwq` container (`WideWriter`/`WideReader`, opcodes, the binary `BOPS` block) |
| `interp.py` | Python reference interpreter over the op-list |
| `golden/fixed.py` | **Q32.32 math core**, `FP_PRECISION = 32` |
| `golden/ops.py` | Q32.32 transformer ops (`_tdiv` for Rust-compatible division) |
| `golden/hybrid.py` | comparator core + fusion gate + the judgment boundary conversion |
| `golden/test_fixed.py`, `golden/test_ops.py` | the golden's own accuracy tests vs float |
| `hf_helpers.py` | the five HF plumbing functions extracted from the int8 pipeline |
| `build_gates.py` | `static_float_scan` — the f32/f64 determinism gate |
| `wide_wasm/` | the Rust runtime (`fixed.rs`, `ops.rs`, `reader.rs`, `interp.rs`, `hybrid.rs`, `lib.rs`) |
| `wide_wasm/tests/` | Rust↔Python bit-identity tests + their fixture generators (`embed_slot.rs` pins the embed-slot layout) |
| `wide_wasm/rust-toolchain.toml` | pinned compiler (1.98.1) — a package's hash depends on the embed host's bytes |
| `wide_wasm/Dockerfile.hosts`, `build-hosts.sh` | reproducible Docker build of the two embed hosts (twice, sha256-compared, float-scanned) |
| `wide_wasm/hosts/` | the COMMITTED pinned embed hosts + `hosts.lock.json` (sha256, size, toolchain, image digest, rustflags) |

The Go side is at the root of this repository. The packager core is `pkg/wasm/wideembed`;
the official build/publish step is `pkg/adapterbuild` + `cmd/adapterbuild` (pinned hosts only,
catalog class, twice-package gate, per-intent publish with an overwrite guard). `cmd/wideembed`
remains as the dev packager for arbitrary hosts and warns when the host is not pinned.

`pkg/wasm/runtime` (`WideRuntime`, `WideHybridRuntime`, and the self-contained-package loaders
`NewWideEmbedded` / `NewWideHybridEmbedded`) and the `pkg/scoring/...` packages are copies of the
Telegraph node's own code, kept identical apart from import paths, so `adapterbuild` checks a
package exactly the way a node loads it. When the node's copies change, copy them here again.

## Running it

```bash
# Rust↔Python bit-identity (ops, full model, and the hybrid half)
cd wide_runtime/wide_wasm && cargo test --release

# build the shared wasm (a ~52 KB INTERPRETER + comparator — the model is a separate .imwq).
# Default features: this binary must stay byte-identical; it is the deterministic comparator.
cargo build --release --target wasm32-unknown-unknown

# the two EMBED HOSTS are committed (hosts/ + hosts.lock.json). Rebuild them only on a deliberate
# toolchain/source bump — in Docker, twice, sha256-compared, float-scanned:
PYTHON=python3 ./build-hosts.sh            # rewrites hosts/ + hosts.lock.json; commit them together
PYTHON=python3 ./build-hosts.sh --check    # reproduce and compare against the committed hosts
cargo test --release --features embedded   # also runs embed_slot.rs

# the f32/f64 determinism gate — must be clean (from the repository root from here on)
cd ../.. && python -m wide_runtime.build_gates \
  wide_runtime/wide_wasm/target/wasm32-unknown-unknown/release/wide_wasm.wasm

# the Python golden's own accuracy tests vs float
python -m wide_runtime.golden.test_fixed
python -m wide_runtime.golden.test_ops

# regenerate the golden fixtures (only when the golden itself changes)
python wide_runtime/wide_wasm/tests/gen_fixture.py
python wide_runtime/wide_wasm/tests/gen_model_fixture.py
python wide_runtime/wide_wasm/tests/gen_hybrid_fixture.py
python wide_runtime/wide_wasm/tests/gen_rope_model_fixture.py

# train an adapter from preference pairs (JSONL: prompt/chosen/rejected). GPU recommended.
# --check-data runs only the dataset gate. Exit 0 = PASS, 1 = a gate FAILED.
python -m wide_runtime.train_adapter \
  --intent VULNERABILITY_TRIAGE --data pairs.jsonl --out runs/VULNERABILITY_TRIAGE

# convert an adapter (~1.2 GB out, several minutes) — into runs/, which is gitignored.
# --adapter takes a local directory or a Hugging Face repo[/subfolder].
python -m wide_runtime.convert_adapter \
  --base answerdotai/ModernBERT-base \
  --adapter runs/VULNERABILITY_TRIAGE/adapter \
  --out runs/VULNERABILITY_TRIAGE/VULNERABILITY_TRIAGE.imwq

# package ONE self-contained wasm per intent (interpreter + embedded model) — what gets registered.
# Class from the intent catalog, pinned host from hosts.lock.json, built twice, verified by load.
go build -o runs/adapterbuild ./cmd/adapterbuild
runs/adapterbuild --intent VULNERABILITY_TRIAGE \
  --imwq runs/VULNERABILITY_TRIAGE/VULNERABILITY_TRIAGE.imwq --out runs/VULNERABILITY_TRIAGE \
  [--adapter ... --base-model ... --backbone-version ... --lora-hash ... --dataset-version ...]
# ... and publish to wasm/<INTENT>/<INTENT>.wasm (+ .manifest.json) in an S3-compatible store
OBJECT_STORE_ENDPOINT=http://localhost:9000 OBJECT_STORE_BUCKET=<bucket> \
OBJECT_STORE_ACCESS_KEY=... OBJECT_STORE_SECRET_KEY=... \
  runs/adapterbuild --intent VULNERABILITY_TRIAGE --imwq ... --out ... --publish [--overwrite]

# dev packager for an arbitrary (unpinned) host — never for registration
go run ./cmd/wideembed --host <embed-host.wasm> --imwq <model.imwq> --class hybrid --out <pkg.wasm>

# the parity/refusal tests (skip unless the builds are named); the committed hosts work here
WIDE_WASM=wide_runtime/wide_wasm/target/wasm32-unknown-unknown/release/wide_wasm.wasm \
WIDE_EMBED_HOST_JUDGMENT=wide_runtime/wide_wasm/hosts/wide_embed_judgment.wasm \
WIDE_EMBED_HOST_HYBRID=wide_runtime/wide_wasm/hosts/wide_embed_hybrid.wasm go test ./pkg/wasm/wideembed -v
go test ./pkg/adapterbuild/... ./cmd/adapterbuild   # lock refusal, class, twice-build, overwrite guard
```

## The `.imwq` format

Little-endian throughout.

```
'IMWQ'                    4 bytes magic
mjson_len : u32           length of the JSON meta that follows
mjson     : bytes         JSON (version/config/tensor-names/ops) — for DEBUGGING only;
                          the Rust runtime does not parse it, it uses the block below
'BOPS'                    4 bytes
num_ops   : u32
ops[num_ops]              60 bytes each:
    op:u16, flags:u16,
    in0:i32, in1:i32, out:i32, weight_idx:i32, bias_idx:i32, aux_idx:i32,
    arg0:i64, arg1:i64, arg2:i64, arg3:i64
config    : 10 x i32      arch, hidden, layers, heads, kv_heads, head_dim, inter,
                          max_seq, rope_theta_milli, rope
num_blobs : u32
per blob  : ndim:u32, dims[ndim]:u64, n:u64, data:i64[n]   (Q32.32 weights)
```

Opcodes: `END=0 EMBED=1 QMATMUL=2 ADD=3 LAYERNORM=4 RMSNORM=5 ACT_GELU=6 ROPE=7 POS_ADD=8
ATTN=9 SWIGLU=10 POOL=11 LAST_TOKEN=12 SCORE=13 ACT_SILU=15`. Flags: `CAUSAL=1`,
`MEAN_POOL=4`.

## WASM ABI

```
imwq_alloc(len) -> ptr
imwq_free(ptr, len)
imwq_score(imw_ptr, imw_len, tok_ptr, n_tokens) -> i64        // raw Q32.32, i64::MIN on reject
imwq_debug_counts(imw_ptr, imw_len) -> i64                    // (tensors<<32)|ops
imwq_score_hybrid(imw_ptr, imw_len, tok_ptr, n_tokens,
                  ground_truth_value, miner_value, spec_ptr) -> i64   // [0,10000], -(err)
```

`imwq_score_hybrid` mirrors int8's `imw_score_hybrid` with the two artifact arguments
prepended — int8 needed none because its model was baked into the module.

```
imw_score_deterministic(ground_truth_value, miner_value, spec_ptr) -> i64  // [0,10000]
imw_alloc(len) -> ptr                                         // alias of imwq_alloc
imw_free(ptr, len)                                            // alias of imwq_free
```

`imw_score_deterministic` is the comparator-only path the deterministic intents load: no
model, no `.imwq`, just an integer compare of two already-scaled values. It was ported from
the int8 crate, where it had always lived — the comparator half was never int8-specific, so
retiring int8 quantisation retired the model path and left this untouched. Before the port
this repo could build the hybrid wasm but not the one most intents actually use, and
`ComparatorScorer` had to be handed a module published from another repo (hence
`COMPARATOR_WASM` being an env var the tests skip on).

Its `spec_ptr` reads 2 i64s — `[comparator, tolerance]` — the first two fields of the
hybrid's 10-int buffer, so one `comparator.Fusion.Buffer()` serves both exports and a
deterministic intent cannot drift from its hybrid sibling's thresholds.

`imw_alloc`/`imw_free` are thin aliases of the `imwq_` pair, because `DeterministicRuntime`
predates the wide runtime and binds the unprefixed spelling. Renaming the Go side instead
would break every already-published int8-era comparator module the validator still loads.

One 52 KB module therefore satisfies all three scoring interfaces.

## Self-contained packages (one wasm per intent)

The split shape above (52 KB interpreter + separate 1.2 GB `.imwq`) is the dev path. What
ships — and what is registered on-chain — is ONE `.wasm` per intent with the model inside, the
same shape the old int8 build had, so the registered keccak256 covers model and code together.

**Mechanism: post-link data segment, not `include_bytes!`.** `cmd/wideembed`
(`pkg/wasm/wideembed`) takes a pinned *embed host* and appends two active data segments:

1. the 56-byte embed slot (`IMWQEMB1`, version, ptr, len, sha256 of the model) over the host's
   `IMWQ_EMBED_DESC` static — whose address the host reports via `imwq_embed_desc_addr`, and
   which the packager first checks still holds the `IMWQHOST` sentinel;
2. the `.imwq` bytes, at the host's original initial-memory size (== `__heap_end`, the only
   pre-existing region dlmalloc 0.2 ever hands out; everything else comes from `memory.grow`,
   which starts past the raised minimum, so the allocator can never overwrite the model);

and raises the memory section's minimum to cover the model. Every other section is copied
verbatim. The interpreter is never recompiled per intent (no 1.2 GB constant through rustc),
packaging takes under a minute even for the real model, and the output is a pure function of
(host, model, class).

**Parity is structural.** At run time `NewWideEmbedded` calls `imwq_embedded()` to get
`(ptr, len)`, cross-checks the slot and the model's sha256, parses it (`imwq_debug_counts`),
then scores through the *unchanged* `imwq_score` / `imwq_score_hybrid` — the same function over
the same bytes as the split path. Measured on the real VULNERABILITY_TRIAGE adapter (8 tokens):
external raw `-10589197770` / hybrid `3548` bps, embedded raw `-10589197770` / `3548` bps
(`pkg/wasm/wideembed` `TestRealArtifactParity`; the fixture-level `TestEmbeddedMatchesExternal`
also matches the Python golden codes).

**Features.**

| build | features | exports |
|---|---|---|
| shared interpreter / comparator | (none — default) | `imwq_score`, `imwq_score_hybrid`, `imw_score_deterministic`, allocators |
| judgment embed host | `embedded` | `imwq_score`, `imwq_embedded`, `imwq_embed_desc_addr`, allocators |
| hybrid embed host | `embedded-hybrid` | the above + `imwq_score_hybrid` |

There is deliberately no `default = [...]`: cargo hashes the enabled feature set into
`-C metadata`, which renames every mangled symbol in the name section, so even a default-on
feature would move the shared binary's hash (verified: the default build's sha256 is unchanged
by the feature work). An embed host never exports `imw_score_deterministic`, so a package cannot
be mistaken for the comparator module, and a package declares its class by whether it exports
`imwq_score_hybrid` (the node refuses a package whose class disagrees with the intent's).

```
imwq_embedded() -> i64          // (ptr<<32)|len of the resident model, i64::MIN if none/corrupt
imwq_embed_desc_addr() -> u32   // address of the 56-byte embed slot (packager + loader cross-check)
```

**4 GiB budget.** The packager refuses — never truncates — a model whose
`model end + parse copy (WideModel::parse copies every tensor) + activation bound + 64 MiB`
exceeds the wasm32 ceiling. For VULNERABILITY_TRIAGE at 256 tokens: need 3.12 GiB (model end
1.12 + parse 1.12 + activation bound 0.81 + margin), headroom 0.88 GiB. Measured linear memory
after an 8-token score: 2.264 GiB (headroom 1.74 GiB) on both paths.

**Host RAM.** wazero keeps its own copy of every data segment for the instance's lifetime, so a
package costs ~1.1 GiB more resident host memory per intent than the split path (measured Go heap
in use after scoring: 3.87 GiB embedded vs 2.75 GiB external). A loader that applies the model
segment itself would remove that; not done yet.

**Reproducing a package hash.** You need the exact embed-host bytes. rustc embeds source paths
in the binary (an absolute `.cargo\registry\...\dlmalloc.rs` path, and `src\ops.rs` with Windows
separators), so the same toolchain on another machine produces a different host. That is why the
official hosts are built by `wide_wasm/build-hosts.sh` in Docker — `rust:1.98.1-slim-bookworm`
pinned by digest, fixed `WORKDIR /build` and `CARGO_HOME`, `--remap-path-prefix` for both,
`cargo build --locked` (the compile itself `--offline` after a separate fetch) — built twice with
`--no-cache` and required byte-identical, then float-scanned. The result is committed under
`wide_wasm/hosts/` with `hosts.lock.json`; `cmd/adapterbuild` refuses any host whose sha256 is not
in the lock, and every package manifest records `host_sha256` plus the lock's own sha256. Anyone
with the committed host and the `.imwq` reproduces the registered keccak (packaging is a pure
function of host, model, class). Validators are unaffected — they verify the keccak of the bytes
they fetch.

## Automated build + publish (`cmd/adapterbuild`)

(2026-10-03) The build/publish step is wired end to end:

```
.imwq (Q32.32 convert: convert_adapter.merge_to_wide)
  -> class from the intent catalog (pkg/scoring/catalog; --class must agree; deterministic refused)
  -> pinned embed host for that class (hosts.lock.json sha256 check)
  -> wideembed.Package TWICE, byte-compared (reproducibility gate), first build verified by load
  -> <out>/<INTENT>.wasm + <INTENT>.wasm.manifest.json (wideembed manifest + intent, class,
     provenance, hosts-lock reference, builder version, reproducibility record)
  -> [--publish] wasm/<INTENT>/<INTENT>.wasm + .manifest.json in the S3-compatible store
     (OBJECT_STORE_ENDPOINT/_BUCKET/_ACCESS_KEY/_SECRET_KEY/_PUBLIC_BASE_URL/_SECURE)
  -> registerWasm(keccak256, url, intent) — still manual
```

Per-intent keys are overwritten in place, and a node re-fetches the on-chain URL on a cache miss,
so replacing the bytes behind a key breaks every older registration of that URL. Publish therefore
refuses when the key already holds a different object (it logs the old object's sha256 / etag)
unless `--overwrite` is passed; re-publishing identical bytes is a no-op for the package. Any
failure — including publish — exits non-zero with nothing on stdout. `--artifact <INTENT>.wasm
--publish` re-hashes an existing build against its manifest and the current lock and publishes it
without re-packaging.

Measured on VULNERABILITY_TRIAGE (1,205,314,279-byte `.imwq`): package 1,205,366,070 B, both
builds sha256 `1efe5326…7c8e`, keccak256 `0x87c17b85…aa07`, ~1 min for both builds + verify;
fetched back from the object store it scores raw `-10589197770` / hybrid `3548` bps — identical to the split
path.

## Two bugs this port found and fixed

Both were latent in the upstream tree, both were silent, and both were caught by adding a
golden fixture that uses RoPE (the previous fixtures were all `rope=False`, while every real
ModernBERT adapter uses it — the converted VULNERABILITY_TRIAGE artifact has 44 ROPE ops).

**1. The RoPE table was all zeros on wasm32 — the deployment target.**
`interp.rs::build_rope` filled the cos/sin tables with f64 trig under
`#[cfg(not(target_arch = "wasm32"))]`. Native builds filled them and were correct; the wasm32
build compiled the fill out, so `ops::rope` computed `x1*0 - x2*0 = 0` and **zeroed Q and K
in every attention layer**. Nothing errored, and `cargo test` stayed green because it runs
native. Measured on the synthetic fixture: the correct score is `+484013641`, the wasm32
build returned `-2057092746` — different magnitude, different sign.

Fixed by baking the tables into the `.imwq` at conversion time (where float is already
allowed, at the same boundary that converts every weight), so both runtimes read identical
integers. An artifact without them is now *refused* on wasm32 rather than scored.

**2. Mean-POOL divided the wrong way in the Python golden.**
`interp.py` used `//` (floors toward -inf) where the Rust runtime uses `/` on an i128
(truncates toward zero). On the old fixture every pooled column sum happened to be positive,
so the two agreed; with RoPE, 5 of 16 columns go negative and the golden came out 1 ULP off.
Fixed by using `_tdiv`, the convention the rest of the golden already follows.

Worth recording what #2 was *not*: an off-by-one in a fixed-point pipeline looks like a
rounding-mode mismatch (Python `round()` is banker's/ties-to-even, Rust `.round()` is
half-away-from-zero). That was measured and ruled out — no entry of a real RoPE table lands
on an exact `.5` tie, so both modes produce byte-identical tables. The Rust fallback was
aligned to ties-to-even anyway, and the convention is pinned by
`rope_golden.rs::float_to_q32_rounds_ties_to_even_like_python`, because "currently tie-free"
is not a contract.

## Traps that have already been paid for

1. **`FP_PRECISION` must be 32 in BOTH** `golden/fixed.py` and `wide_wasm/src/fixed.rs`.
   Changing one without the other breaks bit-identity silently.
2. **Integer division.** Python `//` floors toward -inf; Rust/C `/` truncates toward zero.
   The golden uses `_tdiv` so they match. Any new division must match, including in the
   fusion gate and in mean-POOL (which is where this was actually violated — see above).
3. **Float -> Q32.32 rounding** is BANKER'S (ties-to-even), because the golden's
   `fp_from_float` is `int(round(x * FP_ONE))` and Python's `round()` is ties-to-even. Rust's
   `f64::round()` is half-away-from-zero and must not be used; `round_ties_even()` is the
   matching primitive. The golden is canonical here: every converted weight already goes
   through `fp_from_float`, so "fixing" Python would alter every existing artifact.
4. **`fp_inv_sqrt`'s octave target** is `FP_PRECISION`, not a literal 48 — the hardcoded
   Q16.48 value returned ~0 under Q32.32.
5. **Attention scale** must be `fp_inv_sqrt(fp_from_int(head_dim))` computed in integers, not
   `fp_from_float(1/sqrt(hd))`, which diverges between the two runtimes.
6. **GELU constants** (`C_SQRT_2_PI`, `C_044715`) are hardcoded in `fixed.rs` and must equal
   Python's `fp_from_float(...)` under the current `FP_ONE`. Recompute them if the format
   ever changes.
7. **The Python golden wraps to i64** (`_wrap_i64`) so overflow surfaces in the fast Python
   test instead of hiding until wasm. Keep that — it is what would have caught the Q16.48
   overflow before it shipped.
8. **Truth comes BEFORE miner** at the `imwq_score_hybrid` boundary. The Go wrapper takes
   `(minerValue, truthValue)` in host-natural order and swaps at the call site. A swap is
   silent under RelTol: it only changes the denominator, so it still returns a plausible
   score.
9. **The spec buffer is exactly 10 i64s, positional, no length argument**:
   `[comparator, tolerance, fusion_mode, pass, low, floor, span, w_c, w_j, reserved]`. A
   short buffer is read past its end; a reordered one fuses against the wrong thresholds.
   Neither errors. Pinned by `hybrid_golden.rs::spec_buffer_layout_is_positional` and
   `wide_hybrid_test.go::TestWideHybridSpecMatchesTheGoProducer` against
   `comparator.Fusion.Buffer()`.
10. **The score head emits an unbounded LOGIT**, not a `[0,1]` value. The boundary conversion
   applies a Q32.32 sigmoid. A plain divide-and-clamp puts every negative-logit miner at 0,
   where they all tie at the bottom — indistinguishable from every miner answering badly.

## Artifacts are not committed

A converted `.imwq` is ~1.2 GB (int64 weights, 8× the int8 build). Build it under `runs/`,
which is gitignored. It is rebuildable from this code, which is the thing worth committing.

The same goes for packages (`cmd/adapterbuild` output, ~1.2 GB each): build them under
`runs/` too. The embed hosts are the exception — ~50 KB each, committed under
`wide_wasm/hosts/` with `hosts.lock.json`, because a package's registered hash is only
reproducible from those exact bytes.
