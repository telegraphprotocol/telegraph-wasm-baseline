> **Retired.** This is the V1 MiniLM `rank_answer` scoring module. Telegraph V2 nodes reject it at registration (`MiniLM rank_answer scoring modules are retired`). To build a scoring module today, see the [repository README](../../README.md).

# Telegraph WASM Scoring Module — Baseline

Telegraph's production WASM scoring module: the program that judges how good
a miner's answer is. It embeds `question` / `ground_truth` / `miner_answer`
with a MiniLM-L6-v2 sentence transformer, compares them with cosine
similarity, adds a BM25 lexical-overlap signal and a length-quality signal,
and combines all four into one composite score.

For a minimal, from-scratch example of what a scoring module needs to
implement (just word-overlap, no embeddings), see the `rust-module/` example
in the `wasm-scoring-module` examples repo, this is what a real, non-toy
one looks like once it needs to judge semantic meaning rather than exact
word overlap.

```
telegraph-wasm-baseline/
├── src/
│   ├── lib.rs         exports: rank_answer, breakdown_answer,
│   │                  embed, cosine_sim, bm25_score, alloc, dealloc
│   ├── embed.rs        MiniLM-L6-v2 inference (two modes, see below)
│   ├── tokenizer.rs     BERT-style tokenizer feeding embed.rs
│   ├── bm25.rs          single-document BM25 lexical scorer
│   ├── math.rs          cosine similarity, sigmoid, L2 norm — pure libm, no host calls
│   └── allocator.rs     no_std global allocator + panic handler
├── build.rs             compiles vocab.txt into a binary lookup table (real_weights mode only)
├── vocab.txt             BERT uncased vocabulary (30,522 tokens)
├── weights/
│   └── minilm_l6_v2_q8.bin   INT8-quantized MiniLM-L6-v2 weights
├── Cargo.toml
└── Cargo.lock
```

Every file above has a module-level doc comment explaining what it does and
why it's built that way, read the source directly for the details
(particularly `embed.rs`, which documents the exact MiniLM-L6-v2 graph it
reimplements and the binary weights format it reads).

## Two build modes

**Projection mode (default)** — no real model weights, no Python. Token IDs
get hashed into a deterministic pseudo-embedding. Structurally identical
output shape to real inference (384-dim, L2-normalised), but not
semantically meaningful (two sentences about the same topic won't
necessarily score high similarity). Useful for exercising the rest of the
pipeline fast, not for judging real answer quality.

```bash
rustup target add wasm32-unknown-unknown   # once
cargo build --release --target wasm32-unknown-unknown
```

**Real weights mode** — runs actual MiniLM-L6-v2 inference (6-layer
transformer, INT8-quantized) using the weights already included in
`weights/minilm_l6_v2_q8.bin`, so this builds out of the box with no extra
export step:

```bash
cargo build --release --target wasm32-unknown-unknown --features real_weights
```

Output either way: `target/wasm32-unknown-unknown/release/telegraph_scoring.wasm`

## Testing it

Load the built `.wasm` with any [wazero](https://wazero.io)-based host and
call `rank_answer` (write `question`/`ground_truth`/`miner_answer` into the
module's memory via its own `alloc`, then call it), the same way Telegraph's
own validator does. The `go-tester` CLI in the `wasm-scoring-module`
examples repo is a small standalone tool that does exactly this if you want
something ready-made.

Each source file also has its own `#[cfg(test)]` module with unit tests for
that file's logic (cosine/sigmoid edge cases in `math.rs`, BM25 scoring in
`bm25.rs`, tokenizer padding/truncation in `tokenizer.rs`, embedding
determinism in `embed.rs`). `cargo test` does not currently run them,
`#[panic_handler]` in `allocator.rs` is unconditional and collides with
`std`'s own panic handler when compiling for the native test target instead
of `wasm32-unknown-unknown`. Gating that handler behind `#[cfg(not(test))]`
would fix it. Read the tests directly in each file's bottom section in the
meantime.

## Exports beyond the minimal three

A minimal scoring module only needs `alloc` / `dealloc` / `rank_answer`.
This module also exports:

- `breakdown_answer` — returns the four individual signals (relevance,
  correctness, lexical, length) plus the composite, for debugging why a
  score came out the way it did instead of just the final number.
- `embed` / `cosine_sim` / `bm25_score` — the individual building blocks,
  exposed standalone so callers/tests can exercise one signal in isolation.
