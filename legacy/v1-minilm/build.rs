//! Build script.
//!
//! For the default (projection) build, this does nothing — `word_to_id` uses
//! FNV1a hashing at runtime, no vocab file needed.
//!
//! For `--features real_weights`, this reads `vocab.txt` (produced by
//! `scripts/export_minilm_weights.py`, one token per line, line number =
//! token ID — the standard BERT/HuggingFace `vocab.txt` format) and emits a
//! binary lookup table to `$OUT_DIR/vocab_table.bin`, which `tokenizer.rs`
//! embeds via `include_bytes!` and binary-searches at runtime.
//!
//! Doing the sort at build time (not runtime) means `tokenizer.rs` never
//! pays sorting cost and never touches the filesystem inside the WASM
//! module itself — WASM has no filesystem, so this MUST happen in build.rs,
//! not in a `once_cell`-style lazy-init inside the crate.
//!
//! ## vocab_table.bin format
//!
//! A sequence of fixed-size records, one per vocab entry, SORTED
//! lexicographically by the token bytes (so tokenizer.rs can binary-search):
//!
//!   [u16 len][u32 token_id][bytes padded to MAX_TOKEN_LEN with 0x00]
//!
//! Fixed-size records (rather than a separate offset table) keep the reader
//! in tokenizer.rs to a single `chunks_exact` + direct indexing — no offset
//! table to also embed and keep in sync.
//!
//! MAX_TOKEN_LEN is generous (48 bytes) — BERT's uncased vocab has a small
//! number of long tokens (mostly `##`-prefixed subword continuations and a
//! handful of full words); anything longer is truncated with a build-time
//! warning rather than failing the build, since a handful of rare
//! unreachable long tokens is a non-issue in practice (they'd simply never
//! be looked up if user text can't produce that literal token id, but we'd
//! rather truncate than panic on a checkpoint dump we don't fully control).

use std::env;
use std::fs;
use std::path::Path;

const MAX_TOKEN_LEN: usize = 48;

fn main() {
    println!("cargo:rerun-if-changed=vocab.txt");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_REAL_WEIGHTS");

    // Only build the vocab table when real_weights is active — the default
    // projection build has no use for it and shouldn't require vocab.txt to
    // exist at all.
    if env::var("CARGO_FEATURE_REAL_WEIGHTS").is_err() {
        return;
    }

    let vocab_path = "vocab.txt";
    let text = fs::read_to_string(vocab_path).unwrap_or_else(|e| {
        panic!(
            "build.rs: real_weights feature is enabled but couldn't read {vocab_path}: {e}\n\
             Run `python3 scripts/export_minilm_weights.py` first — it writes both \
             weights/minilm_l6_v2_q8.bin and vocab.txt."
        )
    });

    let mut records: Vec<(String, u32)> = text
        .lines()
        .enumerate()
        .map(|(id, tok)| (tok.to_string(), id as u32))
        .collect();

    if records.is_empty() {
        panic!("build.rs: vocab.txt is empty");
    }

    // Sort lexicographically by token bytes so tokenizer.rs can binary-search.
    // Stable sort is fine — vocab.txt should never contain duplicate tokens,
    // but if it somehow does, keep the lowest (canonical) id by sorting id
    // as the tiebreaker.
    records.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()).then(a.1.cmp(&b.1)));

    let mut out: Vec<u8> = Vec::with_capacity(records.len() * (2 + 4 + MAX_TOKEN_LEN));
    let mut truncated = 0u32;

    for (tok, id) in &records {
        let bytes = tok.as_bytes();
        let len = if bytes.len() > MAX_TOKEN_LEN {
            truncated += 1;
            MAX_TOKEN_LEN
        } else {
            bytes.len()
        };

        out.extend_from_slice(&(len as u16).to_le_bytes());
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&bytes[..len]);
        out.resize(out.len() + (MAX_TOKEN_LEN - len), 0u8);
    }

    if truncated > 0 {
        println!(
            "cargo:warning=build.rs: {truncated} vocab token(s) longer than \
             MAX_TOKEN_LEN={MAX_TOKEN_LEN} were truncated"
        );
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set by cargo");
    let dest = Path::new(&out_dir).join("vocab_table.bin");
    fs::write(&dest, &out).unwrap_or_else(|e| panic!("build.rs: failed writing {dest:?}: {e}"));

    println!(
        "cargo:warning=build.rs: embedded {} vocab entries ({} bytes) from {vocab_path}",
        records.len(),
        out.len()
    );
}
