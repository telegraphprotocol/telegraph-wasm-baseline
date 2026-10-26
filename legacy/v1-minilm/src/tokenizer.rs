//! Tokenizer for MiniLM-L6-v2 (BERT-style).
//!
//! Produces the three arrays a BERT model expects:
//!   input_ids      — token IDs, padded to MAX_SEQ_LEN
//!   attention_mask — 1 for real tokens, 0 for padding
//!   token_type_ids — all 0 (single-sentence inputs)
//!
//! ## Vocabulary mapping
//!
//! ### Projection mode (default, no `real_weights` feature)
//!
//! One token per word, via FNV-1a hashing folded into [103, VOCAB_SIZE) —
//! stable and deterministic, but not a real vocabulary lookup. Matches
//! `embed.rs`'s projection fallback, which doesn't care what the IDs mean.
//!
//! ### Real weights mode (`--features real_weights`)
//!
//! Real WordPiece tokenization (greedy longest-match-first, BERT-style) with
//! IDs resolved via binary search against a vocab table embedded at compile
//! time by `build.rs` (from `vocab.txt`, same file `export_minilm_weights.py`
//! writes alongside the weights). A word may produce multiple sub-tokens
//! (e.g. `"playing"` → `"play"` + `"##ing"`); a word with no valid WordPiece
//! split at all becomes a single `[UNK]`.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// BERT special token IDs (standard uncased vocab)
pub const TOKEN_CLS: u32 = 101;
pub const TOKEN_SEP: u32 = 102;
pub const TOKEN_PAD: u32 = 0;
pub const TOKEN_UNK: u32 = 100;

/// MiniLM-L6-v2 maximum sequence length (model hard limit)
pub const MAX_SEQ_LEN: usize = 128;

/// Standard BERT uncased vocabulary size
const VOCAB_SIZE: u32 = 30_522;

/// Output of the tokenizer.
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Vec<u32>,
}

/// Tokenise `text` into a padded BERT encoding of length `MAX_SEQ_LEN`.
pub fn tokenize(text: &str) -> Encoding {
    let mut ids: Vec<u32> = Vec::with_capacity(MAX_SEQ_LEN);
    ids.push(TOKEN_CLS);

    'words: for word in split_words(text) {
        for id in token_ids_for_word(&word) {
            if ids.len() >= MAX_SEQ_LEN - 1 {
                break 'words; // reserve slot for [SEP]
            }
            ids.push(id);
        }
    }

    ids.push(TOKEN_SEP);

    let seq_len = ids.len();
    ids.resize(MAX_SEQ_LEN, TOKEN_PAD);

    let attention_mask: Vec<u32> = (0..MAX_SEQ_LEN)
        .map(|i| if i < seq_len { 1 } else { 0 })
        .collect();

    let token_type_ids = alloc::vec![0u32; MAX_SEQ_LEN];

    Encoding {
        input_ids: ids,
        attention_mask,
        token_type_ids,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection-mode word → id (FNV hash, one id per word)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "real_weights"))]
fn token_ids_for_word(word: &str) -> Vec<u32> {
    alloc::vec![word_to_id(word)]
}

/// Map a word to a pseudo-vocab token ID via FNV-1a, folded into
/// [103, VOCAB_SIZE) — avoids all special token IDs. Only used in
/// projection mode; real_weights mode uses `wordpiece_tokenize` instead.
#[cfg(not(feature = "real_weights"))]
fn word_to_id(word: &str) -> u32 {
    let h = fnv1a(word.as_bytes());
    (h % (VOCAB_SIZE as u64 - 103) + 103) as u32
}

#[cfg(not(feature = "real_weights"))]
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Real-weights mode: WordPiece tokenization against the embedded vocab
// ─────────────────────────────────────────────────────────────────────────────
//
// vocab_table.bin record format (written by build.rs), fixed-size records
// so lookup is a plain binary search with no separate offset table:
//   [u16 len][u32 token_id][bytes, padded to MAX_TOKEN_LEN with 0x00]
// sorted lexicographically by token bytes.

#[cfg(feature = "real_weights")]
const MAX_TOKEN_LEN: usize = 48; // must match build.rs's MAX_TOKEN_LEN
#[cfg(feature = "real_weights")]
const RECORD_SIZE: usize = 2 + 4 + MAX_TOKEN_LEN;
#[cfg(feature = "real_weights")]
const MAX_INPUT_CHARS_PER_WORD: usize = 200; // BERT default — longer "words"

// (e.g. garbage/URLs with no whitespace) are treated as unknown rather than
// spending O(len²) trying to WordPiece-split them.

#[cfg(feature = "real_weights")]
static VOCAB_TABLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vocab_table.bin"));

#[cfg(feature = "real_weights")]
fn token_ids_for_word(word: &str) -> Vec<u32> {
    wordpiece_tokenize(word)
}

/// Greedy longest-match-first WordPiece, matching BERT's reference algorithm:
/// for each word, repeatedly take the longest prefix (continuation pieces
/// after the first are prefixed `##`) found in the vocab; if any piece can't
/// be matched, the whole word becomes a single `[UNK]`.
#[cfg(feature = "real_weights")]
fn wordpiece_tokenize(word: &str) -> Vec<u32> {
    let chars: Vec<char> = word.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    if chars.len() > MAX_INPUT_CHARS_PER_WORD {
        return alloc::vec![TOKEN_UNK];
    }

    let mut sub_tokens: Vec<u32> = Vec::new();
    let mut start = 0usize;

    while start < chars.len() {
        let mut end = chars.len();
        let mut found: Option<u32> = None;

        while start < end {
            let piece: String = if start == 0 {
                chars[start..end].iter().collect()
            } else {
                let mut s = String::from("##");
                s.extend(chars[start..end].iter());
                s
            };
            if let Some(id) = lookup_token(&piece) {
                found = Some(id);
                break;
            }
            end -= 1;
        }

        match found {
            Some(id) => {
                sub_tokens.push(id);
                start = end;
            }
            // No valid split from `start` onward — BERT's rule is the whole
            // word becomes [UNK], not just the unmatched remainder.
            None => return alloc::vec![TOKEN_UNK],
        }
    }

    sub_tokens
}

/// Binary search `VOCAB_TABLE` for `tok`. O(log vocab_size), no allocation.
#[cfg(feature = "real_weights")]
fn lookup_token(tok: &str) -> Option<u32> {
    let needle = tok.as_bytes();
    if needle.len() > MAX_TOKEN_LEN {
        return None;
    }

    let n = VOCAB_TABLE.len() / RECORD_SIZE;
    let mut lo = 0usize;
    let mut hi = n;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let rec = &VOCAB_TABLE[mid * RECORD_SIZE..(mid + 1) * RECORD_SIZE];
        let len = u16::from_le_bytes([rec[0], rec[1]]) as usize;
        let rec_tok = &rec[6..6 + len];

        match needle.cmp(rec_tok) {
            core::cmp::Ordering::Less => hi = mid,
            core::cmp::Ordering::Greater => lo = mid + 1,
            core::cmp::Ordering::Equal => {
                return Some(u32::from_le_bytes([rec[2], rec[3], rec[4], rec[5]]));
            }
        }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Word splitting (shared by both modes)
// ─────────────────────────────────────────────────────────────────────────────

/// Split `text` into lowercase alphanumeric word tokens.
fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.push(if ch.is_ascii_uppercase() {
                (ch as u8 + 32) as char
            } else {
                ch
            });
        } else if !current.is_empty() {
            words.push(core::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

// ── Tests ──────────────────────────────────────────────────────────────────
// These run against the default (projection) build — real_weights mode
// needs an actual vocab_table.bin from a real export, so it's exercised by
// integration testing after `export_minilm_weights.py` has been run, not by
// `cargo test` in this crate directly.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_padded_to_max_seq_len() {
        let enc = tokenize("hello world");
        assert_eq!(enc.input_ids.len(), MAX_SEQ_LEN);
        assert_eq!(enc.attention_mask.len(), MAX_SEQ_LEN);
        assert_eq!(enc.token_type_ids.len(), MAX_SEQ_LEN);
    }

    #[test]
    fn starts_with_cls() {
        let enc = tokenize("test");
        assert_eq!(enc.input_ids[0], TOKEN_CLS);
    }

    #[test]
    fn last_real_token_is_sep() {
        let enc = tokenize("test input");
        let last = enc.attention_mask.iter().rposition(|&m| m == 1).unwrap();
        assert_eq!(enc.input_ids[last], TOKEN_SEP);
    }

    #[test]
    fn same_word_same_id() {
        let a = tokenize("paris");
        let b = tokenize("paris");
        assert_eq!(a.input_ids[1], b.input_ids[1]);
    }

    #[test]
    fn long_input_truncated_to_max_seq_len() {
        let long = "word ".repeat(200);
        let enc = tokenize(&long);
        assert_eq!(enc.input_ids.len(), MAX_SEQ_LEN);
    }

    #[test]
    fn attention_mask_matches_real_tokens() {
        let enc = tokenize("hello");
        // [CLS] hello [SEP] → 3 real tokens (projection mode: 1 id per word)
        let real_cnt = enc.attention_mask.iter().filter(|&&m| m == 1).count();
        assert_eq!(real_cnt, 3);
    }
}
