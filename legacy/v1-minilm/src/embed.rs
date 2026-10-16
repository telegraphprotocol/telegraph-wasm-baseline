//! MiniLM-L6-v2 embedding inference.
//!
//! ## Two build modes
//!
//! ### Projection mode (default, no `real_weights` feature)
//!
//! Uses a deterministic random-projection to convert token IDs to a 384-dim
//! float32 vector. This is NOT semantically meaningful — two sentences about
//! the same topic will not necessarily score high cosine similarity. It is
//! however:
//!   - Structurally correct (same shape, L2-normalised output)
//!   - Deterministic (same input → same output, always)
//!   - Fast (no weight matrix multiplications)
//!   - Useful for end-to-end testing of all surrounding infrastructure
//!
//! ### Real weights mode (`--features real_weights`)
//!
//! Runs a full INT8-quantized MiniLM-L6-v2 encoder forward pass using
//! weights embedded at compile time from `weights/minilm_l6_v2_q8.bin`.
//! Generate with: `python3 scripts/export_minilm_weights.py`.
//!
//! This is a from-scratch re-implementation of a BERT-style encoder, NOT a
//! generic ONNX/tract runtime — it must match MiniLM-L6-v2's actual graph
//! exactly (6 layers, hidden=384, heads=12, intermediate=1536, post-LayerNorm,
//! GELU activation) or pretrained weights will produce numerically arbitrary
//! output despite being "real". Specifically this means:
//!   - multi-head attention, split into NUM_HEADS heads of HEAD_DIM each
//!     (NOT one big hidden_size-wide attention — that's a different, untrained
//!     computation and real weights won't make sense fed through it)
//!   - a separate attention-output dense projection after concatenating heads
//!   - LayerNorm after the attention residual AND after the FFN residual
//!     (BERT/MiniLM is post-LN) — omitting this lets activation magnitude
//!     drift across layers in a way the pretrained weights never saw
//!   - bias vectors on every linear layer (Q, K, V, attn-out, FFN1, FFN2)
//!   - position embeddings + token-type embeddings + an embedding-layer
//!     LayerNorm, added before layer 0 — without position embeddings, mean
//!     pooling is order-invariant and "dog bites man" == "man bites dog"
//!
//! Output is L2-normalised in both modes so cosine similarity = dot product.

extern crate alloc;

use crate::tokenizer::{Encoding, MAX_SEQ_LEN};
use alloc::vec::Vec;

pub const EMBED_DIM: usize = 384;

// Real weights embedded at compile time (activated by feature flag)
#[cfg(feature = "real_weights")]
static WEIGHTS: &[u8] = include_bytes!("../weights/minilm_l6_v2_q8.bin");

/// Run MiniLM inference on `encoding`. Returns L2-normalised float32[384].
pub fn run(encoding: &Encoding) -> [f32; EMBED_DIM] {
    #[cfg(feature = "real_weights")]
    return run_transformer(encoding);

    #[cfg(not(feature = "real_weights"))]
    return run_projection(encoding);
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic random-projection embedding.
///
/// For each output dimension d:
///   output[d] = Σ mask[i] · R[d, token_id[i] % P] · position_decay(i)
///
/// R is a pseudo-random matrix derived from a fixed seed via an LCG, giving
/// stable column vectors per (dim, token_id) pair.
fn run_projection(encoding: &Encoding) -> [f32; EMBED_DIM] {
    const PROJ_COLS: usize = 512;
    const SEED: u64 = 0xDEAD_BEEF_CAFE_1337;

    let mut output = [0f32; EMBED_DIM];

    for d in 0..EMBED_DIM {
        let mut val = 0f32;
        for (i, (&id, &mask)) in encoding
            .input_ids
            .iter()
            .zip(encoding.attention_mask.iter())
            .enumerate()
        {
            if mask == 0 {
                continue;
            }
            let col = (id as usize) % PROJ_COLS;
            let w = lcg_f32(SEED ^ ((d as u64) << 32) ^ (col as u64));
            // Mild position decay so earlier tokens carry slightly more weight
            val += w / (i as f32 + 1.0);
        }
        output[d] = val;
    }

    crate::math::normalise(&mut output);
    output
}

/// LCG-based float in [−1, 1] from a u64 seed.
#[inline]
fn lcg_f32(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Map top 24 bits to [1.0, 2.0) via IEEE 754 mantissa trick, then shift
    let bits = ((x >> 40) as u32) | 0x3F80_0000;
    (f32::from_bits(bits) - 1.5) * 2.0 // → [−1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Real transformer path — matches MiniLM-L6-v2's actual BERT-style encoder
// ─────────────────────────────────────────────────────────────────────────────
//
// Binary format ("MLM2"), all multi-byte ints little-endian:
//
//   magic                 [4]u8    = "MLM2"
//   num_layers             u32     = 6
//   hidden_size            u32     = 384
//   num_heads               u32     = 12
//   intermediate_size      u32     = 1536
//   vocab_size              u32    = 30522
//   num_positions            u32   = MAX_SEQ_LEN (128) — export script must
//                                    slice the checkpoint's position table
//                                    (512 rows) down to the first 128; we
//                                    never index beyond MAX_SEQ_LEN
//
//   ── Embedding block ──
//   word_embeddings         : QLinearTable  (vocab_size × hidden_size)
//   position_embeddings     : QLinearTable  (num_positions × hidden_size)
//   token_type_embeddings   : QLinearTable  (1 × hidden_size) — only row 0,
//                              since this tokenizer always emits
//                              token_type_ids = 0 (single-sentence input)
//   emb_ln_gamma             : f32[hidden_size]  (NOT quantized — LayerNorm
//   emb_ln_beta               : f32[hidden_size]  params are applied every
//                                                   sublayer, so precision
//                                                   matters more than the
//                                                   ~1.5KB/table saved)
//
//   ── Per layer (× num_layers) ──
//   q_weight   : QLinear(hidden_size → hidden_size)     q_bias   : f32[hidden_size]
//   k_weight   : QLinear(hidden_size → hidden_size)     k_bias   : f32[hidden_size]
//   v_weight   : QLinear(hidden_size → hidden_size)     v_bias   : f32[hidden_size]
//   out_weight : QLinear(hidden_size → hidden_size)     out_bias : f32[hidden_size]
//   attn_ln_gamma : f32[hidden_size]   attn_ln_beta : f32[hidden_size]
//   ffn1_weight : QLinear(hidden_size → intermediate_size)  ffn1_bias : f32[intermediate_size]
//   ffn2_weight : QLinear(intermediate_size → hidden_size)  ffn2_bias : f32[hidden_size]
//   out_ln_gamma  : f32[hidden_size]   out_ln_beta  : f32[hidden_size]
//
// Where:
//   QLinearTable(rows × cols) = f32 scale + rows*cols i8 bytes, row-major
//   QLinear(in → out)         = f32 scale + out*in i8 bytes, row-major [out][in]
//     (dequantized value = i8_byte as f32 * scale)

const LN_EPS: f32 = 1e-12;

#[cfg(feature = "real_weights")]
fn run_transformer(encoding: &Encoding) -> [f32; EMBED_DIM] {
    let w = WEIGHTS;
    let mut c = 0usize; // byte cursor through the weights file

    // ── Header ───────────────────────────────────────────────────────────────
    assert_eq!(&w[c..c + 4], b"MLM2", "weights magic mismatch (stale/old-format weights.bin?)");
    c += 4;
    let num_layers = read_u32(w, &mut c) as usize; // 6
    let hidden_size = read_u32(w, &mut c) as usize; // 384
    let num_heads = read_u32(w, &mut c) as usize; // 12
    let intermediate_size = read_u32(w, &mut c) as usize; // 1536
    let vocab_size = read_u32(w, &mut c) as usize; // 30522
    let num_positions = read_u32(w, &mut c) as usize;

    assert_eq!(
        num_positions, MAX_SEQ_LEN,
        "weights.bin position table size doesn't match tokenizer::MAX_SEQ_LEN"
    );
    assert_eq!(hidden_size % num_heads, 0, "hidden_size must divide evenly by num_heads");
    let head_dim = hidden_size / num_heads;

    // ── Embedding layer: word + position + token_type, then LayerNorm ────────
    let word_emb = read_qtable(w, &mut c, vocab_size, hidden_size);
    let pos_emb = read_qtable(w, &mut c, num_positions, hidden_size);
    let type_emb = read_qtable(w, &mut c, 1, hidden_size); // row 0 only

    let emb_ln_gamma = read_f32_vec(w, &mut c, hidden_size);
    let emb_ln_beta = read_f32_vec(w, &mut c, hidden_size);

    // Trim to the real (non-padding) sequence length. `tokenize()` always
    // places real tokens first and padding after, so attention_mask is a
    // prefix of 1s — counting them gives the true length in one pass.
    //
    // Processing all MAX_SEQ_LEN=128 positions unconditionally (as this
    // function originally did) means a 5-word question pays the same
    // attention/FFN cost as a 128-token wall of text. Padding positions were
    // already masked out of attention (-inf before softmax, so they never
    // influence real positions' context vectors) and already excluded from
    // mean_pool — they contributed NOTHING to the output, just wasted cycles.
    // Dropping them is mathematically identical output for real tokens, at a
    // fraction of the cost: attention score cost is O(seq_len²), so this is
    // typically a 5-20x reduction for realistic question/answer lengths.
    let real_len = encoding
        .attention_mask
        .iter()
        .take_while(|&&m| m == 1)
        .count()
        .max(1);
    let attention_mask = &encoding.attention_mask[..real_len]; // now all 1s

    let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(real_len);
    for i in 0..real_len {
        let id = encoding.input_ids[i];
        let mut row = alloc::vec![0f32; hidden_size];
        let w_row = &word_emb[(id as usize % vocab_size) * hidden_size..][..hidden_size];
        let p_row = &pos_emb[(i % num_positions) * hidden_size..][..hidden_size];
        let t_row = &type_emb[0..hidden_size]; // always token_type 0
        for d in 0..hidden_size {
            row[d] = w_row[d] + p_row[d] + t_row[d];
        }
        layer_norm(&mut row, &emb_ln_gamma, &emb_ln_beta);
        hidden.push(row);
    }

    // ── Transformer layers ────────────────────────────────────────────────────
    for _ in 0..num_layers {
        hidden = transformer_layer(
            w,
            &mut c,
            &hidden,
            attention_mask,
            num_heads,
            head_dim,
            hidden_size,
            intermediate_size,
        );
    }

    // ── Mean pooling (all real_len rows are real — no padding left to mask) ───
    let pooled = mean_pool(&hidden, attention_mask);

    // ── Copy into fixed-size array and normalise ──────────────────────────────
    let mut out = [0f32; EMBED_DIM];
    out.copy_from_slice(&pooled[..EMBED_DIM]);
    crate::math::normalise(&mut out);
    out
}

#[cfg(feature = "real_weights")]
fn read_u32(w: &[u8], c: &mut usize) -> u32 {
    let v = u32::from_le_bytes(w[*c..*c + 4].try_into().unwrap());
    *c += 4;
    v
}

#[cfg(feature = "real_weights")]
fn read_f32(w: &[u8], c: &mut usize) -> f32 {
    let v = f32::from_le_bytes(w[*c..*c + 4].try_into().unwrap());
    *c += 4;
    v
}

/// Read `n` raw (non-quantized) f32 values — used for LayerNorm gamma/beta
/// and every bias vector, where precision matters more than the small size
/// saved by quantizing.
#[cfg(feature = "real_weights")]
fn read_f32_vec(w: &[u8], c: &mut usize, n: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..n).map(|_| read_f32(w, c)).collect();
    v
}

/// Read an INT8-quantized embedding table: [scale f32][rows × cols i8 bytes].
/// Returns dequantized f32, row-major [rows][cols], flattened.
#[cfg(feature = "real_weights")]
fn read_qtable(w: &[u8], c: &mut usize, rows: usize, cols: usize) -> Vec<f32> {
    let scale = read_f32(w, c);
    let n = rows * cols;
    let mat: Vec<f32> = w[*c..*c + n].iter().map(|&b| (b as i8) as f32 * scale).collect();
    *c += n;
    mat
}

/// Read INT8 linear weight block: [scale f32][out_dim × in_dim i8 bytes],
/// row-major [out_dim][in_dim] — matches `matmul_row`'s indexing.
#[cfg(feature = "real_weights")]
fn read_linear(w: &[u8], c: &mut usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    read_qtable(w, c, out_dim, in_dim)
}

#[cfg(feature = "real_weights")]
#[allow(clippy::too_many_arguments)]
fn transformer_layer(
    w: &[u8],
    c: &mut usize,
    hidden: &[Vec<f32>],
    attention_mask: &[u32],
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Vec<Vec<f32>> {
    let seq_len = hidden.len();

    // ── Load this layer's weights ─────────────────────────────────────────────
    let q_w = read_linear(w, c, hidden_size, hidden_size);
    let q_b = read_f32_vec(w, c, hidden_size);
    let k_w = read_linear(w, c, hidden_size, hidden_size);
    let k_b = read_f32_vec(w, c, hidden_size);
    let v_w = read_linear(w, c, hidden_size, hidden_size);
    let v_b = read_f32_vec(w, c, hidden_size);
    let out_w = read_linear(w, c, hidden_size, hidden_size);
    let out_b = read_f32_vec(w, c, hidden_size);
    let attn_ln_gamma = read_f32_vec(w, c, hidden_size);
    let attn_ln_beta = read_f32_vec(w, c, hidden_size);
    let ffn1_w = read_linear(w, c, hidden_size, intermediate_size);
    let ffn1_b = read_f32_vec(w, c, intermediate_size);
    let ffn2_w = read_linear(w, c, intermediate_size, hidden_size);
    let ffn2_b = read_f32_vec(w, c, hidden_size);
    let out_ln_gamma = read_f32_vec(w, c, hidden_size);
    let out_ln_beta = read_f32_vec(w, c, hidden_size);

    // ── Project Q, K, V once per position (not recomputed per (i,j) pair) ────
    let q: Vec<Vec<f32>> = hidden.iter().map(|h| matmul_row_bias(h, &q_w, &q_b, hidden_size)).collect();
    let k: Vec<Vec<f32>> = hidden.iter().map(|h| matmul_row_bias(h, &k_w, &k_b, hidden_size)).collect();
    let v: Vec<Vec<f32>> = hidden.iter().map(|h| matmul_row_bias(h, &v_w, &v_b, hidden_size)).collect();

    let scale_f = libm::sqrtf(head_dim as f32);

    // ── Multi-head self-attention ─────────────────────────────────────────────
    let mut attn_out: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let mut context_i = alloc::vec![0f32; hidden_size];

        for h in 0..num_heads {
            let hs = h * head_dim;
            let he = hs + head_dim;
            let q_head = &q[i][hs..he];

            // Attention scores for head h, position i, against all positions j.
            // Padding positions (attention_mask == 0) are masked out with -inf
            // so softmax assigns them ~0 weight — otherwise padding tokens
            // (which are real, if meaningless, KV vectors) would dilute the
            // real tokens' attention distribution.
            let mut scores: Vec<f32> = (0..seq_len)
                .map(|j| {
                    if attention_mask[j] == 0 {
                        f32::NEG_INFINITY
                    } else {
                        crate::math::dot(q_head, &k[j][hs..he]) / scale_f
                    }
                })
                .collect();
            softmax(&mut scores);

            for (j, &wj) in scores.iter().enumerate() {
                if wj == 0.0 {
                    continue;
                }
                let v_head = &v[j][hs..he];
                for (ci, &vi) in context_i[hs..he].iter_mut().zip(v_head.iter()) {
                    *ci += wj * vi;
                }
            }
        }

        attn_out.push(context_i);
    }

    // ── Attention output projection + residual + LayerNorm ───────────────────
    let mut normed1: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let projected = matmul_row_bias(&attn_out[i], &out_w, &out_b, hidden_size);
        let mut row: Vec<f32> = projected
            .iter()
            .zip(hidden[i].iter())
            .map(|(&a, &b)| a + b) // residual
            .collect();
        layer_norm(&mut row, &attn_ln_gamma, &attn_ln_beta);
        normed1.push(row);
    }

    // ── FFN (GELU) + residual + LayerNorm ─────────────────────────────────────
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let mid: Vec<f32> = matmul_row_bias(&normed1[i], &ffn1_w, &ffn1_b, intermediate_size)
            .iter()
            .map(|&x| gelu(x))
            .collect();
        let delta = matmul_row_bias(&mid, &ffn2_w, &ffn2_b, hidden_size);
        let mut row: Vec<f32> = delta
            .iter()
            .zip(normed1[i].iter())
            .map(|(&a, &b)| a + b) // residual
            .collect();
        layer_norm(&mut row, &out_ln_gamma, &out_ln_beta);
        out.push(row);
    }

    out
}

#[cfg(feature = "real_weights")]
fn matmul_row_bias(input: &[f32], weights: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|o| crate::math::dot(input, &weights[o * in_dim..(o + 1) * in_dim]) + bias[o])
        .collect()
}

/// In-place LayerNorm over a single row: normalise to zero-mean/unit-variance,
/// then scale by `gamma` and shift by `beta`. eps = 1e-12 (BERT/MiniLM default).
#[cfg(feature = "real_weights")]
fn layer_norm(row: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = row.len() as f32;
    let mean: f32 = row.iter().sum::<f32>() / n;
    let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / libm::sqrtf(var + LN_EPS);
    for (i, x) in row.iter_mut().enumerate() {
        *x = (*x - mean) * inv_std * gamma[i] + beta[i];
    }
}

#[cfg(feature = "real_weights")]
fn mean_pool(hidden: &[Vec<f32>], mask: &[u32]) -> Vec<f32> {
    let dim = hidden[0].len();
    let mut sum = alloc::vec![0f32; dim];
    let mut count = 0f32;
    for (h, &m) in hidden.iter().zip(mask.iter()) {
        if m == 1 {
            for (s, &v) in sum.iter_mut().zip(h.iter()) {
                *s += v;
            }
            count += 1.0;
        }
    }
    if count > 0.0 {
        sum.iter_mut().for_each(|s| *s /= count);
    }
    sum
}

#[cfg(feature = "real_weights")]
fn softmax(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in v.iter_mut() {
        // NEG_INFINITY - max(which may also be -inf if every position is
        // masked, e.g. a fully-padded row that never happens in practice
        // since position 0 is always attention_mask==1) -> exp(NaN-ish);
        // guard explicitly rather than relying on float semantics here.
        *x = if x.is_finite() || max.is_finite() {
            libm::expf(*x - max)
        } else {
            0.0
        };
        sum += *x;
    }
    if sum > 0.0 {
        v.iter_mut().for_each(|x| *x /= sum);
    }
}

#[cfg(feature = "real_weights")]
fn gelu(x: f32) -> f32 {
    // Approximate GELU used by BERT
    const C: f32 = 0.797_884_6; // sqrt(2/π)
    0.5 * x * (1.0 + libm::tanhf(C * (x + 0.044_715 * x * x * x)))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    #[test]
    fn output_is_384_dims() {
        let enc = tokenize("hello world");
        let vec = run(&enc);
        assert_eq!(vec.len(), EMBED_DIM);
    }

    #[test]
    fn output_is_unit_length() {
        let enc = tokenize("the capital of france is paris");
        let vec = run(&enc);
        let norm = crate::math::l2_norm(&vec);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected unit vector, norm={norm}"
        );
    }

    #[test]
    fn same_input_same_output() {
        let enc_a = tokenize("determinism test");
        let enc_b = tokenize("determinism test");
        let a = run(&enc_a);
        let b = run(&enc_b);
        for (ai, bi) in a.iter().zip(b.iter()) {
            assert_eq!(
                ai.to_bits(),
                bi.to_bits(),
                "outputs differ for identical input"
            );
        }
    }
}
