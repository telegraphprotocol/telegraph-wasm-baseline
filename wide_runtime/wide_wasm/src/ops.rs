//! Wide Q32.32 transformer ops — Rust port of wide_runtime/golden/ops.py, bit-identical.
//!
//! All values i64 Q32.32; accumulation uses i128 then one >>32 (matmul, attention). No int8,
//! no requant between layers, no LUTs (activations computed via fixed.rs). Flat Vec<i64>
//! buffers, row-major [rows*cols]. Accumulation order is fixed/sequential (determinism).

#![allow(dead_code)]

use crate::fixed::{self as F, FP_PRECISION};

#[cfg(target_arch = "wasm32")]
use alloc::vec;
#[cfg(target_arch = "wasm32")]
use alloc::vec::Vec;

/// y[s,o] = ( Σ_k x[s,k]*w[o,k] ) >> 32. x:[seq,in] w:[out,in] -> [seq,out]. Optional Q32.32
/// bias added after the shift. Raw i128 accumulation, single shift (int-llm matmul_vec).
pub fn qmatmul(x: &[i64], seq: usize, in_dim: usize, w: &[i64], out_dim: usize,
               bias: Option<&[i64]>) -> Vec<i64> {
    let mut out = vec![0i64; seq * out_dim];
    for s in 0..seq {
        let xr = &x[s * in_dim..s * in_dim + in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc: i128 = 0;
            for k in 0..in_dim {
                acc += (xr[k] as i128) * (wr[k] as i128); // exact, order fixed
            }
            let mut y = (acc >> FP_PRECISION) as i64;
            if let Some(b) = bias {
                y += b[o];
            }
            out[s * out_dim + o] = y;
        }
    }
    out
}

pub fn act_gelu(x: &[i64]) -> Vec<i64> {
    x.iter().map(|&v| F::fp_gelu_tanh(v)).collect()
}

pub fn act_silu(x: &[i64]) -> Vec<i64> {
    x.iter().map(|&v| F::fp_silu(v)).collect()
}

pub fn residual_add(a: &[i64], b: &[i64]) -> Vec<i64> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

pub fn elementwise_mul(a: &[i64], b: &[i64]) -> Vec<i64> {
    a.iter().zip(b).map(|(&x, &y)| F::fp_mul(x, y)).collect()
}

/// RMSNorm over the last dim (int-llm rmsnorm): mean_sq=(Σx^2>>32)/dim; inv=fp_inv_sqrt(ms+eps);
/// out = x*inv*gamma.
pub fn rmsnorm(x: &[i64], seq: usize, dim: usize, gamma: &[i64], eps_q: i64) -> Vec<i64> {
    let mut out = vec![0i64; seq * dim];
    for s in 0..seq {
        let row = &x[s * dim..s * dim + dim];
        let mut sum_sq: i128 = 0;
        for &v in row {
            sum_sq += (v as i128) * (v as i128);
        }
        let mean_sq = ((sum_sq >> FP_PRECISION) as i64) / (dim as i64);
        let inv = F::fp_inv_sqrt(mean_sq + eps_q);
        for j in 0..dim {
            out[s * dim + j] = F::fp_mul(F::fp_mul(row[j], inv), gamma[j]);
        }
    }
    out
}

/// LayerNorm over the last dim: subtract mean, /std, affine (beta optional).
pub fn layernorm(x: &[i64], seq: usize, dim: usize, gamma: &[i64], beta: Option<&[i64]>,
                 eps_q: i64) -> Vec<i64> {
    let mut out = vec![0i64; seq * dim];
    for s in 0..seq {
        let row = &x[s * dim..s * dim + dim];
        let mut sum: i128 = 0;
        for &v in row {
            sum += v as i128;
        }
        let mean = (sum / (dim as i128)) as i64;
        let mut var_sum: i128 = 0;
        for &v in row {
            let c = v - mean;
            var_sum += ((c as i128) * (c as i128)) >> FP_PRECISION;
        }
        let var = (var_sum >> 0) as i64 / (dim as i64);
        let inv = F::fp_inv_sqrt(var + eps_q);
        for j in 0..dim {
            let c = row[j] - mean;
            let normed = F::fp_mul(c, inv);
            let mut val = F::fp_mul(normed, gamma[j]);
            if let Some(b) = beta {
                val += b[j];
            }
            out[s * dim + j] = val;
        }
    }
    out
}

/// Single-head attention, all Q32.32. q,k,v: [seq, hd]. scale_q = 1/sqrt(hd) as Q32.32.
/// scores = (q·k >>32)*scale; softmax(max-subtract, fp_safe_exp, fp_div); ctx = Σ w*v.
pub fn softmax_attn(q: &[i64], k: &[i64], v: &[i64], seq: usize, hd: usize,
                    scale_q: i64, causal: bool) -> Vec<i64> {
    let neg = -F::fp_from_int(30);
    let mut out = vec![0i64; seq * hd];
    let mut scores = vec![0i64; seq];
    let mut exps = vec![0i64; seq];
    for i in 0..seq {
        for j in 0..seq {
            if causal && j > i {
                scores[j] = neg;
                continue;
            }
            let mut dot: i128 = 0;
            for d in 0..hd {
                dot += (q[i * hd + d] as i128) * (k[j * hd + d] as i128);
            }
            scores[j] = F::fp_mul((dot >> FP_PRECISION) as i64, scale_q);
        }
        let mut m = scores[0];
        for j in 1..seq {
            if scores[j] > m {
                m = scores[j];
            }
        }
        let mut denom: i128 = 0;
        for j in 0..seq {
            exps[j] = F::fp_safe_exp(scores[j] - m);
            denom += exps[j] as i128;
        }
        let denom = if denom <= 0 { 1 } else { denom as i64 };
        for d in 0..hd {
            let mut acc: i64 = 0;
            for j in 0..seq {
                let w = F::fp_div(exps[j], denom);
                acc += F::fp_mul(w, v[j * hd + d]);
            }
            out[i * hd + d] = acc;
        }
    }
    out
}

/// RoPE rotate-half on [seq, n_heads*hd] with precomputed Q32.32 cos/sin [seq, hd].
pub fn rope(x: &[i64], seq: usize, n_heads: usize, hd: usize, cos: &[i64], sin: &[i64]) -> Vec<i64> {
    let half = hd / 2;
    let width = n_heads * hd;
    let mut out = x.to_vec();
    for s in 0..seq {
        for h in 0..n_heads {
            let base = s * width + h * hd;
            for i in 0..half {
                let x1 = x[base + i];
                let x2 = x[base + half + i];
                let c = cos[s * hd + i];
                let sn = sin[s * hd + i];
                out[base + i] = F::fp_mul(x1, c) - F::fp_mul(x2, sn);
                out[base + half + i] = F::fp_mul(x2, c) + F::fp_mul(x1, sn);
            }
        }
    }
    out
}
