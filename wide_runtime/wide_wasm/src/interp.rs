//! Wide op-list interpreter — Rust mirror of wide_runtime/interp.py (WideInterp).
//!
//! Runs a parsed WideModel over token ids, keeping every slot in Q32.32 (Vec<i64>), returns
//! the final SCORE slot's scalar as an i64 Q32.32 (the host does the one float conversion).
//! Bit-identical to the Python interpreter by construction (same ops, same order).

#![allow(dead_code)]

#[cfg(target_arch = "wasm32")]
use alloc::{vec, vec::Vec};

use crate::fixed as F;
use crate::ops;
use crate::reader::{WideModel, WideOp};

// opcode integers — match format.py Op / wide_runtime interp
const END: u16 = 0;
const EMBED: u16 = 1;
const QMATMUL: u16 = 2;
const ADD: u16 = 3;
const LAYERNORM: u16 = 4;
const RMSNORM: u16 = 5;
const ACT_GELU: u16 = 6;
const ROPE: u16 = 7;
const POS_ADD: u16 = 8;
const ATTN: u16 = 9;
const SWIGLU: u16 = 10;
const POOL: u16 = 11;
const LAST_TOKEN: u16 = 12;
const SCORE: u16 = 13;
const ACT_SILU: u16 = 15;

const FLAG_CAUSAL: u16 = 1;
const FLAG_MEAN_POOL: u16 = 1 << 2;

/// Convert a real number to Q32.32 with BANKER'S rounding (ties-to-even), matching the Python
/// golden's `fp_from_float` == `int(round(x * FP_ONE))`.
///
/// Rust's `f64::round()` is half-away-from-zero and would differ by 1 ULP on an exact .5 tie.
/// `round_ties_even` is the matching primitive. Native-only: the wasm32 build has no floats
/// at all, which is the determinism guarantee, and is why RoPE tables are baked at conversion
/// time instead of computed here.
#[cfg(not(target_arch = "wasm32"))]
fn round_ties_even_to_q32(x: f64) -> i64 {
    (x * (F::FP_ONE as f64)).round_ties_even() as i64
}

/// A slot value: flat Q32.32 data with a (rows, cols) shape.
#[derive(Clone)]
struct Slot {
    data: Vec<i64>,
    rows: usize,
    cols: usize,
}

pub struct Interp<'a> {
    m: &'a WideModel,
    slots: Vec<Option<Slot>>,
    rope_cos: Vec<i64>,
    rope_sin: Vec<i64>,
    rope_built: bool,
}

impl<'a> Interp<'a> {
    pub fn new(m: &'a WideModel) -> Self {
        // slot ids are small dense ints assigned by the converter; size generously.
        let max_slot = m.ops.iter().map(|o| o.out.max(o.in0).max(o.in1).max(o.arg3 as i32)).max().unwrap_or(0);
        let n = (max_slot + 1).max(1) as usize;
        Interp { m, slots: (0..n).map(|_| None).collect(), rope_cos: Vec::new(),
                 rope_sin: Vec::new(), rope_built: false }
    }

    fn get(&self, id: i32) -> &Slot {
        self.slots[id as usize].as_ref().expect("slot unset")
    }
    fn set(&mut self, id: i32, s: Slot) {
        self.slots[id as usize] = Some(s);
    }
    fn tensor(&self, idx: i32) -> &[i64] {
        &self.m.tensors[idx as usize].data
    }
    fn tshape(&self, idx: i32) -> (usize, usize) {
        let d = &self.m.tensors[idx as usize].dims;
        if d.len() == 2 { (d[0] as usize, d[1] as usize) } else { (d[0] as usize, 1) }
    }

    /// Run over token ids; returns the final score as i64 Q32.32. Panics (traps) on a bad model.
    pub fn run(&mut self, tokens: &[i64]) -> i64 {
        let mut final_slot: i32 = -1;
        for i in 0..self.m.ops.len() {
            let op = self.m.ops[i];
            match op.op {
                END => break,
                EMBED => {
                    let (_v, hidden) = self.tshape(op.weight_idx);
                    let emb = self.tensor(op.weight_idx);
                    let seq = tokens.len();
                    let mut data = vec![0i64; seq * hidden];
                    for (s, &t) in tokens.iter().enumerate() {
                        let base = (t as usize) * hidden;
                        data[s * hidden..s * hidden + hidden]
                            .copy_from_slice(&emb[base..base + hidden]);
                    }
                    self.set(op.out, Slot { data, rows: seq, cols: hidden });
                }
                QMATMUL => self.qmatmul(&op),
                ADD => {
                    let a = self.get(op.in0); let b = self.get(op.in1);
                    let data = ops::residual_add(&a.data, &b.data);
                    let (r, c) = (a.rows, a.cols);
                    self.set(op.out, Slot { data, rows: r, cols: c });
                }
                RMSNORM => {
                    let x = self.get(op.in0);
                    let (r, c) = (x.rows, x.cols);
                    let g = self.tensor(op.weight_idx);
                    let data = ops::rmsnorm(&x.data, r, c, g, op.arg0);
                    self.set(op.out, Slot { data, rows: r, cols: c });
                }
                LAYERNORM => {
                    let x = self.get(op.in0);
                    let (r, c) = (x.rows, x.cols);
                    let g = self.tensor(op.weight_idx);
                    let beta = if op.bias_idx >= 0 { Some(self.tensor(op.bias_idx)) } else { None };
                    let data = ops::layernorm(&x.data, r, c, g, beta, op.arg0);
                    self.set(op.out, Slot { data, rows: r, cols: c });
                }
                ACT_GELU => {
                    let x = self.get(op.in0);
                    let (r, c) = (x.rows, x.cols);
                    let data = ops::act_gelu(&x.data);
                    self.set(op.out, Slot { data, rows: r, cols: c });
                }
                ACT_SILU => {
                    let x = self.get(op.in0);
                    let (r, c) = (x.rows, x.cols);
                    let data = ops::act_silu(&x.data);
                    self.set(op.out, Slot { data, rows: r, cols: c });
                }
                ROPE => self.rope(&op),
                POS_ADD => {
                    let x = self.get(op.in0);
                    let (r, c) = (x.rows, x.cols);
                    let pos = self.tensor(op.weight_idx);
                    let data = ops::residual_add(&x.data, &pos[..r * c]);
                    self.set(op.out, Slot { data, rows: r, cols: c });
                }
                ATTN => self.attn(&op),
                SWIGLU => self.swiglu(&op),
                POOL => {
                    let x = self.get(op.in0);
                    let (r, c) = (x.rows, x.cols);
                    let mut out = vec![0i64; c];
                    if op.flags & FLAG_MEAN_POOL != 0 {
                        for j in 0..c {
                            let mut acc: i128 = 0;
                            for s in 0..r { acc += x.data[s * c + j] as i128; }
                            out[j] = (acc / r as i128) as i64;
                        }
                    } else {
                        out.copy_from_slice(&x.data[..c]);
                    }
                    self.set(op.out, Slot { data: out, rows: 1, cols: c });
                }
                LAST_TOKEN => {
                    let x = self.get(op.in0);
                    let (r, c) = (x.rows, x.cols);
                    let pad = op.arg0;
                    let mut p = r - 1;
                    for s in (0..r).rev() {
                        if tokens[s] != pad { p = s; break; }
                    }
                    let row = x.data[p * c..p * c + c].to_vec();
                    self.set(op.out, Slot { data: row, rows: 1, cols: c });
                }
                SCORE => {
                    self.qmatmul(&op);
                    final_slot = op.out;
                }
                _ => panic!("unimplemented opcode"),
            }
        }
        let s = self.get(final_slot);
        s.data[0]
    }

    fn qmatmul(&mut self, op: &WideOp) {
        let x = self.get(op.in0);
        let (seq, in_dim) = (x.rows, x.cols);
        let (out_dim, w_in) = self.tshape(op.weight_idx);
        debug_assert_eq!(in_dim, w_in);
        let w = self.tensor(op.weight_idx);
        let bias = if op.bias_idx >= 0 { Some(self.tensor(op.bias_idx)) } else { None };
        let data = ops::qmatmul(&x.data, seq, in_dim, w, out_dim, bias);
        self.set(op.out, Slot { data, rows: seq, cols: out_dim });
    }

    fn build_rope(&mut self, head_dim: usize) {
        if self.rope_built { return; }
        let max_seq = self.m.cfg.max_seq as usize;
        let theta = self.m.cfg.rope_theta_milli as f64 / 1000.0;
        // NOTE: cos/sin are precomputed at BUILD time in the .imwq for real models; this
        // fallback recomputes via float which is boundary-only. For the wide runtime the
        // converter should bake the table as a tensor. Kept simple here.
        let half = head_dim / 2;
        let mut cos = vec![0i64; max_seq * head_dim];
        let mut sin = vec![0i64; max_seq * head_dim];
        // float used ONLY here (table bake), not in the forward pass — same as Python golden.
        #[cfg(not(target_arch = "wasm32"))]
        for pos in 0..max_seq {
            for i in 0..half {
                let freq = 1.0f64 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let ang = pos as f64 * freq;
                // ROUNDING CONVENTION: match the Python golden's `fp_from_float`, which is
                // `int(round(x * FP_ONE))` — and Python's `round()` is BANKER'S rounding
                // (ties-to-even: round(0.5)=0, round(2.5)=2). Rust's `f64::round()` is
                // half-away-from-zero, so on any value landing exactly on .5 the two would
                // differ by 1 ULP and propagate a divergence through attention.
                //
                // The golden is canonical here, not this fallback: every converted weight
                // already goes through fp_from_float, so changing Python would alter every
                // artifact rather than fix anything.
                //
                // In practice no entry of a real RoPE table lands on an exact tie (measured:
                // 0 of 512 on the test fixture), and baked tables mean this code does not run
                // for any current artifact — but "currently unreachable and currently
                // tie-free" is not a contract, and this is the last float->Q32.32 conversion
                // left in the Rust tree.
                let c = round_ties_even_to_q32(ang.cos());
                let s = round_ties_even_to_q32(ang.sin());
                cos[pos * head_dim + i] = c; cos[pos * head_dim + i + half] = c;
                sin[pos * head_dim + i] = s; sin[pos * head_dim + i + half] = s;
            }
        }
        let _ = (theta, half);
        self.rope_cos = cos; self.rope_sin = sin; self.rope_built = true;
    }

    fn rope(&mut self, op: &WideOp) {
        let head_dim = op.arg2 as usize;
        let n_heads = op.arg3 as usize;

        // Prefer the cos/sin tables BAKED into the artifact by the converter
        // (weight_idx = cos, aux_idx = sin). The fallback below computes them with
        // f64 trig, which is compiled out on wasm32 — the module is integer-only by
        // construction — so on the real deployment target it left the tables all
        // zeros, `ops::rope` computed x*0 - x*0 = 0, and Q and K were zeroed in every
        // attention layer. Nothing errored and the score stayed plausible; native
        // builds filled the table and passed, which is exactly why `cargo test` never
        // caught it. Any artifact produced by the current converter carries the tables.
        if op.weight_idx >= 0 && op.aux_idx >= 0 {
            let x = self.get(op.in0);
            let (seq, cols) = (x.rows, x.cols);
            let data = {
                let cos = self.tensor(op.weight_idx);
                let sin = self.tensor(op.aux_idx);
                let x = self.get(op.in0);
                ops::rope(&x.data, seq, n_heads, head_dim, cos, sin)
            };
            self.set(op.out, Slot { data, rows: seq, cols });
            return;
        }

        // An artifact from before the tables were baked, running on wasm32, would
        // silently score with attention destroyed. Refuse instead: a rejected artifact is
        // recoverable, a confident wrong score is not. Checked BEFORE build_rope so the
        // wasm32 build does not first allocate a max_seq x head_dim table (8 MB for
        // ModernBERT-base) that it is about to refuse to use.
        #[cfg(target_arch = "wasm32")]
        panic!("this .imwq predates baked RoPE tables and cannot run on wasm32");

        #[cfg(not(target_arch = "wasm32"))]
        self.build_rope(head_dim);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let x = self.get(op.in0);
            let (seq, cols) = (x.rows, x.cols);
            let data = ops::rope(&x.data, seq, n_heads, head_dim, &self.rope_cos, &self.rope_sin);
            self.set(op.out, Slot { data, rows: seq, cols });
        }
    }

    fn attn(&mut self, op: &WideOp) {
        let head_dim = op.arg0 as usize;
        let n_heads = op.arg1 as usize;
        let n_kv = op.arg2 as usize;
        let v_slot = op.arg3 as i32;
        let causal = op.flags & FLAG_CAUSAL != 0;
        let q = self.get(op.in0).clone();
        let k = self.get(op.in1).clone();
        let v = self.get(v_slot).clone();
        let seq = q.rows;
        let scale = {
            // 1/sqrt(head_dim) as Q32.32
            F::fp_inv_sqrt(F::fp_from_int(head_dim as i64))
        };
        let group = n_heads / n_kv;
        let width = n_heads * head_dim;
        let mut out = vec![0i64; seq * width];
        // gather per-head q/k/v slices then run softmax_attn
        for h in 0..n_heads {
            let kv = h / group;
            let mut qh = vec![0i64; seq * head_dim];
            let mut kh = vec![0i64; seq * head_dim];
            let mut vh = vec![0i64; seq * head_dim];
            for s in 0..seq {
                for d in 0..head_dim {
                    qh[s * head_dim + d] = q.data[s * width + h * head_dim + d];
                    kh[s * head_dim + d] = k.data[s * (n_kv * head_dim) + kv * head_dim + d];
                    vh[s * head_dim + d] = v.data[s * (n_kv * head_dim) + kv * head_dim + d];
                }
            }
            let ctx = ops::softmax_attn(&qh, &kh, &vh, seq, head_dim, scale, causal);
            for s in 0..seq {
                for d in 0..head_dim {
                    out[s * width + h * head_dim + d] = ctx[s * head_dim + d];
                }
            }
        }
        self.set(op.out, Slot { data: out, rows: seq, cols: width });
    }

    fn swiglu(&mut self, op: &WideOp) {
        if op.flags & 1 != 0 {
            // encoder GeGLU: in0 = already-activated gate, in1 = up. multiply.
            let g = self.get(op.in0); let u = self.get(op.in1);
            let (r, c) = (g.rows, g.cols);
            let data = ops::elementwise_mul(&g.data, &u.data);
            self.set(op.out, Slot { data, rows: r, cols: c });
        } else {
            // decoder SwiGLU: in0=x, weight_idx=gate, aux_idx=up
            let x = self.get(op.in0).clone();
            let (og, wg_in) = self.tshape(op.weight_idx);
            let _ = wg_in;
            let gate_raw = ops::qmatmul(&x.data, x.rows, x.cols, self.tensor(op.weight_idx), og, None);
            let gate = ops::act_silu(&gate_raw);
            let (ou, _) = self.tshape(op.aux_idx);
            let up = ops::qmatmul(&x.data, x.rows, x.cols, self.tensor(op.aux_idx), ou, None);
            let data = ops::elementwise_mul(&gate, &up);
            self.set(op.out, Slot { data, rows: x.rows, cols: og });
        }
    }
}
