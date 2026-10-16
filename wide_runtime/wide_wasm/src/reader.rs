//! .imwq reader — parses the binary BOPS block + tensor blobs (matches format.py _pack_binary).
//!
//! No JSON parsing: format.py writes a fixed-width binary op-list + config block ('BOPS') the
//! host JSON is only for Python-side debugging. Layout:
//!   'IMWQ' | mjson_len:u32 | mjson bytes (skipped)
//!   'BOPS' | num_ops:u32 | ops[num_ops] (each 60B) | config (10 x i32 = 40B)
//!   num_blobs:u32 | per blob: ndim:u32, dims[ndim]:u64, n:u64, i64[n] little-endian

#![allow(dead_code)]

#[cfg(target_arch = "wasm32")]
use alloc::vec::Vec;

#[derive(Clone, Copy, Default)]
pub struct WideOp {
    pub op: u16,
    pub flags: u16,
    pub in0: i32,
    pub in1: i32,
    pub out: i32,
    pub weight_idx: i32,
    pub bias_idx: i32,
    pub aux_idx: i32,
    pub arg0: i64,
    pub arg1: i64,
    pub arg2: i64,
    pub arg3: i64,
}

#[derive(Clone, Copy, Default)]
pub struct Config {
    pub arch: i32, // 1=encoder 2=decoder
    pub hidden: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub inter: i32,
    pub max_seq: i32,
    pub rope_theta_milli: i32,
    pub rope: i32,
}

pub struct Tensor {
    pub dims: Vec<u64>,
    pub data: Vec<i64>,
}

pub struct WideModel {
    pub ops: Vec<WideOp>,
    pub cfg: Config,
    pub tensors: Vec<Tensor>,
}

struct Cur<'a> {
    b: &'a [u8],
    off: usize,
}
impl<'a> Cur<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.off..self.off + 4].try_into().unwrap());
        self.off += 4;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes(self.b[self.off..self.off + 2].try_into().unwrap());
        self.off += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.b[self.off..self.off + 4].try_into().unwrap());
        self.off += 4;
        v
    }
    fn i64(&mut self) -> i64 {
        let v = i64::from_le_bytes(self.b[self.off..self.off + 8].try_into().unwrap());
        self.off += 8;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.off..self.off + 8].try_into().unwrap());
        self.off += 8;
        v
    }
    fn tag(&mut self, want: &[u8; 4]) -> bool {
        let ok = &self.b[self.off..self.off + 4] == want;
        self.off += 4;
        ok
    }
}

impl WideModel {
    /// Parse a full .imwq image. Returns None on a malformed header (fail closed).
    pub fn parse(bytes: &[u8]) -> Option<WideModel> {
        let mut c = Cur { b: bytes, off: 0 };
        if !c.tag(b"IMWQ") {
            return None;
        }
        let mlen = c.u32() as usize;
        c.off += mlen; // skip JSON meta
        if !c.tag(b"BOPS") {
            return None;
        }
        let n_ops = c.u32() as usize;
        let mut ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            let op = c.u16();
            let flags = c.u16();
            let in0 = c.i32();
            let in1 = c.i32();
            let out = c.i32();
            let weight_idx = c.i32();
            let bias_idx = c.i32();
            let aux_idx = c.i32();
            let arg0 = c.i64();
            let arg1 = c.i64();
            let arg2 = c.i64();
            let arg3 = c.i64();
            ops.push(WideOp { op, flags, in0, in1, out, weight_idx, bias_idx, aux_idx,
                              arg0, arg1, arg2, arg3 });
        }
        let cfg = Config {
            arch: c.i32(), hidden: c.i32(), layers: c.i32(), heads: c.i32(), kv_heads: c.i32(),
            head_dim: c.i32(), inter: c.i32(), max_seq: c.i32(), rope_theta_milli: c.i32(),
            rope: c.i32(),
        };
        let nb = c.u32() as usize;
        let mut tensors = Vec::with_capacity(nb);
        for _ in 0..nb {
            let ndim = c.u32() as usize;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(c.u64());
            }
            let n = c.u64() as usize;
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                data.push(c.i64());
            }
            tensors.push(Tensor { dims, data });
        }
        Some(WideModel { ops, cfg, tensors })
    }
}
