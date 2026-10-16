//! Q32.32 fixed-point core — Rust port of wide_runtime/golden/fixed.py, bit-identical.
//!
//! One wide format everywhere: i64 in Q32.32 (32 fractional bits, FP_ONE = 2^32). Weights and
//! activations are all this type — no int8. Every multiply is (a*b) as i128 >> 32, matching
//! int-llm/fp_math.h and the Python golden exactly. Rust has native i128 so no two-limb backend
//! is needed on wasm32. NO f32/f64 anywhere (the build gate scans for float opcodes).
//!
//! The determinism contract: `fp_mul` uses i128 intermediate then arithmetic right shift by 48.
//! Rust `>>` on a signed integer is an arithmetic shift (floor toward -inf), identical to
//! Python `>>` and C's gcc/clang ASR — so Python golden and this port produce the SAME bits.

#![allow(dead_code)]

// Q32.32 (see golden/fixed.py): 32 integer bits (±2.1e9, ample for ModernBERT's ~27000
// activations that overflowed Q32.32's 15 integer bits) and 32 fractional bits (~2.3e-10).
pub const FP_PRECISION: u32 = 32;
pub const FP_ONE: i64 = 1i64 << FP_PRECISION; // 2^32
pub const FP_HALF: i64 = 1i64 << (FP_PRECISION - 1);
pub const INT64_MAX: i64 = i64::MAX;

#[inline]
pub fn fp_from_int(n: i64) -> i64 {
    n << FP_PRECISION
}

/// a * b in Q32.32: (a*b) >> 32 via i128 intermediate, arithmetic shift (matches golden fp_mul).
#[inline]
pub fn fp_mul(a: i64, b: i64) -> i64 {
    (((a as i128) * (b as i128)) >> FP_PRECISION) as i64
}

/// a / b in Q32.32 = (a<<48)/b, truncating toward zero on the magnitude (matches golden fp_div).
#[inline]
pub fn fp_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let num = ((a as i128).unsigned_abs()) << FP_PRECISION;
    let q = (num / (b as i128).unsigned_abs()) as i128;
    let neg = (a < 0) != (b < 0);
    (if neg { -q } else { q }) as i64
}

/// floor(log2(x)) for positive x  == 63 - clz.
#[inline]
fn msb(x: i64) -> u32 {
    63 - (x as u64).leading_zeros()
}

/// 1/sqrt(x) in Q32.32 via range-reduced Newton — faithful port of golden fp_inv_sqrt
/// (the range reduction is what keeps x86/arm bit-identical; port it exactly).
pub fn fp_inv_sqrt(x: i64) -> i64 {
    if x <= 0 {
        return 0;
    }
    let m = msb(x) as i32;
    let p = FP_PRECISION as i32;
    let k = (m - p) >> 1; // floor((msb-P)/2); tied to the format, not a literal 48 (the old Q16.48 value)
    let xn = if k >= 0 { x >> (2 * k) } else { x << (-2 * k) };

    let msb_n = msb(xn);
    let mut y = if msb_n == FP_PRECISION { FP_ONE } else { FP_HALF };
    let three = fp_from_int(3);
    for _ in 0..8 {
        let y2 = (((y as i128) * (y as i128)) >> FP_PRECISION) as i64;
        let xy2 = (((xn as i128) * (y2 as i128)) >> FP_PRECISION) as i64;
        let factor = three - xy2;
        y = (((y as i128) * (factor as i128)) >> (FP_PRECISION + 1)) as i64;
        if y <= 0 {
            y = 1;
            break;
        }
    }
    if k >= 0 {
        let r = y >> k;
        if r > 0 {
            r
        } else {
            0
        }
    } else {
        let r = (y as i128) << (-k);
        if r > INT64_MAX as i128 {
            INT64_MAX
        } else if r > 0 {
            r as i64
        } else {
            0
        }
    }
}

/// sqrt(x) in Q32.32 = isqrt(x << 48). Integer sqrt via bit-by-bit (exact), matching golden.
pub fn fp_sqrt(x: i64) -> i64 {
    if x <= 0 {
        return 0;
    }
    let n = (x as i128) << FP_PRECISION;
    // integer sqrt of a u128
    let mut op = n as u128;
    let mut res: u128 = 0;
    let mut one: u128 = 1u128 << 126;
    while one > op {
        one >>= 2;
    }
    while one != 0 {
        if op >= res + one {
            op -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }
    res as i64
}

/// exp(x) = (1 + x/2^k)^(2^k), k=14 squarings (golden fp_exp_dyadic).
fn fp_exp_dyadic(x: i64, k: u32) -> i64 {
    let base = FP_ONE + (x >> k);
    let mut r = base;
    for _ in 0..k {
        r = (((r as i128) * (r as i128)) >> FP_PRECISION) as i64;
    }
    r
}

pub fn fp_exp(x: i64) -> i64 {
    if x >= 0 {
        fp_exp_dyadic(x, 14)
    } else {
        let pos = fp_exp_dyadic(-x, 14);
        if pos == 0 {
            0
        } else {
            fp_div(FP_ONE, pos)
        }
    }
}

pub fn fp_safe_exp(x: i64) -> i64 {
    let max_x = fp_from_int(10);
    let x = if x > max_x { max_x } else { x };
    if x < -max_x {
        0
    } else {
        fp_exp(x)
    }
}

pub fn fp_sigmoid(x: i64) -> i64 {
    let ten = fp_from_int(10);
    if x > ten {
        return FP_ONE;
    }
    if x < -ten {
        return 0;
    }
    let exp_neg = fp_safe_exp(-x);
    fp_div(FP_ONE, FP_ONE + exp_neg)
}

pub fn fp_silu(x: i64) -> i64 {
    fp_mul(x, fp_sigmoid(x))
}

/// GELU (tanh approx): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715 x^3))), tanh via sigmoid.
/// Constants are the SAME Q32.32 integers the Python golden uses (bit-identical).
pub fn fp_gelu_tanh(x: i64) -> i64 {
    // fp_from_float(..) under Q32.32 (FP_ONE=2^32) — taken from the Python golden, verified
    // equal so Python and Rust stay bit-identical.
    const C_SQRT_2_PI: i64 = 3426888095; // round(0.7978845608028654 * 2^32)
    const C_044715: i64 = 192049463; // round(0.044715 * 2^32)
    let x3 = fp_mul(fp_mul(x, x), x);
    let inner = fp_mul(C_SQRT_2_PI, x + fp_mul(C_044715, x3));
    let tanh = 2 * fp_sigmoid(2 * inner) - FP_ONE;
    fp_mul(fp_mul(FP_HALF, x), FP_ONE + tanh)
}
