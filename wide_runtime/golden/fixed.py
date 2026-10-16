"""Q32.32 fixed-point primitives — Python golden reference, ported from int-llm/fp_math.h.

This is the WIDE-integer core that replaces the int8 activation path. Everything is an
int64-valued Python int in Q32.32: 16 integer bits, 32 fractional bits, so a real number r is
stored as round(r * 2**32). Weights AND activations use this one format end-to-end — no int8
requantisation between layers, which is what destroys ranking in the int8 path.

Why this fixes the ranking bug: int8 gives 256 levels per value, and the reward signal (fine
differences separating good vs bad answers) is smaller than the accumulated int8 rounding
error across ~22 layers. Q32.32 carries 32 fractional bits (~3.5e-15 resolution), so the
signal survives every layer — exactly as int-llm demonstrated.

Determinism contract (must match the Rust port bit-for-bit — imw_wasm Stage 5):
  - fp_mul is (a*b) arithmetic-shifted right by 32. In C this is an int128 intermediate; in
    Python int is arbitrary precision so it is exact. The Rust side uses the two-limb int128
    backend that computes the SAME bits.
  - Arithmetic right shift here means floor-division toward -inf for the shift (>> on Python
    ints already floors toward -inf), matching int-llm's fp_w_asr (native signed >>, which
    gcc/clang define as arithmetic shift). We replicate that exactly with Python >>.
  - Fixed operation order; no float anywhere in the compute path (floats appear only at the
    conversion boundary and in display/tests).

Ported faithfully from int-llm/fp_math.h — function names mirror the C for cross-checking:
  fp_mul, fp_div, fp_exp/fp_safe_exp, fp_inv_sqrt, fp_sqrt, fp_sigmoid, fp_silu.
"""

from __future__ import annotations

# Q32.32: 32 integer bits, 32 fractional bits. Chosen over Q32.32 because real reward-model
# activations (ModernBERT) reach ~27000 — Q32.32's 15 integer bits (±32767) left no headroom
# and the intermediate products wrapped i64 in the Rust/wasm runtime (all scores collapsed to
# 0). Q32.32 gives ±2.1e9 integer range (ample) and 2^-32 ~ 2.3e-10 resolution (still far finer
# than float32's ~1e-7). fp_mul/matmul accumulate in i128, so the wide intermediates are safe.
FP_PRECISION = 32
FP_ONE = 1 << FP_PRECISION                 # 2**32 == 1.0
FP_HALF = 1 << (FP_PRECISION - 1)
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)

# Q32.32 integer part is 31 bits + sign.
FP_MAX_INT = (1 << 31) - 1


def _wrap_i64(v: int) -> int:
    """Wrap a Python int into i64 two's-complement — MATCHES Rust/wasm i64 overflow exactly.
    The golden MUST wrap like the real runtime, or overflow bugs hide in Python (arbitrary
    precision) and only surface in wasm — which is exactly what let the Q32.32 overflow ship.
    Applied to every value that lands back in an i64 slot (fp_mul result, adds, etc.)."""
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >= (1 << 63) else v


def fp_from_int(n: int) -> int:
    """Integer n as Q32.32 (n << 48). |n| must be <= 32767."""
    return int(n) << FP_PRECISION


def fp_from_float(x: float) -> int:
    """Real x -> Q32.32 (round-to-nearest). CONVERSION/boundary use only, never in compute."""
    return int(round(float(x) * FP_ONE))


def to_float(q: int) -> float:
    """Q32.32 -> float. DISPLAY/TEST only — never feed back into the integer path."""
    return q / FP_ONE


# --------------------------------------------------------------------------- #
#  Core arithmetic — the two operations everything else is built on
# --------------------------------------------------------------------------- #

def fp_mul(a: int, b: int) -> int:
    """a * b in Q32.32. (a*b) >> 32 with arithmetic (floor) shift, matching fp_math.h fp_mul.

    Python int is exact and arbitrary precision, so the int128 intermediate is free and exact.
    Python's >> on a negative int floors toward -inf — identical to C's arithmetic right shift
    on the gcc/clang backends int-llm targets. This is the single most-used op; its rounding
    (truncate toward -inf) defines the whole runtime's determinism. The result is wrapped to
    i64 so the golden matches the Rust/wasm runtime bit-for-bit even on overflow (a real value
    exceeding the Q32.32 range wraps identically in both, rather than silently staying exact
    in Python)."""
    return _wrap_i64((a * b) >> FP_PRECISION)


def fp_div(a: int, b: int) -> int:
    """a / b in Q32.32 = (a << 48) / b, truncating toward zero on the magnitude (matches
    fp_math.h fp_div, which shifts the magnitude then reapplies the sign to avoid the UB of
    left-shifting a negative and to keep INT64_MIN safe)."""
    if b == 0:
        return 0
    num = abs(a) << FP_PRECISION
    q = num // abs(b)                       # magnitude, truncate toward zero
    neg = (a < 0) != (b < 0)
    return -q if neg else q


def fp_abs(a: int) -> int:
    return -a if a < 0 else a


# --------------------------------------------------------------------------- #
#  Inverse square root — range-reduced Newton (fp_math.h:658). Used by RMSNorm/LayerNorm.
# --------------------------------------------------------------------------- #

def _bit_length_minus1(x: int) -> int:
    """msb index = 63 - clz(x) for a positive int (i.e. floor(log2(x)))."""
    return x.bit_length() - 1


def fp_inv_sqrt(x: int) -> int:
    """1/sqrt(x) in Q32.32 via direct Newton y = y*(3 - x*y^2)/2, range-reduced first.

    Faithful port of fp_math.h fp_inv_sqrt (the range reduction is the fix for a real
    arm64-vs-x86 divergence — porting it exactly keeps our two runtimes bit-identical). Reduce
    x = xn * 2^(2k) with xn in [2^32, 2^34) (real [1,4)); 1/sqrt(x) = 1/sqrt(xn) * 2^(-k).
    Saturates to INT64_MAX when the result leaves Q32.32 range.
    """
    if x <= 0:
        return 0
    msb = _bit_length_minus1(x)
    # normalise xn into [2^FP_PRECISION, 2^(FP_PRECISION+2)) (real [1,4)); the octave target
    # is FP_PRECISION, not the literal 48 — that hardcoded Q32.32 value silently broke inv_sqrt
    # when the format changed to Q32.32 (returned ~0). Keep it tied to the format.
    k = (msb - FP_PRECISION) >> 1            # floor((msb-P)/2); may be negative (arith shift)
    xn = (x >> (2 * k)) if k >= 0 else (x << (-2 * k))

    msb_n = _bit_length_minus1(xn)
    y = FP_ONE if msb_n == FP_PRECISION else FP_HALF   # sub-octave initial guess -> Newton basin
    three = fp_from_int(3)
    for _ in range(8):
        y2 = (y * y) >> FP_PRECISION         # <= ~2^32
        xy2 = (xn * y2) >> FP_PRECISION      # <= ~2^34
        factor = three - xy2
        y = (y * factor) >> (FP_PRECISION + 1)   # (y*factor >>32) >>1
        if y <= 0:
            y = 1
            break

    if k >= 0:
        r = y >> k
        return r if r > 0 else 0
    r = y << (-k)
    if r > INT64_MAX:
        return INT64_MAX
    return r if r > 0 else 0


def fp_sqrt(x: int) -> int:
    """sqrt(x) in Q32.32. sqrt(x_real) = sqrt(x_q / 2^48) = isqrt(x_q << 48). Exact integer
    sqrt via Python's math.isqrt on the widened operand (matches fp_math.h fp_sqrt's result;
    Newton+CLZ in C, exact isqrt here — same value)."""
    if x <= 0:
        return 0
    from math import isqrt
    return isqrt(x << FP_PRECISION)


# --------------------------------------------------------------------------- #
#  Exponential — dyadic squaring (fp_math.h:714). Used by softmax and sigmoid/SiLU.
# --------------------------------------------------------------------------- #

def fp_exp_dyadic(x: int, k: int = 14) -> int:
    """exp(x) = (1 + x/2^k)^(2^k) via k repeated squarings. k=14 gives ~42-bit accuracy."""
    base = FP_ONE + (x >> k)
    result = base
    for _ in range(k):
        result = (result * result) >> FP_PRECISION
    return result


def fp_exp(x: int) -> int:
    """exp(x) for any sign; negatives via reciprocal (matches fp_math.h fp_exp)."""
    if x >= 0:
        return fp_exp_dyadic(x, 14)
    pos = fp_exp_dyadic(-x, 14)
    if pos == 0:
        return 0
    return fp_div(FP_ONE, pos)


def fp_safe_exp(x: int) -> int:
    """exp(x) clamped to [-10, 10] for softmax/sigmoid safety (fp_math.h fp_safe_exp).
    exp(10)~22026 fits Q32.32's 15 integer bits; below -10, exp<4.5e-5 -> return 0."""
    max_x = fp_from_int(10)
    if x > max_x:
        x = max_x
    if x < -max_x:
        return 0
    return fp_exp(x)


# --------------------------------------------------------------------------- #
#  Sigmoid / SiLU — exp + div (fp_math.h:911). Replaces the int8 activation LUT.
# --------------------------------------------------------------------------- #

def fp_sigmoid(x: int) -> int:
    """1/(1+e^-x) in Q32.32, saturating past |x|>10 (fp_math.h fp_sigmoid)."""
    ten = fp_from_int(10)
    if x > ten:
        return FP_ONE
    if x < -ten:
        return 0
    exp_neg = fp_safe_exp(-x)
    return fp_div(FP_ONE, FP_ONE + exp_neg)


def fp_silu(x: int) -> int:
    """SiLU(x) = x * sigmoid(x) in Q32.32 (fp_math.h fp_silu). The SwiGLU gate activation."""
    return fp_mul(x, fp_sigmoid(x))


def fp_gelu_tanh(x: int) -> int:
    """GELU (tanh approximation) in Q32.32: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).

    int-llm is Llama-only (SiLU) so fp_math.h has no GELU; ModernBERT needs it. Built from the
    same computed primitives (tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1). All Q32.32, no
    LUT, deterministic.
    """
    # constants in Q32.32 (boundary conversion, done once)
    C_SQRT_2_PI = fp_from_float(0.7978845608028654)   # sqrt(2/pi)
    C_044715 = fp_from_float(0.044715)
    x3 = fp_mul(fp_mul(x, x), x)
    inner = fp_mul(C_SQRT_2_PI, x + fp_mul(C_044715, x3))
    # tanh(z) = 2*sigmoid(2z) - 1
    tanh = 2 * fp_sigmoid(2 * inner) - FP_ONE
    return fp_mul(fp_mul(FP_HALF, x), FP_ONE + tanh)
