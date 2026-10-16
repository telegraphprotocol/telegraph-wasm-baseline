//! Pure math utilities for the scoring pipeline.
//!
//! All float operations use `libm` (pure-Rust software implementation) so
//! results are IEEE 754 bit-identical across every platform and every
//! validator node — guaranteed determinism with no host libm involvement.

/// Cosine similarity between two equal-length float32 slices.
///
/// Returns a value in [0, 1]. Negative similarity (opposite-direction vectors)
/// is clamped to 0 — for the validator this means "no match" rather than
/// "anti-match", which is the correct semantic for scoring miner answers.
///
/// Returns 0 for zero vectors rather than NaN.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine: dimension mismatch");

    let mut dot = 0f32;
    let mut norm_a = 0f32;
    let mut norm_b = 0f32;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let sim = dot / (libm::sqrtf(norm_a) * libm::sqrtf(norm_b));
    clamp01(sim)
}

/// Logistic sigmoid σ(x) = 1 / (1 + e^{−x}).
/// Used for smooth length-quality scoring.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

/// Clamp `v` into [0, 1].
#[inline]
pub fn clamp01(v: f32) -> f32 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 (Euclidean) norm.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    libm::sqrtf(v.iter().map(|&x| x * x).sum::<f32>())
}

/// Normalise a slice to unit L2 length in-place. No-ops on zero vectors.
#[inline]
pub fn normalise(v: &mut [f32]) {
    let n = l2_norm(v);
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

// ── Tests (run with `cargo test`, native target) ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = [1.0f32, 2.0, 3.0];
        let s = cosine(&v, &v);
        assert!((s - 1.0).abs() < 1e-6, "identical vectors → 1.0, got {s}");
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_clamped_to_zero() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = [0.0f32, 0.0];
        let b = [1.0f32, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalise_unit_length() {
        let mut v = [3.0f32, 4.0]; // norm = 5
        normalise(&mut v);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
    }
}
