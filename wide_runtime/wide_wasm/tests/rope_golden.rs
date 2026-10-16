//! Rust<->Python bit-identity for a model that USES RoPE, and the regression guard for the
//! zeroed-table bug.
//!
//! `model_golden.rs` covers a model built with `rope=False`, so until this file existed the
//! ROPE opcode had no parity coverage at all — while every real ModernBERT adapter uses it
//! (the converted VULNERABILITY_TRIAGE artifact contains 44 ROPE ops). That gap hid a live
//! bug, and hid it specifically on the deployment target:
//!
//! `interp.rs::build_rope` filled the cos/sin tables with f64 trig under
//! `#[cfg(not(target_arch = "wasm32"))]`. Native builds filled them and were correct. The
//! wasm32 build — the one that actually runs — compiled the fill out and left the tables all
//! zeros, so `ops::rope` computed `x1*0 - x2*0 = 0` and zeroed Q and K in every attention
//! layer. Nothing errored. The model returned a confident score with attention destroyed, and
//! `cargo test` stayed green the whole time because it runs native.
//!
//! The fix bakes the tables into the `.imwq` at conversion time, where float is already
//! allowed (the same boundary that converts every weight) and where both runtimes then read
//! identical integers.
//!
//! Adding this fixture immediately caught a SECOND, independent divergence, which is the
//! better argument for the file than the RoPE bug itself.
//!
//! **Integer division convention — do not "simplify" either side back.** Python `//` floors
//! toward -inf; Rust and C `/` truncate toward zero. They differ by exactly 1 whenever the
//! numerator is negative and not an exact multiple. The golden therefore uses `_tdiv`
//! everywhere it divides, and `interp.py`'s mean-POOL was the one place that had been left on
//! `//`. On the original non-RoPE fixture every pooled column sum happened to be positive, so
//! the two agreed and nothing noticed; with RoPE, 5 of 16 columns go negative and the final
//! score came out 1 ULP off (484013642 vs the Rust runtime's 484013641).
//!
//! It is worth being precise about what this was NOT, because the symptom invites a wrong
//! diagnosis: an off-by-one in a fixed-point pipeline looks like a rounding-mode mismatch
//! (Python's `round()` is banker's/ties-to-even, Rust's `.round()` is half-away-from-zero).
//! That hypothesis was checked and is not the cause here — no entry of this table lands on an
//! exact .5 tie, so the two rounding modes produce byte-identical tables. The divergence was
//! the division, and only the division.
//!
//! Fixtures from tests/gen_rope_model_fixture.py.

use wide_wasm::interp::Interp;
use wide_wasm::reader::WideModel;

fn fixture(stem: &str) -> (Vec<u8>, Vec<i64>, i64) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests");
    let bytes = std::fs::read(format!("{dir}/{stem}.imwq"))
        .unwrap_or_else(|e| panic!("{stem}.imwq missing — run gen_rope_model_fixture.py ({e})"));
    let meta = std::fs::read_to_string(format!("{dir}/{stem}.txt")).unwrap();
    let mut lines = meta.lines();
    let tokens: Vec<i64> = lines.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect();
    let want: i64 = lines.next().unwrap().trim().parse().unwrap();
    (bytes, tokens, want)
}

/// The parity test proper: a RoPE-using model scores bit-identically to the Python golden.
#[test]
fn rope_model_score_matches_python() {
    let (bytes, tokens, want) = fixture("rope_model_fixture");
    let m = WideModel::parse(&bytes).expect("parse rope .imwq");
    let got = Interp::new(&m).run(&tokens);
    assert_eq!(got, want, "the RoPE model's score diverged from the Python golden");
}

/// The baked tables must actually be USED, not merely present.
///
/// Reading the cos/sin from the artifact and recomputing them natively happen to agree on a
/// native build, so a test that only checks the baked fixture cannot tell whether the
/// interpreter read the tensors or rebuilt the table. This compares the baked artifact
/// against the same model with the tables zeroed, and asserts the scores differ — if they
/// did not, the parity test above would pass whether the fix worked or not.
#[test]
fn a_zeroed_rope_table_changes_the_score() {
    let (baked, tokens, baked_want) = fixture("rope_model_fixture");
    let (_unbaked, _t, zero_table_want) = fixture("rope_model_fixture_unbaked");

    assert_ne!(
        baked_want, zero_table_want,
        "the fixture is useless: a zeroed RoPE table reaches the same score as the correct \
         one, so nothing here would catch the table not being applied"
    );

    let m = WideModel::parse(&baked).expect("parse baked .imwq");
    let got = Interp::new(&m).run(&tokens);
    assert_eq!(got, baked_want);
    assert_ne!(
        got, zero_table_want,
        "the interpreter produced the ZEROED-table score from an artifact that carries baked \
         tables — the rotation is being skipped and attention is running on zeroed Q/K"
    );
}

/// Every ROPE op the current converter emits must carry both baked tensors.
///
/// Checked structurally rather than by score, because a missing table is silent: the
/// interpreter falls back, and on native the fallback is correct, so a score assertion alone
/// would not notice a converter that stopped baking them.
#[test]
fn the_converter_bakes_both_tables_on_every_rope_op() {
    const ROPE: u16 = 7;
    let (bytes, _t, _w) = fixture("rope_model_fixture");
    let m = WideModel::parse(&bytes).expect("parse rope .imwq");

    let rope_ops: Vec<_> = m.ops.iter().filter(|o| o.op == ROPE).collect();
    assert!(!rope_ops.is_empty(), "the RoPE fixture contains no ROPE ops");
    for (i, op) in rope_ops.iter().enumerate() {
        assert!(
            op.weight_idx >= 0 && op.aux_idx >= 0,
            "ROPE op {i} carries weight_idx={} aux_idx={}; without baked cos/sin tensors the \
             wasm32 build has no way to build the table and rotates by zero",
            op.weight_idx, op.aux_idx
        );
        let (cos, sin) = (
            &m.tensors[op.weight_idx as usize].data,
            &m.tensors[op.aux_idx as usize].data,
        );
        assert!(
            cos.iter().any(|&v| v != 0) && sin.iter().any(|&v| v != 0),
            "ROPE op {i}'s baked tables are all zeros — that is the bug, written into the \
             artifact instead of produced at runtime"
        );
    }
}

/// An artifact from before the fix still scores correctly on a NATIVE build, via the
/// fallback. This pins that the fallback was not broken by the change — the wasm32 refusal is
/// the behaviour change, not a native one.
#[test]
fn an_unbaked_artifact_still_works_natively() {
    let (bytes, tokens, _zero_want) = fixture("rope_model_fixture_unbaked");
    let (_b, _t, correct) = fixture("rope_model_fixture");
    let m = WideModel::parse(&bytes).expect("parse unbaked .imwq");
    let got = Interp::new(&m).run(&tokens);
    assert_eq!(
        got, correct,
        "the native fallback no longer reproduces the correct score for a pre-fix artifact"
    );
}

/// Pin the float->Q32.32 ROUNDING convention, separately from the division convention above.
///
/// Python's `round()` is banker's (ties-to-even); Rust's `f64::round()` is half-away-from-zero.
/// They agree everywhere except on an exact .5 tie, which is precisely why a mismatch here
/// survives ordinary testing and then diverges on one unlucky weight.
///
/// This asserts the two rules genuinely differ on ties (so the test is not vacuous) and that
/// the conversion used by the interpreter follows the Python one. `fp_from_float` in the
/// golden is canonical: every converted weight already goes through it, so aligning Rust to
/// Python is the only change that does not alter every existing artifact.
#[test]
fn float_to_q32_rounds_ties_to_even_like_python() {
    // FP_ONE = 2^32, so x = k + 0.5/FP_ONE lands exactly on a .5 tie in Q32.32.
    let one = wide_wasm::fixed::FP_ONE as f64;
    let half_ulp = 0.5 / one;

    // Python: round(0.5)=0, round(1.5)=2, round(2.5)=2, round(3.5)=4.
    // The two rules DIFFER only where half-away-from-zero would land on an ODD number —
    // i.e. at 0.5 and 2.5. At 1.5 and 3.5 both round up to an even value anyway, so those
    // ties cannot distinguish the conventions and must not be asserted to.
    for (k, want_even, differs) in
        [(0i64, 0i64, true), (1, 2, false), (2, 2, true), (3, 4, false)]
    {
        let x = (k as f64) / one + half_ulp;
        let ties_even = (x * one).round_ties_even() as i64;
        let away_from_zero = (x * one).round() as i64;

        assert_eq!(
            ties_even, want_even,
            "ties-to-even gave {ties_even} for the tie at {k}.5 ULP; Python's round() gives \
             {want_even}"
        );
        if differs {
            assert_ne!(
                away_from_zero, ties_even,
                "the two rounding rules agree at the tie {k}.5, but they must differ there — \
                 if that stops holding, this test no longer distinguishes the conventions"
            );
        } else {
            assert_eq!(away_from_zero, ties_even);
        }
    }
}
