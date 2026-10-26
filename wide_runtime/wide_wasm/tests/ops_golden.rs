//! Rust↔Python bit-identity golden test for the wide Q32.32 ops.
//!
//! Reads tests/fixture.json (flat keys, emitted by gen_fixture.py from the Python golden) and
//! asserts the Rust ops produce BYTE-IDENTICAL i64 outputs — the determinism guarantee. Tiny
//! hand-rolled reader over flat "key":value / "key":[..] pairs (no serde dep).

use wide_wasm::ops;

fn load() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixture.json"))
        .expect("fixture.json missing — run gen_fixture.py")
}

fn arr(j: &str, key: &str) -> Vec<i64> {
    let pat = format!("\"{}\":[", key);
    let start = j.find(&pat).unwrap_or_else(|| panic!("key {key} not found")) + pat.len();
    let end = j[start..].find(']').unwrap() + start;
    j[start..end]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<i64>().unwrap())
        .collect()
}

fn scalar(j: &str, key: &str) -> i64 {
    // match "key": followed immediately by a digit/minus (not "key":[ )
    let pat = format!("\"{}\":", key);
    let start = j.find(&pat).unwrap_or_else(|| panic!("scalar {key} not found")) + pat.len();
    let rest = &j[start..];
    let end = rest.find(|c| c == ',' || c == '}').unwrap();
    rest[..end].trim().parse::<i64>().unwrap()
}

#[test]
fn qmatmul_matches_python() {
    let j = load();
    let got = ops::qmatmul(&arr(&j, "qm_x"), scalar(&j, "qm_seq") as usize,
                           scalar(&j, "qm_in") as usize, &arr(&j, "qm_w"),
                           scalar(&j, "qm_out") as usize, None);
    assert_eq!(got, arr(&j, "qm_o"), "qmatmul diverged from Python golden");
}

#[test]
fn rmsnorm_matches_python() {
    let j = load();
    let got = ops::rmsnorm(&arr(&j, "rm_x"), scalar(&j, "rm_seq") as usize,
                           scalar(&j, "rm_dim") as usize, &arr(&j, "rm_g"), scalar(&j, "rm_eps"));
    assert_eq!(got, arr(&j, "rm_o"), "rmsnorm diverged");
}

#[test]
fn layernorm_matches_python() {
    let j = load();
    let got = ops::layernorm(&arr(&j, "ln_x"), scalar(&j, "ln_seq") as usize,
                             scalar(&j, "ln_dim") as usize, &arr(&j, "ln_g"),
                             Some(&arr(&j, "ln_b")), scalar(&j, "ln_eps"));
    assert_eq!(got, arr(&j, "ln_o"), "layernorm diverged");
}

#[test]
fn attn_matches_python() {
    let j = load();
    let got = ops::softmax_attn(&arr(&j, "at_q"), &arr(&j, "at_k"), &arr(&j, "at_v"),
                                scalar(&j, "at_seq") as usize, scalar(&j, "at_hd") as usize,
                                scalar(&j, "at_scale"), false);
    assert_eq!(got, arr(&j, "at_o"), "softmax_attn diverged");
}

#[test]
fn acts_match_python() {
    let j = load();
    let x = arr(&j, "ac_x");
    assert_eq!(ops::act_gelu(&x), arr(&j, "ac_gelu"), "gelu diverged");
    assert_eq!(ops::act_silu(&x), arr(&j, "ac_silu"), "silu diverged");
}
