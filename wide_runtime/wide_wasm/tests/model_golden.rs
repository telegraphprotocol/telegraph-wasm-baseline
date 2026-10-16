//! End-to-end Rust↔Python bit-identity: parse the synthetic 1-layer encoder .imwq, run the
//! Rust interpreter, assert the raw i64 Q32.32 score equals the Python golden byte-for-byte.
//! Fixtures from tests/gen_model_fixture.py.

use wide_wasm::interp::Interp;
use wide_wasm::reader::WideModel;

#[test]
fn model_score_matches_python() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests");
    let bytes = std::fs::read(format!("{dir}/model_fixture.imwq"))
        .expect("model_fixture.imwq missing — run gen_model_fixture.py");
    let meta = std::fs::read_to_string(format!("{dir}/model_fixture.txt")).unwrap();
    let mut lines = meta.lines();
    let tokens: Vec<i64> = lines.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect();
    let want: i64 = lines.next().unwrap().trim().parse().unwrap();

    let m = WideModel::parse(&bytes).expect("parse .imwq");
    let got = Interp::new(&m).run(&tokens);
    assert_eq!(got, want, "Rust wide score diverged from Python golden");
}
