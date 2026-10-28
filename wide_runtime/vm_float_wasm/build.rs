use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=TELEGRAPH_ONNX_PATH");
    let model = env::var("TELEGRAPH_ONNX_PATH")
        .expect("TELEGRAPH_ONNX_PATH must name the ONNX model to bake");
    assert!(Path::new(&model).is_file(), "ONNX model does not exist: {model}");
    println!("cargo:rerun-if-changed={model}");
}
