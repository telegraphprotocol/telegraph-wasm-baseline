use std::env;
use std::path::Path;

// A ModernBERT-base FP32 model is ~600 MB; its graph alone is ~150 KB. Anything under this
// floor is a graph without its weights, which would still compile and package cleanly.
const MIN_MODEL_BYTES: u64 = 16 << 20;

fn main() {
    println!("cargo:rerun-if-env-changed=TELEGRAPH_ONNX_PATH");
    println!("cargo:rerun-if-env-changed=TELEGRAPH_ALLOW_SMALL_ONNX");
    let model = env::var("TELEGRAPH_ONNX_PATH")
        .expect("TELEGRAPH_ONNX_PATH must name the ONNX model to bake");
    assert!(Path::new(&model).is_file(), "ONNX model does not exist: {model}");
    println!("cargo:rerun-if-changed={model}");

    // The bake embeds only this one file (include_bytes!), so weights stored beside it are lost.
    let sidecar = format!("{model}.data");
    assert!(
        !Path::new(&sidecar).exists(),
        "{sidecar} exists: this ONNX keeps its weights in a separate file, which the bake cannot \
         embed. Re-export a single self-contained file (wide_runtime.export_onnx does this)."
    );
    let bytes = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
    assert!(
        bytes >= MIN_MODEL_BYTES || env::var_os("TELEGRAPH_ALLOW_SMALL_ONNX").is_some(),
        "{model} is only {bytes} bytes: a ModernBERT reward model is ~600 MB, so this looks like a \
         graph exported without its weights (external data). Re-export a single self-contained file, \
         or set TELEGRAPH_ALLOW_SMALL_ONNX=1 for a deliberately tiny test model."
    );
}
