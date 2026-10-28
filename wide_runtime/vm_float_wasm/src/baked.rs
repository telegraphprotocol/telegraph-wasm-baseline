//! Telegraph ABI-v2 float adapter guest.
//!
//! The ONNX model is embedded at compile time. The node runs this as a WASI
//! command with `baked /in/input.json 1`; the JSON contains int64 token IDs and
//! an attention mask. Exactly one finite f32 score is printed as its raw bits.

use oxionnx::{Session, Tensor};
use std::collections::HashMap;
use std::env;
use std::fs;

static MODEL: &[u8] = include_bytes!(env!("TELEGRAPH_ONNX_PATH"));

fn parse_i64_array(raw: &str, key: &str) -> Result<Vec<i64>, String> {
    let quoted = format!("\"{key}\"");
    let key_at = raw.find(&quoted).ok_or_else(|| format!("missing {key}"))?;
    let tail = &raw[key_at + quoted.len()..];
    let colon = tail.find(':').ok_or_else(|| format!("missing ':' after {key}"))?;
    let tail = &tail[colon + 1..];
    let left = tail.find('[').ok_or_else(|| format!("{key} is not an array"))?;
    let tail = &tail[left + 1..];
    let right = tail.find(']').ok_or_else(|| format!("unterminated {key} array"))?;
    let body = tail[..right].trim();
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|s| s.trim().parse::<i64>().map_err(|e| format!("invalid {key}: {e}")))
        .collect()
}

fn score(session: &Session, ids: &[i64], mask: &[i64]) -> Result<f32, String> {
    let shape = vec![1, ids.len()];
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_ids",
        Tensor::new(ids.iter().map(|&v| v as f32).collect(), shape.clone()),
    );
    inputs.insert(
        "attention_mask",
        Tensor::new(mask.iter().map(|&v| v as f32).collect(), shape),
    );
    let outputs = session.run(&inputs).map_err(|e| format!("inference failed: {e}"))?;
    let logits = outputs
        .get("logits")
        .or_else(|| outputs.values().next())
        .ok_or_else(|| "model produced no output".to_string())?;
    logits.data.first().copied().ok_or_else(|| "model produced an empty score".to_string())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return Err("usage: baked <input.json> [runs] (model is baked into this wasm)".into());
    }
    let runs = args.get(2).map(|s| s.parse::<usize>()).transpose()
        .map_err(|e| format!("invalid runs: {e}"))?.unwrap_or(1);
    if runs == 0 {
        return Err("runs must be at least 1".into());
    }
    let raw = fs::read_to_string(&args[1]).map_err(|e| format!("read input json: {e}"))?;
    let ids = parse_i64_array(&raw, "input_ids")?;
    let mask = parse_i64_array(&raw, "attention_mask")?;
    if ids.is_empty() || ids.len() != mask.len() {
        return Err(format!("input_ids/attention_mask length mismatch: {}/{}", ids.len(), mask.len()));
    }

    let session = Session::from_bytes(MODEL).map_err(|e| format!("load ONNX model: {e}"))?;
    let mut result = 0.0_f32;
    for _ in 0..runs {
        result = score(&session, &ids, &mask)?;
    }
    if !result.is_finite() {
        return Err(format!("model produced a non-finite score: {result}"));
    }
    println!("HEX         : 0x{:08x}  (score {:+.10})", result.to_bits(), result);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("baked: {e}");
        std::process::exit(1);
    }
}
