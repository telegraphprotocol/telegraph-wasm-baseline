//! Foundation engine (Telegraph ABI 2, "lora-v1"): a WASI reactor that keeps ONE Foundation
//! Model ONNX graph (LoRA branches and reward head as graph inputs) and scores Evaluator
//! adapters on it. An adapter is only the tensors fed at score time; the foundation is never
//! mutated. The node side is Telegraph pkg/wasm/runtime/foundation.go.
//!
//! Exports (all i32 unless noted):
//!   telegraph_abi() -> 2
//!   tg_alloc(len) -> ptr, tg_free(ptr, len)   (libc owns the names alloc/free)
//!   load_foundation(ptr, len) -> 0 | <0       (once per instance; the host frees the bytes after)
//!   foundation_spec() -> len                   (JSON in the message buffer)
//!   load_adapter(ptr, len) -> handle > 0 | <0  (a safetensors delta package; re-validated here)
//!   score(handle, ids_ptr, n) -> i64           (f32 bits in 0..2^32, or <0)
//!   unload_adapter(handle) -> 0 | <0
//!   message_ptr() / message_len()             (last error text or the spec JSON)
//!
//! Graph inputs the foundation must declare (by name; order is irrelevant):
//!   input_ids, attention_mask            [batch, seq]
//!   lora.scale [1]   head.weight [hidden, 1]   head.bias [1]
//!   lora.<l>.<target>.A [in, R_MAX]   lora.<l>.<target>.B [R_MAX, out]
//! for target in attn.Wqkv, attn.Wo, mlp.Wi, mlp.Wo. A delta package (safetensors, F32, the
//! package fields in __metadata__) carries the same names with A [in, r] and B [r, out]
//! (r <= R_MAX, pre-transposed); the engine zero-pads them.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use oxionnx::{Session, Tensor};

const ABI: i32 = 2;
const MAX_TOKENS: usize = 512;
const TARGETS: [&str; 4] = ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"];

struct Target {
    a_name: String,
    b_name: String,
    fin: usize,
    fout: usize,
}

struct Foundation {
    session: Session,
    targets: Vec<Target>,
    hidden: usize,
    layers: usize,
    r_max: usize,
    dims: BTreeMap<&'static str, (usize, usize)>,
}

struct State {
    foundation: Option<Foundation>,
    adapters: HashMap<i32, Vec<(String, Tensor)>>,
    next_handle: i32,
    message: Vec<u8>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State {
        foundation: None, adapters: HashMap::new(), next_handle: 1, message: Vec::new(),
    });
}

fn set_message(s: &str) {
    STATE.with(|st| st.borrow_mut().message = s.as_bytes().to_vec());
}

#[no_mangle]
pub extern "C" fn telegraph_abi() -> i32 {
    ABI
}

#[no_mangle]
pub extern "C" fn tg_alloc(len: i32) -> i32 {
    if len <= 0 {
        return 0;
    }
    let mut v: Vec<u8> = Vec::with_capacity(len as usize);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p as i32
}

#[no_mangle]
pub extern "C" fn tg_free(ptr: i32, len: i32) {
    if ptr != 0 && len > 0 {
        unsafe { drop(Vec::from_raw_parts(ptr as *mut u8, 0, len as usize)) };
    }
}

#[no_mangle]
pub extern "C" fn message_ptr() -> i32 {
    STATE.with(|st| st.borrow().message.as_ptr() as i32)
}

#[no_mangle]
pub extern "C" fn message_len() -> i32 {
    STATE.with(|st| st.borrow().message.len() as i32)
}

fn bytes<'a>(ptr: i32, len: i32) -> &'a [u8] {
    if ptr == 0 || len <= 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) }
}

fn static_shape(sess: &Session, name: &str) -> Result<Vec<usize>, String> {
    let info = sess.input_info().iter().find(|i| i.name == name)
        .ok_or_else(|| format!("foundation has no input {name}"))?;
    info.shape.iter().map(|d| d.ok_or_else(|| format!("input {name} has a dynamic dim"))).collect()
}

fn layout(session: Session) -> Result<Foundation, String> {
    let names: Vec<String> = session.input_names().to_vec();
    for need in ["input_ids", "attention_mask", "lora.scale", "head.weight", "head.bias"] {
        if !names.iter().any(|n| n == need) {
            return Err(format!("foundation has no input {need}"));
        }
    }
    if !session.output_names().iter().any(|n| n == "logits") {
        return Err("foundation has no output logits".into());
    }
    let hw = static_shape(&session, "head.weight")?;
    if hw.len() != 2 || hw[1] != 1 {
        return Err(format!("head.weight shape {hw:?}, want [hidden, 1]"));
    }
    let hidden = hw[0];
    if static_shape(&session, "head.bias")? != vec![1] || static_shape(&session, "lora.scale")? != vec![1] {
        return Err("head.bias and lora.scale must be [1]".into());
    }
    let lora_inputs = names.iter().filter(|n| n.starts_with("lora.") && *n != "lora.scale").count();
    if lora_inputs == 0 || lora_inputs % (2 * TARGETS.len()) != 0 {
        return Err(format!("{lora_inputs} LoRA inputs is not a whole number of layers"));
    }
    let layers = lora_inputs / (2 * TARGETS.len());
    let mut targets = Vec::new();
    let mut dims: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let mut r_max = 0;
    for l in 0..layers {
        for t in TARGETS {
            let a_name = format!("lora.{l}.{t}.A");
            let b_name = format!("lora.{l}.{t}.B");
            let a = static_shape(&session, &a_name)?;
            let b = static_shape(&session, &b_name)?;
            if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                return Err(format!("{a_name} {a:?} / {b_name} {b:?} are not [in, R] / [R, out]"));
            }
            if r_max == 0 {
                r_max = a[1];
            }
            if a[1] != r_max {
                return Err(format!("{a_name} has rank dim {}, others {r_max}", a[1]));
            }
            let d = (a[0], b[1]);
            if *dims.entry(t).or_insert(d) != d {
                return Err(format!("{t} has different shapes across layers"));
            }
            targets.push(Target { a_name, b_name, fin: d.0, fout: d.1 });
        }
    }
    if r_max == 0 {
        return Err("R_MAX is 0".into());
    }
    Ok(Foundation { session, targets, hidden, layers, r_max, dims })
}

#[no_mangle]
pub extern "C" fn load_foundation(ptr: i32, len: i32) -> i32 {
    let loaded = STATE.with(|st| st.borrow().foundation.is_some());
    if loaded {
        set_message("foundation already loaded");
        return -1;
    }
    let session = match Session::from_bytes(bytes(ptr, len)) {
        Ok(s) => s,
        Err(e) => {
            set_message(&format!("foundation parse failed: {e:?}"));
            return -2;
        }
    };
    match layout(session) {
        Ok(f) => {
            STATE.with(|st| st.borrow_mut().foundation = Some(f));
            0
        }
        Err(e) => {
            set_message(&format!("foundation layout: {e}"));
            -3
        }
    }
}

#[no_mangle]
pub extern "C" fn foundation_spec() -> i32 {
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        let Some(f) = st.foundation.as_ref() else { return -1 };
        let mut targets = serde_json::Map::new();
        for (t, (i, o)) in &f.dims {
            targets.insert(t.to_string(), serde_json::json!([i, o]));
        }
        let spec = serde_json::json!({
            "abi": ABI, "r_max": f.r_max, "hidden_size": f.hidden, "num_layers": f.layers,
            "targets": targets, "max_tokens": MAX_TOKENS,
        });
        st.message = spec.to_string().into_bytes();
        st.message.len() as i32
    })
}

fn read_f32s(data: &[u8]) -> Result<Vec<f32>, String> {
    let mut v = Vec::with_capacity(data.len() / 4);
    for c in data.chunks_exact(4) {
        let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        if !x.is_finite() {
            return Err("non-finite value".into());
        }
        v.push(x);
    }
    Ok(v)
}

/// Parses and checks a safetensors delta package against the loaded foundation; returns the
/// graph feed. Errors are "reason: detail", with the node's reason names.
fn adapter_feed(f: &Foundation, pkg: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    if pkg.len() < 9 {
        return Err("bad_container: shorter than a safetensors header".into());
    }
    let hl = u64::from_le_bytes(pkg[..8].try_into().unwrap()) as usize;
    if hl == 0 || hl > (1 << 20) || 8 + hl > pkg.len() || pkg[8] != b'{' {
        return Err("bad_container: not a safetensors file".into());
    }
    let hdr: serde_json::Value = serde_json::from_slice(&pkg[8..8 + hl]).map_err(|e| format!("bad_container: {e}"))?;
    let obj = hdr.as_object().ok_or("bad_container: header is not an object")?;
    let meta = obj.get("__metadata__").and_then(|m| m.as_object()).ok_or("missing_metadata: no __metadata__")?;
    let ms = |k: &str| -> Result<&str, String> {
        meta.get(k).and_then(|v| v.as_str()).ok_or_else(|| format!("missing_metadata: {k}"))
    };
    if ms("abi")? != "lora-v1" {
        return Err(format!("bad_abi: {}", ms("abi")?));
    }
    if ms("format_version")? != "1" {
        return Err(format!("bad_version: format version {}", ms("format_version")?));
    }
    let rank: usize = ms("rank")?.parse().map_err(|_| "bad_header: rank".to_string())?;
    let alpha: f64 = ms("alpha")?.parse().map_err(|_| "bad_header: alpha".to_string())?;
    if rank == 0 || rank > f.r_max {
        return Err(format!("rank_too_large: rank {rank}, R_MAX {}", f.r_max));
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err("bad_header: alpha".into());
    }
    let data = &pkg[8 + hl..];
    let mut entries: Vec<(String, Vec<usize>, usize, usize)> = Vec::new();
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        if t["dtype"] != "F32" {
            return Err(format!("bad_dtype: tensor {name} is {}, want F32", t["dtype"]));
        }
        let shape: Vec<usize> = t["shape"].as_array().ok_or("bad_header: shape")?
            .iter().map(|d| d.as_u64().map(|x| x as usize)).collect::<Option<_>>().ok_or("bad_header: shape")?;
        let off = t["data_offsets"].as_array().ok_or("bad_offsets: data_offsets")?;
        let (s, e) = (off[0].as_u64().ok_or("bad_offsets")? as usize, off[1].as_u64().ok_or("bad_offsets")? as usize);
        if e < s || e - s != 4 * shape.iter().product::<usize>() || e > data.len() {
            return Err(format!("bad_offsets: tensor {name}"));
        }
        entries.push((name.clone(), shape, s, e));
    }
    // The tensors must tile the data exactly: sorted by offset, contiguous, nothing left.
    entries.sort_by_key(|e| e.2);
    let mut off = 0usize;
    for (name, _, s, e) in &entries {
        if *s != off {
            return Err(format!("bad_offsets: tensor {name} at {s}, expected {off}"));
        }
        off = *e;
    }
    if off != data.len() {
        return Err("trailing_bytes: data after the last tensor".into());
    }
    let mut raw: HashMap<String, (Vec<usize>, &[u8])> = HashMap::new();
    for (name, shape, s, e) in entries {
        raw.insert(name, (shape, &data[s..e]));
    }
    if raw.len() != 2 * f.targets.len() + 2 {
        return Err(format!("bad_shape: {} tensors, want {}", raw.len(), 2 * f.targets.len() + 2));
    }
    let take = |name: &str, want: &[usize]| -> Result<Vec<f32>, String> {
        let (shape, b) = raw.get(name).ok_or_else(|| format!("bad_shape: missing {name}"))?;
        if shape.as_slice() != want {
            return Err(format!("bad_shape: {name} {shape:?}, want {want:?}"));
        }
        read_f32s(b).map_err(|e| format!("non_finite: {name}: {e}"))
    };
    let r_max = f.r_max;
    let mut feed = Vec::with_capacity(3 + 2 * f.targets.len());
    feed.push(("lora.scale".to_string(), Tensor::new(vec![(alpha / rank as f64) as f32], vec![1])));
    feed.push(("head.weight".to_string(), Tensor::new(take("head.weight", &[f.hidden, 1])?, vec![f.hidden, 1])));
    feed.push(("head.bias".to_string(), Tensor::new(take("head.bias", &[1])?, vec![1])));
    for t in &f.targets {
        let a = take(&t.a_name, &[t.fin, rank])?; // [in, r]
        let b = take(&t.b_name, &[rank, t.fout])?; // [r, out]
        // Zero-pad the rank dim to R_MAX: exact zeros, so the padded branch adds nothing.
        let mut ap = vec![0f32; t.fin * r_max]; // [in, R_MAX]
        for i in 0..t.fin {
            ap[i * r_max..i * r_max + rank].copy_from_slice(&a[i * rank..(i + 1) * rank]);
        }
        let mut bp = vec![0f32; r_max * t.fout]; // [R_MAX, out]
        bp[..rank * t.fout].copy_from_slice(&b);
        feed.push((t.a_name.clone(), Tensor::new(ap, vec![t.fin, r_max])));
        feed.push((t.b_name.clone(), Tensor::new(bp, vec![r_max, t.fout])));
    }
    Ok(feed)
}

#[no_mangle]
pub extern "C" fn load_adapter(ptr: i32, len: i32) -> i32 {
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        let Some(f) = st.foundation.as_ref() else {
            st.message = b"no_foundation: no foundation loaded".to_vec();
            return -1;
        };
        match adapter_feed(f, bytes(ptr, len)) {
            Ok(feed) => {
                let h = st.next_handle;
                st.next_handle += 1;
                st.adapters.insert(h, feed);
                h
            }
            Err(e) => {
                st.message = e.into_bytes();
                -2
            }
        }
    })
}

#[no_mangle]
pub extern "C" fn unload_adapter(handle: i32) -> i32 {
    STATE.with(|st| if st.borrow_mut().adapters.remove(&handle).is_some() { 0 } else { -1 })
}

#[no_mangle]
pub extern "C" fn score(handle: i32, ids_ptr: i32, n: i32) -> i64 {
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        let n = n.max(0) as usize;
        if n == 0 || n > MAX_TOKENS {
            st.message = format!("token count {n} outside 1..{MAX_TOKENS}").into_bytes();
            return -3;
        }
        let ids: Vec<f32> = bytes(ids_ptr, (4 * n) as i32).chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect();
        let result = {
            let Some(f) = st.foundation.as_ref() else { return -1 };
            let Some(feed) = st.adapters.get(&handle) else { return -2 };
            let mut inputs: HashMap<&str, Tensor> = HashMap::with_capacity(feed.len() + 2);
            inputs.insert("input_ids", Tensor::new(ids, vec![1, n]));
            inputs.insert("attention_mask", Tensor::new(vec![1.0; n], vec![1, n]));
            for (name, t) in feed {
                // The session takes owned tensors: one clone per score, the adapter's copy is never mutated.
                inputs.insert(name.as_str(), t.clone());
            }
            f.session.run(&inputs).map_err(|e| format!("run failed: {e:?}"))
                .and_then(|out| out.get("logits").map(|t| t.data[0]).ok_or_else(|| "no logits output".to_string()))
        };
        match result {
            Ok(v) => v.to_bits() as i64,
            Err(e) => {
                st.message = e.into_bytes();
                -4
            }
        }
    })
}
