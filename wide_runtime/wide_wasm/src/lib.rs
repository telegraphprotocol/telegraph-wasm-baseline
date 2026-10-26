//! wide-wasm — deterministic Q32.32 integer runtime (wide activations + weights).
//!
//! Rust port of pipelines/wide_runtime/golden (fixed.py, ops.py, interp.py). Bit-identical to
//! the Python golden by construction (same i128-intermediate math, same op/accumulation order,
//! same truncate-toward-zero division). Compiles to wasm32-unknown-unknown; no f32/f64 in the
//! forward pass (integer-only, the determinism guarantee).

// wasm builds are no_std (no OS); native builds keep std for the test harness.
#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

#[cfg(target_arch = "wasm32")]
mod wasm_rt {
    use dlmalloc::GlobalDlmalloc;

    #[global_allocator]
    static ALLOCATOR: GlobalDlmalloc = GlobalDlmalloc;

    /// Required for no_std. A trap surfaces as an instantiation error in the host.
    #[panic_handler]
    fn panic(_info: &core::panic::PanicInfo) -> ! {
        core::arch::wasm32::unreachable()
    }
}

pub mod fixed;
pub mod hybrid;
pub mod ops;
pub mod interp;
pub mod reader;

use alloc::vec::Vec;

/// Score a wide `.imwq` image on `tokens`, returning the raw i64 Q32.32 score code (the host
/// applies the single boundary float multiply). Returns None on a malformed artifact (fail
/// closed — a rejected file must never look like a valid score).
pub fn score_bytes(imwq: &[u8], tokens: &[i64]) -> Option<i64> {
    let m = reader::WideModel::parse(imwq)?;
    Some(interp::Interp::new(&m).run(tokens))
}

// --------------------------------------------------------------------------- //
//  Host ABI (mirrors int8 imw_wasm): alloc(n) -> write bytes -> score(...).
// --------------------------------------------------------------------------- //

/// Allocate `len` bytes in linear memory; the host writes the artifact/tokens into it.
#[no_mangle]
pub extern "C" fn imwq_alloc(len: usize) -> *mut u8 {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Free a buffer from `imwq_alloc`.
/// # Safety
/// `ptr`/`len` must come from a prior `imwq_alloc(len)`, freed at most once.
#[no_mangle]
pub unsafe extern "C" fn imwq_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len != 0 {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

/// Score a wide `.imwq` (`imw_ptr`,`imw_len`) on `n_tokens` little-endian i64s at `tok_ptr`.
///
/// Returns the raw i64 Q32.32 score code — the host applies the single boundary conversion
/// (`pkg/wasm/runtime/wide.go`: sigmoid for a logit head, then scale to basis points).
///
/// On a rejected artifact it returns `i64::MIN`. That exact sentinel rather than a small
/// negative, because the score itself is a signed logit and is routinely negative: the real
/// CODE_REVIEW adapter returns about -2.32 in Q32.32. A `-1` sentinel would be
/// indistinguishable from a legitimate score, and the host would read "rejected the file" as
/// "this miner scored badly".
/// # Safety
/// Both pointers must reference buffers of the stated lengths in this module's linear memory.
#[no_mangle]
pub unsafe extern "C" fn imwq_score(
    imw_ptr: *const u8,
    imw_len: usize,
    tok_ptr: *const i64,
    n_tokens: usize,
) -> i64 {
    let imw = core::slice::from_raw_parts(imw_ptr, imw_len);
    let tokens = core::slice::from_raw_parts(tok_ptr, n_tokens);
    match score_bytes(imw, tokens) {
        Some(score) => score,
        None => i64::MIN, // rejected artifact — a clearly-invalid sentinel
    }
}

/// Debug: parse and return `(num_tensors << 32) | num_ops`, or `i64::MIN` on reject.
/// # Safety
/// `imw_ptr`/`imw_len` must reference a valid buffer in this module's linear memory.
#[no_mangle]
pub unsafe extern "C" fn imwq_debug_counts(imw_ptr: *const u8, imw_len: usize) -> i64 {
    let imw = core::slice::from_raw_parts(imw_ptr, imw_len);
    match reader::WideModel::parse(imw) {
        Some(m) => ((m.tensors.len() as i64) << 32) | (m.ops.len() as i64),
        None => i64::MIN,
    }
}

// --------------------------------------------------------------------------- //
//  Hybrid scoring: comparator + fusion gate over the wide judgment.
// --------------------------------------------------------------------------- //

/// Error codes returned across the wasm boundary as `-(code)`, so the host can tell "rejected
/// the input" from "scored it". A rejected file must never look like a valid score.
///
/// Only the hybrid path uses these. `imwq_score` returns a raw signed logit and therefore
/// cannot carry a small negative sentinel, so it uses `i64::MIN` instead; the hybrid path
/// returns a score in [0, SCORE_ONE] which is non-negative by construction, so any negative
/// return is unambiguously an error.
#[repr(i64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridError {
    /// The `.imwq` artifact did not parse.
    BadArtifact = 1,
    /// No tokens were supplied — a missing input, not an empty judgment.
    NoTokens = 2,
}

/// Per-intent hybrid spec, the ten integers the host's spec buffer carries.
pub struct HybridSpec {
    pub comparator: hybrid::Comparator,
    pub tolerance: i64,
    pub fusion: hybrid::Fusion,
    pub fusion_const: hybrid::FusionConst,
}

/// Map the interpreter's raw Q32.32 score code into the [0, SCORE_ONE] judgment domain.
///
/// This MUST match `WideToScoreDomain`/`fpSigmoid` in `pkg/wasm/runtime/wide.go`, because a
/// hybrid score and a bare `imwq_score` run through the host conversion have to agree on what
/// a given model output is worth — otherwise the same adapter grades differently depending on
/// which export the caller used.
///
/// The sigmoid is the load-bearing part. The score head is a sequence-classification reward
/// head (`num_labels=1`) and emits an UNBOUNDED logit: the real CODE_REVIEW adapter returns
/// about -2.32. Mapped by a plain divide-and-clamp, every miner with a negative logit lands at
/// 0 and they all tie at the bottom, which is indistinguishable from every miner answering
/// badly. The sigmoid is monotone, so it changes what a score reads as but never who wins
/// (except past |logit| > 10, where it saturates).
pub fn judgment_to_score_domain(code: i64) -> i64 {
    // fp_sigmoid puts the value in [0, FP_ONE]; scale to basis points with one wide multiply
    // so the intermediate cannot overflow (FP_ONE * 10000 needs 46 bits, fine, but the shift
    // ordering matters for truncation and i128 keeps it exact).
    let unit = fixed::fp_sigmoid(code);
    let bps = ((unit as i128) * (hybrid::SCORE_ONE as i128)) >> fixed::FP_PRECISION;
    bps.clamp(0, hybrid::SCORE_ONE as i128) as i64
}

/// Full hybrid score: run the wide model on `tokens` -> judgment; integer-compare `miner_value`
/// against the passed-in `ground_truth_value` -> comparator_core; fuse -> [0, SCORE_ONE].
/// Pure integer end to end.
pub fn score_hybrid_bytes(
    imwq: &[u8],
    tokens: &[i64],
    ground_truth_value: i64,
    miner_value: i64,
    spec: &HybridSpec,
) -> Result<i64, HybridError> {
    if tokens.is_empty() {
        return Err(HybridError::NoTokens);
    }
    let m = reader::WideModel::parse(imwq).ok_or(HybridError::BadArtifact)?;
    let code = interp::Interp::new(&m).run(tokens);
    let judgment = judgment_to_score_domain(code);
    let core = hybrid::comparator_core(
        spec.comparator, miner_value, ground_truth_value, spec.tolerance);
    Ok(hybrid::fuse(spec.fusion, core, judgment, &spec.fusion_const))
}

/// Hybrid scoring across the wasm boundary. Mirrors int8's `imw_score_hybrid` signature so the
/// Go host's `HybridRuntime` shape carries over, with the one structural difference the wide
/// format forces: the resident `.imwq` is passed in by pointer, exactly as `imwq_score` takes
/// it. In the shared ~52 KB interpreter the host uploads it (`LoadArtifact`); in a
/// self-contained package (`embedded-hybrid` host + cmd/wideembed) the model is already in
/// linear memory and the host passes the `(ptr, len)` that `imwq_embedded` reports — the SAME
/// function over the SAME bytes, which is what makes the two paths bit-identical.
///
/// Absent from a judgment-only embed host (`--features embedded`), so a judgment package
/// cannot be installed for a hybrid intent: the missing export fails the load.
///
/// Returns the fused score in [0, SCORE_ONE], or `-(HybridError)` on a rejected input.
///
/// **Argument order: `ground_truth_value` comes BEFORE `miner_value`.** This is not the
/// host-natural order and it is deliberate — it matches int8's export, and the Go wrapper
/// takes `(minerValue, truthValue)` and swaps at the call site. A swap here is SILENT: under
/// RelTol it only changes the denominator, so the module still returns a plausible score. Do
/// not "fix" either side to match the other.
///
/// **`spec_ptr` is exactly 10 i64s, positional, with no length argument:**
/// `[comparator, tolerance, fusion_mode, pass, low, floor, span, w_c, w_j, reserved]`.
/// A short buffer is read past its end; a reordered one fuses against the wrong thresholds.
/// Neither errors. `comparator.Fusion.Buffer()` in `pkg/scoring/comparator/specs_db.go` is the
/// Go side that produces it, and the layout is pinned by a test on both sides.
///
/// # Safety
/// `imw_ptr`/`tok_ptr` must reference buffers of the stated lengths in this module's linear
/// memory; `spec_ptr` must reference at least 10 i64s.
#[cfg(any(not(feature = "embedded"), feature = "embedded-hybrid"))]
#[no_mangle]
pub unsafe extern "C" fn imwq_score_hybrid(
    imw_ptr: *const u8,
    imw_len: usize,
    tok_ptr: *const i64,
    n_tokens: usize,
    ground_truth_value: i64,
    miner_value: i64,
    spec_ptr: *const i64,
) -> i64 {
    let imw = core::slice::from_raw_parts(imw_ptr, imw_len);
    let tokens = core::slice::from_raw_parts(tok_ptr, n_tokens);
    let s = core::slice::from_raw_parts(spec_ptr, 10);
    let spec = HybridSpec {
        comparator: hybrid::Comparator::from_i32(s[0] as i32),
        tolerance: s[1],
        fusion: hybrid::Fusion::from_i32(s[2] as i32),
        fusion_const: hybrid::FusionConst {
            pass: s[3], low: s[4], floor: s[5], span: s[6], w_c: s[7], w_j: s[8],
        },
    };
    match score_hybrid_bytes(imw, tokens, ground_truth_value, miner_value, &spec) {
        Ok(score) => score,
        Err(e) => -(e as i64),
    }
}

/// Deterministic (comparator-only) score — NO model, no `.imwq` artifact.
///
/// Ported from the int8 crate's `imw_score_deterministic`, which the wide build previously
/// did not carry. The comparator half was never int8-specific: it is an integer compare of
/// two already-scaled values, so retiring int8 quantisation retired the model path and left
/// this untouched. Without it, this repo could build the hybrid wasm but not the one the
/// eight deterministic intents actually load, and `ComparatorScorer` had to be handed a
/// module published from another repo.
///
/// `spec_ptr` points to 2 i64s: `[comparator, tolerance]` — the first two fields of the
/// 10-int hybrid buffer, so one `comparator.Fusion.Buffer()` serves both exports and a
/// deterministic intent cannot drift from its hybrid sibling's thresholds.
///
/// Ground truth comes FIRST, matching `imwq_score_hybrid` and the Go side's
/// `DeterministicRuntime.ScoreDeterministic`. The swap is silent under RelTol — it only
/// changes the denominator — so both ends stay on one convention rather than being
/// "fixed" to meet in the middle.
///
/// # Safety
/// `spec_ptr` must reference at least 2 i64s.
#[cfg(not(feature = "embedded"))]
#[no_mangle]
pub unsafe extern "C" fn imw_score_deterministic(
    ground_truth_value: i64,
    miner_value: i64,
    spec_ptr: *const i64,
) -> i64 {
    let s = core::slice::from_raw_parts(spec_ptr, 2);
    let cmp = hybrid::Comparator::from_i32(s[0] as i32);
    hybrid::comparator_core(cmp, miner_value, ground_truth_value, s[1])
}

/// `imw_alloc`/`imw_free` — the allocator names the DETERMINISTIC loader binds.
///
/// `DeterministicRuntime` predates the wide runtime and looks up the int8 crate's unprefixed
/// spelling, while everything wide exports carries the `imwq_` prefix. These are thin
/// aliases rather than a second allocator: same `imwq_alloc` underneath, so a buffer handed
/// back through either name frees correctly. Renaming the Go side instead would break every
/// already-published int8-era comparator module the validator still loads.
///
/// # Safety
/// As `imwq_alloc`/`imwq_free`: `ptr`/`len` must be a pair this module returned.
#[cfg(not(feature = "embedded"))]
#[no_mangle]
pub extern "C" fn imw_alloc(len: usize) -> *mut u8 {
    imwq_alloc(len)
}

/// See `imw_alloc`.
///
/// # Safety
/// `ptr` must have come from `imw_alloc`/`imwq_alloc` with the same `len`.
#[cfg(not(feature = "embedded"))]
#[no_mangle]
pub unsafe extern "C" fn imw_free(ptr: *mut u8, len: usize) {
    imwq_free(ptr, len)
}

// --------------------------------------------------------------------------- //
//  Embedded model (self-contained package): `--features embedded[-hybrid]`.
// --------------------------------------------------------------------------- //

/// The embed slot a self-contained package's model is located through.
///
/// cmd/wideembed (pkg/wasm/wideembed) instantiates the bare embed host, asks
/// `imwq_embed_desc_addr` where this slot lives, checks it still holds the `IMWQHOST`
/// sentinel, and then appends two ACTIVE data segments to the module: one over these 56 bytes
/// (`IMWQEMB1`, version, ptr, len, sha256) and one holding the `.imwq` itself above the
/// original initial memory. Active segments are applied in order at instantiation, so the
/// second write wins over the sentinel. The model sits above `__heap_end`, which is the only
/// pre-existing region dlmalloc ever hands out (everything else comes from `memory.grow`,
/// which starts past the raised initial size), so the allocator can never overwrite it.
#[cfg(feature = "embedded")]
pub mod embed {
    /// Byte layout pinned by the packager (`pkg/wasm/wideembed`): 8+4+4+4+4+32 = 56 bytes.
    #[repr(C, align(8))]
    pub struct EmbedDesc {
        pub magic: [u8; 8],
        pub version: u32,
        pub ptr: u32,
        pub len: u32,
        pub reserved: u32,
        pub sha256: [u8; 32],
    }

    /// Magic of an UNFILLED slot (the bare embed host).
    pub const HOST_MAGIC: [u8; 8] = *b"IMWQHOST";
    /// Magic of a slot the packager filled.
    pub const FILLED_MAGIC: [u8; 8] = *b"IMWQEMB1";
    /// Slot layout version the packager writes.
    pub const VERSION: u32 = 1;

    /// `static mut` and read only through `read_volatile`: an immutable static with a known
    /// initialiser would be constant-folded, and the packager's overwrite would be ignored —
    /// every package would silently report "no model".
    pub static mut IMWQ_EMBED_DESC: EmbedDesc = EmbedDesc {
        magic: HOST_MAGIC,
        version: 0,
        ptr: 0,
        len: 0,
        reserved: 0,
        sha256: [0; 32],
    };

    /// Read the slot without letting the compiler assume its initial value.
    pub fn read() -> EmbedDesc {
        // SAFETY: a plain volatile read of a static the module never writes; the only writer
        // is the data segment applied at instantiation, before any export can run.
        unsafe { core::ptr::read_volatile(core::ptr::addr_of!(IMWQ_EMBED_DESC)) }
    }
}

/// Locate the model a self-contained package carries: `(ptr << 32) | len`.
///
/// Returns `i64::MIN` when there is no usable model — the bare embed host (slot still
/// `IMWQHOST`), an unknown slot version, a pointer/length that runs past linear memory, or
/// bytes at `ptr` that do not start with the `IMWQ` magic. The host treats that as a load
/// error, never as "score with whatever is there". The full parse (op count) is the host's
/// next check, via `imwq_debug_counts(ptr, len)`.
#[cfg(feature = "embedded")]
#[no_mangle]
pub extern "C" fn imwq_embedded() -> i64 {
    let d = embed::read();
    if d.magic != embed::FILLED_MAGIC || d.version != embed::VERSION || d.ptr == 0 || d.len < 8 {
        return i64::MIN;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mem = core::arch::wasm32::memory_size(0) * 65536;
        match (d.ptr as usize).checked_add(d.len as usize) {
            Some(end) if end <= mem => {}
            _ => return i64::MIN,
        }
        // SAFETY: bounds checked against the current memory size just above.
        let head = unsafe { core::slice::from_raw_parts(d.ptr as *const u8, 4) };
        if head != b"IMWQ" {
            return i64::MIN;
        }
    }
    ((d.ptr as i64) << 32) | (d.len as i64)
}

/// Packager-only: the linear-memory address of the embed slot. cmd/wideembed calls this on
/// the bare host so it never has to guess at linker layout.
#[cfg(feature = "embedded")]
#[no_mangle]
pub extern "C" fn imwq_embed_desc_addr() -> u32 {
    core::ptr::addr_of!(embed::IMWQ_EMBED_DESC) as usize as u32
}
