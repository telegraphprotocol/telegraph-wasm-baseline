//! Global allocator and panic handler for `wasm32-unknown-unknown`.
//!
//! `no_std` targets require an explicit allocator. We use `dlmalloc` which
//! is small, well-tested, and has a proven WASM track record.

use dlmalloc::GlobalDlmalloc;

#[global_allocator]
static ALLOCATOR: GlobalDlmalloc = GlobalDlmalloc;

/// Panic handler — required for `no_std`.
/// WASM traps on `unreachable`, which surfaces as a wazero error in Go.
///
/// Gated out of native test builds: under `cargo test` the crate links against
/// std, which supplies its own `#[panic_handler]`, and defining two is a hard
/// error. Without this gate the per-file `#[cfg(test)]` unit tests cannot run.
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
