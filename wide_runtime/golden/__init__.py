"""Q32.32 golden reference: the fixed-point core and the transformer ops.

This package is the SPEC the Rust runtime (`wide_wasm/src/fixed.rs`, `ops.rs`) must match
bit-for-bit. `wide_wasm/tests/ops_golden.rs` and `model_golden.rs` assert exactly that
against fixtures generated from here.
"""
