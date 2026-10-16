//! The embed slot is a byte layout shared with the Go packager (pkg/wasm/wideembed), which
//! writes it as a raw 56-byte data segment. Nothing on the wasm side can detect a layout
//! drift — a moved field reads as a different pointer or length — so the layout is pinned
//! here and in wideembed's own test.
//!
//! Only meaningful with `--features embedded` (or `embedded-hybrid`); the default build has no
//! slot and this file compiles to nothing.
#![cfg(feature = "embedded")]

use core::mem::{align_of, offset_of, size_of};
use wide_wasm::embed::{EmbedDesc, FILLED_MAGIC, HOST_MAGIC, VERSION};

#[test]
fn slot_layout_matches_the_packager() {
    assert_eq!(size_of::<EmbedDesc>(), 56);
    assert_eq!(align_of::<EmbedDesc>(), 8);
    assert_eq!(offset_of!(EmbedDesc, magic), 0);
    assert_eq!(offset_of!(EmbedDesc, version), 8);
    assert_eq!(offset_of!(EmbedDesc, ptr), 12);
    assert_eq!(offset_of!(EmbedDesc, len), 16);
    assert_eq!(offset_of!(EmbedDesc, reserved), 20);
    assert_eq!(offset_of!(EmbedDesc, sha256), 24);
    assert_eq!(&HOST_MAGIC, b"IMWQHOST");
    assert_eq!(&FILLED_MAGIC, b"IMWQEMB1");
    assert_eq!(VERSION, 1);
}

/// The bare embed host (slot never filled) must report "no model", never a (0, 0) pair the
/// host could mistake for a resident artifact.
#[test]
fn unfilled_slot_reports_no_model() {
    assert_eq!(wide_wasm::imwq_embedded(), i64::MIN);
    assert_eq!(&wide_wasm::embed::read().magic, b"IMWQHOST");
}
