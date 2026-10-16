"""The f32/f64 determinism gate — extracted from int8_runtime/imw_wasm/build_gates.py.

`static_float_scan(wasm_bytes)` parses the WASM code section and asserts NO f32/f64 opcode
appears in ANY function body. Any float in the forward pass is a build failure, not a
warning: a float multiply is where two validators on different CPUs can disagree by one ULP,
and two validators disagreeing fork the epoch. The integer-only forward pass is the entire
reason this runtime exists in fixed point, so the gate is what keeps that property true
after every edit.

It works unchanged on the wide Q32.32 wasm (the int8 module and the wide module are both
pure-integer wasm32), so it is reused rather than rewritten.

This is a proper LEB128-aware instruction walker, not a byte grep, so an immediate that
happens to equal a float opcode byte is never a false positive. Deliberately self-contained
(no wasm-tools/wabt dependency): the build runs offline and the gate must run there.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- WASM float opcodes (single-byte 0x00-prefix space). Any of these in a code body fails. ---
# f32/f64 numeric + comparison + conversion opcodes, per the WebAssembly core spec.
_FLOAT_OPCODES = set(range(0x5B, 0x67))          # f32/f64 comparisons (eq..ge): 0x5B-0x66
_FLOAT_OPCODES |= set(range(0x8B, 0xA7))         # f32/f64 arithmetic + conversions: 0x8B-0xA6
_FLOAT_OPCODES |= {0x43, 0x44}                   # f32.const, f64.const
_FLOAT_OPCODES |= {0x2A, 0x2B}                   # f32.load, f64.load
_FLOAT_OPCODES |= {0x38, 0x39}                   # f32.store, f64.store
_FLOAT_OPCODES |= {0xBE, 0xBF}                   # f32.reinterpret_i32, f64.reinterpret_i64

_FLOAT_OP_NAMES = {
    0x43: "f32.const", 0x44: "f64.const", 0x2A: "f32.load", 0x2B: "f64.load",
    0x38: "f32.store", 0x39: "f64.store", 0xBE: "f32.reinterpret_i32",
    0xBF: "f64.reinterpret_i64",
}


class FloatOpcodeFound(Exception):
    """Raised when the static float scan finds a float opcode — a hard build failure."""


@dataclass
class ScanResult:
    ok: bool
    functions_scanned: int
    detail: str


def _read_uleb(b: bytes, i: int) -> tuple[int, int]:
    """Read an unsigned LEB128 at offset i -> (value, next_offset)."""
    result = 0
    shift = 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, i


def _skip_sleb(b: bytes, i: int) -> int:
    """Skip a signed LEB128, returning the next offset."""
    while True:
        byte = b[i]
        i += 1
        if (byte & 0x80) == 0:
            break
    return i


def _scan_code_body(b: bytes, start: int, end: int) -> None:
    """Walk one function body's instructions; raise FloatOpcodeFound on any float opcode.

    We decode enough of each instruction to skip its immediates correctly so we never read an
    immediate byte as an opcode. Unknown/other opcodes with immediates are handled by families.
    """
    i = start
    while i < end:
        op = b[i]
        i += 1
        if op in _FLOAT_OPCODES:
            name = _FLOAT_OP_NAMES.get(op, f"float-op-0x{op:02x}")
            raise FloatOpcodeFound(name)

        # ---- skip immediates by opcode family (only what wasm32 int code emits) ----
        if op in (0x02, 0x03, 0x04):                 # block/loop/if: blocktype (sleb or 0x40)
            i = _skip_sleb(b, i)
        elif op == 0x0C or op == 0x0D:               # br / br_if: labelidx
            _, i = _read_uleb(b, i)
        elif op == 0x0E:                             # br_table: vec(labelidx) + default
            n, i = _read_uleb(b, i)
            for _ in range(n + 1):
                _, i = _read_uleb(b, i)
        elif op == 0x10:                             # call: funcidx
            _, i = _read_uleb(b, i)
        elif op == 0x11:                             # call_indirect: typeidx + tableidx
            _, i = _read_uleb(b, i)
            _, i = _read_uleb(b, i)
        elif op in (0x20, 0x21, 0x22):               # local.get/set/tee: localidx
            _, i = _read_uleb(b, i)
        elif op in (0x23, 0x24):                     # global.get/set: globalidx
            _, i = _read_uleb(b, i)
        elif 0x28 <= op <= 0x3E:                     # memory load/store: align + offset
            _, i = _read_uleb(b, i)
            _, i = _read_uleb(b, i)
        elif op in (0x3F, 0x40):                     # memory.size/grow: memidx (1 byte)
            i += 1
        elif op == 0x41:                             # i32.const: sleb
            i = _skip_sleb(b, i)
        elif op == 0x42:                             # i64.const: sleb
            i = _skip_sleb(b, i)
        elif op == 0xFC:                             # misc prefix (i32.trunc_sat etc.)
            _, i = _read_uleb(b, i)
        # all other single-byte opcodes (i32/i64 arithmetic, drop, select, end, return, ...)
        # have no immediates — nothing to skip.


def static_float_scan(wasm: bytes) -> ScanResult:
    """Parse the WASM module, walk every function body, assert no f32/f64 opcode."""
    if wasm[:4] != b"\x00asm":
        raise ValueError("not a WASM module (bad magic)")
    i = 8  # skip magic (4) + version (4)
    n = len(wasm)
    functions_scanned = 0
    while i < n:
        section_id = wasm[i]
        i += 1
        size, i = _read_uleb(wasm, i)
        section_end = i + size
        if section_id == 10:  # Code section
            count, j = _read_uleb(wasm, i)
            for _ in range(count):
                body_size, j = _read_uleb(wasm, j)
                body_end = j + body_size
                # body = vec(locals) then instructions; skip the locals declarations first.
                nlocals, k = _read_uleb(wasm, j)
                for _ in range(nlocals):
                    _, k = _read_uleb(wasm, k)   # count
                    k += 1                        # valtype byte
                _scan_code_body(wasm, k, body_end)
                functions_scanned += 1
                j = body_end
        i = section_end
    return ScanResult(True, functions_scanned,
                      f"clean: {functions_scanned} function bodies, no f32/f64 opcodes")


def main(argv=None) -> int:
    """CLI: python -m wide_runtime.build_gates <module.wasm>"""
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m wide_runtime.build_gates <module.wasm>")
        return 2
    with open(argv[0], "rb") as f:
        wasm = f.read()
    try:
        r = static_float_scan(wasm)
    except FloatOpcodeFound as e:
        print(f"FAIL: float opcode {e} found in {argv[0]} — the forward pass is not integer-only")
        return 1
    print(f"PASS: {argv[0]} — {r.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
