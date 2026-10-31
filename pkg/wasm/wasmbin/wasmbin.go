// Package wasmbin is a minimal, dependency-free reader/writer for the WebAssembly binary
// format — just enough to classify a module by its imports, exports and custom sections
// WITHOUT compiling it, and to build small modules in tests.
//
// Classification has to be cheap because a registration can be a ~600 MB legacy full-model
// module that must be refused from its first MiB: handing that to wazero just to ask "which
// ABI is this?" would decode, copy and compile the whole thing. The import and export sections precede the code and data
// sections, so Imports and ExportNames read only section headers until they reach them —
// O(header), regardless of how big the model is.
//
// Every read is bounds-checked and returns an error rather than panicking: the input is
// untrusted bytes fetched from an author's URL.
package wasmbin

import (
	"errors"
	"fmt"
)

// Section ids (WebAssembly core spec §5.5.2).
const (
	SecCustom    byte = 0
	SecType      byte = 1
	SecImport    byte = 2
	SecFunction  byte = 3
	SecTable     byte = 4
	SecMemory    byte = 5
	SecGlobal    byte = 6
	SecExport    byte = 7
	SecStart     byte = 8
	SecElement   byte = 9
	SecCode      byte = 10
	SecData      byte = 11
	SecDataCount byte = 12
)

// Export kinds (§5.5.10).
const (
	ExportFunc   byte = 0
	ExportTable  byte = 1
	ExportMemory byte = 2
	ExportGlobal byte = 3
)

// Magic is the 8-byte module preamble: "\0asm" + version 1.
var Magic = []byte{0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00}

// ErrNotWasm is returned for bytes that do not start with the wasm preamble.
var ErrNotWasm = errors.New("wasmbin: not a wasm module (bad preamble)")

// Section locates one section inside the module bytes.
//
//	b[Start:PayloadStart] is the id byte + LEB128 size; b[PayloadStart:End] is the payload.
type Section struct {
	ID           byte
	Start        int
	PayloadStart int
	End          int
}

// Payload returns the section's payload bytes from b.
func (s Section) Payload(b []byte) []byte { return b[s.PayloadStart:s.End] }

// IsWasm reports whether b starts with the wasm preamble.
func IsWasm(b []byte) bool {
	if len(b) < len(Magic) {
		return false
	}
	for i := range Magic {
		if b[i] != Magic[i] {
			return false
		}
	}
	return true
}

// ReadU32 decodes an unsigned LEB128 u32 at b[off:], returning the value and bytes consumed.
func ReadU32(b []byte, off int) (uint32, int, error) {
	var v uint64
	for i := 0; i < 5; i++ {
		if off+i >= len(b) {
			return 0, 0, fmt.Errorf("wasmbin: truncated LEB128 at offset %d", off)
		}
		c := b[off+i]
		v |= uint64(c&0x7f) << (7 * i)
		if c&0x80 == 0 {
			if v > 0xffffffff {
				return 0, 0, fmt.Errorf("wasmbin: LEB128 u32 overflow at offset %d", off)
			}
			return uint32(v), i + 1, nil
		}
	}
	return 0, 0, fmt.Errorf("wasmbin: LEB128 u32 longer than 5 bytes at offset %d", off)
}

// AppendU32 appends v as a MINIMAL unsigned LEB128. Minimal encoding is what makes a rewritten
// module a unique function of its inputs (wasm also accepts padded forms).
func AppendU32(dst []byte, v uint32) []byte {
	for {
		c := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			dst = append(dst, c|0x80)
			continue
		}
		return append(dst, c)
	}
}

// AppendI32 appends v as a minimal signed LEB128 (the immediate of i32.const).
func AppendI32(dst []byte, v int32) []byte {
	x := int64(v)
	for {
		c := byte(x & 0x7f)
		x >>= 7
		if (x == 0 && c&0x40 == 0) || (x == -1 && c&0x40 != 0) {
			return append(dst, c)
		}
		dst = append(dst, c|0x80)
	}
}

// ReadI32 decodes a signed LEB128 i32 at b[off:].
func ReadI32(b []byte, off int) (int32, int, error) {
	var v int64
	var shift uint
	for i := 0; i < 5; i++ {
		if off+i >= len(b) {
			return 0, 0, fmt.Errorf("wasmbin: truncated SLEB128 at offset %d", off)
		}
		c := b[off+i]
		v |= int64(c&0x7f) << shift
		shift += 7
		if c&0x80 == 0 {
			if shift < 64 && c&0x40 != 0 {
				v |= -1 << shift
			}
			return int32(v), i + 1, nil
		}
	}
	return 0, 0, fmt.Errorf("wasmbin: SLEB128 i32 longer than 5 bytes at offset %d", off)
}

// nextSection decodes the section header at off. It does NOT require the payload to be
// present in b (callers that only need headers up to the export section can tolerate a file
// truncated later); it reports End as declared.
func nextSection(b []byte, off int) (Section, error) {
	if off >= len(b) {
		return Section{}, fmt.Errorf("wasmbin: section header past end at %d", off)
	}
	id := b[off]
	size, n, err := ReadU32(b, off+1)
	if err != nil {
		return Section{}, err
	}
	ps := off + 1 + n
	return Section{ID: id, Start: off, PayloadStart: ps, End: ps + int(size)}, nil
}

// Sections walks every section of a complete module. A section whose declared size runs past
// the end of b is an error (a truncated download).
func Sections(b []byte) ([]Section, error) {
	if !IsWasm(b) {
		return nil, ErrNotWasm
	}
	var out []Section
	off := len(Magic)
	for off < len(b) {
		s, err := nextSection(b, off)
		if err != nil {
			return nil, err
		}
		if s.End > len(b) {
			return nil, fmt.Errorf("wasmbin: section %d declares %d payload bytes but the module ends at %d (truncated?)",
				s.ID, s.End-s.PayloadStart, len(b))
		}
		out = append(out, s)
		off = s.End
	}
	return out, nil
}

// CustomSection returns the data of the first custom section (id 0) called name, and whether
// one was found. Only section headers are decoded on the way; payloads are skipped by size,
// so this stays cheap on a multi-GB module.
func CustomSection(b []byte, name string) ([]byte, bool, error) {
	secs, err := Sections(b)
	if err != nil {
		return nil, false, err
	}
	for _, s := range secs {
		if s.ID != SecCustom {
			continue
		}
		p := s.Payload(b)
		n, k, err := ReadU32(p, 0)
		if err != nil {
			return nil, false, fmt.Errorf("wasmbin: custom section name: %w", err)
		}
		if uint64(k)+uint64(n) > uint64(len(p)) {
			return nil, false, fmt.Errorf("wasmbin: custom section name runs past the section")
		}
		if string(p[k:k+int(n)]) == name {
			return p[k+int(n):], true, nil
		}
	}
	return nil, false, nil
}

// AppendCustomSection appends a custom section (id 0) called name carrying data.
func AppendCustomSection(dst []byte, name string, data []byte) []byte {
	payload := AppendU32(nil, uint32(len(name)))
	payload = append(payload, name...)
	payload = append(payload, data...)
	return AppendSection(dst, SecCustom, payload)
}

// ExportNames returns name -> export kind for every export, reading section headers only up
// to (and including) the export section. Sections after it are never touched, so this is
// cheap on a multi-GB module and works on a file truncated after its exports. A module with
// no export section yields an empty map.
func ExportNames(b []byte) (map[string]byte, error) {
	if !IsWasm(b) {
		return nil, ErrNotWasm
	}
	off := len(Magic)
	for off < len(b) {
		s, err := nextSection(b, off)
		if err != nil {
			return nil, err
		}
		if s.End > len(b) {
			return nil, fmt.Errorf("wasmbin: section %d runs past the end of the module before the export section", s.ID)
		}
		switch {
		case s.ID == SecExport:
			return parseExports(s.Payload(b))
		case s.ID == SecCustom || s.ID < SecExport:
			// keep walking: custom sections may appear anywhere; ids 1-6 precede exports
		default:
			// Past the export section's slot (start/element/datacount/code/data): no exports.
			return map[string]byte{}, nil
		}
		off = s.End
	}
	return map[string]byte{}, nil
}

// Import kinds (§5.5.9) — the same byte values as the export kinds.
const (
	ImportFunc   byte = 0
	ImportTable  byte = 1
	ImportMemory byte = 2
	ImportGlobal byte = 3
)

// Import is one entry of the import section.
type Import struct {
	Module string
	Name   string
	Kind   byte
}

// Imports returns every import, in order, reading section headers only up to
// (and including) the import section. Like ExportNames it never touches the
// code or data sections, so it is cheap on a module whose data segment is
// hundreds of MB, and it works on a file truncated after its imports. A module
// with no import section yields an empty slice.
func Imports(b []byte) ([]Import, error) {
	if !IsWasm(b) {
		return nil, ErrNotWasm
	}
	off := len(Magic)
	for off < len(b) {
		s, err := nextSection(b, off)
		if err != nil {
			return nil, err
		}
		if s.End > len(b) {
			return nil, fmt.Errorf("wasmbin: section %d runs past the end of the module before the import section", s.ID)
		}
		switch {
		case s.ID == SecImport:
			return parseImports(s.Payload(b))
		case s.ID == SecCustom || s.ID == SecType:
			// keep walking: custom sections may appear anywhere; type precedes imports
		default:
			// Past the import section's slot: the module imports nothing.
			return []Import{}, nil
		}
		off = s.End
	}
	return []Import{}, nil
}

func parseImports(p []byte) ([]Import, error) {
	n, k, err := ReadU32(p, 0)
	if err != nil {
		return nil, err
	}
	off := k
	readName := func(what string) (string, error) {
		nl, k, err := ReadU32(p, off)
		if err != nil {
			return "", err
		}
		off += k
		if off+int(nl) > len(p) {
			return "", fmt.Errorf("wasmbin: import %s runs past the import section", what)
		}
		s := string(p[off : off+int(nl)])
		off += int(nl)
		return s, nil
	}
	skipLimits := func() error {
		if off >= len(p) {
			return fmt.Errorf("wasmbin: import limits truncated")
		}
		flag := p[off]
		off++
		if _, k, err := ReadU32(p, off); err != nil { // min
			return err
		} else {
			off += k
		}
		if flag&0x01 != 0 { // has max
			_, k, err := ReadU32(p, off)
			if err != nil {
				return err
			}
			off += k
		}
		return nil
	}
	// n is untrusted: an entry is at least 4 bytes (two empty names, kind, index).
	out := make([]Import, 0, capHint(n, len(p)-off, 4))
	for i := uint32(0); i < n; i++ {
		mod, err := readName("module name")
		if err != nil {
			return nil, err
		}
		name, err := readName("field name")
		if err != nil {
			return nil, err
		}
		if off >= len(p) {
			return nil, fmt.Errorf("wasmbin: import %s.%s has no kind byte", mod, name)
		}
		kind := p[off]
		off++
		switch kind {
		case ImportFunc:
			_, k, err := ReadU32(p, off) // type index
			if err != nil {
				return nil, err
			}
			off += k
		case ImportTable:
			if off >= len(p) {
				return nil, fmt.Errorf("wasmbin: import %s.%s table type truncated", mod, name)
			}
			off++ // reftype
			if err := skipLimits(); err != nil {
				return nil, err
			}
		case ImportMemory:
			if err := skipLimits(); err != nil {
				return nil, err
			}
		case ImportGlobal:
			if off+2 > len(p) {
				return nil, fmt.Errorf("wasmbin: import %s.%s global type truncated", mod, name)
			}
			off += 2 // valtype, mutability
		default:
			return nil, fmt.Errorf("wasmbin: import %s.%s has unknown kind 0x%02x", mod, name, kind)
		}
		out = append(out, Import{Module: mod, Name: name, Kind: kind})
	}
	if off != len(p) {
		return nil, fmt.Errorf("wasmbin: %d trailing bytes in import section", len(p)-off)
	}
	return out, nil
}

// capHint bounds a preallocation for n untrusted entries by the most that
// remaining payload bytes could encode at minSize bytes each.
func capHint(n uint32, remaining, minSize int) int {
	if remaining < 0 {
		remaining = 0
	}
	if max := remaining / minSize; uint64(n) > uint64(max) {
		return max
	}
	return int(n)
}

func parseExports(p []byte) (map[string]byte, error) {
	n, k, err := ReadU32(p, 0)
	if err != nil {
		return nil, err
	}
	off := k
	// n is untrusted: preallocate no more than the payload could hold (an entry
	// is at least 3 bytes), so a crafted count cannot ask for gigabytes.
	out := make(map[string]byte, capHint(n, len(p)-off, 3))
	for i := uint32(0); i < n; i++ {
		nl, k, err := ReadU32(p, off)
		if err != nil {
			return nil, err
		}
		off += k
		if off+int(nl) > len(p) {
			return nil, fmt.Errorf("wasmbin: export name runs past the export section")
		}
		name := string(p[off : off+int(nl)])
		off += int(nl)
		if off >= len(p) {
			return nil, fmt.Errorf("wasmbin: export %q has no kind byte", name)
		}
		kind := p[off]
		off++
		if _, k, err = ReadU32(p, off); err != nil { // index
			return nil, err
		}
		off += k
		out[name] = kind
	}
	return out, nil
}

// Limits is a memory type's limits (§5.3.4). Only 32-bit, non-shared memories are
// representable here; anything else is rejected by ParseMemory.
type Limits struct {
	Min    uint32
	Max    uint32
	HasMax bool
}

// ParseMemory decodes a memory section payload that declares exactly one memory.
func ParseMemory(p []byte) (Limits, error) {
	n, k, err := ReadU32(p, 0)
	if err != nil {
		return Limits{}, err
	}
	if n != 1 {
		return Limits{}, fmt.Errorf("wasmbin: memory section declares %d memories, want exactly 1", n)
	}
	off := k
	if off >= len(p) {
		return Limits{}, fmt.Errorf("wasmbin: memory section truncated")
	}
	flag := p[off]
	off++
	if flag != 0x00 && flag != 0x01 {
		return Limits{}, fmt.Errorf("wasmbin: unsupported memory limits flag 0x%02x (shared/memory64 not supported)", flag)
	}
	var l Limits
	if l.Min, k, err = ReadU32(p, off); err != nil {
		return Limits{}, err
	}
	off += k
	if flag == 0x01 {
		l.HasMax = true
		if l.Max, k, err = ReadU32(p, off); err != nil {
			return Limits{}, err
		}
		off += k
	}
	if off != len(p) {
		return Limits{}, fmt.Errorf("wasmbin: %d trailing bytes in memory section", len(p)-off)
	}
	return l, nil
}

// EncodeMemory encodes a one-memory section payload.
func EncodeMemory(l Limits) []byte {
	out := AppendU32(nil, 1)
	if l.HasMax {
		out = append(out, 0x01)
		out = AppendU32(out, l.Min)
		return AppendU32(out, l.Max)
	}
	out = append(out, 0x00)
	return AppendU32(out, l.Min)
}

// AppendSection appends a full section (id, minimal LEB128 size, payload) to dst.
func AppendSection(dst []byte, id byte, payload []byte) []byte {
	dst = append(dst, id)
	dst = AppendU32(dst, uint32(len(payload)))
	return append(dst, payload...)
}

// ActiveDataSegment encodes one active data segment for memory 0 at a constant offset
// (flag 0, `i32.const offset; end`, then the bytes) — the form wasm-ld itself emits.
// The offset is written as the i32 two's-complement of the u32 address, as the spec requires.
func ActiveDataSegment(offset uint32, data []byte) []byte {
	out := []byte{0x00, 0x41}
	out = AppendI32(out, int32(offset))
	out = append(out, 0x0b)
	out = AppendU32(out, uint32(len(data)))
	return append(out, data...)
}

// DataSegment is one decoded data segment (active, memory 0, i32.const offset).
type DataSegment struct {
	Offset uint32
	// DataStart/DataEnd locate the segment's bytes within the data section PAYLOAD.
	DataStart, DataEnd int
}

// ParseDataSegments decodes a data section payload whose segments are all of the simple
// active form (flag 0 with an i32.const offset). Anything else — passive segments, explicit
// memory indices, non-constant offsets — is reported as an error rather than guessed at.
func ParseDataSegments(p []byte) ([]DataSegment, error) {
	n, k, err := ReadU32(p, 0)
	if err != nil {
		return nil, err
	}
	off := k
	// n is untrusted: cap the preallocation by what the payload could hold (a
	// segment is at least 4 bytes) so a crafted count cannot exhaust memory.
	segs := make([]DataSegment, 0, capHint(n, len(p)-off, 4))
	for i := uint32(0); i < n; i++ {
		if off+2 > len(p) {
			return nil, fmt.Errorf("wasmbin: data segment %d truncated", i)
		}
		if p[off] != 0x00 || p[off+1] != 0x41 {
			return nil, fmt.Errorf("wasmbin: data segment %d is not an active i32.const segment (flag 0x%02x, op 0x%02x)", i, p[off], p[off+1])
		}
		off += 2
		v, k, err := ReadI32(p, off)
		if err != nil {
			return nil, err
		}
		off += k
		if off >= len(p) || p[off] != 0x0b {
			return nil, fmt.Errorf("wasmbin: data segment %d offset expression is not a lone i32.const", i)
		}
		off++
		sz, k, err := ReadU32(p, off)
		if err != nil {
			return nil, err
		}
		off += k
		if off+int(sz) > len(p) {
			return nil, fmt.Errorf("wasmbin: data segment %d runs past the data section", i)
		}
		segs = append(segs, DataSegment{Offset: uint32(v), DataStart: off, DataEnd: off + int(sz)})
		off += int(sz)
	}
	if off != len(p) {
		return nil, fmt.Errorf("wasmbin: %d trailing bytes in data section", len(p)-off)
	}
	return segs, nil
}
