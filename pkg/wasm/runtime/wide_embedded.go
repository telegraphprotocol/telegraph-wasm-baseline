package runtime

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

const (
	// fnWideEmbedded is the export a self-contained wide package locates its
	// resident model through: () -> i64 = (ptr<<32)|len, or i64::MIN when the
	// module carries no usable model. Its presence is also the kind
	// discriminator (see IsWideEmbeddedWasm).
	fnWideEmbedded = "imwq_embedded"

	// fnWideEmbedDescAddr returns the linear-memory address of the embed slot.
	// The packager needs it to know where to write; the loader uses it to read
	// back the slot's sha256 and cross-check it against the resident bytes.
	fnWideEmbedDescAddr = "imwq_embed_desc_addr"
)

// The embed slot: a 56-byte record inside the embed host's data, overwritten
// at instantiation by a data segment the packager appends. The layout is pinned
// on the Rust side by wide_wasm/tests/embed_slot.rs (struct EmbedDesc) and here
// by TestEmbedSlotRoundTrip; neither side can detect a drift at run time.
//
//	off  0  magic    [8]byte  "IMWQHOST" (bare host) | "IMWQEMB1" (filled)
//	off  8  version  u32      EmbedSlotVersion
//	off 12  ptr      u32      linear-memory address of the .imwq
//	off 16  len      u32      its length in bytes
//	off 20  reserved u32      0
//	off 24  sha256   [32]byte sha256 of the .imwq bytes
const (
	EmbedSlotSize    = 56
	EmbedSlotVersion = 1
)

var (
	// EmbedHostMagic marks an unfilled slot (a bare embed host).
	EmbedHostMagic = [8]byte{'I', 'M', 'W', 'Q', 'H', 'O', 'S', 'T'}
	// EmbedFilledMagic marks a slot the packager filled.
	EmbedFilledMagic = [8]byte{'I', 'M', 'W', 'Q', 'E', 'M', 'B', '1'}
)

// EmbedSlot is the decoded embed slot.
type EmbedSlot struct {
	Magic   [8]byte
	Version uint32
	Ptr     uint32
	Len     uint32
	SHA256  [32]byte
}

// Encode serialises the slot to its 56-byte wire form.
func (s EmbedSlot) Encode() []byte {
	b := make([]byte, EmbedSlotSize)
	copy(b[0:8], s.Magic[:])
	binary.LittleEndian.PutUint32(b[8:], s.Version)
	binary.LittleEndian.PutUint32(b[12:], s.Ptr)
	binary.LittleEndian.PutUint32(b[16:], s.Len)
	// b[20:24] reserved = 0
	copy(b[24:56], s.SHA256[:])
	return b
}

// ParseEmbedSlot decodes a 56-byte slot.
func ParseEmbedSlot(b []byte) (EmbedSlot, error) {
	if len(b) != EmbedSlotSize {
		return EmbedSlot{}, fmt.Errorf("wasm/runtime: embed slot is %d bytes, want %d", len(b), EmbedSlotSize)
	}
	var s EmbedSlot
	copy(s.Magic[:], b[0:8])
	s.Version = binary.LittleEndian.Uint32(b[8:])
	s.Ptr = binary.LittleEndian.Uint32(b[12:])
	s.Len = binary.LittleEndian.Uint32(b[16:])
	copy(s.SHA256[:], b[24:56])
	return s, nil
}

// NewWideEmbedded instantiates a SELF-CONTAINED wide package (interpreter +
// resident .imwq, built by cmd/wideembed) and adopts the model it carries, so
// imwq_score works with no LoadArtifact call.
//
// Before returning it verifies, and errors loudly on any failure:
//   - the module exports imwq_embedded (otherwise it is a bare interpreter or a
//     MiniLM module — wrong path);
//   - imwq_embedded reports a model (a bare embed host, or a slot that is
//     corrupt/out of bounds, reports none);
//   - the slot read back from memory agrees with that report, and the sha256 it
//     records matches the resident bytes (catches a mis-packaged segment);
//   - the resident model parses to a non-zero op count (DebugCounts), exactly as
//     LoadArtifact verifies an uploaded one.
//
// Scoring then calls the SAME imwq_score export over the SAME bytes as the
// split path, which is why the two are bit-identical.
func NewWideEmbedded(ctx context.Context, pkg []byte) (*WideRuntime, error) {
	r, err := instantiateWide(ctx, pkg)
	if err != nil {
		return nil, err
	}
	if err := r.adoptEmbedded(ctx); err != nil {
		r.Close(context.Background()) //nolint:errcheck
		return nil, err
	}
	return r, nil
}

// NewWideHybridEmbedded is NewWideEmbedded for a hybrid package: it additionally
// requires the imwq_score_hybrid export, so a judgment-only package cannot be
// installed for a hybrid intent (it would silently drop the comparator half).
func NewWideHybridEmbedded(ctx context.Context, pkg []byte) (*WideHybridRuntime, error) {
	r, err := NewWideEmbedded(ctx, pkg)
	if err != nil {
		return nil, err
	}
	fn := r.mod.ExportedFunction(fnWideScoreHybrid)
	if fn == nil {
		r.Close(context.Background()) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: missing required export %q — this is a JUDGMENT-only "+
			"self-contained package (built with --features embedded), not a hybrid one", fnWideScoreHybrid)
	}
	return &WideHybridRuntime{WideRuntime: r, fnScoreHybrid: fn}, nil
}

// adoptEmbedded locates, verifies and adopts the module's resident model.
func (r *WideRuntime) adoptEmbedded(ctx context.Context) error {
	locate := r.mod.ExportedFunction(fnWideEmbedded)
	if locate == nil {
		return fmt.Errorf("wasm/runtime: not a self-contained wide package (no %q export) — "+
			"for a bare interpreter use NewWide and pass the .imwq to LoadArtifact", fnWideEmbedded)
	}
	descAddr := r.mod.ExportedFunction(fnWideEmbedDescAddr)
	if descAddr == nil {
		return fmt.Errorf("wasm/runtime: self-contained wide package lacks %q", fnWideEmbedDescAddr)
	}

	res, err := locate.Call(ctx)
	if err != nil {
		return fmt.Errorf("wasm/runtime: %s: %w", fnWideEmbedded, err)
	}
	if len(res) == 0 {
		return fmt.Errorf("wasm/runtime: %s returned no values", fnWideEmbedded)
	}
	packed := int64(res[0])
	if packed == wideRejected {
		return fmt.Errorf("wasm/runtime: the module carries no embedded model (%s returned i64::MIN): "+
			"a bare embed host that was never packaged, or a corrupt/out-of-bounds embed slot", fnWideEmbedded)
	}
	ptr, size := uint32(uint64(packed)>>32), uint32(uint64(packed))

	memSize := uint64(r.mem.Size())
	if ptr == 0 || size == 0 || uint64(ptr)+uint64(size) > memSize {
		return fmt.Errorf("wasm/runtime: embedded model (ptr=%d len=%d) lies outside linear memory (%d bytes)",
			ptr, size, memSize)
	}

	// Cross-check against the slot itself and its recorded digest.
	dres, err := descAddr.Call(ctx)
	if err != nil || len(dres) == 0 {
		return fmt.Errorf("wasm/runtime: %s: %v", fnWideEmbedDescAddr, err)
	}
	raw, ok := r.mem.Read(uint32(dres[0]), EmbedSlotSize)
	if !ok {
		return fmt.Errorf("wasm/runtime: embed slot at %d is outside linear memory", uint32(dres[0]))
	}
	slot, err := ParseEmbedSlot(raw)
	if err != nil {
		return err
	}
	if slot.Magic != EmbedFilledMagic || slot.Version != EmbedSlotVersion || slot.Ptr != ptr || slot.Len != size {
		return fmt.Errorf("wasm/runtime: embed slot disagrees with %s (slot magic=%q v=%d ptr=%d len=%d; reported ptr=%d len=%d)",
			fnWideEmbedded, slot.Magic[:], slot.Version, slot.Ptr, slot.Len, ptr, size)
	}
	model, ok := r.mem.Read(ptr, size)
	if !ok {
		return fmt.Errorf("wasm/runtime: reading the %d-byte embedded model at %d failed", size, ptr)
	}
	if !bytes.HasPrefix(model, []byte(imwqMagic)) {
		return fmt.Errorf("wasm/runtime: embedded model does not start with the IMWQ magic")
	}
	if sum := sha256.Sum256(model); sum != slot.SHA256 {
		return fmt.Errorf("wasm/runtime: embedded model sha256 %x does not match the slot's %x — the package is corrupt",
			sum[:8], slot.SHA256[:8])
	}

	r.imwPtr, r.imwLen, r.embedded = ptr, size, true
	if _, _, err := r.DebugCounts(ctx); err != nil {
		r.imwPtr, r.imwLen, r.embedded = 0, 0, false
		return fmt.Errorf("wasm/runtime: the embedded model did not parse: %w", err)
	}
	return nil
}

// EmbeddedModelSHA256 returns the sha256 of the resident model, read from linear
// memory. For diagnostics (wasmprobe, the packager's verify step).
func (r *WideRuntime) EmbeddedModelSHA256() ([32]byte, error) {
	if r == nil || r.imwPtr == 0 {
		return [32]byte{}, fmt.Errorf("wasm/runtime: no resident model")
	}
	b, ok := r.mem.Read(r.imwPtr, r.imwLen)
	if !ok {
		return [32]byte{}, fmt.Errorf("wasm/runtime: resident model outside linear memory")
	}
	return sha256.Sum256(b), nil
}

// ResidentModel reports where the resident model sits: (ptr, len). Zero when
// none is loaded.
func (r *WideRuntime) ResidentModel() (ptr, size uint32) {
	if r == nil {
		return 0, 0
	}
	return r.imwPtr, r.imwLen
}

// MemoryBytes is the module's current linear-memory size. dlmalloc never
// shrinks memory, so read after a score it is the high-water mark — the number
// to compare against the wasm32 4 GiB ceiling.
func (r *WideRuntime) MemoryBytes() uint64 {
	if r == nil || r.mem == nil {
		return 0
	}
	return uint64(r.mem.Size())
}
