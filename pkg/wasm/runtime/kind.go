package runtime

import (
	"fmt"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
)

// ArtifactKind names the shapes of scoring artifact the promote path can
// receive, so the scorer, the candidate evaluator and Stage 1 validation branch
// on ONE definition rather than each re-checking bytes their own way. A wrong
// artifact that "loads" and returns a confident number is the failure this
// whole layer exists to avoid, so the kind is decided in exactly one place.
type ArtifactKind uint8

const (
	// KindMiniLMWasm is a self-contained MiniLM .wasm module scored through a
	// wasmrt.Pool via the rank_answer ABI. This is the historical/default kind.
	KindMiniLMWasm ArtifactKind = iota

	// KindWideIMWQ is a bare wide Q32.32 model image (.imwq artifact), NOT a
	// wasm module: it is uploaded into a shared wide interpreter's linear memory
	// via WideRuntime.LoadArtifact and scored via imwq_score / imwq_score_hybrid.
	// The dev/split path.
	KindWideIMWQ

	// KindWideEmbedded is a SELF-CONTAINED wide package: one .wasm carrying the
	// wide interpreter AND one intent's .imwq as a resident data segment (built
	// by cmd/wideembed). It is recognised by the imwq_embedded export, loaded by
	// NewWideEmbedded / NewWideHybridEmbedded, and needs no LoadArtifact and no
	// shared interpreter — the registered hash covers model and code together.
	//
	// Appended after KindWideIMWQ so existing values do not move.
	KindWideEmbedded
)

// String names the kind for logs and errors.
func (k ArtifactKind) String() string {
	switch k {
	case KindMiniLMWasm:
		return "minilm-wasm"
	case KindWideIMWQ:
		return "wide-imwq"
	case KindWideEmbedded:
		return "wide-embedded"
	default:
		return fmt.Sprintf("ArtifactKind(%d)", uint8(k))
	}
}

// imwqMagic is the four ASCII bytes a wide Q32.32 model image begins with. It
// is the SAME check WideRuntime.LoadArtifact makes before uploading an
// artifact; kept here as the single source of truth so the promote-path kind
// branch and the loader cannot drift apart.
const imwqMagic = "IMWQ"

// ArtifactKindOf classifies raw promote-path bytes:
//
//   - leading "IMWQ" magic              -> KindWideIMWQ (bare model image)
//   - wasm exporting imwq_embedded      -> KindWideEmbedded (self-contained package)
//   - anything else                     -> KindMiniLMWasm
//
// Deliberately a cheap, total function: it never errors and never compiles the
// module (the embedded probe reads only section headers up to the export
// section, so a 1.2 GB package costs microseconds). Anything that is not
// positively wide is handed to the MiniLM loader, which reports its own error
// if the bytes are not a valid rank_answer module. A wide-looking module that is
// broken past its export section (e.g. truncated) is classified wide and then
// fails loudly in NewWideEmbedded — never silently re-routed.
func ArtifactKindOf(b []byte) ArtifactKind {
	if IsWideIMWQ(b) {
		return KindWideIMWQ
	}
	if IsWideEmbeddedWasm(b) {
		return KindWideEmbedded
	}
	return KindMiniLMWasm
}

// IsWideIMWQ reports whether b is a bare wide Q32.32 model image (starts with
// the IMWQ magic).
func IsWideIMWQ(b []byte) bool {
	return len(b) >= len(imwqMagic) && string(b[:len(imwqMagic)]) == imwqMagic
}

// IsWideEmbeddedWasm reports whether b is a wasm module exporting the
// imwq_embedded function — the discriminator of a self-contained wide package
// (or of a bare embed host, which then fails NewWideEmbedded loudly with "no
// embedded model"). It inspects the export section only; it does not validate
// or compile the module.
func IsWideEmbeddedWasm(b []byte) bool {
	names, err := wasmbin.ExportNames(b)
	if err != nil {
		return false
	}
	kind, ok := names[fnWideEmbedded]
	return ok && kind == wasmbin.ExportFunc
}

// IsWideArtifact reports whether b takes the wide path in either form (bare
// .imwq or self-contained package). This is the one predicate Scorer.Promote
// and the candidate evaluator use to decide "wide, not MiniLM".
func IsWideArtifact(b []byte) bool {
	k := ArtifactKindOf(b)
	return k == KindWideIMWQ || k == KindWideEmbedded
}

// WideEmbeddedIsHybrid reports whether a self-contained wide package carries
// the hybrid half (exports imwq_score_hybrid). A package DECLARES its class by
// its exports — a judgment build (--features embedded) has no hybrid export, a
// hybrid build (--features embedded-hybrid) has one — so the installer and the
// gate can refuse a package submitted for the wrong class of intent instead of
// silently scoring it on the other path.
func WideEmbeddedIsHybrid(b []byte) (bool, error) {
	names, err := wasmbin.ExportNames(b)
	if err != nil {
		return false, fmt.Errorf("wasm/runtime: reading exports of a wide package: %w", err)
	}
	if k, ok := names[fnWideEmbedded]; !ok || k != wasmbin.ExportFunc {
		return false, fmt.Errorf("wasm/runtime: not a self-contained wide package (no %q export)", fnWideEmbedded)
	}
	k, ok := names[fnWideScoreHybrid]
	return ok && k == wasmbin.ExportFunc, nil
}
