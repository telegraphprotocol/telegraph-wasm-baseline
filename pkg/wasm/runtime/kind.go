package runtime

import (
	"fmt"
	"strings"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
)

// ArtifactKind names the shapes of scoring artifact the promote path can
// receive, so the scorer, the candidate evaluator and Stage 1 validation branch
// on ONE definition rather than each re-checking bytes their own way. A wrong
// artifact that "loads" and returns a confident number is the failure this
// whole layer exists to avoid, so the kind is decided in exactly one place.
type ArtifactKind uint8

const (
	// KindOther is anything that is not a model adapter: a comparator module
	// (deterministic intents), a retired MiniLM rank_answer module, or bytes
	// that are not a module at all. Callers route it by intent class.
	KindOther ArtifactKind = iota

	// KindFloatModel is a LEGACY full-model float adapter: a WASI command
	// (exports `_start`) carrying a whole ONNX float32 model and its engine.
	// No longer accepted: it is recognised only to be rejected by name
	// (LegacyFullModelReason). Adapters are KindLoRAAdapter deltas on the
	// shared Foundation Model.
	KindFloatModel

	// KindLoRAAdapter is an Evaluator adapter for the shared Foundation Model:
	// a safetensors LoRA delta package (lora_adapter.go), scored by LoRARuntime.
	KindLoRAAdapter

	// KindRetiredQ32 is a Q32.32 wide artifact — a bare .imwq model image
	// (IMWQ magic) or a module exporting imwq_* functions (the wide
	// interpreter, or a self-contained wide package). The format is retired:
	// it is refused at registration and never loaded or scored.
	KindRetiredQ32
)

// RetiredQ32Reason is the one message every path that meets a Q32.32 wide
// artifact reports (Stage 1, the gate, promotion, startup hydration).
const RetiredQ32Reason = "Q32.32 wide adapters are retired; package the adapter as a safetensors delta for the shared Foundation Model"

// LegacyFullModelReason is the one message every path that meets a legacy
// full-model float adapter (a WASI command with the whole model baked in)
// reports: Stage 1, the fetch (without downloading the ~600 MB), the gate,
// promotion and startup hydration. The format is no longer accepted; an
// adapter already registered in it must be repackaged as a delta and
// re-registered.
const LegacyFullModelReason = "legacy_full_model: full-model float adapters are no longer accepted; repackage the adapter as a safetensors delta for the shared Foundation Model and re-register it"

// String names the kind for logs and errors.
func (k ArtifactKind) String() string {
	switch k {
	case KindOther:
		return "other"
	case KindFloatModel:
		return "float-model"
	case KindLoRAAdapter:
		return "lora-adapter"
	case KindRetiredQ32:
		return "retired-q32"
	default:
		return fmt.Sprintf("ArtifactKind(%d)", uint8(k))
	}
}

// q32Magic is the four ASCII bytes a Q32.32 model image (.imwq) begins with.
const q32Magic = "IMWQ"

// q32ExportPrefix prefixes every export of a Q32.32 wide module.
const q32ExportPrefix = "imwq_"

// ArtifactKindOf classifies raw promote-path bytes:
//
//   - a safetensors header whose __metadata__ abi is lora-* -> KindLoRAAdapter
//   - leading "IMWQ" magic, or a module exporting imwq_*  -> KindRetiredQ32
//   - a module exporting a `_start` function              -> KindFloatModel
//   - anything else                                       -> KindOther
//
// Deliberately a cheap, total function: it never errors and never compiles the
// module. It reads section headers only up to the export section, which
// precedes the code and data sections, so it is microseconds on a 600 MB
// legacy module and works on the first megabyte of a download. Whether a delta
// package is actually loadable is ParseLoRAAdapter's (and the engine's) job;
// this only decides which path it takes.
func ArtifactKindOf(b []byte) ArtifactKind {
	if IsLoRAPackage(b) {
		return KindLoRAAdapter
	}
	if IsRetiredQ32(b) {
		return KindRetiredQ32
	}
	if IsFloatModel(b) {
		return KindFloatModel
	}
	return KindOther
}

// comparatorExport is the export of the comparator ABI (deterministic
// intents), which did not change with the adapter format.
const comparatorExport = "imw_score_deterministic"

// IsRetiredQ32 reports whether b is a Q32.32 wide ADAPTER artifact: a bare
// .imwq image, or a wasm module with imwq_* exports that is not also a
// comparator module.
//
// The exception matters: the shared Q32.32 interpreter build (wide_wasm.wasm)
// also exports imw_score_deterministic and may be registered as a
// deterministic intent's comparator. As a comparator its ABI is unchanged, so
// it is routed by intent class like any comparator module — never refused, and
// never skipped at restart, as a retired adapter. A judgment/hybrid
// registration of it is still refused (HasQ32Exports).
func IsRetiredQ32(b []byte) bool {
	if len(b) >= len(q32Magic) && string(b[:len(q32Magic)]) == q32Magic {
		return true
	}
	names, err := wasmbin.ExportNames(b)
	if err != nil {
		return false
	}
	if k, ok := names[comparatorExport]; ok && k == wasmbin.ExportFunc {
		return false
	}
	return hasQ32Export(names)
}

// HasQ32Exports reports whether b is a bare .imwq or a module with any imwq_*
// export, comparator or not. Stage 1 uses it for judgment/hybrid intents,
// where no imwq_* module is acceptable.
func HasQ32Exports(b []byte) bool {
	if len(b) >= len(q32Magic) && string(b[:len(q32Magic)]) == q32Magic {
		return true
	}
	names, err := wasmbin.ExportNames(b)
	return err == nil && hasQ32Export(names)
}

func hasQ32Export(names map[string]byte) bool {
	for n := range names {
		if strings.HasPrefix(n, q32ExportPrefix) {
			return true
		}
	}
	return false
}

// IsFloatModel reports whether b is a LEGACY full-model float adapter: a wasm
// module (not a retired Q32.32 one) exporting a `_start` function, a WASI
// command. Such an artifact is rejected (LegacyFullModelReason).
func IsFloatModel(b []byte) bool {
	names, err := wasmbin.ExportNames(b)
	if err != nil || hasQ32Export(names) {
		return false
	}
	k, ok := names[fnStart]
	return ok && k == wasmbin.ExportFunc
}
