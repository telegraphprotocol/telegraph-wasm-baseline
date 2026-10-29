package runtime

import (
	"context"
	"fmt"
)

// ModelRuntime is the scoring surface of a loaded Evaluator adapter
// (*LoRARuntime: a safetensors delta on the shared Foundation Model).
type ModelRuntime interface {
	ScoreTokens(ctx context.Context, ids []int64) (ModelScore, error)
	MaxTokens() int
	Meta() AdapterMeta
	MemoryBytes() uint64
	Identity() ScoreIdentity
	Close(ctx context.Context) error
}

// NewModelRuntime loads artifact as an Evaluator adapter: a safetensors delta
// package on the process's active foundation. A legacy full-model float wasm
// and a retired Q32.32 artifact are refused by name; anything else is not an
// adapter. wantClass is the intent's class ("" accepts either).
func NewModelRuntime(ctx context.Context, artifact []byte, wantClass string) (ModelRuntime, error) {
	switch ArtifactKindOf(artifact) {
	case KindLoRAAdapter:
		return NewLoRARuntime(ctx, ActiveFoundation(), artifact, wantClass)
	case KindFloatModel:
		return nil, fmt.Errorf("wasm/runtime: %s", LegacyFullModelReason)
	case KindRetiredQ32:
		return nil, fmt.Errorf("wasm/runtime: %s", RetiredQ32Reason)
	default:
		return nil, fmt.Errorf("wasm/runtime: not an Evaluator adapter (not a safetensors delta package)")
	}
}
