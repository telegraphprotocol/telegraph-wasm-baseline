package runtime

import (
	"context"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/ethereum/go-ethereum/crypto"
)

// LoRARuntime is one loaded Evaluator adapter on the shared Foundation Model:
// the scoring surface the scorers use (ScoreTokens, MaxTokens, Meta,
// MemoryBytes, Close), so the scorers, the gate and the competition do not
// care which one serves an intent. It holds the Foundation and its validated
// safetensors delta package; the engine keeps the adapter's tensors per instance, loaded
// on first use there.
type LoRARuntime struct {
	f       *Foundation
	adapter *LoRAAdapter
	hash    string // keccak256 of the package (wasm_hash)
	once    sync.Once
	closed  bool
	mu      sync.Mutex
}

// NewLoRARuntime validates pkg against f (every rule of ParseLoRAAdapter,
// with the intent's class) and loads it into one engine instance, so a
// package the engine refuses fails here, not at the first score. wantClass
// "" accepts either class.
func NewLoRARuntime(ctx context.Context, f *Foundation, pkg []byte, wantClass string) (*LoRARuntime, error) {
	if f == nil {
		return nil, &LoRARejectError{Reason: LoRAReasonNoFoundation, Detail: "this node runs no Foundation Model (" + FoundationModelPathEnv + " unset)"}
	}
	a, err := ParseLoRAAdapter(pkg, f.Expect(wantClass))
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: lora adapter: %w", err)
	}
	rt := &LoRARuntime{f: f, adapter: a, hash: hex.EncodeToString(crypto.Keccak256(pkg))}
	inst, err := f.acquire(ctx)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: foundation: %w", err)
	}
	_, lerr := inst.ensureAdapter(ctx, f, rt)
	if lerr != nil && inst.mod.IsClosed() {
		inst.kill()
	}
	f.release(inst)
	if lerr != nil {
		return nil, fmt.Errorf("wasm/runtime: lora adapter: %w", lerr)
	}
	return rt, nil
}

// Foundation is the shared foundation this adapter runs on.
func (r *LoRARuntime) Foundation() *Foundation { return r.f }

// Adapter is the validated package.
func (r *LoRARuntime) Adapter() *LoRAAdapter { return r.adapter }

// Meta is the adapter's metadata in the legacy section's shape.
func (r *LoRARuntime) Meta() AdapterMeta {
	if r == nil {
		return AdapterMeta{}
	}
	return r.adapter.Meta()
}

// MaxTokens is the token cap this adapter is scored with.
func (r *LoRARuntime) MaxTokens() int {
	if r == nil || r.adapter.Header.MaxTokens <= 0 {
		return DefaultFloatMaxTokens
	}
	return r.adapter.Header.MaxTokens
}

// MemoryBytes is what this adapter adds to the resident foundation: the
// package itself, the engine's padded copy of its tensors (R_MAX/rank times
// the deltas) and the per-score clone of them. The foundation's own memory is
// accounted once, by the budget, not per adapter.
func (r *LoRARuntime) MemoryBytes() uint64 {
	if r == nil {
		return 0
	}
	pad := uint64(1)
	if rk := r.adapter.Header.Rank; rk > 0 && r.f.spec.RMax > rk {
		pad = uint64((r.f.spec.RMax + rk - 1) / rk)
	}
	delta := uint64(r.adapter.DeltaBytes)
	return uint64(len(r.adapter.Raw)) + 2*pad*delta
}

// GuestBytes is what the adapter holds in an engine instance while resident:
// its tensors zero-padded to R_MAX (the per-score clone is transient).
func (r *LoRARuntime) GuestBytes() uint64 {
	if r == nil {
		return 0
	}
	pad := uint64(1)
	if rk := r.adapter.Header.Rank; rk > 0 && r.f.spec.RMax > rk {
		pad = uint64((r.f.spec.RMax + rk - 1) / rk)
	}
	return pad * uint64(r.adapter.DeltaBytes)
}

// Identity is everything a score of this adapter is a function of besides
// the input: the foundation (F) and its engine, the adapter (A), the
// tokenizer (T), the ABI, the precision and the token limit.
func (r *LoRARuntime) Identity() ScoreIdentity {
	if r == nil {
		return ScoreIdentity{}
	}
	return ScoreIdentity{FoundationHash: r.f.Hash(), FoundationVersion: r.f.Version(), EngineHash: r.f.EngineHash(),
		AdapterHash: r.hash, TokenizerHash: r.f.TokenizerHash(), ABI: FoundationABIVersion, Precision: ScorePrecision, MaxTokens: r.MaxTokens()}
}

// ScoreTokens scores ids (truncated to MaxTokens) on the
// foundation and returns the exact f32 bits.
func (r *LoRARuntime) ScoreTokens(ctx context.Context, ids []int64) (ModelScore, error) {
	if r == nil || r.adapter == nil {
		return ModelScore{}, fmt.Errorf("wasm/runtime: lora adapter is not loaded")
	}
	r.mu.Lock()
	closed := r.closed
	r.mu.Unlock()
	if closed {
		return ModelScore{}, fmt.Errorf("wasm/runtime: lora adapter is closed")
	}
	if len(ids) == 0 {
		return ModelScore{}, fmt.Errorf("wasm/runtime: score called with no tokens — the text was not tokenised")
	}
	for i, id := range ids {
		if id < 0 {
			return ModelScore{}, fmt.Errorf("wasm/runtime: token %d is negative (%d)", i, id)
		}
	}
	ids = TruncateTokens(ids, r.MaxTokens())
	return r.f.score(ctx, r, ids)
}

// Close unloads the adapter from every engine instance. Idempotent.
func (r *LoRARuntime) Close(ctx context.Context) error {
	if r == nil {
		return nil
	}
	r.once.Do(func() {
		r.mu.Lock()
		r.closed = true
		r.mu.Unlock()
		r.f.unload(r)
	})
	return nil
}
