package runtime

import (
	"context"
	"fmt"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

const (
	fnScoreDeterministic = "imw_score_deterministic"
	fnIMWAlloc           = "imw_alloc"
	fnIMWFree            = "imw_free"

	// ScoreBasisPoints is the fixed-point domain the comparator returns:
	// 10000 == a perfect score of 1.0.
	ScoreBasisPoints = 10000

	// specBufBytes is the size of the two-i64 spec buffer
	// [comparator_code, tolerance] passed to the comparator.
	specBufBytes = 16
)

// DeterministicRuntime holds one instantiated comparator WASM module.
//
// It is separate from Runtime rather than an extra optional export on it
// because the two modules are genuinely different binaries with disjoint
// exports. Runtime hard-requires rank_answer, alloc and dealloc — the
// semantic scorer's memory handshake — none of which the comparator module
// has: it takes two integers and a pointer, allocates nothing, and exports
// only imw_score_deterministic. Loading the comparator through Runtime.New
// would fail on the required-export check.
//
// The same binary backs every deterministic intent (see
// pkg/scoring/comparator), so one loaded instance serves all of them; the
// per-intent difference is entirely in the spec buffer passed per call.
//
// Safe for sequential use. The comparator is pure integer arithmetic with no
// allocation, but calls write the spec buffer to a fixed offset in linear
// memory, so concurrent calls on one instance would race on that buffer.
type DeterministicRuntime struct {
	wrt         wazero.Runtime
	ownsRuntime bool
	mod         api.Module
	mem         api.Memory
	fnScore     api.Function
	fnAlloc     api.Function
	fnFree      api.Function
}

// NewDeterministic instantiates the comparator module from wasmBytes.
func NewDeterministic(ctx context.Context, wasmBytes []byte) (*DeterministicRuntime, error) {
	return newDeterministic(ctx, wasmBytes, cachedRuntimeConfig())
}

// NewDeterministicInterruptible is NewDeterministic for an UNTRUSTED comparator
// that must be abortable when ctx expires — the comparator promotion gate,
// which runs a module someone registered on-chain before anything has vetted
// what it computes. By default wazero never checks the context once a call is
// running, so a module that loops would pin its OS thread and freeze the node
// (see NewInterruptible). The per-call cost of the check is irrelevant here:
// the gate makes a few thousand calls on integer arithmetic.
func NewDeterministicInterruptible(ctx context.Context, wasmBytes []byte) (*DeterministicRuntime, error) {
	// The cache key covers WithCloseOnContextDone (it changes codegen), so this
	// never shares an entry with the non-interruptible build of the same bytes.
	return newDeterministic(ctx, wasmBytes, cachedRuntimeConfig().WithCloseOnContextDone(true))
}

func newDeterministic(ctx context.Context, wasmBytes []byte, cfg wazero.RuntimeConfig) (*DeterministicRuntime, error) {
	if len(wasmBytes) == 0 {
		return nil, fmt.Errorf("wasm/runtime: empty comparator module")
	}

	wrt := wazero.NewRuntimeWithConfig(ctx, cfg)
	mod, err := wrt.Instantiate(ctx, wasmBytes)
	if err != nil {
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: instantiating comparator module: %w", err)
	}

	fn := mod.ExportedFunction(fnScoreDeterministic)
	if fn == nil {
		mod.Close(ctx) //nolint:errcheck
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: missing required export %q", fnScoreDeterministic)
	}

	mem := mod.Memory()
	if mem == nil {
		mod.Close(ctx) //nolint:errcheck
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: comparator module exports no memory")
	}

	// imw_alloc/imw_free are the module's own allocator. The spec buffer is
	// written through it rather than to a fixed offset, matching the reference
	// host (fx_score.mjs): offset 0 is not guaranteed to be free, and writing
	// there would corrupt whatever the module has placed at the start of its
	// linear memory.
	alloc := mod.ExportedFunction(fnIMWAlloc)
	free := mod.ExportedFunction(fnIMWFree)
	if alloc == nil || free == nil {
		mod.Close(ctx) //nolint:errcheck
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: comparator module missing %q/%q", fnIMWAlloc, fnIMWFree)
	}

	return &DeterministicRuntime{
		wrt:         wrt,
		ownsRuntime: true,
		mod:         mod,
		mem:         mem,
		fnScore:     fn,
		fnAlloc:     alloc,
		fnFree:      free,
	}, nil
}

// ScoreDeterministic compares a miner value against a truth value and returns
// the score in basis points, 0..10000.
//
// Both values must already be on the same fixed-point scale — the WASM cannot
// tell them apart from correctly-scaled ones, so a mismatch does not error, it
// silently scores 0 and looks exactly like a wrong answer. Scaling is the
// caller's responsibility (comparator.ScaleFloat).
func (r *DeterministicRuntime) ScoreDeterministic(ctx context.Context, minerValue, truthValue int64, spec [2]int64) (int32, error) {
	if r.fnScore == nil {
		return 0, fmt.Errorf("wasm/runtime: %s not exported", fnScoreDeterministic)
	}
	if truthValue == 0 {
		// RelTol divides by the truth value. A zero truth is a failure to
		// fetch ground truth, not a statement that every miner is wrong, so
		// it must not be scored.
		return 0, fmt.Errorf("wasm/runtime: truth value is zero; ground truth was not resolved")
	}

	// Allocate the spec buffer through the module's own allocator.
	allocRes, err := r.fnAlloc.Call(ctx, specBufBytes)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: %s: %w", fnIMWAlloc, err)
	}
	if len(allocRes) == 0 || allocRes[0] == 0 {
		return 0, fmt.Errorf("wasm/runtime: %s returned a null pointer", fnIMWAlloc)
	}
	specPtr := uint32(allocRes[0])
	defer r.fnFree.Call(ctx, uint64(specPtr), specBufBytes) //nolint:errcheck

	if !r.mem.WriteUint64Le(specPtr, uint64(spec[0])) {
		return 0, fmt.Errorf("wasm/runtime: writing comparator code to memory")
	}
	if !r.mem.WriteUint64Le(specPtr+8, uint64(spec[1])) {
		return 0, fmt.Errorf("wasm/runtime: writing tolerance to memory")
	}

	// Argument order is (ground_truth_value, miner_value, spec_ptr) — truth
	// FIRST. See imw_score_deterministic in imw_wasm/src/lib.rs. The order is
	// easy to get backwards and mostly harmless when it is: RelTol divides by
	// |truth|, so swapping the two only changes the denominator and still
	// returns a plausible score. It is wrong by a few percent near the
	// tolerance edge, which is exactly where ranking decisions are made.
	res, err := r.fnScore.Call(ctx, uint64(truthValue), uint64(minerValue), uint64(specPtr))
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: %s: %w", fnScoreDeterministic, err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: %s returned no values", fnScoreDeterministic)
	}

	// The export signals rejection as a NEGATIVE return carrying -(ScoreError),
	// not as a trap — so a raw cast would turn error code 6 (NoScore) into a
	// score of -6 and, once divided by 10000, into a silent ~0 that is
	// indistinguishable from a miner that answered badly.
	raw := int64(res[0])
	if raw < 0 {
		return 0, fmt.Errorf("wasm/runtime: %s rejected the input (ScoreError %d)", fnScoreDeterministic, -raw)
	}
	if raw > ScoreBasisPoints {
		return 0, fmt.Errorf("wasm/runtime: %s returned out-of-domain score %d", fnScoreDeterministic, raw)
	}
	return int32(raw), nil
}

// Close releases the module and, if it owns one, the wazero runtime.
func (r *DeterministicRuntime) Close(ctx context.Context) error {
	if r.mod != nil {
		if err := r.mod.Close(ctx); err != nil {
			return err
		}
		r.mod = nil
	}
	if r.ownsRuntime && r.wrt != nil {
		if err := r.wrt.Close(ctx); err != nil {
			return err
		}
		r.wrt = nil
	}
	return nil
}

// ScoreToFloat converts a basis-point score to the [0,1] float domain used by
// RankedMiner.Score, so comparator results land in the same slot as semantic
// ones and the ranker needs no change.
func ScoreToFloat(bps int32) float32 {
	return float32(bps) / float32(ScoreBasisPoints)
}
