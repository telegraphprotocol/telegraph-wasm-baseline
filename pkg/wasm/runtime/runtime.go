// Package runtime loads and drives the Telegraph WASM scoring module.
//
// Responsibilities:
//   - Instantiate the WASM module via wazero
//   - Manage WASM linear memory (alloc / dealloc / grow)
//   - Expose typed Go functions that map to each WASM export
//
// Callers should use Pool rather than Runtime directly in production.
package runtime

import (
	"context"
	"fmt"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

const (
	fnRankAnswer       = "rank_answer"
	fnRankAnswerCached = "rank_answer_cached"
	fnBreakdownAnswer  = "breakdown_answer"
	fnEmbed            = "embed"
	fnCosineSim        = "cosine_sim"
	fnBM25Score        = "bm25_score"
	fnAlloc            = "alloc"
	fnDealloc          = "dealloc"

	// EmbedDim is the output dimension of the MiniLM-L6-v2 model.
	EmbedDim = 384

	// BreakdownDim is the number of signals returned by breakdown_answer.
	// Layout: [relevance, correctness, lexical, length_quality, composite]
	BreakdownDim = 5

	// MaxTextBytes caps individual string inputs (128 KiB).
	MaxTextBytes = 128 * 1024

	// MaxVecDim caps raw vector inputs to CosineSim.
	MaxVecDim = 16_384
)

// Runtime holds one instantiated WASM module.
// Safe for sequential use; use Pool for concurrent access.
type Runtime struct {
	wrt wazero.Runtime
	// ownsRuntime says whether Close may close wrt. True for a Runtime built
	// by New(), which creates a runtime for itself; false for pooled instances,
	// which share one runtime owned by the Pool. Closing a shared runtime from
	// a single instance would tear down every sibling instance with it.
	ownsRuntime  bool
	mod          api.Module
	mem          api.Memory
	fnRank       api.Function
	fnRankCached api.Function
	fnBreakdown  api.Function
	fnEmbed      api.Function
	fnCosine     api.Function
	fnBM25       api.Function
	fnAlloc      api.Function
	fnDealloc    api.Function
}

// New compiles and instantiates the WASM module on a wazero runtime of its own,
// which it owns and releases on Close.
//
// Prefer NewPool for more than one instance: calling New in a loop compiles the
// bytes once per instance and keeps that many copies of the data segments.
func New(ctx context.Context, wasmBytes []byte) (*Runtime, error) {
	return newWithConfig(ctx, wasmBytes, wazero.NewRuntimeConfig())
}

// NewInterruptible is New for a module that must be abortable when ctx expires.
//
// By default wazero never looks at the context once a call is running, so a
// module whose exported function does not return pins its OS thread: the Go
// call never comes back, and runtime-generated machine code is exempt from
// async preemption, so the next GC stop-the-world blocks behind it and the
// whole process stops answering. That is the node freeze.
//
// This costs roughly 22x on an embedding module — the check is a trampoline at
// every Wasm loop header — so it is deliberately NOT the default. Use it only
// where a module is untrusted and the work is small: Stage 1 structural
// validation, which is the gate that decides whether a module is allowed
// anywhere near the scoring path at all. Callers must budget for the slowdown
// (see validationTimeout).
//
// Residual risk, deliberately accepted: Stage 2 candidate evaluation still uses
// plain New, so a module that loops only on inputs unlike Stage 1s six fixed
// probes passes the gate and can still freeze the process during replay. This is
// not merely a perf trade — it is a targeted-adversary hole. The realistic fix is
// a supervisor that hard-kills on a wall-clock bound, not making the hot path
// interruptible.
func NewInterruptible(ctx context.Context, wasmBytes []byte) (*Runtime, error) {
	return newWithConfig(ctx, wasmBytes,
		wazero.NewRuntimeConfig().WithCloseOnContextDone(true))
}

func newWithConfig(ctx context.Context, wasmBytes []byte, cfg wazero.RuntimeConfig) (*Runtime, error) {
	wrt := wazero.NewRuntimeWithConfig(ctx, cfg)

	compiled, err := wrt.CompileModule(ctx, wasmBytes)
	if err != nil {
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: compile: %w", err)
	}
	defer compiled.Close(ctx) //nolint:errcheck

	r, err := instantiate(ctx, wrt, compiled, "telegraph-scoring")
	if err != nil {
		wrt.Close(ctx) //nolint:errcheck
		return nil, err
	}
	r.ownsRuntime = true

	return r, nil
}

// instantiate builds one Runtime from an already-compiled module on wrt, under
// the module name `name`.
//
// It never closes wrt on failure — wrt may be shared with sibling instances,
// and only its owner may close it. Callers that own wrt (New, and NewPool on
// its error path) close it themselves.
func instantiate(ctx context.Context, wrt wazero.Runtime, compiled wazero.CompiledModule, name string) (*Runtime, error) {
	mod, err := wrt.InstantiateModule(ctx, compiled,
		wazero.NewModuleConfig().WithName(name),
	)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: instantiate: %w", err)
	}

	mem := mod.Memory()
	if mem == nil {
		mod.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: module exports no linear memory")
	}

	r := &Runtime{wrt: wrt, mod: mod, mem: mem}

	// Required exports — fail fast if missing. rank_answer is the only
	// scoring function every caller actually depends on: the live scoring
	// path (pkg/scoring/scorer.go's ScoreOne), Stage 1 structural
	// validation, and Stage 2 replay evaluation all call it. alloc/dealloc
	// aren't scoring logic at all, they're the memory handshake every other
	// export needs to receive string input.
	for _, req := range []struct {
		name string
		dst  *api.Function
	}{
		{fnRankAnswer, &r.fnRank},
		{fnAlloc, &r.fnAlloc},
		{fnDealloc, &r.fnDealloc},
	} {
		fn := mod.ExportedFunction(req.name)
		if fn == nil {
			mod.Close(ctx) //nolint:errcheck
			return nil, fmt.Errorf("wasm/runtime: missing required export %q", req.name)
		}
		*req.dst = fn
	}

	// Optional exports — a module may not have these. Callers of the
	// corresponding Go methods (BreakdownAnswer, Embed, CosineSim,
	// BM25Score, RankAnswerCached) get a plain error if the export is
	// missing, not a panic — see each method below. Note these are set
	// independently of each other: a module can have embed without
	// rank_answer_cached, or vice versa. Callers that need BOTH (e.g.
	// candidate_eval.go's cached scoring path) must check
	// SupportsCachedRanking() rather than probing one export to infer the
	// other's presence. breakdown_answer is only ever called by
	// Scorer.BreakdownOne, an admin/debug introspection path — neither Stage
	// 1 validation nor Stage 2 evaluation calls it, so it doesn't need to be
	// required for a module to load and score live traffic.
	r.fnBreakdown = mod.ExportedFunction(fnBreakdownAnswer)
	r.fnEmbed = mod.ExportedFunction(fnEmbed)
	r.fnCosine = mod.ExportedFunction(fnCosineSim)
	r.fnBM25 = mod.ExportedFunction(fnBM25Score)
	r.fnRankCached = mod.ExportedFunction(fnRankAnswerCached)

	return r, nil
}

// SupportsCachedRanking reports whether the loaded module exports both
// embed and rank_answer_cached — the two functions the cached scoring path
// (pkg/scoring/candidate_eval.go's scoreRows) actually needs. These two
// exports are set independently in New() (either can be present without
// the other), so probing one to infer the other's availability is unsafe —
// a module with embed but not rank_answer_cached would pass an Embed-only
// probe, then hard-fail deep in the scoring loop when rank_answer_cached
// turns out to be missing, instead of falling back cleanly up front.
func (r *Runtime) SupportsCachedRanking() bool {
	return r.fnEmbed != nil && r.fnRankCached != nil
}

// RankAnswer calls the WASM rank_answer export. Returns composite score in [0, 1].
func (r *Runtime) RankAnswer(ctx context.Context, question, groundTruth, minerAnswer string) (float32, error) {
	if err := checkText("question", question); err != nil {
		return 0, err
	}
	if err := checkText("ground_truth", groundTruth); err != nil {
		return 0, err
	}
	if err := checkText("miner_answer", minerAnswer); err != nil {
		return 0, err
	}

	qPtr, qLen, free1, err := r.writeStr(ctx, question)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write question: %w", err)
	}
	defer free1(ctx)

	gtPtr, gtLen, free2, err := r.writeStr(ctx, groundTruth)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write ground_truth: %w", err)
	}
	defer free2(ctx)

	maPtr, maLen, free3, err := r.writeStr(ctx, minerAnswer)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write miner_answer: %w", err)
	}
	defer free3(ctx)

	res, err := r.fnRank.Call(ctx,
		uint64(qPtr), uint64(qLen),
		uint64(gtPtr), uint64(gtLen),
		uint64(maPtr), uint64(maLen),
	)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: rank_answer: %w", err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: rank_answer returned no values")
	}
	return decodeF32(res[0]), nil
}

// RankAnswerRaw calls rank_answer directly, skipping the non-empty guard that
// RankAnswer applies to miner_answer. This is used ONLY by candidate WASM
// structural validation (pkg/listener) to probe a module's own blank-input
// handling — the normal scoring path (Scorer/Pool.RankAnswer) already blocks
// empty answers before they'd ever reach a WASM module, candidate or live,
// so this method has no effect on production scoring behavior.
func (r *Runtime) RankAnswerRaw(ctx context.Context, question, groundTruth, minerAnswer string) (float32, error) {
	if err := checkText("question", question); err != nil {
		return 0, err
	}
	if err := checkText("ground_truth", groundTruth); err != nil {
		return 0, err
	}
	if len(minerAnswer) > MaxTextBytes {
		return 0, fmt.Errorf("wasm/runtime: miner_answer too large (%d > %d bytes)", len(minerAnswer), MaxTextBytes)
	}
	// Deliberately no non-empty check on minerAnswer — that's the point.

	qPtr, qLen, free1, err := r.writeStr(ctx, question)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write question: %w", err)
	}
	defer free1(ctx)

	gtPtr, gtLen, free2, err := r.writeStr(ctx, groundTruth)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write ground_truth: %w", err)
	}
	defer free2(ctx)

	maPtr, maLen, free3, err := r.writeStr(ctx, minerAnswer)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write miner_answer: %w", err)
	}
	defer free3(ctx)

	res, err := r.fnRank.Call(ctx,
		uint64(qPtr), uint64(qLen),
		uint64(gtPtr), uint64(gtLen),
		uint64(maPtr), uint64(maLen),
	)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: rank_answer: %w", err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: rank_answer returned no values")
	}
	return decodeF32(res[0]), nil
}

// RankAnswerCached calls rank_answer_cached, reusing precomputed question and
// ground-truth embedding vectors (typically from a prior Embed call) instead
// of re-embedding that text. Only miner_answer is freshly embedded inside
// WASM. Intended for callers that score many miner answers against the same
// (question, ground_truth) pair — e.g. Stage 2 replay evaluation
// (pkg/scoring/candidate_eval.go) scoring every miner that answered a given
// intent — where re-embedding the same question/ground_truth on every call
// is pure waste (embedding is the dominant cost of scoring).
//
// Returns an error if the loaded WASM doesn't export rank_answer_cached
// (e.g. the test-double module, or any module built before this export
// existed) — callers should fall back to RankAnswer in that case, not treat
// this as a hard failure. Callers scoring a batch should check
// SupportsCachedRanking() up front rather than relying on this per-call
// error, since a mid-batch failure here after already partially scoring via
// the cached path is harder to recover cleanly from.
//
// Computes the composite score identically to RankAnswer — same weights,
// same formula, defined once in lib.rs — this is NOT a separate
// reimplementation on the Go side.
func (r *Runtime) RankAnswerCached(ctx context.Context, qVec, gtVec []float32, groundTruth, minerAnswer string) (float32, error) {
	if r.fnRankCached == nil {
		return 0, fmt.Errorf("wasm/runtime: rank_answer_cached not exported")
	}
	if len(qVec) != EmbedDim {
		return 0, fmt.Errorf("wasm/runtime: qVec must be length %d, got %d", EmbedDim, len(qVec))
	}
	if len(gtVec) != EmbedDim {
		return 0, fmt.Errorf("wasm/runtime: gtVec must be length %d, got %d", EmbedDim, len(gtVec))
	}
	if err := checkText("ground_truth", groundTruth); err != nil {
		return 0, err
	}
	if err := checkText("miner_answer", minerAnswer); err != nil {
		return 0, err
	}

	qPtr, freeQ, err := r.writeF32Vec(ctx, qVec)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write qVec: %w", err)
	}
	defer freeQ(ctx)

	gtVecPtr, freeGtVec, err := r.writeF32Vec(ctx, gtVec)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write gtVec: %w", err)
	}
	defer freeGtVec(ctx)

	gtPtr, gtLen, freeGtStr, err := r.writeStr(ctx, groundTruth)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write ground_truth: %w", err)
	}
	defer freeGtStr(ctx)

	maPtr, maLen, freeMa, err := r.writeStr(ctx, minerAnswer)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: write miner_answer: %w", err)
	}
	defer freeMa(ctx)

	res, err := r.fnRankCached.Call(ctx,
		uint64(qPtr), uint64(gtVecPtr),
		uint64(gtPtr), uint64(gtLen),
		uint64(maPtr), uint64(maLen),
	)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: rank_answer_cached: %w", err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: rank_answer_cached returned no values")
	}
	return decodeF32(res[0]), nil
}

// BreakdownSignals holds the five per-signal scores returned by BreakdownAnswer.
// Field order matches WASM BREAKDOWN_BUF layout indices 0-4.
type BreakdownSignals struct {
	Relevance     float32 // cosine(question,     miner_answer) — weight 0.25
	Correctness   float32 // cosine(ground_truth, miner_answer) — weight 0.50
	Lexical       float32 // bm25(ground_truth,   miner_answer) — weight 0.15
	LengthQuality float32 // sigmoid length penalty             — weight 0.10
	Composite     float32 // weighted sum, clamped to [0,1]
}

// BreakdownAnswer calls the WASM breakdown_answer export and returns all five
// scoring signals in one WASM call. This is the canonical source of the
// composite formula — pkg/scoring must not reimplement it.
func (r *Runtime) BreakdownAnswer(ctx context.Context, question, groundTruth, minerAnswer string) (*BreakdownSignals, error) {
	if r.fnBreakdown == nil {
		return nil, fmt.Errorf("wasm/runtime: breakdown_answer not exported")
	}
	if err := checkText("question", question); err != nil {
		return nil, err
	}
	if err := checkText("ground_truth", groundTruth); err != nil {
		return nil, err
	}
	// minerAnswer may be empty — WASM returns all-zeros in that case, not an error

	qPtr, qLen, free1, err := r.writeStr(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: write question: %w", err)
	}
	defer free1(ctx)

	gtPtr, gtLen, free2, err := r.writeStr(ctx, groundTruth)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: write ground_truth: %w", err)
	}
	defer free2(ctx)

	maPtr, maLen, free3, err := r.writeStr(ctx, minerAnswer)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: write miner_answer: %w", err)
	}
	defer free3(ctx)

	res, err := r.fnBreakdown.Call(ctx,
		uint64(qPtr), uint64(qLen),
		uint64(gtPtr), uint64(gtLen),
		uint64(maPtr), uint64(maLen),
	)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: breakdown_answer: %w", err)
	}
	if len(res) == 0 {
		return nil, fmt.Errorf("wasm/runtime: breakdown_answer returned no values")
	}

	bufPtr := uint32(res[0])
	signals := &BreakdownSignals{}
	vals := [BreakdownDim]*float32{
		&signals.Relevance,
		&signals.Correctness,
		&signals.Lexical,
		&signals.LengthQuality,
		&signals.Composite,
	}
	for i, dst := range vals {
		v, ok := r.mem.ReadFloat32Le(bufPtr + uint32(i)*4)
		if !ok {
			return nil, fmt.Errorf("wasm/runtime: breakdown_answer read signal[%d] failed", i)
		}
		*dst = v
	}

	return signals, nil
}

// Embed runs MiniLM-L6-v2 on text and returns the 384-dim vector.
func (r *Runtime) Embed(ctx context.Context, text string) ([]float32, error) {
	if r.fnEmbed == nil {
		return nil, fmt.Errorf("wasm/runtime: embed not exported")
	}
	if err := checkText("text", text); err != nil {
		return nil, err
	}
	ptr, length, free, err := r.writeStr(ctx, text)
	if err != nil {
		return nil, err
	}
	defer free(ctx)

	res, err := r.fnEmbed.Call(ctx, uint64(ptr), uint64(length))
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: embed call: %w", err)
	}
	if len(res) == 0 {
		return nil, fmt.Errorf("wasm/runtime: embed returned no values")
	}
	vecPtr := uint32(res[0])
	vec := make([]float32, EmbedDim)
	for i := range vec {
		v, ok := r.mem.ReadFloat32Le(vecPtr + uint32(i)*4)
		if !ok {
			return nil, fmt.Errorf("wasm/runtime: embed read dim %d failed", i)
		}
		vec[i] = v
	}
	return vec, nil
}

// CosineSim computes cosine similarity of two float32 vectors.
func (r *Runtime) CosineSim(ctx context.Context, a, b []float32) (float32, error) {
	if r.fnCosine == nil {
		return 0, fmt.Errorf("wasm/runtime: cosine_sim not exported")
	}
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("wasm/runtime: invalid vector lengths %d/%d", len(a), len(b))
	}
	if len(a) > MaxVecDim {
		return 0, fmt.Errorf("wasm/runtime: vector length %d exceeds max %d", len(a), MaxVecDim)
	}
	dim := uint32(len(a))
	offA, offB := uint32(0), dim*4
	if err := r.growIfNeeded(offB + dim*4); err != nil {
		return 0, err
	}
	for i, v := range a {
		r.mem.WriteFloat32Le(offA+uint32(i)*4, v) //nolint:errcheck
	}
	for i, v := range b {
		r.mem.WriteFloat32Le(offB+uint32(i)*4, v) //nolint:errcheck
	}
	res, err := r.fnCosine.Call(ctx, uint64(offA), uint64(offB), uint64(dim))
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: cosine_sim: %w", err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: cosine_sim returned no values")
	}
	return decodeF32(res[0]), nil
}

// BM25Score returns the BM25 lexical score of doc against query.
func (r *Runtime) BM25Score(ctx context.Context, query, doc string) (float32, error) {
	if r.fnBM25 == nil {
		return 0, fmt.Errorf("wasm/runtime: bm25_score not exported")
	}
	qPtr, qLen, free1, err := r.writeStr(ctx, query)
	if err != nil {
		return 0, err
	}
	defer free1(ctx)
	dPtr, dLen, free2, err := r.writeStr(ctx, doc)
	if err != nil {
		return 0, err
	}
	defer free2(ctx)
	res, err := r.fnBM25.Call(ctx, uint64(qPtr), uint64(qLen), uint64(dPtr), uint64(dLen))
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: bm25_score: %w", err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: bm25_score returned no values")
	}
	return decodeF32(res[0]), nil
}

// writeF32Vec allocates space in WASM memory via the module's own alloc()
// export, writes vec into it, and returns the pointer plus a free func.
//
// Unlike CosineSim's approach above (writing floats at hardcoded offsets 0
// and dim*4 without going through alloc()), this always asks the module's
// own allocator for the space — it can't collide with static data or
// allocator bookkeeping the linker placed at low addresses, since the
// module's own alloc() is the one deciding where free space is.
func (r *Runtime) writeF32Vec(ctx context.Context, vec []float32) (ptr uint32, free func(context.Context), err error) {
	size := uint32(len(vec)) * 4
	res, err := r.fnAlloc.Call(ctx, uint64(size))
	if err != nil {
		return 0, nil, fmt.Errorf("wasm/runtime: alloc for f32 vec (%d bytes): %w", size, err)
	}
	if len(res) == 0 {
		return 0, nil, fmt.Errorf("wasm/runtime: alloc returned no values")
	}
	p := uint32(res[0])
	for i, v := range vec {
		if !r.mem.WriteFloat32Le(p+uint32(i)*4, v) {
			return 0, nil, fmt.Errorf("wasm/runtime: write f32 vec failed at index %d", i)
		}
	}
	freeFn := func(ctx context.Context) {
		_, _ = r.fnDealloc.Call(ctx, uint64(p), uint64(size)) //nolint:errcheck
	}
	return p, freeFn, nil
}

// Close releases this instance's module, and the underlying wazero runtime
// too if this Runtime owns it (i.e. it came from New rather than a Pool).
// A pooled instance shares its runtime with its siblings and leaves closing
// it to Pool.Close.
func (r *Runtime) Close(ctx context.Context) error {
	var first error
	if r.mod != nil {
		if err := r.mod.Close(ctx); err != nil {
			first = fmt.Errorf("wasm/runtime: module close: %w", err)
		}
	}
	if r.wrt != nil && r.ownsRuntime {
		if err := r.wrt.Close(ctx); err != nil && first == nil {
			first = fmt.Errorf("wasm/runtime: runtime close: %w", err)
		}
	}
	return first
}
