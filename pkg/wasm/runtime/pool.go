package runtime

import (
	"context"
	"fmt"
	"sync"

	"github.com/tetratelabs/wazero"
)

// Pool manages N Runtime instances and issues them on demand via a buffered
// channel. This allows concurrent callers to score miners in parallel without
// serialising through a single mutex.
//
// Typical usage:
//
//	pool, err := NewPool(ctx, wasmBytes, runtime.NumCPU())
//	defer pool.Close(ctx)
//
//	score, err := pool.RankAnswer(ctx, question, groundTruth, minerAnswer)
type Pool struct {
	ch   chan *Runtime
	size int

	// wrt is the one wazero runtime holding every instance in this pool, and
	// the pool owns it — the instances do not (see Runtime.ownsRuntime).
	// Instances share only the immutable compiled artifact; each still gets its
	// own linear memory, so scoring stays isolated. Closing wrt is what
	// releases the pool's compiled code, so Close must not skip it.
	wrt wazero.Runtime

	// closeOnce/closeErr replace the former mu+done pair. The invariant that
	// makes this safe: there are always exactly `size` Runtime tokens live —
	// each is either sitting in ch, checked out, or in flight back into ch
	// via checkin (which always fires exactly once per checkout, via defer
	// in every caller). So by the time Close's drain loop has received
	// `size` items, every outstanding checkin has already completed and
	// there is nothing left to race against close(ch). That means checkin
	// itself needs no lock and no "are we closed" guard — it can be a plain
	// unconditional send — and Close only needs idempotency protection,
	// which sync.Once gives without holding any lock across the blocking
	// drain.
	//
	// The previous implementation held p.mu for the entire drain in Close,
	// which checkin also needed to return a runtime — so Close(), called
	// concurrently with any in-flight scoring call, deadlocked: Close held
	// the lock waiting for a runtime that could only come back via checkin,
	// and checkin blocked forever waiting for the lock Close was holding.
	// That was harmless while Close only ran at process shutdown, but
	// Scorer.Promote (pkg/scoring/scorer.go) now calls Close on the old
	// pool from a background goroutine specifically while scoring may still
	// be in flight, making the deadlock reachable in normal operation.
	closeOnce sync.Once
	closeErr  error
}

// NewPool compiles `wasmBytes` once and instantiates `size` Runtime instances
// from that single compiled artifact, placing them into the pool. If any
// instance fails to initialise, everything created so far is released and the
// error is returned.
func NewPool(ctx context.Context, wasmBytes []byte, size int) (*Pool, error) {
	if size <= 0 {
		size = 1
	}

	wrt := wazero.NewRuntime(ctx)

	compiled, err := wrt.CompileModule(ctx, wasmBytes)
	if err != nil {
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime pool: compile: %w", err)
	}
	// Released once every instance has been built. Instantiated modules keep
	// working after their CompiledModule is closed — the pre-existing New()
	// relies on the same property — so holding it for the pool's whole life
	// would just pin memory we no longer need.
	defer compiled.Close(ctx) //nolint:errcheck

	ch := make(chan *Runtime, size)

	for i := 0; i < size; i++ {
		// Anonymous, not named. Two reasons, and the second is the important
		// one. wazero rejects a second module instantiated under a name already
		// live in the same runtime, so sharing a runtime rules out reusing one
		// name. And a *named* module is entered in the runtime's name table,
		// where sibling instances become resolvable as imports — a module
		// importing "telegraph-scoring-0" would link against another pool
		// slot's memory and functions, the one way pool slots could share
		// mutable state. An empty name skips the table entirely.
		rt, err := instantiate(ctx, wrt, compiled, "")
		if err != nil {
			// Closing the shared runtime tears down every instance already
			// built into it, so there is nothing else to drain.
			wrt.Close(ctx) //nolint:errcheck
			return nil, fmt.Errorf("wasm/runtime pool: create instance %d: %w", i, err)
		}
		ch <- rt
	}

	return &Pool{ch: ch, size: size, wrt: wrt}, nil
}

// RankAnswer checks out a Runtime, calls RankAnswer, then returns it.
func (p *Pool) RankAnswer(ctx context.Context, question, groundTruth, minerAnswer string) (float32, error) {
	rt, err := p.checkout(ctx)
	if err != nil {
		return 0, err
	}
	defer p.checkin(rt)
	return rt.RankAnswer(ctx, question, groundTruth, minerAnswer)
}

// BreakdownAnswer checks out a Runtime, calls BreakdownAnswer, then returns it.
// Use this instead of reimplementing the composite formula in Go.
func (p *Pool) BreakdownAnswer(ctx context.Context, question, groundTruth, minerAnswer string) (*BreakdownSignals, error) {
	rt, err := p.checkout(ctx)
	if err != nil {
		return nil, err
	}
	defer p.checkin(rt)
	return rt.BreakdownAnswer(ctx, question, groundTruth, minerAnswer)
}

// Embed checks out a Runtime, calls Embed, then returns it.
func (p *Pool) Embed(ctx context.Context, text string) ([]float32, error) {
	rt, err := p.checkout(ctx)
	if err != nil {
		return nil, err
	}
	defer p.checkin(rt)
	return rt.Embed(ctx, text)
}

// SupportsCachedRanking reports whether the pooled module exports both embed
// and rank_answer_cached, so callers can pre-embed a question/ground_truth pair
// once and score many answers against it. Uses a background-context checkout;
// the check is a cheap in-memory export lookup.
func (p *Pool) SupportsCachedRanking() bool {
	rt, err := p.checkout(context.Background())
	if err != nil {
		return false
	}
	defer p.checkin(rt)
	return rt.SupportsCachedRanking()
}

// RankAnswerCached checks out a Runtime and scores minerAnswer against
// pre-computed question/ground_truth embedding vectors. The vectors are derived
// from the module weights, so they are valid on any instance of the same
// module: producing them via Embed on one pooled instance and consuming them
// here on another is safe.
func (p *Pool) RankAnswerCached(ctx context.Context, qVec, gtVec []float32, groundTruth, minerAnswer string) (float32, error) {
	rt, err := p.checkout(ctx)
	if err != nil {
		return 0, err
	}
	defer p.checkin(rt)
	return rt.RankAnswerCached(ctx, qVec, gtVec, groundTruth, minerAnswer)
}

// CosineSim checks out a Runtime, calls CosineSim, then returns it.
func (p *Pool) CosineSim(ctx context.Context, a, b []float32) (float32, error) {
	rt, err := p.checkout(ctx)
	if err != nil {
		return 0, err
	}
	defer p.checkin(rt)
	return rt.CosineSim(ctx, a, b)
}

// BM25Score checks out a Runtime, calls BM25Score, then returns it.
func (p *Pool) BM25Score(ctx context.Context, query, doc string) (float32, error) {
	rt, err := p.checkout(ctx)
	if err != nil {
		return 0, err
	}
	defer p.checkin(rt)
	return rt.BM25Score(ctx, query, doc)
}

// Acquire checks out a Runtime for direct use. The caller MUST call Release
// when done. Prefer the typed methods above unless you need sequential calls
// on the same instance.
func (p *Pool) Acquire(ctx context.Context) (*Runtime, error) {
	return p.checkout(ctx)
}

// Release returns a Runtime previously obtained via Acquire.
func (p *Pool) Release(rt *Runtime) {
	p.checkin(rt)
}

// Close shuts down all pooled runtimes. Blocks until all checked-out runtimes
// have been returned. Safe to call multiple times, including concurrently:
// only the first call performs the drain; any other concurrent or subsequent
// callers block on sync.Once until that first call finishes and then receive
// its result.
//
// Callers relying on Close draining safely while other goroutines may still
// be mid-checkout (e.g. Scorer.Promote closing a superseded pool while
// in-flight scoring calls still hold runtimes from it) depend on checkin
// never blocking and never racing close(ch) — see the closeOnce field
// comment above for why that invariant holds.
func (p *Pool) Close(ctx context.Context) error {
	p.closeOnce.Do(func() {
		var first error
		for i := 0; i < p.size; i++ {
			rt := <-p.ch
			if err := rt.Close(ctx); err != nil && first == nil {
				first = err
			}
		}
		close(p.ch)
		// The instances share p.wrt and none of them owns it, so the pool
		// closes it here — after the drain, so nothing is still executing in
		// it. This is what actually frees the pool's compiled code.
		if p.wrt != nil {
			if err := p.wrt.Close(ctx); err != nil && first == nil {
				first = fmt.Errorf("wasm/runtime pool: runtime close: %w", err)
			}
		}
		p.closeErr = first
	})
	return p.closeErr
}

// Size returns the number of Runtime instances in the pool.
func (p *Pool) Size() int { return p.size }

func (p *Pool) checkout(ctx context.Context) (*Runtime, error) {
	select {
	case rt := <-p.ch:
		return rt, nil
	case <-ctx.Done():
		return nil, fmt.Errorf("wasm/runtime pool: context cancelled: %w", ctx.Err())
	}
}

// checkin returns rt to the pool. Deliberately lock-free and unconditional —
// see the closeOnce field comment on Pool for why this is safe even when a
// concurrent Close() is draining the pool.
func (p *Pool) checkin(rt *Runtime) {
	p.ch <- rt
}
