package runtime

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/big"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

const (
	fnWideScore       = "imwq_score"
	fnWideDebugCounts = "imwq_debug_counts"
	fnWideAlloc       = "imwq_alloc"
	fnWideFree        = "imwq_free"

	// WideFracBits is the number of fractional bits in the Q32.32 fixed-point
	// format the wide runtime computes in: an int64 with 32 integer bits
	// (±2.1e9) and 32 fractional bits (~2.3e-10 resolution).
	//
	// It must equal FP_PRECISION on BOTH sides of the conversion pipeline —
	// golden/fixed.py and wide_wasm/src/fixed.rs — or the host divides by a
	// different power of two than the module multiplied by, and every score is
	// off by a factor of 2^k without anything erroring.
	//
	// Q16.48 was the first attempt and overflowed: real ModernBERT activations
	// reach ~27,000, which does not fit 15 integer bits, and the intermediate
	// products wrapped i64 and collapsed every score to zero. Do not narrow
	// the integer half again without re-measuring activation range.
	WideFracBits = 32

	// wideRejected is what imwq_score and imwq_debug_counts return for an
	// artifact they refuse to parse. i64::MIN, chosen because it cannot be
	// confused with a real Q32.32 score.
	wideRejected = -9223372036854775808
)

// WideRuntime holds the deterministic Q32.32 ("wide") inference runtime with
// one model artifact resident in its linear memory.
//
// This is the fourth IMW-convention runtime, and it replaces the int8 path
// (EmbeddedRuntime) for judgment scoring rather than sitting beside it as an
// option. The reason is not performance, it is that int8 does not rank:
// activations are requantised to 256 levels between every layer, the reward
// signal is smaller than the accumulated rounding error, and the measured
// Spearman correlation against the float model is ~0.001 — indistinguishable
// from shuffling the miners. Calibration, SmoothQuant and a wider score op
// were all tried and none recovered it. Keeping weights AND activations wide
// in Q32.32 end to end leaves the signal intact: op-level error against float
// is ~3e-10 rather than ~1%.
//
// It stays fully integer, so it keeps the property int8 was chosen for in the
// first place: no f32/f64 opcode appears in the forward pass, so every
// validator computes bit-identical scores and the epoch cannot fork on
// floating-point rounding.
//
// The model reaches linear memory one of two ways, and the scoring code cannot
// tell them apart (by design — that is the parity guarantee):
//
//   - split (dev): a ~52KB interpreter (NewWide) plus a separate ~1.2GB .imwq
//     uploaded ONCE by LoadArtifact, which keeps the pointer — re-uploading
//     1.2GB per score would dominate the round.
//   - self-contained (production): ONE .wasm per intent built by cmd/wideembed,
//     carrying the interpreter and the .imwq as a resident data segment, the
//     same shape the old int8 build had. NewWideEmbedded adopts the (ptr, len)
//     the module reports via imwq_embedded; no LoadArtifact. The registered hash
//     then covers model and code together.
//
// Either way scoring calls the same export over the same bytes:
//
//	imwq_score(imw_ptr, imw_len, tok_ptr, n_tokens) -> i64
//
// Tokenisation is the host's job, as for the hybrid and embedded paths, and
// must use the vocabulary the adapter was trained against — a mismatch does
// not fail, it scores meaningless ids confidently.
//
// Safe for concurrent use in the same sense as the other runtimes: not. Each
// call allocates in the module's own allocator, so callers must serialise.
type WideRuntime struct {
	wrt         wazero.Runtime
	ownsRuntime bool
	mod         api.Module
	mem         api.Memory

	fnScore  api.Function
	fnCounts api.Function
	fnAlloc  api.Function
	fnFree   api.Function

	// Output says how to read the score head's raw value. Defaults to
	// WideOutputLogit, the zero value, because that is what the trained
	// reward adapters emit.
	Output WideOutput

	// imwPtr/imwLen locate the resident .imwq artifact. Zero until
	// LoadArtifact succeeds — or, for a self-contained package, until
	// NewWideEmbedded adopts the model the module carries.
	imwPtr uint32
	imwLen uint32

	// embedded is true when the model came from the package itself (a
	// KindWideEmbedded module) rather than from LoadArtifact.
	embedded bool
}

// Embedded reports whether the resident model is the one the module carries
// (self-contained package) rather than one uploaded by LoadArtifact.
func (r *WideRuntime) Embedded() bool { return r != nil && r.embedded }

// NewWide instantiates the wide Q32.32 interpreter from wasmBytes.
//
// The returned runtime holds no model: it cannot score until LoadArtifact has
// been called. Separating the two is what lets wasmprobe answer "is this a
// wide interpreter" without a 1.2GB artifact to hand. For a self-contained
// package (interpreter + embedded model) use NewWideEmbedded instead.
func NewWide(ctx context.Context, wasmBytes []byte) (*WideRuntime, error) {
	return instantiateWide(ctx, wasmBytes)
}

// instantiateWide is the shared instantiate-and-bind step behind NewWide and
// NewWideEmbedded: it requires imwq_score, an exported memory and the
// imwq_alloc/imwq_free pair, and returns a runtime with no model adopted.
func instantiateWide(ctx context.Context, wasmBytes []byte) (*WideRuntime, error) {
	if len(wasmBytes) == 0 {
		return nil, fmt.Errorf("wasm/runtime: empty wide module")
	}

	// One runtime per load (isolated linear memory), but compiled code comes
	// from the process-wide cache, so reloading the same adapter every epoch
	// does not re-compile it (see compile_cache.go).
	// Interruptible: an adapter is untrusted code fetched from an on-chain
	// registration, and one that loops forever must be stopped when its
	// caller's deadline passes instead of pinning a thread (and the intent's
	// promotion lock) for good. The module is closed when that happens, so a
	// runtime whose call timed out is not reused. The compilation cache keys on
	// this setting, so it shares entries only with other interruptible loads.
	wrt := wazero.NewRuntimeWithConfig(ctx, cachedRuntimeConfig().WithCloseOnContextDone(true))
	mod, err := wrt.Instantiate(ctx, wasmBytes)
	if err != nil {
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: instantiating wide module: %w", err)
	}

	closeAll := func() {
		mod.Close(ctx) //nolint:errcheck
		wrt.Close(ctx) //nolint:errcheck
	}

	fn := mod.ExportedFunction(fnWideScore)
	if fn == nil {
		closeAll()
		return nil, fmt.Errorf("wasm/runtime: missing required export %q — "+
			"is this an int8 imw_* module rather than a wide Q32.32 build?", fnWideScore)
	}
	mem := mod.Memory()
	if mem == nil {
		closeAll()
		return nil, fmt.Errorf("wasm/runtime: wide module exports no memory")
	}
	alloc := mod.ExportedFunction(fnWideAlloc)
	free := mod.ExportedFunction(fnWideFree)
	if alloc == nil || free == nil {
		closeAll()
		return nil, fmt.Errorf("wasm/runtime: wide module missing %q/%q", fnWideAlloc, fnWideFree)
	}

	return &WideRuntime{
		wrt: wrt, ownsRuntime: true, mod: mod, mem: mem,
		fnScore: fn, fnCounts: mod.ExportedFunction(fnWideDebugCounts),
		fnAlloc: alloc, fnFree: free,
	}, nil
}

// LoadArtifact uploads a .imwq model image into the module's linear memory and
// keeps it there for the lifetime of the runtime.
//
// Once, not per call: the image is ~1.2GB for ModernBERT-base (8× the int8
// build, because the weights are int64), and copying that for every miner
// response would cost more than the inference.
//
// The upload is verified by parsing it inside the module rather than trusting
// the byte count. A truncated or mis-framed artifact still writes fine and
// then makes every subsequent score return the rejection sentinel, which reads
// as "this miner scored nothing" rather than "the model never loaded".
func (r *WideRuntime) LoadArtifact(ctx context.Context, imwq []byte) error {
	if len(imwq) == 0 {
		return fmt.Errorf("wasm/runtime: empty .imwq artifact")
	}
	if r.imwPtr != 0 {
		if r.embedded {
			return fmt.Errorf("wasm/runtime: this is a self-contained wide package that already carries its model "+
				"(%d bytes); a second artifact cannot be loaded into it", r.imwLen)
		}
		return fmt.Errorf("wasm/runtime: wide runtime already holds an artifact (%d bytes)", r.imwLen)
	}
	if len(imwq) < 4 || string(imwq[:4]) != "IMWQ" {
		// Caught here rather than inside the module so the error names the
		// likely cause: an int8 .imw or a plain .wasm handed to the wide path.
		return fmt.Errorf("wasm/runtime: artifact does not start with the IMWQ magic — "+
			"is this an int8 .imw rather than a wide .imwq? (first bytes: %q)",
			imwq[:min(4, len(imwq))])
	}

	size := uint32(len(imwq))
	res, err := r.fnAlloc.Call(ctx, uint64(size))
	if err != nil {
		return fmt.Errorf("wasm/runtime: %s(%d) for the model artifact: %w", fnWideAlloc, size, err)
	}
	if len(res) == 0 || res[0] == 0 {
		return fmt.Errorf("wasm/runtime: %s returned a null pointer for %d bytes — "+
			"wasm32 linear memory is capped at 4GB and this artifact is %.2f GB",
			fnWideAlloc, size, float64(size)/(1<<30))
	}
	ptr := uint32(res[0])

	if !r.mem.Write(ptr, imwq) {
		r.fnFree.Call(ctx, uint64(ptr), uint64(size)) //nolint:errcheck
		return fmt.Errorf("wasm/runtime: writing the %d-byte artifact at %d failed", size, ptr)
	}
	r.imwPtr, r.imwLen = ptr, size

	if _, _, err := r.DebugCounts(ctx); err != nil {
		r.fnFree.Call(ctx, uint64(ptr), uint64(size)) //nolint:errcheck
		r.imwPtr, r.imwLen = 0, 0
		return fmt.Errorf("wasm/runtime: the uploaded artifact did not parse: %w", err)
	}
	return nil
}

// DebugCounts reports how many tensors and ops the module parsed out of the
// resident artifact.
//
// Diagnostic, and the load-time check: a model that parses to zero ops is a
// valid-looking file that scores every input identically.
func (r *WideRuntime) DebugCounts(ctx context.Context) (tensors, ops uint32, err error) {
	if r.fnCounts == nil {
		return 0, 0, fmt.Errorf("wasm/runtime: %s not exported", fnWideDebugCounts)
	}
	if r.imwPtr == 0 {
		return 0, 0, fmt.Errorf("wasm/runtime: no artifact loaded")
	}
	res, err := r.fnCounts.Call(ctx, uint64(r.imwPtr), uint64(r.imwLen))
	if err != nil {
		return 0, 0, fmt.Errorf("wasm/runtime: %s: %w", fnWideDebugCounts, err)
	}
	if len(res) == 0 {
		return 0, 0, fmt.Errorf("wasm/runtime: %s returned no values", fnWideDebugCounts)
	}
	packed := int64(res[0])
	if packed == wideRejected {
		return 0, 0, fmt.Errorf("wasm/runtime: %s rejected the artifact", fnWideDebugCounts)
	}
	tensors = uint32(packed >> 32)
	ops = uint32(packed & 0xffffffff)
	if ops == 0 {
		return tensors, 0, fmt.Errorf("wasm/runtime: the artifact parsed to zero ops; "+
			"it would score every input identically (tensors=%d)", tensors)
	}
	return tensors, ops, nil
}

// ScoreRaw runs the resident model over tokens and returns the raw Q32.32
// score code, before the boundary conversion.
//
// Exposed for the same reason as ScoreEmbeddedRaw: when every miner scores the
// same, this separates "the adapter returned a constant" from "the conversion
// saturated". Through Score alone those are indistinguishable, and they have
// completely different causes — one needs a retrained adapter, the other is a
// host bug.
func (r *WideRuntime) ScoreRaw(ctx context.Context, tokens []int64) (int64, error) {
	if r.fnScore == nil {
		return 0, fmt.Errorf("wasm/runtime: %s not exported", fnWideScore)
	}
	if r.imwPtr == 0 {
		return 0, fmt.Errorf("wasm/runtime: %s called with no artifact loaded", fnWideScore)
	}
	if len(tokens) == 0 {
		// As on the hybrid and embedded paths: an empty token array is a
		// missing input, not an empty judgment. The model would score the
		// empty sequence and return something plausible.
		return 0, fmt.Errorf("wasm/runtime: %s called with no tokens — the text was not tokenised", fnWideScore)
	}

	tokPtr, freeTok, err := r.writeI64s(ctx, tokens)
	if err != nil {
		return 0, err
	}
	defer freeTok()

	res, err := r.fnScore.Call(ctx,
		uint64(r.imwPtr), uint64(r.imwLen),
		uint64(tokPtr), uint64(len(tokens)),
	)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: %s: %w", fnWideScore, err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: %s returned no values", fnWideScore)
	}
	code := int64(res[0])
	if code == wideRejected {
		// Fail closed. Unlike the int8 path there is no error code to report:
		// the module returns one sentinel for every rejection, so the artifact
		// is the thing to look at.
		return 0, fmt.Errorf("wasm/runtime: %s rejected the artifact (i64::MIN sentinel)", fnWideScore)
	}
	return code, nil
}

// WideOutput says what the score head's raw Q32.32 value MEANS, which decides
// how the boundary conversion maps it into basis points.
//
// This is not a preference, it is a property of how the adapter was trained,
// and guessing it wrong is silent in the worst way. A reward head trained with
// a sequence-classification objective (num_labels=1) emits an unbounded
// LOGIT — the real CODE_REVIEW adapter returns about -2.32 on ordinary input.
// Run through the unit mapping, every such miner clamps to 0 bps and the
// ranking vanishes: all of them tie at the bottom, which reads exactly like
// "every miner answered badly".
type WideOutput uint8

const (
	// WideOutputLogit is an unbounded score-head output, mapped through a
	// sigmoid. The default, because it is what the reward adapters actually
	// emit.
	WideOutputLogit WideOutput = iota

	// WideOutputUnit is a head that already emits [0,1] — for instance one
	// ending in a sigmoid of its own. Mapped by a plain clamp; applying a
	// second sigmoid would compress everything into the middle of the range
	// and flatten the differences between miners.
	WideOutputUnit
)

// Score runs the resident model over tokens and returns the judgment in basis
// points, 0..10000.
//
// tokens must be the rendered text encoded with the SAME vocabulary the
// adapter was trained against — see tokenizer.Load's digest pin. A mismatched
// vocabulary does not error here.
//
// There is deliberately no ground-truth argument, for the same reason as
// ScoreEmbedded: the adapter is a reward model trained on (prompt, chosen) /
// (prompt, rejected) pairs, so it judges an answer on its own merits rather
// than by similarity to a reference.
func (r *WideRuntime) Score(ctx context.Context, tokens []int64) (int32, error) {
	code, err := r.ScoreRaw(ctx, tokens)
	if err != nil {
		return 0, err
	}
	return WideToScoreDomain(code, r.Output), nil
}

// WideToScoreDomain maps a raw Q32.32 score code into basis points.
//
// This is the single documented boundary conversion, and the only place the
// fixed-point representation is left behind. In Q32.32 the real value is
// simply code / 2^WideFracBits — no per-model scale, unlike the int8 path
// which had to carry an (out_mul, out_shift) pair out of the module because
// its scale was learned at calibration time.
//
// What the real value means is the part that varies, so it is named rather
// than assumed: see WideOutput.
//
// Every step stays in exact integer arithmetic. Not for speed — for the same
// reason the runtime itself is integer-only: a float here would put a
// platform-dependent rounding step between the model and the leaderboard, and
// two validators disagreeing by one basis point fork the epoch.
func WideToScoreDomain(code int64, out WideOutput) int32 {
	if out == WideOutputLogit {
		code = fpSigmoid(code)
	}
	return wideUnitToBasisPoints(code)
}

// wideUnitToBasisPoints maps a Q32.32 value already in [0,1] to basis points:
//
//	clamp(code * SCORE_ONE >> WideFracBits, 0, SCORE_ONE)
//
// Multiplying before shifting keeps it exact; the product is computed wide
// because it overflows int64 for any code above ~9.2e14, which in Q32.32 is
// only 214516 — well inside the range a raw head output can reach.
func wideUnitToBasisPoints(code int64) int32 {
	prod := new(big.Int).Mul(big.NewInt(code), big.NewInt(int64(ScoreBasisPoints)))
	prod.Rsh(prod, WideFracBits)

	if prod.Sign() < 0 {
		return 0
	}
	if prod.Cmp(big.NewInt(int64(ScoreBasisPoints))) > 0 {
		return ScoreBasisPoints
	}
	return int32(prod.Int64())
}

// The Q32.32 helpers below mirror golden/fixed.py (and wide_wasm/src/fixed.rs)
// so the host's boundary conversion uses the SAME fixed-point arithmetic the
// model was computed in, rather than a float sigmoid at the last step.
//
// They are deliberately a port rather than a call into the module: the wide
// WASM exports no sigmoid, and the conversion has to happen somewhere. Ported
// exactly — the shift direction and the truncation convention are part of the
// contract, not implementation details:
//
//   - fp_div truncates the MAGNITUDE toward zero and reapplies the sign.
//
// big.Int rather than a two-limb int128: this runs once per score, against a
// model that takes seconds, so exactness is worth more than the cycles.

const wideFPOne = int64(1) << WideFracBits

// fpDiv is a/b in Q32.32, truncating the magnitude toward zero.
func fpDiv(a, b int64) int64 {
	if b == 0 {
		return 0
	}
	num := new(big.Int).Abs(big.NewInt(a))
	num.Lsh(num, WideFracBits)
	num.Quo(num, new(big.Int).Abs(big.NewInt(b)))
	if (a < 0) != (b < 0) {
		num.Neg(num)
	}
	return num.Int64()
}

// fpExpDyadic is exp(x) = (1 + x/2^k)^(2^k) by k repeated squarings, k=14 for
// ~42-bit accuracy. Mirrors fp_exp_dyadic.
func fpExpDyadic(x int64) int64 {
	const k = 14
	r := big.NewInt(wideFPOne + (x >> k))
	for i := 0; i < k; i++ {
		r.Mul(r, r)
		r.Rsh(r, WideFracBits)
	}
	if !r.IsInt64() {
		// Unreachable for the clamped input fpSafeExp allows (exp(10) is about
		// 22026), but a wrapped i64 here would come back as a negative
		// "probability" and invert the ranking rather than failing.
		return math.MaxInt64
	}
	return r.Int64()
}

// fpSafeExp is exp(x) clamped to [-10, 10], mirroring fp_safe_exp.
func fpSafeExp(x int64) int64 {
	maxX := int64(10) * wideFPOne
	if x > maxX {
		x = maxX
	}
	if x < -maxX {
		return 0
	}
	if x >= 0 {
		return fpExpDyadic(x)
	}
	pos := fpExpDyadic(-x)
	if pos == 0 {
		return 0
	}
	return fpDiv(wideFPOne, pos)
}

// fpSigmoid is 1/(1+e^-x) in Q32.32, saturating past |x| > 10. Mirrors
// fp_sigmoid.
//
// Monotone, which is what makes it safe as a ranking transform: it changes the
// numbers a miner sees but never the order they finish in — except in the
// saturated tails, where |logit| > 10 means a difference far below anything the
// leaderboard distinguishes anyway.
func fpSigmoid(x int64) int64 {
	ten := int64(10) * wideFPOne
	if x > ten {
		return wideFPOne
	}
	if x < -ten {
		return 0
	}
	return fpDiv(wideFPOne, wideFPOne+fpSafeExp(-x))
}

// writeI64s allocates through the module's own allocator and writes vals.
// Same reasoning as the other runtimes: a module with a 1.2GB artifact
// resident has plenty in linear memory that a fixed offset would corrupt.
func (r *WideRuntime) writeI64s(ctx context.Context, vals []int64) (uint32, func(), error) {
	size := uint32(len(vals) * 8)
	res, err := r.fnAlloc.Call(ctx, uint64(size))
	if err != nil {
		return 0, nil, fmt.Errorf("wasm/runtime: %s(%d): %w", fnWideAlloc, size, err)
	}
	if len(res) == 0 || res[0] == 0 {
		return 0, nil, fmt.Errorf("wasm/runtime: %s returned a null pointer", fnWideAlloc)
	}
	ptr := uint32(res[0])

	buf := make([]byte, size)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(v))
	}
	if !r.mem.Write(ptr, buf) {
		r.fnFree.Call(ctx, uint64(ptr), uint64(size)) //nolint:errcheck
		return 0, nil, fmt.Errorf("wasm/runtime: writing %d bytes at %d failed", size, ptr)
	}
	return ptr, func() { r.fnFree.Call(ctx, uint64(ptr), uint64(size)) }, nil //nolint:errcheck
}

// Close releases the module and, when this runtime owns it, the wazero runtime.
//
// The artifact is not freed separately: closing the module drops its whole
// linear memory, so an explicit imwq_free would be touching memory that is
// about to be discarded.
func (r *WideRuntime) Close(ctx context.Context) error {
	if r == nil {
		return nil
	}
	r.imwPtr, r.imwLen, r.embedded = 0, 0, false
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
