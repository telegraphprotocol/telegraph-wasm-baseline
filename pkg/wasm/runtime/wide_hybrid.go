package runtime

import (
	"context"
	"fmt"

	"github.com/tetratelabs/wazero/api"
)

const (
	fnWideScoreHybrid = "imwq_score_hybrid"

	// wideHybridSpecInts is the length of the spec buffer imwq_score_hybrid
	// reads: [comparator, tolerance, fusion, pass, low, floor, span, w_c, w_j,
	// _reserved]. Identical to the int8 path's hybridSpecInts, and for the same
	// reason: the WASM reads exactly ten i64s from the pointer with no length
	// argument, so a shorter buffer is read past its end. It does not error, it
	// fuses against whatever happens to sit in linear memory next.
	//
	// Deliberately its own constant rather than an alias, so that changing one
	// ABI cannot silently move the other.
	wideHybridSpecInts = 10
)

// Wide hybrid error codes, returned across the boundary as -(code).
//
// The bare imwq_score path cannot use small negative sentinels: it returns a
// signed logit that is routinely negative, so it signals rejection with
// i64::MIN. The hybrid path returns a score in [0, ScoreBasisPoints], which is
// non-negative by construction, so any negative return is unambiguously an
// error and can carry a reason.
const (
	wideHybridErrBadArtifact = 1
	wideHybridErrNoTokens    = 2
)

// WideHybridRuntime fuses a comparator verdict on a checkable fact with the
// wide Q32.32 model's judgment of the response text.
//
// It is the wide equivalent of HybridRuntime, and the structural difference
// between them is the same one that separates WideRuntime from
// EmbeddedRuntime. An int8 hybrid module has its adapter baked in, so the
// ~159MB .wasm IS the model and instantiating it is the whole setup. A wide
// module is a ~51KB interpreter and the model is a separate ~1.2GB .imwq
// artifact. So this type embeds WideRuntime and inherits its LoadArtifact:
// the artifact is uploaded ONCE and its pointer kept, because re-uploading a
// gigabyte per miner response would cost far more than the inference.
//
// The export it calls:
//
//	imwq_score_hybrid(imw_ptr, imw_len, tok_ptr, n_tokens,
//	                  ground_truth_value, miner_value, spec_ptr) -> i64
//
// That signature mirrors int8's imw_score_hybrid with the two artifact
// arguments prepended — int8 had no need for them because its model was in the
// module.
//
// Why fuse inside the WASM rather than scoring and fusing in Go: the fusion is
// part of what every validator must agree on bit-for-bit. Done host-side it
// would be Go integer arithmetic on one machine and Go integer arithmetic on
// another, which is fine today and is exactly the kind of thing that drifts.
// Inside the module it is covered by the same float-opcode gate as the forward
// pass.
//
// Safe for concurrent use in the same sense as the other runtimes: not.
// Callers must serialise.
type WideHybridRuntime struct {
	*WideRuntime

	fnScoreHybrid api.Function
}

// NewWideHybrid instantiates a wide Q32.32 module that also exports the hybrid
// path.
//
// The returned runtime holds no model: LoadArtifact must be called before it
// can score, exactly as for WideRuntime.
func NewWideHybrid(ctx context.Context, wasmBytes []byte) (*WideHybridRuntime, error) {
	base, err := NewWide(ctx, wasmBytes)
	if err != nil {
		return nil, err
	}
	fn := base.mod.ExportedFunction(fnWideScoreHybrid)
	if fn == nil {
		base.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: missing required export %q — "+
			"this is a wide interpreter without the comparator half, "+
			"or an int8 imw_* module", fnWideScoreHybrid)
	}
	return &WideHybridRuntime{WideRuntime: base, fnScoreHybrid: fn}, nil
}

// ScoreHybrid fuses the comparator's verdict with the resident model's
// judgment and returns the score in basis points, 0..10000.
//
// tokens are the rendered text encoded with the SAME vocabulary the adapter was
// trained on — see tokenizer.RenderForJudgment, whose template is part of the
// model's input contract rather than a formatting choice. A mismatched
// vocabulary does not error; it scores confidently from meaningless ids.
//
// minerValue and truthValue must already be on the intent's fixed-point scale,
// exactly as for the deterministic path.
func (r *WideHybridRuntime) ScoreHybrid(
	ctx context.Context,
	tokens []int64,
	minerValue, truthValue int64,
	spec [wideHybridSpecInts]int64,
) (int32, error) {
	if r.fnScoreHybrid == nil {
		return 0, fmt.Errorf("wasm/runtime: %s not exported", fnWideScoreHybrid)
	}
	if r.imwPtr == 0 {
		return 0, fmt.Errorf("wasm/runtime: %s called with no artifact loaded", fnWideScoreHybrid)
	}
	if len(tokens) == 0 {
		// As on every other scoring path: an empty token array is a missing
		// input, not an empty judgment. The model would score the empty
		// sequence and return something entirely plausible.
		return 0, fmt.Errorf("wasm/runtime: %s called with no tokens — "+
			"the miner's text was not tokenised", fnWideScoreHybrid)
	}

	tokPtr, freeTok, err := r.writeI64s(ctx, tokens)
	if err != nil {
		return 0, err
	}
	defer freeTok()

	specPtr, freeSpec, err := r.writeI64s(ctx, spec[:])
	if err != nil {
		return 0, err
	}
	defer freeSpec()

	// Ground truth comes BEFORE the miner value at the boundary. The Go
	// signature takes (minerValue, truthValue) in host-natural order and swaps
	// here, matching HybridRuntime.ScoreHybrid and ScoreDeterministic. A swap
	// is SILENT: under RelTol it only changes the denominator, so the module
	// still returns a plausible score. Do not "fix" either side to match the
	// other — pick one convention and keep both ends on it.
	res, err := r.fnScoreHybrid.Call(ctx,
		uint64(r.imwPtr), uint64(r.imwLen),
		uint64(tokPtr), uint64(len(tokens)),
		uint64(truthValue), uint64(minerValue),
		uint64(specPtr),
	)
	if err != nil {
		return 0, fmt.Errorf("wasm/runtime: %s: %w", fnWideScoreHybrid, err)
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("wasm/runtime: %s returned no values", fnWideScoreHybrid)
	}

	// A NEGATIVE return is -(error code), not a score. Casting it raw would
	// turn "the artifact did not parse" into a small negative that reads, once
	// clamped, as a miner who answered badly.
	raw := int64(res[0])
	if raw < 0 {
		return 0, fmt.Errorf("wasm/runtime: %s rejected the input: %s",
			fnWideScoreHybrid, wideHybridErrorText(-raw))
	}
	if raw > ScoreBasisPoints {
		return 0, fmt.Errorf("wasm/runtime: %s returned out-of-domain score %d",
			fnWideScoreHybrid, raw)
	}
	return int32(raw), nil
}

// wideHybridErrorText names a rejection code from the module.
func wideHybridErrorText(code int64) string {
	switch code {
	case wideHybridErrBadArtifact:
		return "the .imwq artifact did not parse (code 1)"
	case wideHybridErrNoTokens:
		return "no tokens were supplied (code 2)"
	default:
		return fmt.Sprintf("unknown error code %d", code)
	}
}
