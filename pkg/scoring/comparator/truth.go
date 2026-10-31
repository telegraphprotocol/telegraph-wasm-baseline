package comparator

import (
	"context"
	"time"
)

// TruthSource yields the reference value an intent's miners are scored
// against, already scaled to the intent's fixed-point domain.
//
// In production there is exactly one implementation: the eval-pool material
// source (scoring.MaterialTruthSource, a RoundSource), which samples the
// intent's truth-source material per round. The built-in per-intent fetchers
// (ECB, OSRM, World Bank, CAISO, Chainlink PoR, ...) and the intent_specs
// truth_source/truth_params factory were removed from scoring: ground truth
// comes only from material. Tests wire stubs through this interface.
type TruthSource interface {
	FetchReference(ctx context.Context, quote string, pin Pin) (Reference, error)
}

// Pin identifies the point at which ground truth is read: "" = latest, else
// an ISO yyyy-mm-dd date. Part of the TruthSource contract; the material
// source ignores it (a round's reference is pinned by the epoch instead, and
// stored once per round).
type Pin struct {
	Date string // ISO yyyy-mm-dd, or "" for latest
}

// LatestPin reads the latest value.
func LatestPin() Pin { return Pin{} }

// PinForEpoch returns the pin used when scoring an epoch (latest).
func PinForEpoch(time.Time) Pin { return LatestPin() }

// DatedPin pins a specific day. Unused by the scoring path today.
func DatedPin(at time.Time) Pin {
	return Pin{Date: at.UTC().Format("2006-01-02")}
}

// Reference is a resolved ground-truth value together with the provenance
// needed to audit or reproduce it.
//
// It mirrors what fetchRateBps returns in the reference host fx_price.mjs
// (rateBps / rawRate / pair / date / source), so a score computed here can be
// checked against a run of fx_score.mjs on the same inputs.
type Reference struct {
	RateBps int64   // the value passed to the WASM as ground_truth_value
	RawRate float64 // the unscaled rate, for logging and diagnostics
	Pair    string  // e.g. "USD/EUR"
	Date    string  // the reference date the feed actually served
	Source  string  // provenance, e.g. "truth_sources:3/3"

	// CorpusVersion and Sources are set for a reference built from
	// truth-source material: the eval-pool snapshot sampled, and one audit
	// record per sampled source. Stored with the reference
	// (round_truth_snapshots) and read back from it.
	CorpusVersion string
	Sources       []SourceAnswer
}
