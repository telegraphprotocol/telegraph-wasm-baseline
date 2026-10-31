package comparator

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
)

// Deterministic ground truth comes from EVALUATION MATERIAL: truth sources
// uploaded to the eval pool (label_kind truth_source), sampled per round and
// combined by median (see scoring.MaterialTruthSource). This file holds the
// comparator-side pieces that path needs: the round-aware truth interface, the
// per-source audit record carried on a Reference, and the parse of a source's
// answer, which is the SAME parser path a miner's answer takes.

// RoundSource is a TruthSource whose reference depends on the round being
// scored (the epoch decides which material is active and which sources are
// sampled). The scorers call FetchRoundReference instead of FetchReference
// whenever the truth source implements it.
type RoundSource interface {
	TruthSource
	FetchRoundReference(ctx context.Context, epochID uint32, intentID, key string) (Reference, error)
}

// SourceAnswer is the audit record of one sampled truth source in one round:
// which material row it was, who contributed it, what it answered, what that
// parsed to, and how far it sat from the combined reference.
type SourceAnswer struct {
	RowHash   string `json:"row_hash"`
	CaseID    string `json:"case_id"`
	Submitter string `json:"submitter"`
	// Raw is the source's answer, truncated (MaxSourceAnswerAuditBytes).
	Raw string `json:"raw,omitempty"`
	// Parsed is true when the answer yielded a usable value; Error says why
	// not otherwise (request build, call, or parse failure).
	Parsed bool    `json:"parsed"`
	Value  float64 `json:"value,omitempty"`
	// Scaled is Value in the intent's fixed-point domain (the intent's scale,
	// exactly as a miner's value is scaled).
	Scaled int64 `json:"scaled,omitempty"`
	// DeviationBps is (Scaled - reference) * 10000 / reference: how far this
	// source sat from the median, in basis points. Informational: there is
	// no agreement requirement.
	DeviationBps int64 `json:"deviation_bps"`
	// WithinTolerance reports |DeviationBps| <= the intent's comparator
	// tolerance (RelTol intents only; recorded, never enforced).
	WithinTolerance bool   `json:"within_tolerance"`
	Error           string `json:"error,omitempty"`
}

// MaxSourceAnswerAuditBytes bounds the raw answer stored per source per round.
const MaxSourceAnswerAuditBytes = 4096

// ErrSourceParserUnavailable: no parser model is installed, so a truth
// source's answer cannot be read. The source counts as failed for the round.
var ErrSourceParserUnavailable = errors.New("response parser not installed (PARSER_LLM_ENABLED): a truth source's answer cannot be read")

// ParseSourceValue reads the scored value out of a truth source's answer with
// the parser model and the intent's response format: the same parse a miner's
// answer gets (scored field, key params, declared unit, verbatim guard,
// range). There is no fallback: a missing parser, a missing format or an
// unreachable model is a failure of that source for this round.
//
// label names the source in the parser's logs (never a credential).
func ParseSourceValue(ctx context.Context, intentID, label, key, body, unit string) (float64, error) {
	if strings.TrimSpace(body) == "" {
		return 0, fmt.Errorf("truth source %s returned an empty answer", label)
	}
	p := activeResponseParser()
	if p == nil {
		return 0, ErrSourceParserUnavailable
	}
	format, _, ok := FormatFor(intentID)
	if !ok || !format.HasScored() {
		return 0, fmt.Errorf("intent %s has no response format with a scored field", intentID)
	}
	cfg, _ := ConfigFor(intentID)
	res, err := p.Parse(ctx, responseparser.Request{
		IntentID: intentID, Key: key, Body: body, Slug: label,
		Format: format, KeyParams: cfg.KeyParams[key], Unit: unit,
	})
	if err != nil {
		return 0, fmt.Errorf("parsing truth source %s via the %s response format: %w", label, intentID, err)
	}
	return checkMinerValue(intentID, label, res.Value)
}

// DeclaredUnit returns semantics.signal_mapping.unit from a YAML (or JSON)
// document of the miner schema, or "". A truth source declares the unit it
// quotes in exactly like a miner does.
func DeclaredUnit(doc []byte) string {
	var y minerYAML
	if err := yaml.Unmarshal(doc, &y); err != nil {
		return ""
	}
	return strings.TrimSpace(y.Semantics.SignalMapping.Unit)
}
