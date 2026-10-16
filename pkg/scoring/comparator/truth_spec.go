package comparator

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/fetchspec"
)

// SpecTruthSource resolves ground truth from a declarative fetchspec.Spec
// rather than from hand-written Go.
//
// This is what lets a new intent on an ordinary REST API be a database row
// instead of a code change. Seven of the nine truth sources we ported by hand
// are exactly "call a URL, read a JSON field" — the same shape as a miner —
// and carrying nine bespoke fetchers for that was the thing that did not
// scale.
//
// The two that remain in Go are not an oversight. CAISO returns a ZIP of CSV
// and answers a malformed timestamp with HTTP 200 and an error document;
// Chainlink needs an ABI call and a decimals() read at the same pinned block;
// hashprice is DERIVED from a block header rather than fetched at all. Those
// are behaviours with tests behind them, not values.
type SpecTruthSource struct {
	// Spec declares where the number is.
	Spec fetchspec.Spec

	// KeyParams maps a question key to the parameters that resolve the spec's
	// {placeholders} for it.
	//
	// Keys are opaque strings elsewhere in this package ("US-UNEMP-2019",
	// "ETH-WETH-USDC-005"), so the expansion has to be declared rather than
	// parsed out of the key — parsing would make the key format load-bearing
	// and silently break the moment someone added a key with a dash in it.
	KeyParams map[string]map[string]string

	// Label and Pair describe the reference for logs and the audit trail.
	Label string

	// Scale converts the fetched value to the comparator's integer domain.
	// Held here rather than read from IntentConfig so a source can be tested
	// without the whole registry.
	Scale float64
}

// FetchReference resolves the reference value for one question key.
func (s *SpecTruthSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	params, ok := s.KeyParams[key]
	if !ok {
		// Refusing an unknown key rather than fetching with no substitution is
		// deliberate: an unsubstituted URL would either 404 or, worse, return
		// the upstream's default — a valid answer to a question nobody asked.
		return Reference{}, fmt.Errorf("no parameters declared for question key %q", key)
	}

	value, err := fetchspec.Fetch(ctx, s.Spec, params)
	if err != nil {
		return Reference{}, fmt.Errorf("resolving %s for %s: %w", s.Label, key, err)
	}

	scaled, err := scaleBy(value, s.Scale)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the %s reference for %s: %w", s.Label, key, err)
	}

	return Reference{
		RateBps: scaled,
		RawRate: value,
		Pair:    key,
		Source:  s.Label,
	}, nil
}

// scaleBy applies a fixed-point multiplier and rounds to the nearest integer.
//
// The comparator WASM does pure integer arithmetic, so a float that reaches it
// is a bug. Rounding half away from zero matches the reference hosts.
func scaleBy(v, scale float64) (int64, error) {
	if scale == 0 {
		return 0, fmt.Errorf("scale is zero; every value would collapse to zero")
	}
	scaled := v * scale
	if scaled != scaled { // NaN
		return 0, fmt.Errorf("value %v is not a number", v)
	}
	if scaled > 9.2e18 || scaled < -9.2e18 {
		return 0, fmt.Errorf("value %v at scale %v overflows int64", v, scale)
	}
	if scaled < 0 {
		return int64(scaled - 0.5), nil
	}
	return int64(scaled + 0.5), nil
}

// truthSpecParams is the JSON shape stored in intent_specs.truth_params for a
// spec-driven source.
type truthSpecParams struct {
	fetchspec.Spec `yaml:",inline"`

	// KeyParams is keyed by question key.
	KeyParams map[string]map[string]string `json:"key_params"`
	Label     string                       `json:"label"`
	Scale     float64                      `json:"scale"`
}

// init registers the generic spec-driven source.
//
// An intent naming "spec" carries its whole fetch recipe in truth_params, so
// adding one needs no Go at all. The named sources (caiso, chainlink_por, ...)
// remain for the upstreams whose behaviour cannot be declared.
func init() {
	RegisterTruthFactory("spec", func(params map[string]any) (TruthSource, error) {
		raw, err := json.Marshal(params)
		if err != nil {
			return nil, fmt.Errorf("re-encoding truth_params: %w", err)
		}
		var p truthSpecParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, fmt.Errorf("truth_params is not a fetch spec: %w", err)
		}
		if err := p.Spec.Validate(); err != nil {
			return nil, err
		}
		if len(p.KeyParams) == 0 {
			return nil, fmt.Errorf("truth_params declares no key_params; no question key could be resolved")
		}
		if p.Scale == 0 {
			return nil, fmt.Errorf("truth_params declares scale 0; every reference would be zero")
		}
		return &SpecTruthSource{
			Spec:      p.Spec,
			KeyParams: p.KeyParams,
			Label:     p.Label,
			Scale:     p.Scale,
		}, nil
	})
}
