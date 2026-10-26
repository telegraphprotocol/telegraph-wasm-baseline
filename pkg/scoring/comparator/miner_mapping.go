package comparator

import (
	"fmt"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/fetchspec"
)

// MinerMapping is how a miner says where its answer is.
//
// It is read from the registration YAML the miner already ships and that
// miner_registry.yaml_raw already stores — no new schema, no new upload. Every
// one of the fifty registered miners we surveyed declares
// semantics.signal_mapping.label_field in the same place; the hardcoded
// parsers were re-deriving by hand what the miner had already told us.
//
// This is the half of the design that has to scale. Twenty-seven parsers for
// fifty-one miners was already unreasonable; the registry holds thirteen
// hundred, and each one cannot be a code change, a rebuild and a redeploy.
type MinerMapping struct {
	// Field is the dotted path to the value in the miner's response.
	Field string

	// Verbs and Asserts let an awkward miner describe itself rather than need
	// a bespoke parser: a quoted number, an inverted rate, a value nested in
	// a list, a currency that must match.
	Verbs   []fetchspec.Verb
	Asserts []fetchspec.Assert

	// Scale is an optional per-miner rescale applied BEFORE the intent's own
	// scale. It exists because two miners on one intent legitimately report in
	// different units — GRID serves both dollars and milli-dollars — and the
	// intent's scale cannot describe both.
	Scale float64

	// Unit is the response_format unit the miner quotes in ("" = the
	// format's own unit). Used only by the response-parser path.
	Unit string
}

// minerYAML is the subset of a registration YAML this package reads.
//
// Deliberately partial: the YAML carries endpoints, auth, rate limits and
// on-chain field mappings that scoring has no business interpreting. Decoding
// only what is needed means a miner adding an unrelated key cannot break
// scoring.
type minerYAML struct {
	Slug      string `yaml:"slug"`
	Semantics struct {
		SignalMapping struct {
			LabelField string `yaml:"label_field"`

			// The optional half. A miner that needs none of this declares only
			// label_field, which is the case for every miner we have seen.
			Verbs []struct {
				Op  string  `yaml:"op"`
				Arg float64 `yaml:"arg"`
			} `yaml:"verbs"`
			Asserts []struct {
				Field  string `yaml:"field"`
				Equals string `yaml:"equals"`
			} `yaml:"asserts"`
			Scale float64 `yaml:"scale"`

			// Unit names which of the intent's response_format units the
			// miner quotes in (e.g. "USD per {key}" for an inverted FX
			// quote). Read only by the response-parser path; see
			// LoadMinerUnits.
			Unit string `yaml:"unit"`
		} `yaml:"signal_mapping"`
		SupportedIntents []string `yaml:"supported_intents"`
	} `yaml:"semantics"`
}

// ParseMinerMapping reads a miner's declared response mapping from its
// registration YAML.
//
// A YAML that declares no label_field is an error rather than a default,
// because there is no safe default: guessing a field name would either miss
// (and exclude an honest miner) or hit the wrong one of several numbers in the
// response and score it against the wrong quantity. The GRID miner ships
// lmp_usd_per_mwh and lmp_usd_per_mwh_milli side by side, a thousandfold
// apart, and both are plausible.
func ParseMinerMapping(rawYAML string) (MinerMapping, error) {
	if strings.TrimSpace(rawYAML) == "" {
		return MinerMapping{}, fmt.Errorf("miner has no registration YAML stored")
	}
	var doc minerYAML
	if err := yaml.Unmarshal([]byte(rawYAML), &doc); err != nil {
		return MinerMapping{}, fmt.Errorf("registration YAML is not valid YAML: %w", err)
	}

	sm := doc.Semantics.SignalMapping
	if strings.TrimSpace(sm.LabelField) == "" {
		return MinerMapping{}, fmt.Errorf(
			"registration YAML declares no semantics.signal_mapping.label_field; " +
				"there is no safe default, so the miner cannot be scored")
	}

	m := MinerMapping{Field: sm.LabelField, Scale: sm.Scale, Unit: strings.TrimSpace(sm.Unit)}
	for _, v := range sm.Verbs {
		m.Verbs = append(m.Verbs, fetchspec.Verb{Op: v.Op, Arg: v.Arg})
	}
	for _, a := range sm.Asserts {
		m.Asserts = append(m.Asserts, fetchspec.Assert{Field: a.Field, Equals: a.Equals})
	}

	// Validated through a throwaway spec so a bad verb is caught here, at
	// registration, rather than at scoring time when the miner is already live.
	probe := fetchspec.Spec{URL: "https://validate.local", Field: m.Field,
		Verbs: m.Verbs, Asserts: m.Asserts}
	if err := probe.Validate(); err != nil {
		return MinerMapping{}, err
	}
	return m, nil
}

// ParseWithMapping extracts a miner's value from its response body using its
// declared mapping.
func ParseWithMapping(m MinerMapping, body string) (float64, error) {
	spec := fetchspec.Spec{
		URL: "https://declared.local", Field: m.Field,
		Verbs: m.Verbs, Asserts: m.Asserts,
	}
	v, err := fetchspec.Extract(spec, []byte(body))
	if err != nil {
		return 0, err
	}
	if m.Scale != 0 && m.Scale != 1 {
		v *= m.Scale
	}
	return v, nil
}

// mappingCache holds parsed mappings by slug.
//
// Parsing YAML per miner per question per epoch would be wasteful, and the
// YAML only changes when a miner re-registers.
var mappingCache struct {
	sync.RWMutex
	bySlug map[string]MinerMapping
}

// SetMinerMapping records a miner's declared mapping, replacing any previous
// one. Called when registration loads or refreshes a miner.
func SetMinerMapping(slug string, m MinerMapping) {
	mappingCache.Lock()
	defer mappingCache.Unlock()
	if mappingCache.bySlug == nil {
		mappingCache.bySlug = make(map[string]MinerMapping)
	}
	mappingCache.bySlug[slug] = m
}

// MinerMappingFor returns a miner's declared mapping, if one is known.
func MinerMappingFor(slug string) (MinerMapping, bool) {
	mappingCache.RLock()
	defer mappingCache.RUnlock()
	m, ok := mappingCache.bySlug[slug]
	return m, ok
}

// ClearMinerMappings drops every cached mapping. For tests.
func ClearMinerMappings() {
	mappingCache.Lock()
	defer mappingCache.Unlock()
	mappingCache.bySlug = nil
}

// LoadMinerMappings parses and caches the mappings for a set of miners,
// returning the per-slug errors rather than failing the whole batch.
//
// One miner with a broken YAML must not stop the other fifty from being
// scored, and the errors are returned rather than logged so the caller can
// record them against the miner — a rejection nobody can query is a rejection
// nobody acts on, which is how a 21MB blocklist came to be registered as a
// fraud-risk miner.
func LoadMinerMappings(rawBySlug map[string]string) map[string]error {
	errs := make(map[string]error)
	for slug, raw := range rawBySlug {
		m, err := ParseMinerMapping(raw)
		if err != nil {
			errs[slug] = err
			continue
		}
		SetMinerMapping(slug, m)
		SetMinerUnit(slug, m.Unit)
	}
	return errs
}
