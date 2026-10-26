package adapterbuild

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/family"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// Training records (Level 3 of the family registration guard). The training
// side writes a JSON record of which intents a family adapter was trained on
// and how it did on each; --training-record embeds it in the package's
// telegraph.adapter section (wasmrt.AdapterMeta.Training). It is a safety
// check for OUR builds only: nodes shape-check it and never act on it.

// LoadTrainingRecord reads a training record file. Unknown fields are refused
// (a typo such as "held-out" would otherwise vanish silently), and the record
// must pass wasmrt.TrainingRecord.Validate.
func LoadTrainingRecord(path string) (*wasmrt.TrainingRecord, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("adapterbuild: reading training record: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var rec wasmrt.TrainingRecord
	type plain wasmrt.TrainingRecord // skip UnmarshalJSON so DisallowUnknownFields applies
	if err := dec.Decode((*plain)(&rec)); err != nil {
		return nil, fmt.Errorf("adapterbuild: %s: training record: %w", path, err)
	}
	if err := rec.Validate(); err != nil {
		return nil, fmt.Errorf("adapterbuild: %s: %w", path, err)
	}
	return &rec, nil
}

// ExpectedChildren is the set of intents an adapter registered on intent
// serves, computed offline with the same rule as
// scoring.CatalogFamilyChildrenServed (the Level 1 per-child gate):
//
//   - every intent in the builder's class source (classSource: loaded
//     intent_classes, else the compiled-in catalog; catalog.ListIntents) that
//     equals intent or extends it with "_";
//   - restricted to intent's own class;
//   - minus any child that a LONGER family in families claims by longest
//     prefix (only same-class families count). families is what --families
//     says is installed; the builder cannot see a node, so with none given no
//     child is excluded.
//
// Sorted; intent itself is included.
func ExpectedChildren(intent string, families []string) []string {
	classOf, _ := classSource()
	cls, ok := classOf(intent)
	if !ok {
		return nil
	}
	sameClass := func(id string) bool { c, ok := classOf(id); return ok && c == cls }
	installed := make(map[string]bool, len(families))
	for _, f := range families {
		if f != "" && sameClass(f) {
			installed[f] = true
		}
	}
	return family.ServedChildren(intent, catalog.ListIntents(), sameClass, func(k string) bool { return installed[k] })
}

// CheckTrainingRecord is the builder's training-record gate.
//
// With a record: its family must equal intent (always). With require also set:
// the record is mandatory, and its child intents must equal
// ExpectedChildren(intent, families) exactly — a served child missing from the
// record and a listed child the family does not serve (unknown to the catalog,
// another class, or owned by a longer family) are both refusals. Order,
// metrics and held_out are not compared.
func CheckTrainingRecord(intent string, rec *wasmrt.TrainingRecord, require bool, families []string) error {
	if rec == nil {
		if require {
			return fmt.Errorf("adapterbuild: --require-training-record: no training record given (--training-record <file.json>)")
		}
		return nil
	}
	if err := rec.Validate(); err != nil {
		return fmt.Errorf("adapterbuild: %w", err)
	}
	if rec.Family != intent {
		return fmt.Errorf("adapterbuild: training record is for family %s but --intent is %s", rec.Family, intent)
	}
	if !require {
		return nil
	}
	expected := ExpectedChildren(intent, families)
	want := make(map[string]bool, len(expected))
	for _, c := range expected {
		want[c] = true
	}
	listed := make(map[string]bool, len(rec.Children))
	var notServed []string
	for _, c := range rec.ChildIntents() {
		listed[c] = true
		if !want[c] {
			notServed = append(notServed, c)
		}
	}
	var missing []string
	for _, c := range expected {
		if !listed[c] {
			missing = append(missing, c)
		}
	}
	if len(missing) == 0 && len(notServed) == 0 {
		return nil
	}
	sort.Strings(notServed)
	_, source := classSource()
	return fmt.Errorf("adapterbuild: --require-training-record: the training record's children do not match the children %s serves "+
		"(same class in the %s, longest prefix, longer families %v): missing %v (served but not in the record), "+
		"not served %v (listed but not served by %s: unknown, another class, or owned by a longer family); expected exactly %v",
		intent, source, families, missing, notServed, intent, expected)
}
