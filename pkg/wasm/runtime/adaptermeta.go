package runtime

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/family"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
)

// AdapterMetaSection is the name of the custom section a scoring adapter
// carries its version metadata in. It is written by the package builder
// (cmd/wideembed) and read at registration (Stage 1), so a module built for a
// different scoring ABI is refused before it is ever loaded. Living inside the
// module, it is covered by the registered wasm_hash and needs no contract or
// database change.
const AdapterMetaSection = "telegraph.adapter"

// AdapterABIVersion is the scoring ABI this node implements (see
// docs/dev-docs/scoring-adapter-standard.md). Bump it only together with an
// incompatible change to the exports or their argument layout.
const AdapterABIVersion = 1

// AdapterMeta is the JSON payload of the AdapterMetaSection.
type AdapterMeta struct {
	ABIVersion int    `json:"abi_version"`
	Class      string `json:"class,omitempty"` // "judgment" | "hybrid" | "comparator"
	Tool       string `json:"tool,omitempty"`  // builder that wrote the section, informational
	// Training is the optional, SELF-REPORTED training record our own builds
	// embed (cmd/adapterbuild --training-record). The node only checks that it
	// is well-formed and never uses it to decide anything (promotion, ranking):
	// the per-child eval gate is the real check. omitempty keeps a section
	// without it byte-identical to one written before the field existed, so the
	// field is additive and AdapterABIVersion is unchanged.
	Training *TrainingRecord `json:"training,omitempty"`
}

// TrainingRecord says which intents a (family) adapter was trained on and how
// it did on each. Format: docs/dev-docs/scoring-adapter-standard.md §9.
type TrainingRecord struct {
	Family          string          `json:"family"`
	Children        []TrainingChild `json:"children"`
	Trainer         string          `json:"trainer,omitempty"`
	Backbone        string          `json:"backbone,omitempty"`
	BackboneVersion string          `json:"backbone_version,omitempty"`
}

// TrainingChild is one intent's line in a TrainingRecord.
type TrainingChild struct {
	Intent         string  `json:"intent"`
	DatasetVersion string  `json:"dataset_version,omitempty"`
	Pairs          int     `json:"pairs"`
	Accuracy       float64 `json:"accuracy"`
	Spearman       float64 `json:"spearman"`
	HeldOut        bool    `json:"held_out"`
}

// TrainingRecordError marks a malformed training record, so Stage 1 reports it
// as "training record: ..." whether it failed to decode or to validate.
type TrainingRecordError struct{ Err error }

func (e *TrainingRecordError) Error() string { return "training record: " + e.Err.Error() }
func (e *TrainingRecordError) Unwrap() error { return e.Err }

// UnmarshalJSON decodes a training record, tagging a decode error (wrong JSON
// types) as a TrainingRecordError.
func (t *TrainingRecord) UnmarshalJSON(b []byte) error {
	type plain TrainingRecord
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return &TrainingRecordError{Err: err}
	}
	*t = TrainingRecord(p)
	return nil
}

// Validate reports why t is not a well-formed training record, or nil (also for
// a nil record: absent is valid). It is a shape check only — a family, at least
// one child, every child the family itself or a "_"-extension of it
// (family.IsChild), no duplicates, pairs >= 0, accuracy in [0,1], spearman in
// [-1,1]. It says nothing about whether the numbers are true.
func (t *TrainingRecord) Validate() error {
	if t == nil {
		return nil
	}
	bad := func(format string, a ...any) error { return &TrainingRecordError{Err: fmt.Errorf(format, a...)} }
	if t.Family == "" {
		return bad("family is required")
	}
	if len(t.Children) == 0 {
		return bad("children is empty")
	}
	seen := make(map[string]struct{}, len(t.Children))
	for i, c := range t.Children {
		switch {
		case c.Intent == "":
			return bad("children[%d]: intent is required", i)
		case !family.IsChild(t.Family, c.Intent):
			return bad("children[%d]: %s is not %s or a child of it", i, c.Intent, t.Family)
		case c.Pairs < 0:
			return bad("children[%d] (%s): pairs is negative", i, c.Intent)
		case !(c.Accuracy >= 0 && c.Accuracy <= 1):
			return bad("children[%d] (%s): accuracy %v is outside [0,1]", i, c.Intent, c.Accuracy)
		case !(c.Spearman >= -1 && c.Spearman <= 1):
			return bad("children[%d] (%s): spearman %v is outside [-1,1]", i, c.Intent, c.Spearman)
		}
		if _, dup := seen[c.Intent]; dup {
			return bad("children[%d]: %s is listed twice", i, c.Intent)
		}
		seen[c.Intent] = struct{}{}
	}
	return nil
}

// ChildIntents returns the record's child intent names, in listed order.
func (t *TrainingRecord) ChildIntents() []string {
	if t == nil {
		return nil
	}
	out := make([]string, 0, len(t.Children))
	for _, c := range t.Children {
		out = append(out, c.Intent)
	}
	return out
}

// Encode returns the section payload.
func (m AdapterMeta) Encode() []byte {
	b, _ := json.Marshal(m)
	return b
}

// ReadAdapterMeta returns the module's adapter metadata, or nil when the
// module carries none (an adapter built before the section existed). A section
// that is present but unreadable is an error: the builder wrote something the
// node cannot interpret. A training record of the wrong shape is reported as a
// *TrainingRecordError ("training record: ...").
func ReadAdapterMeta(b []byte) (*AdapterMeta, error) {
	data, ok, err := wasmbin.CustomSection(b, AdapterMetaSection)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}
	var m AdapterMeta
	if err := json.Unmarshal(data, &m); err != nil {
		var te *TrainingRecordError
		if errors.As(err, &te) {
			return nil, te
		}
		return nil, fmt.Errorf("%s section is not valid JSON: %w", AdapterMetaSection, err)
	}
	return &m, nil
}

// Check reports why this node cannot run an adapter declaring m, or nil.
// wantClass, when non-empty, is the class the module's exports say it is. A
// training record, when present, is checked for shape only.
func (m AdapterMeta) Check(wantClass string) error {
	if m.ABIVersion != AdapterABIVersion {
		return fmt.Errorf("built for scoring ABI version %d, this node implements version %d", m.ABIVersion, AdapterABIVersion)
	}
	if wantClass != "" && m.Class != "" && m.Class != wantClass {
		return fmt.Errorf("declares class %q but its exports make it a %q adapter", m.Class, wantClass)
	}
	return m.Training.Validate()
}
