package runtime

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/family"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
)

// AdapterMetaSection is the name of the custom section a scoring module
// carries its version metadata in. It is written by the adapter kit's
// packager and read at registration (Stage 1) and at load, so a module built
// for a different scoring ABI is refused before it is ever run. Living inside
// the module, it is covered by the registered wasm_hash and needs no contract
// or database change.
const AdapterMetaSection = "telegraph.adapter"

// AdapterABIVersion is the scoring ABI this node implements for MODEL adapters
// (judgment and hybrid intents): version 2 (see docs/dev-docs/wasm-scoring.md).
// A delta package carries its metadata in its safetensors header and this
// section is only the shape LoRARuntime.Meta reports it in. Version 1 was the
// Q32.32 wide format, which is retired: a version 1 section on a model adapter
// is refused with RetiredQ32Reason.
const AdapterABIVersion = 2

// ComparatorABIVersion is the ABI of COMPARATOR modules (deterministic
// intents, imw_score_deterministic). Their exports did not change when the
// model format did, so it stays at 1; the section stays optional for them.
const ComparatorABIVersion = 1

// AdapterRuntimeONNXF32 named the LEGACY full-model runtime (an ONNX float32
// graph baked into a WASI command). No longer accepted: a section declaring
// it is refused with LegacyFullModelReason. Model adapters declare LoRAABI.
const AdapterRuntimeONNXF32 = "onnx_f32"

// Adapter classes a section may declare.
const (
	ClassJudgment   = "judgment"
	ClassHybrid     = "hybrid"
	ClassComparator = "comparator"
)

// AdapterMeta is the JSON payload of the AdapterMetaSection.
type AdapterMeta struct {
	ABIVersion int    `json:"abi_version"`
	Class      string `json:"class,omitempty"` // "judgment" | "hybrid" | "comparator"
	Tool       string `json:"tool,omitempty"`  // builder that wrote the section, informational
	// Runtime names the model runtime of a model adapter: LoRAABI ("lora-v1");
	// "onnx_f32" named the retired full-model runtime and is refused. Absent on
	// comparator modules.
	Runtime string `json:"runtime,omitempty"`
	// MaxTokens optionally lowers the token cap the node scores this adapter
	// with (default DefaultFloatMaxTokens, at most MaxFloatMaxTokens).
	MaxTokens int `json:"max_tokens,omitempty"`
	// Training is the optional, SELF-REPORTED training record our own builds
	// embed (the kit's --training-record). The node only checks that it is
	// well-formed and never uses it to decide anything (promotion, ranking):
	// the per-child eval gate is the real check.
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
// module carries none (a comparator module may omit it; a model adapter is a
// safetensors delta package and carries its metadata in its header). A section
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

// Check reports why this node cannot run a module declaring m, or nil.
// wantClass is what the caller needs the module to be:
//
//   - "comparator": a comparator module. abi_version must be
//     ComparatorABIVersion; the class, when set, must be "comparator".
//   - "judgment" / "hybrid" / "" (either): a model adapter.
//     abi_version 1 is the retired Q32.32 format (RetiredQ32Reason); runtime
//     "onnx_f32" is the retired full-model format (LegacyFullModelReason);
//     otherwise it must be AdapterABIVersion with runtime LoRAABI, a class
//     of judgment or hybrid (matching wantClass when one is given) and
//     max_tokens, if set, in [1, MaxFloatMaxTokens].
//
// A training record, when present, is checked for shape only.
func (m AdapterMeta) Check(wantClass string) error {
	if wantClass == ClassComparator {
		if m.ABIVersion != ComparatorABIVersion {
			return fmt.Errorf("built for comparator ABI version %d, this node implements version %d", m.ABIVersion, ComparatorABIVersion)
		}
		if m.Class != "" && m.Class != ClassComparator {
			return fmt.Errorf("declares class %q but its exports make it a %q module", m.Class, ClassComparator)
		}
		return m.Training.Validate()
	}
	if m.ABIVersion == 1 {
		return errors.New(RetiredQ32Reason)
	}
	if m.ABIVersion != AdapterABIVersion {
		return fmt.Errorf("built for scoring ABI version %d, this node implements version %d", m.ABIVersion, AdapterABIVersion)
	}
	if m.Runtime == AdapterRuntimeONNXF32 {
		return errors.New(LegacyFullModelReason)
	}
	if m.Runtime != LoRAABI {
		return fmt.Errorf("declares runtime %q; this node runs only %q adapters", m.Runtime, LoRAABI)
	}
	switch m.Class {
	case ClassJudgment, ClassHybrid:
	case "":
		return fmt.Errorf("declares no class; a float adapter must say %q or %q", ClassJudgment, ClassHybrid)
	default:
		return fmt.Errorf("declares class %q; a float adapter is %q or %q", m.Class, ClassJudgment, ClassHybrid)
	}
	if wantClass != "" && m.Class != wantClass {
		return fmt.Errorf("declares class %q but the intent is a %q intent", m.Class, wantClass)
	}
	if m.MaxTokens < 0 || m.MaxTokens > MaxFloatMaxTokens {
		return fmt.Errorf("max_tokens %d is outside [1, %d]", m.MaxTokens, MaxFloatMaxTokens)
	}
	return m.Training.Validate()
}
