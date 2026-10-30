package responseparser

// Read-only views of a format's on-chain layouts, for the schema API
// (GET /engine/v1/intents/:id). They state the rule DecodeOnChainInputs and
// EncodeOnChain apply - the inputs / fields in order, each in the array of its
// type - so a caller can build a job's params and read its answer without
// reading the code. Nothing here is used to decode or encode; those two
// functions stay the only implementation of the rule.

// Slot arrays of OnChainData.
const (
	SlotIntegers = "integers"
	SlotStrings  = "strings"
	SlotBools    = "bools"
)

// OnChainSlot is one input or answer field in its on-chain position.
type OnChainSlot struct {
	Name string `json:"name"`
	Type string `json:"type"`
	// Slot is the array the value is in; Index its position in that array.
	Slot  string `json:"slot"`
	Index int    `json:"index"`
	// Decimals: a number is the int256 value × 10^Decimals (number only).
	Decimals *int      `json:"decimals,omitempty"`
	Unit     string    `json:"unit,omitempty"`
	Range    []float64 `json:"range,omitempty"`
	Allowed  []string  `json:"allowed,omitempty"`
	// MaxBytes bounds a text value (strings only).
	MaxBytes int `json:"max_bytes,omitempty"`
}

// slotFor is the array a value of type t is laid in.
func slotFor(t string) string {
	switch t {
	case TypeNumber, TypeInteger:
		return SlotIntegers
	case TypeBool:
		return SlotBools
	}
	return SlotStrings // string, enum, date, datetime
}

// OnChainRequestLayout is where each input of f sits in a job's params
// (DecodeOnChainInputs' rule). Empty when the format declares no inputs.
func (f *Format) OnChainRequestLayout() []OnChainSlot {
	if f == nil {
		return nil
	}
	out := make([]OnChainSlot, 0, len(f.Inputs))
	next := map[string]int{}
	for _, in := range f.Inputs {
		slot := slotFor(in.Type)
		s := OnChainSlot{Name: in.Name, Type: in.Type, Slot: slot, Index: next[slot],
			Decimals: in.Decimals, Unit: in.Unit, Range: in.Range, Allowed: in.Allowed}
		if slot == SlotStrings {
			s.MaxBytes = MaxOnChainString
		}
		next[slot]++
		out = append(out, s)
	}
	return out
}

// OnChainAnswerLayout is where each on-chain field of f sits in a job's
// answer (EncodeOnChain's rule), and the fields left off-chain. Call
// OnChainReady first: a format that is not ready has no usable layout.
func (f *Format) OnChainAnswerLayout() (onChain []OnChainSlot, offChain []string) {
	if f == nil {
		return nil, nil
	}
	onChain = make([]OnChainSlot, 0, len(f.Fields))
	next := map[string]int{}
	for _, fl := range f.Fields {
		if fl.OffChain {
			offChain = append(offChain, fl.Name)
			continue
		}
		slot := slotFor(fl.Type)
		s := OnChainSlot{Name: fl.Name, Type: fl.Type, Slot: slot, Index: next[slot],
			Decimals: fl.Decimals, Unit: fl.Unit, Range: fl.Range, Allowed: fl.Allowed}
		if slot == SlotStrings {
			s.MaxBytes = MaxOnChainString
		}
		next[slot]++
		onChain = append(onChain, s)
	}
	return onChain, offChain
}
