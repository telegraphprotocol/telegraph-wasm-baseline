package responseparser

import (
	"bytes"
	"encoding/json"
	"strings"
)

// SystemPrompt is the generic instruction: copy one value, never convert.
//
// Byte-for-byte the SYSTEM string of
// intent_training_experiments/scripts/llm_pick_miner_responses.py, as is the
// user message UserMessage builds. It names no intent, API or field on purpose; everything specific
// arrives in the user message. Do not edit it here without re-running the
// experiments: the model's measured accuracy is for exactly this text.
const SystemPrompt = "You copy one value out of an API response.\n" +
	"You are given the API response and the format of the value wanted: its name, a description, " +
	"its type and the unit(s) it can be in.\n" +
	"Rules:\n" +
	"1. Find the one value that matches the description. Use every detail in it: which item, " +
	"which id or code, which date or period, which direction (from / to).\n" +
	"2. If the response holds the same quantity more than once in different scales or units, take " +
	"the one in a listed unit, preferring the first listed.\n" +
	"3. If there are several entries over time and the description asks for the current or " +
	"latest value, take the entry with the most recent date.\n" +
	"4. Copy the value exactly as it is written in the response. Never convert, round, rescale " +
	"or compute anything, even if the description mentions a different unit.\n" +
	"5. If the description is a yes/no question, do not answer it with yes/no or true/false: copy " +
	"the raw value from the response that the answer depends on.\n" +
	"6. If the type is list, reply a JSON array holding every matching value, each copied exactly. " +
	"If the type is object, reply a JSON object with the keys the description names, each value " +
	"copied exactly.\n" +
	"7. If the value is not in the response, reply null.\n" +
	"Reply with the JSON object only. No explanation, no code fences."

// promptView is the format as shown to the model. Field order and omission
// rules mirror the script's user_message(): name, description and type always;
// units, true_when and allowed only when non-empty. The reply key is the
// neutral "value", never the field name (a name like lookup_ok made the model
// answer the yes/no question instead of copying).
type promptView struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Type        string   `json:"type"`
	Units       []string `json:"units,omitempty"`
	TrueWhen    []string `json:"true_when,omitempty"`
	Allowed     []string `json:"allowed,omitempty"`
}

// UserMessage renders the per-call message: the raw response, the field's
// format and the reply shape. The field must already be filled (Fill).
func UserMessage(fl Field, body string) string {
	view := promptView{
		Name: fl.Name, Description: fl.Description, Type: fl.Type,
		Units: fl.Units(), TrueWhen: fl.TrueWhen, Allowed: fl.Allowed,
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	// Python's json.dumps(..., indent=2, ensure_ascii=False): no HTML escaping
	// of < > &, non-ASCII written as is, two-space indent.
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	_ = enc.Encode(view) // a struct of strings cannot fail to encode
	viewJSON := strings.TrimSuffix(buf.String(), "\n")

	var shape string
	switch fl.Type {
	case TypeList:
		shape = `{"value": ["<each value exactly as written>", ...]}`
	case TypeObject:
		shape = `{"value": {"<key named in the description>": "<value exactly as written>", ...}}`
	default:
		shape = `{"value": "<value exactly as written in the response>"}`
	}
	return "API response:\n" + body + "\n\n" +
		"Format of the value wanted:\n" + viewJSON + "\n\n" +
		"Reply with exactly this JSON:\n" + shape + `   (or {"value": null} if it is not in the response)`
}
