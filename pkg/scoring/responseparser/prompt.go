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
	viewJSON := formatView(fl)

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

// formatView is the field as the model sees it, as JSON.
func formatView(fl Field) string {
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
	return strings.TrimSuffix(buf.String(), "\n")
}

// ProcessSystemPrompt is the instruction for a processed field
// (Field.Processed): work one value out of an API response for a request,
// citing the values used. Copying keeps SystemPrompt; this is for a value the
// response does not hold as written. Code checks every cited value is in the
// response (the request, which the caller writes, is never evidence).
const ProcessSystemPrompt = "You work out one value from an API response, for a request.\n" +
	"You are given the request, the API response and the format of the value wanted: its name, a description, " +
	"its type and the unit(s) it can be in.\n" +
	"Rules:\n" +
	"1. Work out the one value the description asks for, for this request. Use every detail in the request and " +
	"the description: which item, which place, which period or window, which threshold.\n" +
	"2. Use only what is written in the API response; the request only says which item, place or period. " +
	"Never guess or use outside knowledge.\n" +
	"3. If the value is written in the response, copy it exactly. If it has to be worked out (a total, an " +
	"average, a minimum or maximum, a count, a difference, a comparison with the request), work it out from " +
	"the values in the response.\n" +
	"4. A number: reply the number alone, in the first listed unit.\n" +
	"5. A yes/no value (type bool): reply true or false.\n" +
	"6. An enum: reply one of the allowed values.\n" +
	"7. In \"from\", list every value you used, each copied exactly as written in the response.\n" +
	"8. If the response does not hold what the value needs, reply null.\n" +
	"Reply with the JSON object only. No explanation, no code fences."

// ProcessUserMessage renders a processed field's message: the request, the raw
// response, the field's format and the reply shape.
func ProcessUserMessage(fl Field, body, request string) string {
	return "Request:\n" + request + "\n\n" +
		"API response:\n" + body + "\n\n" +
		"Format of the value wanted:\n" + formatView(fl) + "\n\n" +
		"Reply with exactly this JSON:\n" +
		`{"value": "<the value>", "from": ["<each value used, exactly as written>", ...]}` +
		`   (or {"value": null} if the response does not hold it)`
}
