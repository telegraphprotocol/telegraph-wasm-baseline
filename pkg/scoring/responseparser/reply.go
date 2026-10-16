package responseparser

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
)

// Copied is what the model copied out of the response. Exactly one of Text,
// List or Object is meaningful, by Kind.
type Copied struct {
	Kind   CopiedKind
	Text   string
	List   []string
	Object map[string]*string
}

// CopiedKind says what shape the model replied with.
type CopiedKind int

const (
	// CopiedNone: the model said the value is not in the response.
	CopiedNone CopiedKind = iota
	CopiedText
	CopiedList
	CopiedObject
)

var (
	// Port of the script's re.sub(r"^```(?:json)?\s*|\s*```$", "", ...).
	fenceRE = regexp.MustCompile("\\A```(?:json)?\\s*|\\s*```\\z")
	// Port of re.search(r"\{.*\}", s, re.S): first "{" to the LAST "}".
	objectRE = regexp.MustCompile(`(?s)\{.*\}`)
	// Port of re.sub(r"(?s)^.*</think>", "", content): drop a reasoning block.
	thinkRE = regexp.MustCompile(`(?s)\A.*</think>`)
)

// StripThinking removes a leading reasoning block, as the script does to every
// reply before parsing. A non-thinking model never emits one; this keeps a
// thinking model behind the same endpoint from breaking the parse.
func StripThinking(content string) string {
	return strings.TrimSpace(thinkRE.ReplaceAllString(content, ""))
}

// ParseReply reads the model's reply. ok=false means the reply was not the
// {"value": ...} shape at all - a parse failure, which must not be scored.
// ok=true with Kind CopiedNone means the model said the value is absent.
//
// Port of the script's parse(), with one deliberate difference: a JSON number
// is kept as the text the model wrote (json.Number), where Python re-prints it
// through float (1.10 -> "1.1"). The number is the same; keeping the written
// text is what the verbatim guard wants to compare.
func ParseReply(content string) (Copied, bool) {
	s := fenceRE.ReplaceAllString(strings.TrimSpace(content), "")
	candidate := s
	if m := objectRE.FindString(s); m != "" {
		candidate = m
	}
	dec := json.NewDecoder(bytes.NewReader([]byte(candidate)))
	dec.UseNumber()
	var obj any
	if err := dec.Decode(&obj); err != nil {
		return Copied{}, false
	}
	// json.loads rejects trailing data; so must we.
	if rest := strings.TrimSpace(candidate[dec.InputOffset():]); rest != "" {
		return Copied{}, false
	}
	m, isMap := obj.(map[string]any)
	if !isMap {
		return Copied{}, false
	}
	v, has := m["value"]
	if !has {
		return Copied{}, false
	}
	switch x := v.(type) {
	case nil:
		return Copied{Kind: CopiedNone}, true
	case string:
		t := strings.TrimSpace(x)
		switch strings.ToLower(t) {
		case "", "null", "none":
			return Copied{Kind: CopiedNone}, true
		}
		return Copied{Kind: CopiedText, Text: t}, true
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			out = append(out, listItemText(item))
		}
		return Copied{Kind: CopiedList, List: out}, true
	case map[string]any:
		out := make(map[string]*string, len(x))
		for k, item := range x {
			if item == nil {
				out[k] = nil
				continue
			}
			t := objectItemText(item)
			out[k] = &t
		}
		return Copied{Kind: CopiedObject, Object: out}, true
	default:
		return Copied{Kind: CopiedText, Text: scalarText(x)}, true
	}
}

// scalarText renders a non-string scalar: a bool lower-cased (Python
// str(v).lower()), a number as written.
func scalarText(v any) string {
	switch x := v.(type) {
	case bool:
		if x {
			return "true"
		}
		return "false"
	case json.Number:
		return strings.TrimSpace(x.String())
	case nil:
		return "None"
	default:
		b, _ := json.Marshal(x)
		return strings.TrimSpace(string(b))
	}
}

// listItemText mirrors the script: bools lower-cased, everything else str().strip().
func listItemText(v any) string {
	if s, ok := v.(string); ok {
		return strings.TrimSpace(s)
	}
	return scalarText(v)
}

// objectItemText mirrors the script's dict branch, which does NOT lower-case
// bools (str(True) is "True").
func objectItemText(v any) string {
	switch x := v.(type) {
	case string:
		return strings.TrimSpace(x)
	case bool:
		if x {
			return "True"
		}
		return "False"
	default:
		return scalarText(v)
	}
}
