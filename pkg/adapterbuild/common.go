// Package adapterbuild checks an Evaluator adapter — a safetensors delta
// package for the shared Foundation Model, written by
// float_runtime/export_adapter.py — with the Telegraph node's own parser and
// runtime, and prints the keccak256 that is registered on-chain (lora.go).
//
//	Check:   catalog class -> the node's Stage 1 parse against foundation.json
//	         (foundation and tokenizer hashes, class, rank, target modules,
//	         every shape, offsets, finite values) -> the training record vs
//	         the children the family serves -> optionally load and score on
//	         the foundation through the node's runtime (identical bits twice)
//	         -> <out>/<INTENT>.safetensors.manifest.json
//
// Nothing is rewritten: the package IS the registrable artifact. Registration
// (registerWasm with the URL + keccak256) stays manual.
//
// Every failure is loud: a deterministic intent, a class that disagrees with
// the catalog, a package the node would refuse, a mismatched training record
// — each returns an error and nothing is left looking finished.
package adapterbuild

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime/debug"
)

// BuilderName identifies this builder in every manifest.
const (
	BuilderName  = "github.com/telegraphprotocol/telegraph-wasm-baseline/cmd/adapterbuild"
	builderMajor = "adapterbuild/2"
)

// DefaultVerifyInput is the input the package is scored on when no --input is
// given: 10 real ModernBERT ids, [CLS] ... [SEP].
var DefaultVerifyInput = []int64{50281, 4118, 17526, 27, 187, 18658, 436, 13814, 1818, 50282}

// BuilderVersion is "adapterbuild/2" plus the VCS revision the binary was built
// from, when the Go toolchain recorded one ("+dirty" for a modified tree).
func BuilderVersion() string {
	v := builderMajor
	if bi, ok := debug.ReadBuildInfo(); ok {
		var rev, mod string
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				rev = s.Value
			case "vcs.modified":
				mod = s.Value
			}
		}
		if rev != "" {
			if len(rev) > 12 {
				rev = rev[:12]
			}
			v += " " + rev
			if mod == "true" {
				v += "+dirty"
			}
		}
	}
	return v
}

// FileRef names a file by hash.
type FileRef struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// Logf receives human progress lines (the CLI sends them to stderr).
type Logf func(format string, args ...any)

// ReadInput reads an input JSON ({"input_ids": [...], "attention_mask"?: [...]})
// and returns its input_ids and the file's sha256. The node always sends an
// all-ones attention mask of the same length; a file whose mask is anything
// else is refused rather than scored differently from how a node would.
func ReadInput(path string) ([]int64, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, "", fmt.Errorf("adapterbuild: reading --input: %w", err)
	}
	var in struct {
		InputIDs      []int64 `json:"input_ids"`
		AttentionMask []int64 `json:"attention_mask"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return nil, "", fmt.Errorf("adapterbuild: %s: %w", path, err)
	}
	if len(in.InputIDs) == 0 {
		return nil, "", fmt.Errorf("adapterbuild: %s has no input_ids", path)
	}
	if in.AttentionMask != nil {
		if len(in.AttentionMask) != len(in.InputIDs) {
			return nil, "", fmt.Errorf("adapterbuild: %s: attention_mask has %d entries for %d ids", path, len(in.AttentionMask), len(in.InputIDs))
		}
		for i, m := range in.AttentionMask {
			if m != 1 {
				return nil, "", fmt.Errorf("adapterbuild: %s: attention_mask[%d] = %d; the node never pads (all ones)", path, i, m)
			}
		}
	}
	sum := sha256.Sum256(raw)
	return in.InputIDs, hex.EncodeToString(sum[:]), nil
}

func writeFileAtomic(path string, b []byte) error {
	tmp := path + ".partial"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return fmt.Errorf("adapterbuild: writing %s: %w", path, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp) //nolint:errcheck
		return fmt.Errorf("adapterbuild: installing %s: %w", path, err)
	}
	return nil
}
