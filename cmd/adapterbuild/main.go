// Command adapterbuild checks an Evaluator adapter — a safetensors delta
// package for the shared Foundation Model (float_runtime/export_adapter.py
// output) — with the Telegraph node's own parser and prints the keccak256 to
// register. The package is the registrable artifact; nothing is rewritten.
//
//	go build -o adapterbuild ./cmd/adapterbuild
//
//	# check: <INTENT>.safetensors against foundation.json -> <out>/<INTENT>.safetensors.manifest.json
//	adapterbuild lora --intent CODE_REVIEW --package runs/CODE_REVIEW/CODE_REVIEW.safetensors \
//	  --foundation foundation/foundation.json [--out runs/CODE_REVIEW] [--class judgment]
//
//	# also load it on the foundation through the node's runtime and score an input twice (identical bits)
//	adapterbuild lora ... --foundation-onnx foundation/foundation_lora_fp32.onnx --engine foundation_engine.wasm [--input sample_input.json]
//
//	# a family package carrying its training record (our builds; see scoring-adapter-standard.md §9)
//	adapterbuild lora --intent WEATHER ... --training-record runs/WEATHER/training.json \
//	  --require-training-record [--families WEATHER_FORECAST,...]
//
// The class comes from the intent catalog (an explicit --class must agree);
// deterministic intents are refused. The checks are the node's Stage 1
// (foundation and tokenizer hashes, class, rank, target modules, every tensor
// shape, offsets, finite values) plus the training record against the
// children --intent serves (adapterbuild.ExpectedChildren). Nodes never act
// on the record; it is a check on our own builds only.
//
// stdout carries exactly one JSON object on success:
//
//	{intent, class, keccak256, sha256, bytes, package_path, manifest_path, score_hex?, manifest}
//
// Human logs go to stderr. Any failure exits non-zero with nothing on stdout.
// registerWasm (URL + keccak256) stays manual. The `lora` word is optional:
// it is the only mode.
package main

import (
	"fmt"
	"io"
	"os"
	"strings"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 && args[0] == "lora" {
		args = args[1:]
	}
	for _, a := range args {
		if a == "--wasm" || a == "--artifact" || a == "--publish" {
			fmt.Fprintf(stderr, "adapterbuild: FAILED: %s belonged to the full-model build, which nodes no longer accept; "+
				"export a safetensors delta package (float_runtime/export_adapter.py) and check it with `adapterbuild lora --package`\n", a)
			return 2
		}
	}
	return runLoRA(args, stdout, stderr)
}

// splitList parses a comma-separated flag value, dropping blanks.
func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
