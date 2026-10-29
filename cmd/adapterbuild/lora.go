package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// The `lora` mode checks a safetensors delta package (export_adapter.py output) for
// the shared Foundation Model with the node's own parser and prints the
// keccak256 to register:
//
//	adapterbuild lora --intent CODE_REVIEW --package runs/CODE_REVIEW/CODE_REVIEW.safetensors \
//	  --foundation foundation/foundation.json [--out runs/CODE_REVIEW] [--class judgment] \
//	  [--training-record runs/CODE_REVIEW/training.json --require-training-record --families ...] \
//	  [--foundation-onnx foundation/foundation_lora_fp32.onnx --engine foundation_engine.wasm]
//
// Nothing is rewritten: the package is the registrable artifact. With
// --foundation-onnx and --engine the package is also loaded on the foundation
// through the node's runtime and scored twice (identical bits).

// LoRAOutput is the single JSON object printed on success.
type LoRAOutput struct {
	Intent       string                    `json:"intent"`
	Class        string                    `json:"class"`
	Keccak256    string                    `json:"keccak256"`
	SHA256       string                    `json:"sha256"`
	Bytes        int64                     `json:"bytes"`
	PackagePath  string                    `json:"package_path"`
	ManifestPath string                    `json:"manifest_path"`
	ScoreHex     string                    `json:"score_hex,omitempty"`
	Manifest     adapterbuild.LoRAManifest `json:"manifest"`
}

func runLoRA(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("adapterbuild lora", flag.ContinueOnError)
	fs.SetOutput(stderr)
	intent := fs.String("intent", "", "intent ID (required)")
	pkg := fs.String("package", "", "the safetensors delta package (export_adapter.py output) (required)")
	class := fs.String("class", "", "judgment | hybrid (optional; must agree with the intent catalog)")
	foundation := fs.String("foundation", "", "foundation.json from export_foundation.py (required)")
	out := fs.String("out", "", "directory for the manifest (default: next to the package)")
	trainingRecord := fs.String("training-record", "", "training record JSON that must equal the one embedded in the package")
	requireTraining := fs.Bool("require-training-record", false, "refuse unless the package embeds a record whose children are exactly the children --intent serves")
	families := fs.String("families", "", "comma-separated OTHER installed family intents, for --require-training-record")
	modelPath := fs.String("foundation-onnx", "", "the foundation graph: load the package on it and score the default input twice")
	enginePath := fs.String("engine", "", "the foundation engine wasm (with --foundation-onnx)")
	engineHash := fs.String("engine-hash", "", "sha256 the engine must have (default: computed)")
	input := fs.String("input", "", "input JSON {\"input_ids\": [...]} to score (default: a 10-token ModernBERT input)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	logf := func(format string, a ...any) { fmt.Fprintf(stderr, "adapterbuild lora: "+format+"\n", a...) }
	fail := func(err error) int { fmt.Fprintf(stderr, "adapterbuild lora: FAILED: %v\n", err); return 1 }
	if *intent == "" || *pkg == "" || *foundation == "" {
		return fail(errors.New("--intent, --package and --foundation are required"))
	}
	var rec *wasmrt.TrainingRecord
	if *trainingRecord != "" {
		var err error
		if rec, err = adapterbuild.LoadTrainingRecord(*trainingRecord); err != nil {
			return fail(err)
		}
	}
	var ids []int64
	if *input != "" {
		var err error
		if ids, _, err = adapterbuild.ReadInput(*input); err != nil {
			return fail(err)
		}
	}
	start := time.Now()
	art, man, err := adapterbuild.CheckLoRA(context.Background(), adapterbuild.LoRAReq{
		Intent: *intent, PackagePath: *pkg, Class: *class, FoundationPath: *foundation, OutDir: *out,
		Training: rec, RequireTraining: *requireTraining, Families: splitList(*families),
		ModelPath: *modelPath, EnginePath: *enginePath, EngineHash: *engineHash, Input: ids,
	}, logf)
	if err != nil {
		return fail(err)
	}
	logf("checked %s (%d B, keccak256 %s) in %s", art.Path, art.Bytes, art.Keccak256, time.Since(start).Round(time.Millisecond))
	o := LoRAOutput{Intent: *intent, Class: man.Class, Keccak256: art.Keccak256, SHA256: art.SHA256, Bytes: art.Bytes,
		PackagePath: art.Path, ManifestPath: art.ManifestPath, ScoreHex: man.Verification.ScoreHex, Manifest: man}
	b, err := json.Marshal(o)
	if err != nil {
		return fail(err)
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}
