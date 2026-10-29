package adapterbuild

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
)

const loraFixtures = "../../float_runtime/testdata"

func loraFixturePath(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join(loraFixtures, name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("fixture %s missing: %v", name, err)
	}
	return p
}

func useLoRAClasses(t *testing.T) {
	t.Helper()
	catalog.UseClasses(map[string]catalog.Class{"CODE_REVIEW": catalog.ClassJudgment, "VULNERABILITY_TRIAGE": catalog.ClassHybrid})
	t.Cleanup(catalog.ClearClasses)
}

// The lora mode checks a package with the node's parser against foundation.json,
// writes a manifest and reports the keccak256; and refuses a package whose
// header names another intent or class.
func TestCheckLoRA(t *testing.T) {
	useLoRAClasses(t)
	out := t.TempDir()
	art, man, err := CheckLoRA(context.Background(), LoRAReq{
		Intent: "CODE_REVIEW", PackagePath: loraFixturePath(t, "tiny_judgment.safetensors"),
		FoundationPath: loraFixturePath(t, "tiny_foundation.json"), OutDir: out,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(art.Keccak256, "0x") || len(art.Keccak256) != 66 || man.Class != "judgment" || man.Rank != 2 || man.Tensors != 18 || !man.Verification.Stage1 {
		t.Fatalf("art %+v man %+v", art, man)
	}
	if _, err := os.Stat(filepath.Join(out, "tiny_judgment.safetensors.manifest.json")); err != nil {
		t.Fatal(err)
	}
	_, _, err = CheckLoRA(context.Background(), LoRAReq{Intent: "VULNERABILITY_TRIAGE", PackagePath: loraFixturePath(t, "tiny_judgment.safetensors"),
		FoundationPath: loraFixturePath(t, "tiny_foundation.json"), OutDir: out}, nil)
	if err == nil || !strings.Contains(err.Error(), "class_mismatch") {
		t.Fatalf("judgment package on a hybrid intent: %v", err)
	}
	_, _, err = CheckLoRA(context.Background(), LoRAReq{Intent: "VULNERABILITY_TRIAGE", PackagePath: loraFixturePath(t, "tiny_hybrid.safetensors"),
		FoundationPath: loraFixturePath(t, "tiny_foundation.json"), OutDir: out}, nil)
	if err != nil {
		t.Fatalf("hybrid package on a hybrid intent: %v", err)
	}
}

// With the foundation graph and the engine the package is loaded through the
// node's runtime and scored twice with identical bits.
func TestCheckLoRA_LoadsAndScores(t *testing.T) {
	useLoRAClasses(t)
	art, man, err := CheckLoRA(context.Background(), LoRAReq{
		Intent: "CODE_REVIEW", PackagePath: loraFixturePath(t, "tiny_judgment.safetensors"),
		FoundationPath: loraFixturePath(t, "tiny_foundation.json"), OutDir: t.TempDir(),
		ModelPath: loraFixturePath(t, "tiny_foundation.onnx"), EnginePath: loraFixturePath(t, "foundation_engine.wasm"),
		Input:     []int64{1, 5, 9, 13, 17, 2}, // the stub foundation's vocabulary is 64 ids
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !man.Verification.Loaded || man.Verification.Repeats != 2 || !strings.HasPrefix(man.Verification.ScoreHex, "0x") {
		t.Fatalf("verification %+v (%s)", man.Verification, art.Path)
	}
	t.Logf("score %s", man.Verification.ScoreHex)
}
