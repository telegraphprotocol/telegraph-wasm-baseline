package adapterbuild

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// Training records: embedded + echoed, and --require-training-record's exact
// child-set check. The catalog is a fixture family (UseClasses), so the
// builder's class source is the "loaded" one.

func useTrainingCatalog(t *testing.T) {
	t.Helper()
	catalog.UseClasses(map[string]catalog.Class{
		"CODE_REVIEW":         catalog.ClassJudgment,
		"CODE_REVIEW_GO":      catalog.ClassJudgment,
		"CODE_REVIEW_GO_TEST": catalog.ClassJudgment,
		"CODE_REVIEW_RUST":    catalog.ClassJudgment,
		"CODE_REVIEW_PY":      catalog.ClassHybrid, // another class: never served by CODE_REVIEW
		"CODE_REVIEWX":        catalog.ClassJudgment,
		"FX_NOW":              catalog.ClassDeterministic,
	})
	t.Cleanup(catalog.ClearClasses)
}

func trainingFor(family string, children ...string) *wasmrt.TrainingRecord {
	r := &wasmrt.TrainingRecord{Family: family, Trainer: "intent_training_experiments@abc", Backbone: "answerdotai/ModernBERT-base", BackboneVersion: "bb-v2"}
	for _, c := range children {
		r.Children = append(r.Children, wasmrt.TrainingChild{Intent: c, DatasetVersion: "ds-1", Pairs: 10, Accuracy: 0.8, Spearman: 0.4})
	}
	return r
}

var allServed = []string{"CODE_REVIEW", "CODE_REVIEW_GO", "CODE_REVIEW_GO_TEST", "CODE_REVIEW_RUST"}

func TestExpectedChildren(t *testing.T) {
	useTrainingCatalog(t)
	if got := strings.Join(ExpectedChildren("CODE_REVIEW", nil), ","); got != strings.Join(allServed, ",") {
		t.Fatalf("ExpectedChildren(no other families) = %s", got)
	}
	// A longer installed family owns its own children; another-class or
	// unknown "families" change nothing.
	got := strings.Join(ExpectedChildren("CODE_REVIEW", []string{"CODE_REVIEW_GO", "CODE_REVIEW_PY", "NOPE"}), ",")
	if got != "CODE_REVIEW,CODE_REVIEW_RUST" {
		t.Fatalf("ExpectedChildren(with CODE_REVIEW_GO installed) = %s", got)
	}
}

func TestCheckTrainingRecord(t *testing.T) {
	useTrainingCatalog(t)
	ok := []struct {
		name     string
		rec      *wasmrt.TrainingRecord
		require  bool
		families []string
	}{
		{"no record, not required", nil, false, nil},
		{"exact match", trainingFor("CODE_REVIEW", allServed...), true, nil},
		{"exact match, any order", trainingFor("CODE_REVIEW", "CODE_REVIEW_RUST", "CODE_REVIEW", "CODE_REVIEW_GO_TEST", "CODE_REVIEW_GO"), true, nil},
		{"longer family's children are not required", trainingFor("CODE_REVIEW", "CODE_REVIEW", "CODE_REVIEW_RUST"), true, []string{"CODE_REVIEW_GO"}},
		{"not required: child set not compared", trainingFor("CODE_REVIEW", "CODE_REVIEW"), false, nil},
	}
	for _, c := range ok {
		if err := CheckTrainingRecord("CODE_REVIEW", c.rec, c.require, c.families); err != nil {
			t.Errorf("%s: %v", c.name, err)
		}
	}
	bad := []struct {
		name     string
		rec      *wasmrt.TrainingRecord
		require  bool
		families []string
		want     string
	}{
		{"missing record", nil, true, nil, "no training record given"},
		{"family != intent", trainingFor("CODE_REVIEW_GO", "CODE_REVIEW_GO"), false, nil, "training record is for family CODE_REVIEW_GO but --intent is CODE_REVIEW"},
		{"missing child", trainingFor("CODE_REVIEW", "CODE_REVIEW", "CODE_REVIEW_GO", "CODE_REVIEW_GO_TEST"), true, nil, "missing [CODE_REVIEW_RUST]"},
		{"extra child (another class)", trainingFor("CODE_REVIEW", append(append([]string{}, allServed...), "CODE_REVIEW_PY")...), true, nil, "not served [CODE_REVIEW_PY]"},
		{"extra child (unknown)", trainingFor("CODE_REVIEW", append(append([]string{}, allServed...), "CODE_REVIEW_JAVA")...), true, nil, "not served [CODE_REVIEW_JAVA]"},
		{"child owned by a longer family", trainingFor("CODE_REVIEW", allServed...), true, []string{"CODE_REVIEW_GO"}, "not served [CODE_REVIEW_GO CODE_REVIEW_GO_TEST]"},
		{"malformed", trainingFor("CODE_REVIEW"), false, nil, "training record: children is empty"},
	}
	for _, c := range bad {
		err := CheckTrainingRecord("CODE_REVIEW", c.rec, c.require, c.families)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err=%v, want containing %q", c.name, err, c.want)
		}
	}
}

func TestLoadTrainingRecord(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	rec, err := LoadTrainingRecord(write("ok.json", `{"family":"W","children":[{"intent":"W_X","dataset_version":"v","pairs":812,"accuracy":0.81,"spearman":0.44,"held_out":false}],"trainer":"t@1","backbone":"b","backbone_version":"bb-v2"}`))
	if err != nil || rec.Family != "W" || rec.Children[0].Pairs != 812 {
		t.Fatalf("LoadTrainingRecord = %+v, %v", rec, err)
	}
	if _, err := LoadTrainingRecord(write("typo.json", `{"family":"W","children":[{"intent":"W","held-out":true}]}`)); err == nil || !strings.Contains(err.Error(), "held-out") {
		t.Fatalf("unknown field: err=%v", err)
	}
	if _, err := LoadTrainingRecord(write("bad.json", `{"family":"W","children":[{"intent":"V"}]}`)); err == nil || !strings.Contains(err.Error(), "not W or a child") {
		t.Fatalf("invalid record: err=%v", err)
	}
}

// withTraining writes a copy of the tiny judgment fixture with rec embedded
// in its safetensors metadata (as export_adapter --training-record does).
func withTraining(t *testing.T, rec *wasmrt.TrainingRecord) string {
	t.Helper()
	src := filepath.Join(loraFixtures, "tiny_judgment.safetensors")
	b, err := os.ReadFile(src)
	if err != nil {
		t.Skipf("fixture missing: %v", err)
	}
	hl := int(binary.LittleEndian.Uint64(b[:8]))
	var h map[string]any
	if err := json.Unmarshal(b[8:8+hl], &h); err != nil {
		t.Fatal(err)
	}
	rj, _ := json.Marshal(rec)
	h["__metadata__"].(map[string]any)["training"] = string(rj)
	hj, _ := json.Marshal(h)
	n := binary.LittleEndian.AppendUint64(nil, uint64(len(hj)))
	n = append(n, hj...)
	n = append(n, b[8+hl:]...)
	out := filepath.Join(t.TempDir(), "CODE_REVIEW.safetensors")
	if err := os.WriteFile(out, n, 0o644); err != nil {
		t.Fatal(err)
	}
	return out
}

// CheckLoRA refuses before writing a manifest.
func TestCheckLoRARequireTrainingRecordRefuses(t *testing.T) {
	useTrainingCatalog(t)
	fnd := filepath.Join(loraFixtures, "tiny_foundation.json")
	for _, c := range []struct {
		name string
		rec  *wasmrt.TrainingRecord
		want string
	}{
		{"missing", nil, "no training record given"},
		{"extra child", trainingFor("CODE_REVIEW", append(append([]string{}, allServed...), "CODE_REVIEW_PY")...), "not served"},
		{"missing child", trainingFor("CODE_REVIEW", "CODE_REVIEW"), "missing"},
		{"family mismatch", trainingFor("CODE_REVIEW_GO", "CODE_REVIEW_GO"), "--intent is CODE_REVIEW"},
	} {
		pkg := filepath.Join(loraFixtures, "tiny_judgment.safetensors")
		if c.rec != nil {
			pkg = withTraining(t, c.rec)
		}
		out := t.TempDir()
		_, _, err := CheckLoRA(context.Background(), LoRAReq{Intent: "CODE_REVIEW", PackagePath: pkg, FoundationPath: fnd, OutDir: out,
			RequireTraining: true}, t.Logf)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err=%v, want containing %q", c.name, err, c.want)
		}
		if entries, _ := os.ReadDir(out); len(entries) != 0 {
			t.Errorf("%s: a refused check wrote %d files", c.name, len(entries))
		}
	}
}

// The record embedded in the package (read back by the node's own parser)
// lands in the manifest; --training-record must equal it.
func TestCheckLoRAEchoesTrainingRecord(t *testing.T) {
	useTrainingCatalog(t)
	fnd := filepath.Join(loraFixtures, "tiny_foundation.json")
	rec := trainingFor("CODE_REVIEW", allServed...)
	pkg := withTraining(t, rec)
	a, m, err := CheckLoRA(context.Background(), LoRAReq{Intent: "CODE_REVIEW", PackagePath: pkg, FoundationPath: fnd, OutDir: t.TempDir(),
		Training: rec, RequireTraining: true}, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	if m.Training == nil || strings.Join(m.Training.ChildIntents(), ",") != strings.Join(allServed, ",") {
		t.Fatalf("manifest training = %+v", m.Training)
	}
	raw, _ := os.ReadFile(a.ManifestPath)
	var generic map[string]any
	if err := json.Unmarshal(raw, &generic); err != nil {
		t.Fatal(err)
	}
	if tr, ok := generic["training"].(map[string]any); !ok || tr["family"] != "CODE_REVIEW" {
		t.Fatalf("on-disk manifest training = %v", generic["training"])
	}
	other := trainingFor("CODE_REVIEW", "CODE_REVIEW")
	if _, _, err := CheckLoRA(context.Background(), LoRAReq{Intent: "CODE_REVIEW", PackagePath: pkg, FoundationPath: fnd, OutDir: t.TempDir(),
		Training: other}, t.Logf); err == nil || !strings.Contains(err.Error(), "differs from the record embedded") {
		t.Fatalf("mismatched --training-record: %v", err)
	}
	// A package without a record has no training key in its manifest.
	_, m2, err := CheckLoRA(context.Background(), LoRAReq{Intent: "CODE_REVIEW", PackagePath: filepath.Join(loraFixtures, "tiny_judgment.safetensors"),
		FoundationPath: fnd, OutDir: t.TempDir()}, t.Logf)
	if err != nil || m2.Training != nil {
		t.Fatalf("no-record package: training %+v err %v", m2.Training, err)
	}
}
