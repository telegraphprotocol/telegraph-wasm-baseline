package adapterbuild

import (
	"context"
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

// Build refuses before packaging, writing nothing.
func TestBuildRequireTrainingRecordRefuses(t *testing.T) {
	useTrainingCatalog(t)
	out := t.TempDir()
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
		_, _, err := Build(context.Background(), Req{Intent: "CODE_REVIEW", IMWQPath: fixtureIMWQ(t), OutDir: out, HostsDir: committedHosts(t),
			Training: c.rec, RequireTraining: true}, t.Logf)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err=%v, want containing %q", c.name, err, c.want)
		}
		assertNothingWritten(t, out, "CODE_REVIEW")
	}
}

// The record lands in the package's telegraph.adapter section (the one
// section, read back by the node's own reader) and in the manifest.
func TestBuildEmbedsAndEchoesTrainingRecord(t *testing.T) {
	useTrainingCatalog(t)
	rec := trainingFor("CODE_REVIEW", allServed...)
	a, m, err := Build(context.Background(), Req{Intent: "CODE_REVIEW", IMWQPath: fixtureIMWQ(t), OutDir: t.TempDir(),
		HostsDir: committedHosts(t), Training: rec, RequireTraining: true}, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	if m.Training == nil || strings.Join(m.Training.ChildIntents(), ",") != strings.Join(allServed, ",") {
		t.Fatalf("manifest training = %+v", m.Training)
	}
	b, err := os.ReadFile(a.Path)
	if err != nil {
		t.Fatal(err)
	}
	meta, err := wasmrt.ReadAdapterMeta(b)
	if err != nil || meta == nil || meta.Training == nil {
		t.Fatalf("section: %+v, %v", meta, err)
	}
	if got, want := string(meta.Encode()), string((wasmrt.AdapterMeta{ABIVersion: 1, Class: "judgment", Tool: "wideembed", Training: rec}).Encode()); got != want {
		t.Fatalf("section payload:\n got %s\nwant %s", got, want)
	}
	if err := meta.Check("judgment"); err != nil {
		t.Fatalf("node-side Check: %v", err)
	}
	raw, _ := os.ReadFile(a.ManifestPath)
	var generic map[string]any
	if err := json.Unmarshal(raw, &generic); err != nil {
		t.Fatal(err)
	}
	if tr, ok := generic["training"].(map[string]any); !ok || tr["family"] != "CODE_REVIEW" {
		t.Fatalf("on-disk manifest training = %v", generic["training"])
	}
	// A build without a record has no training key (byte-identical section).
	a2, m2, err := Build(context.Background(), Req{Intent: "CODE_REVIEW", IMWQPath: fixtureIMWQ(t), OutDir: t.TempDir(), HostsDir: committedHosts(t)}, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	b2, _ := os.ReadFile(a2.Path)
	meta2, _ := wasmrt.ReadAdapterMeta(b2)
	if m2.Training != nil || meta2 == nil || string(meta2.Encode()) != `{"abi_version":1,"class":"judgment","tool":"wideembed"}` {
		t.Fatalf("no-record build: manifest training %+v, section %s", m2.Training, meta2.Encode())
	}
	if a2.SHA256 == a.SHA256 {
		t.Fatal("the record must change the package bytes (it is inside the hashed module)")
	}
}
