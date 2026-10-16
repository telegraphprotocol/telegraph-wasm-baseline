package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
)

// --training-record is embedded and echoed in the one stdout JSON object;
// --require-training-record refuses a mismatch with nothing on stdout;
// --families feeds the longer-family exclusion.
func TestRunTrainingRecordFlags(t *testing.T) {
	catalog.UseClasses(map[string]catalog.Class{
		"CODE_REVIEW":    catalog.ClassJudgment,
		"CODE_REVIEW_GO": catalog.ClassJudgment,
	})
	t.Cleanup(catalog.ClearClasses)

	r := root(t)
	hosts := filepath.Join(r, filepath.FromSlash(adapterbuild.HostsRelDir))
	imwq := filepath.Join(r, "wide_runtime", "wide_wasm", "tests", "model_fixture.imwq")
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	full := write("full.json", `{"family":"CODE_REVIEW","children":[`+
		`{"intent":"CODE_REVIEW","pairs":10,"accuracy":0.8,"spearman":0.4,"held_out":false},`+
		`{"intent":"CODE_REVIEW_GO","dataset_version":"v1","pairs":812,"accuracy":0.81,"spearman":0.44,"held_out":true}],`+
		`"trainer":"intent_training_experiments@abc","backbone":"answerdotai/ModernBERT-base","backbone_version":"bb-v2"}`)
	familyOnly := write("family-only.json", `{"family":"CODE_REVIEW","children":[{"intent":"CODE_REVIEW","pairs":10,"accuracy":0.8,"spearman":0.4,"held_out":false}]}`)

	base := []string{"--intent", "CODE_REVIEW", "--imwq", imwq, "--hosts-dir", hosts}
	runOK := func(extra ...string) map[string]any {
		t.Helper()
		var so, se bytes.Buffer
		if code := run(append(append(append([]string{}, base...), "--out", t.TempDir()), extra...), &so, &se); code != 0 {
			t.Fatalf("%v: exit %d; stderr:\n%s", extra, code, se.String())
		}
		var o map[string]any
		if err := json.Unmarshal(bytes.TrimSpace(so.Bytes()), &o); err != nil {
			t.Fatalf("stdout is not one JSON object: %v\n%s", err, so.String())
		}
		return o
	}

	o := runOK("--training-record", full, "--require-training-record")
	tr, ok := o["manifest"].(map[string]any)["training"].(map[string]any)
	if !ok || tr["family"] != "CODE_REVIEW" || len(tr["children"].([]any)) != 2 {
		t.Fatalf("output does not echo the training record: %v", o["manifest"])
	}

	// With CODE_REVIEW_GO installed as a longer family, only CODE_REVIEW is
	// required: the family-only record passes, the full one does not.
	runOK("--training-record", familyOnly, "--require-training-record", "--families", "CODE_REVIEW_GO, ")

	for _, c := range []struct {
		name string
		args []string
		want string
	}{
		{"missing record", []string{"--require-training-record"}, "no training record given"},
		{"missing child", []string{"--training-record", familyOnly, "--require-training-record"}, "missing [CODE_REVIEW_GO]"},
		{"child owned by a longer family", []string{"--training-record", full, "--require-training-record", "--families", "CODE_REVIEW_GO"}, "not served [CODE_REVIEW_GO]"},
	} {
		var so, se bytes.Buffer
		code := run(append(append(append([]string{}, base...), "--out", t.TempDir()), c.args...), &so, &se)
		if code == 0 || so.Len() != 0 || !strings.Contains(se.String(), c.want) {
			t.Errorf("%s: exit %d stdout=%q stderr=%s; want refusal mentioning %q", c.name, code, so.String(), se.String(), c.want)
		}
	}
}
