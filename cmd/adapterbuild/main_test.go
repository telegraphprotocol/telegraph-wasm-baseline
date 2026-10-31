package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
)

func root(t *testing.T) string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(f), "..", "..")
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join(root(t), "float_runtime", "testdata", name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("fixture %s missing: %v", name, err)
	}
	return p
}

// withMetadata writes a copy of pkg with key=value set in its safetensors
// __metadata__ (the way export_adapter --training-record embeds a record).
func withMetadata(t *testing.T, pkg, key, value, out string) string {
	t.Helper()
	b, err := os.ReadFile(pkg)
	if err != nil {
		t.Fatal(err)
	}
	hl := int(binary.LittleEndian.Uint64(b[:8]))
	var h map[string]any
	if err := json.Unmarshal(b[8:8+hl], &h); err != nil {
		t.Fatal(err)
	}
	h["__metadata__"].(map[string]any)[key] = value
	hj, _ := json.Marshal(h)
	n := binary.LittleEndian.AppendUint64(nil, uint64(len(hj)))
	n = append(n, hj...)
	n = append(n, b[8+hl:]...)
	if err := os.WriteFile(out, n, 0o644); err != nil {
		t.Fatal(err)
	}
	return out
}

func useClasses(t *testing.T) {
	t.Helper()
	catalog.UseClasses(map[string]catalog.Class{
		"CODE_REVIEW":    catalog.ClassJudgment,
		"CODE_REVIEW_GO": catalog.ClassJudgment,
		"FX_NOW":         catalog.ClassDeterministic,
	})
	t.Cleanup(catalog.ClearClasses)
}

// The stdout contract the training orchestrator and the dev script parse:
// exactly one JSON object, with these fields, and nothing on stdout on
// failure. The `lora` word is optional; the legacy build flags are refused.
func TestRunJSONOutput(t *testing.T) {
	useClasses(t)
	pkg := fixture(t, "tiny_judgment.safetensors")
	fnd := fixture(t, "tiny_foundation.json")
	out := t.TempDir()

	var so, se bytes.Buffer
	code := run([]string{"lora", "--intent", "CODE_REVIEW", "--package", pkg, "--foundation", fnd, "--out", out}, &so, &se)
	if code != 0 {
		t.Fatalf("exit %d; stderr:\n%s", code, se.String())
	}
	lines := strings.Split(strings.TrimSpace(so.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("stdout must be exactly one JSON line, got %d:\n%s", len(lines), so.String())
	}
	var o map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &o); err != nil {
		t.Fatalf("stdout is not JSON: %v\n%s", err, lines[0])
	}
	for _, k := range []string{"intent", "class", "keccak256", "sha256", "bytes", "package_path", "manifest_path", "manifest"} {
		if _, ok := o[k]; !ok {
			t.Errorf("output missing %q", k)
		}
	}
	if o["intent"] != "CODE_REVIEW" || o["class"] != "judgment" || !strings.HasPrefix(o["keccak256"].(string), "0x") {
		t.Fatalf("output values: %v", o)
	}
	man := o["manifest"].(map[string]any)
	if man["keccak256"] != o["keccak256"] || man["abi"] != "lora-v1" {
		t.Fatalf("manifest inside output disagrees: %v", man)
	}
	if _, err := os.Stat(filepath.Join(out, "tiny_judgment.safetensors.manifest.json")); err != nil {
		t.Fatal(err)
	}

	// Without the word: the same.
	so.Reset()
	se.Reset()
	if code := run([]string{"--intent", "CODE_REVIEW", "--package", pkg, "--foundation", fnd, "--out", t.TempDir()}, &so, &se); code != 0 {
		t.Fatalf("bare mode: exit %d; stderr:\n%s", code, se.String())
	}

	// Failures: nothing on stdout, non-zero exit.
	for name, args := range map[string][]string{
		"deterministic intent": {"--intent", "FX_NOW", "--package", pkg, "--foundation", fnd, "--out", out},
		"legacy build flag":    {"--intent", "CODE_REVIEW", "--wasm", "baked.wasm", "--out", out},
	} {
		so.Reset()
		se.Reset()
		if code := run(args, &so, &se); code == 0 || so.Len() != 0 {
			t.Fatalf("%s: exit %d stdout=%q", name, code, so.String())
		}
	}
}

// --training-record must equal the record embedded in the package and is
// echoed in the one stdout JSON object; --require-training-record refuses a
// package without a record or with a mismatched child set, with nothing on
// stdout; --families feeds the longer-family exclusion.
func TestRunTrainingRecordFlags(t *testing.T) {
	useClasses(t)
	fnd := fixture(t, "tiny_foundation.json")
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	fullRec := `{"family":"CODE_REVIEW","children":[` +
		`{"intent":"CODE_REVIEW","pairs":10,"accuracy":0.8,"spearman":0.4,"held_out":false},` +
		`{"intent":"CODE_REVIEW_GO","dataset_version":"v1","pairs":812,"accuracy":0.81,"spearman":0.44,"held_out":true}],` +
		`"trainer":"intent_training_experiments@abc","backbone":"answerdotai/ModernBERT-base","backbone_version":"bb-v2"}`
	familyRec := `{"family":"CODE_REVIEW","children":[{"intent":"CODE_REVIEW","pairs":10,"accuracy":0.8,"spearman":0.4,"held_out":false}]}`
	full := write("full.json", fullRec)
	familyOnly := write("family-only.json", familyRec)
	plain := fixture(t, "tiny_judgment.safetensors")
	fullPkg := withMetadata(t, plain, "training", fullRec, filepath.Join(dir, "full.safetensors"))
	familyPkg := withMetadata(t, plain, "training", familyRec, filepath.Join(dir, "family.safetensors"))

	runOK := func(pkg string, extra ...string) map[string]any {
		t.Helper()
		var so, se bytes.Buffer
		args := append([]string{"lora", "--intent", "CODE_REVIEW", "--package", pkg, "--foundation", fnd, "--out", t.TempDir()}, extra...)
		if code := run(args, &so, &se); code != 0 {
			t.Fatalf("%v: exit %d; stderr:\n%s", extra, code, se.String())
		}
		var o map[string]any
		if err := json.Unmarshal(bytes.TrimSpace(so.Bytes()), &o); err != nil {
			t.Fatalf("stdout is not one JSON object: %v\n%s", err, so.String())
		}
		return o
	}

	o := runOK(fullPkg, "--training-record", full, "--require-training-record")
	tr, ok := o["manifest"].(map[string]any)["training"].(map[string]any)
	if !ok || tr["family"] != "CODE_REVIEW" || len(tr["children"].([]any)) != 2 {
		t.Fatalf("output does not echo the training record: %v", o["manifest"])
	}
	// With CODE_REVIEW_GO installed as a longer family, only CODE_REVIEW is
	// required: the family-only record passes, the full one does not.
	runOK(familyPkg, "--training-record", familyOnly, "--require-training-record", "--families", "CODE_REVIEW_GO, ")

	for _, c := range []struct {
		name string
		pkg  string
		args []string
		want string
	}{
		{"missing record", plain, []string{"--require-training-record"}, "no training record given"},
		{"missing child", familyPkg, []string{"--require-training-record"}, "missing [CODE_REVIEW_GO]"},
		{"child owned by a longer family", fullPkg, []string{"--require-training-record", "--families", "CODE_REVIEW_GO"}, "not served [CODE_REVIEW_GO]"},
		{"record differs from the package", fullPkg, []string{"--training-record", familyOnly}, "differs from the record embedded"},
	} {
		var so, se bytes.Buffer
		args := append([]string{"lora", "--intent", "CODE_REVIEW", "--package", c.pkg, "--foundation", fnd, "--out", t.TempDir()}, c.args...)
		code := run(args, &so, &se)
		if code == 0 || so.Len() != 0 || !strings.Contains(se.String(), c.want) {
			t.Errorf("%s: exit %d stdout=%q stderr=%s; want refusal mentioning %q", c.name, code, so.String(), se.String(), c.want)
		}
	}
}
