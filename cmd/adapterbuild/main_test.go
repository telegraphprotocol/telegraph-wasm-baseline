package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
)

type memStore struct {
	objs  map[string]map[string]string
	sizes map[string]int64
}

func (m *memStore) Stat(_ context.Context, key string) (adapterbuild.ObjectInfo, bool, error) {
	meta, ok := m.objs[key]
	if !ok {
		return adapterbuild.ObjectInfo{}, false, nil
	}
	return adapterbuild.ObjectInfo{Size: m.sizes[key], Metadata: meta}, true, nil
}

func (m *memStore) PutFile(_ context.Context, key, path, _ string, meta map[string]string) error {
	st, err := os.Stat(path)
	if err != nil {
		return err
	}
	m.objs[key], m.sizes[key] = meta, st.Size()
	return nil
}
func (m *memStore) URL(key string) string { return "http://store.test/bucket/" + key }
func (m *memStore) Describe() string      { return "bucket@store.test" }

func root(t *testing.T) string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(f), "..", "..")
}

// The stdout contract the training orchestrator and the dev script parse:
// exactly one JSON object, with these fields, and nothing on stdout on failure.
func TestRunJSONOutput(t *testing.T) {
	r := root(t)
	hosts := filepath.Join(r, filepath.FromSlash(adapterbuild.HostsRelDir))
	imwq := filepath.Join(r, "wide_runtime", "wide_wasm", "tests", "model_fixture.imwq")
	out := t.TempDir()

	store := &memStore{objs: map[string]map[string]string{}, sizes: map[string]int64{}}
	orig := newStore
	newStore = func(context.Context) (adapterbuild.ObjectStore, error) { return store, nil }
	defer func() { newStore = orig }()

	var so, se bytes.Buffer
	code := run([]string{"--intent", "VULNERABILITY_TRIAGE", "--imwq", imwq, "--out", out, "--hosts-dir", hosts,
		"--adapter", "fixture-adapter", "--publish"}, &so, &se)
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
	for _, k := range []string{"intent", "class", "keccak256", "sha256", "bytes", "url", "manifest_path", "manifest", "wasm_path", "publish"} {
		if _, ok := o[k]; !ok {
			t.Errorf("output missing %q", k)
		}
	}
	if o["intent"] != "VULNERABILITY_TRIAGE" || o["class"] != "hybrid" ||
		o["url"] != "http://store.test/bucket/wasm/VULNERABILITY_TRIAGE/VULNERABILITY_TRIAGE.wasm" ||
		!strings.HasPrefix(o["keccak256"].(string), "0x") {
		t.Fatalf("output values: %v", o)
	}
	man := o["manifest"].(map[string]any)
	if man["bundle_keccak256"] != o["keccak256"] || man["provenance"].(map[string]any)["adapter"] != "fixture-adapter" {
		t.Fatalf("manifest inside output disagrees: %v", man)
	}

	// Publish-only mode over the same artifact: re-verified, idempotent.
	so.Reset()
	se.Reset()
	art := filepath.Join(out, "VULNERABILITY_TRIAGE.wasm")
	if code := run([]string{"--intent", "VULNERABILITY_TRIAGE", "--artifact", art, "--hosts-dir", hosts, "--publish"}, &so, &se); code != 0 {
		t.Fatalf("publish-only exit %d; stderr:\n%s", code, se.String())
	}
	var o2 map[string]any
	if err := json.Unmarshal(so.Bytes(), &o2); err != nil || o2["keccak256"] != o["keccak256"] ||
		o2["publish"].(map[string]any)["unchanged"] != true {
		t.Fatalf("publish-only output: %v err=%v", o2, err)
	}

	// Failure: nothing on stdout, non-zero exit.
	so.Reset()
	se.Reset()
	if code := run([]string{"--intent", "FX_NOW", "--imwq", imwq, "--out", out, "--hosts-dir", hosts}, &so, &se); code == 0 || so.Len() != 0 {
		t.Fatalf("deterministic intent: exit %d stdout=%q", code, so.String())
	}
	if !strings.Contains(se.String(), "deterministic") {
		t.Fatalf("stderr does not explain the refusal: %s", se.String())
	}
}
