package adapterbuild

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wideembed"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..")
}

// committedHosts is the real pinned hosts dir (build-hosts.sh output).
func committedHosts(t *testing.T) string {
	t.Helper()
	d := filepath.Join(repoRoot(t), filepath.FromSlash(HostsRelDir))
	if _, err := os.Stat(filepath.Join(d, LockFileName)); err != nil {
		t.Fatalf("committed hosts lock missing: %v", err)
	}
	return d
}

// fixtureIMWQ is the committed tiny model from the Rust golden tests.
func fixtureIMWQ(t *testing.T) string {
	t.Helper()
	return filepath.Join(repoRoot(t), "wide_runtime", "wide_wasm", "tests", "model_fixture.imwq")
}

func TestCommittedHostsMatchLock(t *testing.T) {
	h, err := LoadHosts(committedHosts(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []wideembed.Class{wideembed.ClassJudgment, wideembed.ClassHybrid} {
		b, e, err := h.Host(c)
		if err != nil {
			t.Fatalf("%s: %v", c, err)
		}
		hy, err := wasmrt.WideEmbeddedIsHybrid(b)
		if err != nil {
			t.Fatalf("%s host: %v", c, err)
		}
		if hy != (c == wideembed.ClassHybrid) {
			t.Fatalf("%s host (%s) exports imwq_score_hybrid=%v", c, e.File, hy)
		}
	}
	if !strings.Contains(h.Lock.Image, "@sha256:") {
		t.Fatalf("lock image is not digest-pinned: %q", h.Lock.Image)
	}
}

func TestResolveClass(t *testing.T) {
	ok := []struct{ intent, explicit, want string }{
		{"VULNERABILITY_TRIAGE", "", "hybrid"},
		{"VULNERABILITY_TRIAGE", "hybrid", "hybrid"},
		{"EMAIL_SECURITY", "", "hybrid"},
		{"CODE_REVIEW", "", "judgment"},
		{"CODE_REVIEW", "judgment", "judgment"},
	}
	for _, c := range ok {
		got, err := ResolveClass(c.intent, c.explicit)
		if err != nil || string(got) != c.want {
			t.Errorf("ResolveClass(%s,%q) = %q,%v want %s", c.intent, c.explicit, got, err, c.want)
		}
	}
	bad := []struct{ intent, explicit, wantErr string }{
		{"VULNERABILITY_TRIAGE", "judgment", "disagrees"},
		{"CODE_REVIEW", "hybrid", "disagrees"},
		{"FX_NOW", "", "deterministic"},
		{"ROUTE_ETA", "hybrid", "deterministic"},
		{"NOT_AN_INTENT", "", "not a judgment or hybrid intent"},
		{"", "", "intent is required"},
	}
	for _, c := range bad {
		if _, err := ResolveClass(c.intent, c.explicit); err == nil || !strings.Contains(err.Error(), c.wantErr) {
			t.Errorf("ResolveClass(%s,%q) err=%v, want containing %q", c.intent, c.explicit, err, c.wantErr)
		}
	}
}

// copyHosts makes a writable copy of the committed hosts dir.
func copyHosts(t *testing.T) string {
	t.Helper()
	src := committedHosts(t)
	dst := t.TempDir()
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

func TestBuildRefusesUnpinnedHost(t *testing.T) {
	ctx := context.Background()

	// 1. A host whose bytes changed (one byte flipped in the name section's tail).
	d := copyHosts(t)
	p := filepath.Join(d, "wide_embed_hybrid.wasm")
	b, _ := os.ReadFile(p)
	b[len(b)-1] ^= 0xff
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	out := t.TempDir()
	_, _, err := Build(ctx, Req{Intent: "VULNERABILITY_TRIAGE", IMWQPath: fixtureIMWQ(t), OutDir: out, HostsDir: d}, t.Logf)
	if err == nil || !strings.Contains(err.Error(), "not the pinned") {
		t.Fatalf("tampered host: err=%v, want a pinned-host refusal", err)
	}
	assertNothingWritten(t, out, "VULNERABILITY_TRIAGE")

	// 2. The other class's (genuine, pinned) host swapped into this class's slot.
	d = copyHosts(t)
	j, _ := os.ReadFile(filepath.Join(d, "wide_embed_judgment.wasm"))
	if err := os.WriteFile(filepath.Join(d, "wide_embed_hybrid.wasm"), j, 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err = Build(ctx, Req{Intent: "VULNERABILITY_TRIAGE", IMWQPath: fixtureIMWQ(t), OutDir: out, HostsDir: d}, t.Logf)
	if err == nil || !strings.Contains(err.Error(), "not the pinned") {
		t.Fatalf("swapped host: err=%v, want a pinned-host refusal", err)
	}

	// 3. A lock that pins something else entirely.
	d = copyHosts(t)
	lp := filepath.Join(d, LockFileName)
	lb, _ := os.ReadFile(lp)
	var l Lock
	if err := json.Unmarshal(lb, &l); err != nil {
		t.Fatal(err)
	}
	l.Judgment.SHA256 = strings.Repeat("0", 64)
	lb, _ = json.Marshal(l)
	if err := os.WriteFile(lp, lb, 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err = Build(ctx, Req{Intent: "CODE_REVIEW", IMWQPath: fixtureIMWQ(t), OutDir: out, HostsDir: d}, t.Logf)
	if err == nil || !strings.Contains(err.Error(), "not the pinned") {
		t.Fatalf("re-pinned lock: err=%v, want a pinned-host refusal", err)
	}
	assertNothingWritten(t, out, "CODE_REVIEW")
}

func TestBuildRefusesClassMismatchAndDeterministic(t *testing.T) {
	ctx := context.Background()
	out := t.TempDir()
	for _, c := range []struct{ intent, class, want string }{
		{"VULNERABILITY_TRIAGE", "judgment", "disagrees"},
		{"FX_NOW", "", "deterministic"},
		{"LIVE_SHELF_PRICE", "hybrid", "deterministic"},
	} {
		_, _, err := Build(ctx, Req{Intent: c.intent, Class: c.class, IMWQPath: fixtureIMWQ(t), OutDir: out, HostsDir: committedHosts(t)}, t.Logf)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s/%q: err=%v want %q", c.intent, c.class, err, c.want)
		}
		assertNothingWritten(t, out, c.intent)
	}
}

func TestBuildRefusesOverBudget(t *testing.T) {
	out := t.TempDir()
	_, _, err := Build(context.Background(), Req{Intent: "VULNERABILITY_TRIAGE", IMWQPath: fixtureIMWQ(t),
		OutDir: out, HostsDir: committedHosts(t), LimitBytes: 1 << 20}, t.Logf)
	if err == nil || !strings.Contains(err.Error(), "exceed the wasm32 memory ceiling") {
		t.Fatalf("err=%v, want the 4 GiB budget refusal", err)
	}
	assertNothingWritten(t, out, "VULNERABILITY_TRIAGE")
}

func assertNothingWritten(t *testing.T, dir, intent string) {
	t.Helper()
	for _, n := range []string{ArtifactName(intent), ManifestName(intent), ArtifactName(intent) + ".partial"} {
		if _, err := os.Stat(filepath.Join(dir, n)); err == nil {
			t.Fatalf("%s was written despite the failure", n)
		}
	}
}

// The twice-package gate on the committed tiny fixture, for both classes, plus
// cross-invocation determinism: a second Build reproduces the same bytes.
func TestBuildTwiceDeterministic(t *testing.T) {
	ctx := context.Background()
	for _, c := range []struct{ intent, class string }{
		{"VULNERABILITY_TRIAGE", "hybrid"},
		{"CODE_REVIEW", "judgment"},
	} {
		t.Run(c.intent, func(t *testing.T) {
			prov := Provenance{Adapter: "fixture", BaseModel: "tiny", BackboneVersion: "bb-test", LoRAHash: "abc", DatasetVersion: "v0"}
			out1, out2 := t.TempDir(), t.TempDir()
			a1, m1, err := Build(ctx, Req{Intent: c.intent, IMWQPath: fixtureIMWQ(t), OutDir: out1, HostsDir: committedHosts(t), Provenance: prov}, t.Logf)
			if err != nil {
				t.Fatal(err)
			}
			a2, _, err := Build(ctx, Req{Intent: c.intent, IMWQPath: fixtureIMWQ(t), OutDir: out2, HostsDir: committedHosts(t)}, t.Logf)
			if err != nil {
				t.Fatal(err)
			}
			if a1.SHA256 != a2.SHA256 || a1.Keccak256 != a2.Keccak256 {
				t.Fatalf("two Build invocations differ: %s vs %s", a1.SHA256, a2.SHA256)
			}
			if filepath.Base(a1.Path) != c.intent+".wasm" {
				t.Fatalf("artifact name %s, want %s.wasm", a1.Path, c.intent)
			}
			if string(m1.Class) != c.class || m1.Intent != c.intent || !m1.Verified ||
				!m1.Reproducibility.IdenticalSHA256 || m1.Reproducibility.Builds != 2 ||
				m1.Reproducibility.SHA256s[0] != m1.Reproducibility.SHA256s[1] {
				t.Fatalf("manifest: %+v", m1)
			}
			if m1.Provenance != prov || m1.HostsLock.SHA256 == "" || m1.HostsLock.ImageDigest == "" || m1.BuilderVersion == "" {
				t.Fatalf("manifest provenance/lock/builder not recorded: %+v", m1)
			}
			h, _ := LoadHosts(committedHosts(t))
			if cl, ok := h.Lock.Pins(m1.HostSHA256); !ok || string(cl) != c.class {
				t.Fatalf("manifest host_sha256 %s is not the pinned %s host", m1.HostSHA256, c.class)
			}
			// The on-disk manifest round-trips, flattened wideembed fields included.
			raw, _ := os.ReadFile(a1.ManifestPath)
			var generic map[string]any
			if err := json.Unmarshal(raw, &generic); err != nil {
				t.Fatal(err)
			}
			for _, k := range []string{"schema", "builder", "builder_version", "intent", "class", "host_sha256",
				"imwq_sha256", "bundle_sha256", "bundle_keccak256", "bundle_bytes", "budget", "verified",
				"provenance", "hosts_lock", "reproducibility", "built_at"} {
				if _, ok := generic[k]; !ok {
					t.Errorf("manifest missing %q", k)
				}
			}
			// And it loads as the class it says.
			b, _ := os.ReadFile(a1.Path)
			hy, err := wasmrt.WideEmbeddedIsHybrid(b)
			if err != nil || hy != (c.class == "hybrid") {
				t.Fatalf("package class: hybrid=%v err=%v", hy, err)
			}
			// LoadArtifact accepts the untouched build...
			if _, _, err := LoadArtifact(a1.Path, c.intent, committedHosts(t)); err != nil {
				t.Fatalf("LoadArtifact on a fresh build: %v", err)
			}
			// ...and refuses a tampered one.
			b[len(b)-1] ^= 1
			if err := os.WriteFile(a1.Path, b, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, _, err := LoadArtifact(a1.Path, c.intent, committedHosts(t)); err == nil || !strings.Contains(err.Error(), "manifest says") {
				t.Fatalf("LoadArtifact on tampered bytes: err=%v", err)
			}
			// ...and a mis-named one.
			if _, _, err := LoadArtifact(filepath.Join(out2, ArtifactName(c.intent)), "OTHER_INTENT", committedHosts(t)); err == nil {
				t.Fatal("LoadArtifact accepted an artifact under the wrong intent")
			}
		})
	}
}

// ── overwrite guard ─────────────────────────────────────────────────────────

type fakeObj struct {
	data []byte
	meta map[string]string
}

type fakeStore struct {
	objs map[string]fakeObj
	puts map[string]int
}

func newFakeStore() *fakeStore {
	return &fakeStore{objs: map[string]fakeObj{}, puts: map[string]int{}}
}

func (f *fakeStore) Stat(_ context.Context, key string) (ObjectInfo, bool, error) {
	o, ok := f.objs[key]
	if !ok {
		return ObjectInfo{}, false, nil
	}
	sum := md5.Sum(o.data)
	// Canonicalised header case, like real S3 servers return.
	meta := map[string]string{}
	for k, v := range o.meta {
		meta["X-Amz-Meta-"+strings.ToUpper(k[:1])+k[1:]] = v
	}
	return ObjectInfo{Size: int64(len(o.data)), Metadata: meta, ETag: hex.EncodeToString(sum[:])}, true, nil
}

func (f *fakeStore) PutFile(_ context.Context, key, path, _ string, meta map[string]string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	f.objs[key] = fakeObj{data: b, meta: meta}
	f.puts[key]++
	return nil
}

func (f *fakeStore) URL(key string) string { return "http://fake/bucket/" + key }
func (f *fakeStore) Describe() string      { return "bucket@fake" }

func fakeArtifact(t *testing.T, dir, intent string, content string) Artifact {
	t.Helper()
	p := filepath.Join(dir, ArtifactName(intent))
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	mp := p + ".manifest.json"
	if err := os.WriteFile(mp, []byte(`{"m":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	sha, n, err := sha256File(p)
	if err != nil {
		t.Fatal(err)
	}
	return Artifact{Path: p, ManifestPath: mp, SHA256: sha, Keccak256: "0xk-" + content, Bytes: n}
}

func TestPublishOverwriteGuard(t *testing.T) {
	ctx := context.Background()
	st := newFakeStore()
	const intent = "VULNERABILITY_TRIAGE"
	wasmKey, manKey := Keys(intent)
	if wasmKey != "wasm/VULNERABILITY_TRIAGE/VULNERABILITY_TRIAGE.wasm" ||
		manKey != "wasm/VULNERABILITY_TRIAGE/VULNERABILITY_TRIAGE.wasm.manifest.json" {
		t.Fatalf("keys: %s %s", wasmKey, manKey)
	}
	a := fakeArtifact(t, t.TempDir(), intent, "package-A")
	b := fakeArtifact(t, t.TempDir(), intent, "package-B-different")

	// First publish: uploads.
	r, err := Publish(ctx, st, a, intent, false, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	if r.Unchanged || r.Overwrote || st.puts[wasmKey] != 1 || r.URL != "http://fake/bucket/"+wasmKey {
		t.Fatalf("first publish: %+v puts=%d", r, st.puts[wasmKey])
	}
	if got := st.objs[wasmKey].meta["sha256"]; got != a.SHA256 {
		t.Fatalf("sha256 metadata %q, want %s", got, a.SHA256)
	}

	// Same bytes again: idempotent, the package is not re-uploaded.
	r, err = Publish(ctx, st, a, intent, false, t.Logf)
	if err != nil || !r.Unchanged || st.puts[wasmKey] != 1 || r.PreviousSHA256 != a.SHA256 {
		t.Fatalf("re-publish same bytes: %+v err=%v puts=%d", r, err, st.puts[wasmKey])
	}

	// Different bytes without --overwrite: refused, store untouched, old hash reported.
	_, err = Publish(ctx, st, b, intent, false, t.Logf)
	var ref *ErrOverwriteRefused
	if !errors.As(err, &ref) || ref.ExistingSHA256 != a.SHA256 || ref.NewSHA256 != b.SHA256 {
		t.Fatalf("different bytes: err=%v, want ErrOverwriteRefused naming both hashes", err)
	}
	if string(st.objs[wasmKey].data) != "package-A" || st.puts[wasmKey] != 1 {
		t.Fatal("a refused publish modified the store")
	}

	// With --overwrite: replaced, and the replaced hash is reported.
	r, err = Publish(ctx, st, b, intent, true, t.Logf)
	if err != nil || !r.Overwrote || r.PreviousSHA256 != a.SHA256 || string(st.objs[wasmKey].data) != "package-B-different" {
		t.Fatalf("overwrite: %+v err=%v", r, err)
	}

	// An existing object with NO sha256 metadata (e.g. uploaded by hand) cannot be
	// proven identical — even byte-identical content is refused without --overwrite.
	st2 := newFakeStore()
	raw, _ := os.ReadFile(a.Path)
	st2.objs[wasmKey] = fakeObj{data: raw, meta: map[string]string{}}
	if _, err := Publish(ctx, st2, a, intent, false, t.Logf); !errors.As(err, &ref) {
		t.Fatalf("no-metadata object: err=%v, want ErrOverwriteRefused", err)
	}
}

func TestStoreConfigFromEnv(t *testing.T) {
	set := func(kv map[string]string) {
		for _, k := range []string{"OBJECT_STORE_ENDPOINT", "OBJECT_STORE_BUCKET", "OBJECT_STORE_ACCESS_KEY",
			"OBJECT_STORE_SECRET_KEY", "OBJECT_STORE_PUBLIC_BASE_URL", "OBJECT_STORE_SECURE", "OBJECT_STORE_REGION"} {
			t.Setenv(k, kv[k])
		}
	}
	base := map[string]string{"OBJECT_STORE_ENDPOINT": "http://localhost:9000", "OBJECT_STORE_BUCKET": "b",
		"OBJECT_STORE_ACCESS_KEY": "a", "OBJECT_STORE_SECRET_KEY": "s"}

	set(base)
	c, err := StoreConfigFromEnv()
	if err != nil || c.Endpoint != "localhost:9000" || c.Secure || c.PublicBaseURL != "http://localhost:9000/b" {
		t.Fatalf("http endpoint: %+v %v", c, err)
	}

	m := map[string]string{}
	for k, v := range base {
		m[k] = v
	}
	m["OBJECT_STORE_ENDPOINT"] = "https://s3.example.com"
	m["OBJECT_STORE_PUBLIC_BASE_URL"] = "https://cdn.example.com/wasms/"
	set(m)
	if c, err = StoreConfigFromEnv(); err != nil || !c.Secure || c.PublicBaseURL != "https://cdn.example.com/wasms" {
		t.Fatalf("https + public base: %+v %v", c, err)
	}

	m["OBJECT_STORE_SECURE"] = "false"
	set(m)
	if _, err = StoreConfigFromEnv(); err == nil || !strings.Contains(err.Error(), "contradicts") {
		t.Fatalf("contradictory secure flag: %v", err)
	}

	set(map[string]string{"OBJECT_STORE_ENDPOINT": "localhost:9000"})
	if _, err = StoreConfigFromEnv(); err == nil || !strings.Contains(err.Error(), "OBJECT_STORE_ACCESS_KEY") {
		t.Fatalf("missing vars: %v", err)
	}
}
