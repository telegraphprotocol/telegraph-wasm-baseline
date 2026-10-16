package wideembed

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/comparator"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
)

// Build outputs the tests need. Like every other wide test they skip unless the
// builds are named — the embed hosts are cargo outputs, not committed binaries:
//
//	WIDE_WASM                 shared interpreter (default features)  — the external/split path
//	WIDE_EMBED_HOST_JUDGMENT  --features embedded
//	WIDE_EMBED_HOST_HYBRID    --features embedded-hybrid
func mustRead(t *testing.T, env string) []byte {
	t.Helper()
	p := os.Getenv(env)
	if p == "" {
		t.Skipf("set %s to run this", env)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("%s: %v", env, err)
	}
	return b
}

// fixture returns a committed tiny .imwq from the Rust crate's golden tests.
func fixture(t *testing.T, name string) []byte {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	p := filepath.Join(filepath.Dir(file), "..", "..", "..", "wide_runtime", "wide_wasm", "tests", name)
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("fixture %s: %v", name, err)
	}
	return b
}

// The fixtures' golden raw codes (tests/*_fixture.txt): the Python golden's
// output for these tokens. Parity is checked against these AND between paths.
var fixtures = []struct {
	file   string
	tokens []int64
	golden int64
}{
	{"model_fixture.imwq", []int64{1, 5, 9, 3, 7}, -2998410647},
	{"rope_model_fixture.imwq", []int64{2, 11, 4, 17, 8}, 484013641},
}

func hybridSpec(t *testing.T) [10]int64 {
	t.Helper()
	return comparator.Fusion{Mode: 0, Pass: 8000, Low: 2000, Floor: 3000, Span: 7000}.
		Buffer(comparator.Spec{Code: 1, Tolerance: 5})
}

// THE parity proof: the same model, the same tokens, through the split path
// (shared interpreter + LoadArtifact) and through a self-contained package,
// produces the identical raw Q32.32 code and the identical fused hybrid bps —
// and both equal the Python golden.
func TestEmbeddedMatchesExternal(t *testing.T) {
	interp := mustRead(t, "WIDE_WASM")
	hostJ := mustRead(t, "WIDE_EMBED_HOST_JUDGMENT")
	hostH := mustRead(t, "WIDE_EMBED_HOST_HYBRID")
	ctx := context.Background()
	spec := hybridSpec(t)
	facts := [][2]int64{{100, 100}, {100000, 100}, {103, 100}}

	for _, fx := range fixtures {
		t.Run(fx.file, func(t *testing.T) {
			imwq := fixture(t, fx.file)

			// External (split) path.
			ext, err := wasmrt.NewWideHybrid(ctx, interp)
			if err != nil {
				t.Fatal(err)
			}
			defer ext.Close(ctx) //nolint:errcheck
			if err := ext.LoadArtifact(ctx, imwq); err != nil {
				t.Fatal(err)
			}
			extRaw, err := ext.ScoreRaw(ctx, fx.tokens)
			if err != nil {
				t.Fatal(err)
			}
			var extBps []int32
			for _, f := range facts {
				b, err := ext.ScoreHybrid(ctx, fx.tokens, f[0], f[1], spec)
				if err != nil {
					t.Fatal(err)
				}
				extBps = append(extBps, b)
			}

			// Embedded judgment package.
			pkgJ, manJ, err := Package(ctx, hostJ, imwq, ClassJudgment, Options{})
			if err != nil {
				t.Fatal(err)
			}
			rj, err := wasmrt.NewWideEmbedded(ctx, pkgJ)
			if err != nil {
				t.Fatal(err)
			}
			defer rj.Close(ctx) //nolint:errcheck
			jRaw, err := rj.ScoreRaw(ctx, fx.tokens)
			if err != nil {
				t.Fatal(err)
			}

			// Embedded hybrid package.
			pkgH, manH, err := Package(ctx, hostH, imwq, ClassHybrid, Options{})
			if err != nil {
				t.Fatal(err)
			}
			rh, err := wasmrt.NewWideHybridEmbedded(ctx, pkgH)
			if err != nil {
				t.Fatal(err)
			}
			defer rh.Close(ctx) //nolint:errcheck
			hRaw, err := rh.ScoreRaw(ctx, fx.tokens)
			if err != nil {
				t.Fatal(err)
			}
			var hBps []int32
			for _, f := range facts {
				b, err := rh.ScoreHybrid(ctx, fx.tokens, f[0], f[1], spec)
				if err != nil {
					t.Fatal(err)
				}
				hBps = append(hBps, b)
			}

			t.Logf("raw codes: golden=%d external=%d embedded-judgment=%d embedded-hybrid=%d", fx.golden, extRaw, jRaw, hRaw)
			t.Logf("hybrid bps (facts %v): external=%v embedded=%v", facts, extBps, hBps)
			t.Logf("judgment pkg %d B (model at %d, %d pages); hybrid pkg %d B", manJ.BundleBytes, manJ.Budget.ModelPtr, manJ.MinPages, manH.BundleBytes)
			if extRaw != fx.golden || jRaw != fx.golden || hRaw != fx.golden {
				t.Fatalf("raw code mismatch: golden=%d external=%d judgment=%d hybrid=%d", fx.golden, extRaw, jRaw, hRaw)
			}
			for i := range facts {
				if extBps[i] != hBps[i] {
					t.Fatalf("hybrid bps mismatch for facts %v: external=%d embedded=%d", facts[i], extBps[i], hBps[i])
				}
			}
			if !rj.Embedded() || !rh.Embedded() {
				t.Fatal("embedded runtimes do not report Embedded()")
			}
			if err := rj.LoadArtifact(ctx, imwq); err == nil {
				t.Fatal("LoadArtifact into a self-contained package succeeded; it must refuse a second model")
			}
		})
	}
}

// Packaging is a pure function: the same inputs give the same bytes (and so the
// same registered hash) every time.
func TestPackageIsDeterministic(t *testing.T) {
	hostJ := mustRead(t, "WIDE_EMBED_HOST_JUDGMENT")
	imwq := fixture(t, "model_fixture.imwq")
	ctx := context.Background()
	a, ma, err := Package(ctx, hostJ, imwq, ClassJudgment, Options{SkipVerify: true})
	if err != nil {
		t.Fatal(err)
	}
	b, mb, err := Package(ctx, hostJ, imwq, ClassJudgment, Options{SkipVerify: true})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) || ma.BundleKeccak256 != mb.BundleKeccak256 {
		t.Fatal("two packagings of the same inputs differ")
	}
}

// Wrong artifacts must error loudly at build or load time — never score.
func TestPackageAndLoadRefusals(t *testing.T) {
	hostJ := mustRead(t, "WIDE_EMBED_HOST_JUDGMENT")
	hostH := mustRead(t, "WIDE_EMBED_HOST_HYBRID")
	interp := mustRead(t, "WIDE_WASM")
	imwq := fixture(t, "model_fixture.imwq")
	ctx := context.Background()

	cases := []struct {
		name    string
		host    []byte
		imwq    []byte
		class   Class
		opt     Options
		wantErr string
	}{
		{"judgment class on hybrid host", hostH, imwq, ClassJudgment, Options{}, "--features embedded"},
		{"hybrid class on judgment host", hostJ, imwq, ClassHybrid, Options{}, "embedded-hybrid"},
		{"shared interpreter as host", interp, imwq, ClassJudgment, Options{}, "not a wide_wasm embed host"},
		{"truncated imwq", hostJ, imwq[:len(imwq)-9], ClassJudgment, Options{}, "truncated"},
		{"padded imwq", hostJ, append(append([]byte{}, imwq...), 0, 0), ClassJudgment, Options{}, "trailing bytes"},
		{"not an imwq", hostJ, []byte("\x00asm\x01\x00\x00\x00 a wasm, not a model"), ClassJudgment, Options{}, "IMWQ magic"},
		// The loud over-budget failure, end to end, without a multi-GB file:
		// the tiny fixture against a 1 MiB ceiling.
		{"over memory budget", hostJ, imwq, ClassJudgment, Options{LimitBytes: 1 << 20}, "exceed the wasm32 memory ceiling"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, _, err := Package(ctx, tc.host, tc.imwq, tc.class, tc.opt)
			if err == nil {
				t.Fatalf("Package succeeded (%d bytes); want error containing %q", len(out), tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not mention %q", err, tc.wantErr)
			}
		})
	}

	// Re-packaging a package is refused (its slot is already filled).
	pkg, _, err := Package(ctx, hostJ, imwq, ClassJudgment, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := Package(ctx, pkg, imwq, ClassJudgment, Options{}); err == nil || !strings.Contains(err.Error(), "already a packaged") {
		t.Fatalf("re-packaging a package: err=%v", err)
	}

	// Load-time refusals.
	if _, err := wasmrt.NewWideEmbedded(ctx, hostJ); err == nil || !strings.Contains(err.Error(), "no embedded model") {
		t.Fatalf("bare embed host loaded as a package: err=%v", err)
	}
	if _, err := wasmrt.NewWideEmbedded(ctx, interp); err == nil || !strings.Contains(err.Error(), "not a self-contained") {
		t.Fatalf("shared interpreter loaded as a package: err=%v", err)
	}
	if _, err := wasmrt.NewWideHybridEmbedded(ctx, pkg); err == nil || !strings.Contains(err.Error(), "JUDGMENT-only") {
		t.Fatalf("judgment package loaded as hybrid: err=%v", err)
	}
	// A corrupted model byte inside a finished package: the slot's sha256
	// catches it at load instead of scoring a different model.
	bad := append([]byte{}, pkg...)
	bad[modelFileOffset(t, bad)+len(imwq)-100] ^= 0xff // a weight byte, inside the model segment
	if _, err := wasmrt.NewWideEmbedded(ctx, bad); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("corrupted package loaded: err=%v", err)
	}
	// A truncated package (download cut short) fails to instantiate.
	if _, err := wasmrt.NewWideEmbedded(ctx, pkg[:len(pkg)-1000]); err == nil {
		t.Fatal("truncated package loaded")
	}
}

// goHeap reports the Go heap after a GC. wazero's linear memory and its retained
// data-segment copies are ordinary Go []byte, so this is the host-side cost.
func goHeap() string {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return fmt.Sprintf("go heap in use %.3f GiB (sys %.3f GiB)", gib(ms.HeapInuse), gib(ms.Sys))
}

// modelFileOffset returns where the model segment's bytes start in a package
// file (the last data segment). Custom sections (name, producers) follow the
// data section, so "near the end of the file" is not inside the model.
func modelFileOffset(t *testing.T, pkg []byte) int {
	t.Helper()
	secs, err := wasmbin.Sections(pkg)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range secs {
		if s.ID != wasmbin.SecData {
			continue
		}
		segs, err := wasmbin.ParseDataSegments(s.Payload(pkg))
		if err != nil {
			t.Fatal(err)
		}
		last := segs[len(segs)-1]
		if string(pkg[s.PayloadStart+last.DataStart:][:4]) != "IMWQ" {
			t.Fatal("last data segment is not the model")
		}
		return s.PayloadStart + last.DataStart
	}
	t.Fatal("no data section")
	return 0
}

// CheckBudget with the real VULNERABILITY_TRIAGE numbers: a 1.2 GB model fits,
// a model ~1.6x that size does not, and the error is explicit about why.
func TestCheckBudget(t *testing.T) {
	const ptr = 17 * pageSize
	const realLen = 1205314279
	b, err := CheckBudget(ptr, realLen, realLen, 512<<20, DefaultMargin, Wasm32Limit)
	if err != nil {
		t.Fatalf("real-size model rejected: %v", err)
	}
	if b.HeadroomBytes <= 0 {
		t.Fatalf("headroom %d", b.HeadroomBytes)
	}
	_, err = CheckBudget(ptr, 1900<<20, 1900<<20, 512<<20, DefaultMargin, Wasm32Limit)
	if err == nil || !strings.Contains(err.Error(), "exceed the wasm32 memory ceiling") {
		t.Fatalf("1.9 GB model accepted or wrong error: %v", err)
	}
}

func TestParseIMWQFixture(t *testing.T) {
	imwq := fixture(t, "rope_model_fixture.imwq")
	info, err := ParseIMWQ(imwq)
	if err != nil {
		t.Fatal(err)
	}
	if info.Ops == 0 || info.Tensors == 0 || info.TensorBytes == 0 || info.MaxCols == 0 || info.Slots == 0 {
		t.Fatalf("implausible info %+v", info)
	}
	if info.ParsedBytes < info.TensorBytes {
		t.Fatal("parsed estimate below tensor payload")
	}
	// The blob count field lies: must be caught, not read past.
	bad := append([]byte{}, imwq...)
	binary.LittleEndian.PutUint32(bad[4:], uint32(len(bad))) // mjson_len past EOF
	if _, err := ParseIMWQ(bad); err == nil {
		t.Fatal("accepted an mjson_len past EOF")
	}
}

// The real model end to end: package VULNERABILITY_TRIAGE (or whatever
// WIDE_IMWQ names) with the hybrid host, then score the same 8 tokens through
// the external path and the embedded path and require identical raw codes and
// hybrid bps. ~2.6 s/token per score plus ~40 s per load — minutes, run it in
// the background:
//
//	WIDE_WASM=... WIDE_EMBED_HOST_HYBRID=... WIDE_IMWQ=local/builds/wide/VULNERABILITY_TRIAGE.imwq \
//	  go test ./pkg/wasm/wideembed -run TestRealArtifactParity -v -timeout 60m
func TestRealArtifactParity(t *testing.T) {
	imwqPath := os.Getenv("WIDE_IMWQ")
	if imwqPath == "" {
		t.Skip("set WIDE_IMWQ (plus WIDE_WASM, WIDE_EMBED_HOST_HYBRID) to run against the real model")
	}
	interp := mustRead(t, "WIDE_WASM")
	hostH := mustRead(t, "WIDE_EMBED_HOST_HYBRID")
	ctx, cancel := context.WithTimeout(context.Background(), 55*time.Minute)
	defer cancel()

	tokens := []int64{50281, 510, 2127, 310, 973, 3542, 15, 50282} // 8 tokens
	spec := hybridSpec(t)

	var extRaw int64
	var extBps int32
	func() {
		imwq, err := os.ReadFile(imwqPath)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		ext, err := wasmrt.NewWideHybrid(ctx, interp)
		if err != nil {
			t.Fatal(err)
		}
		defer ext.Close(context.Background()) //nolint:errcheck
		if err := ext.LoadArtifact(ctx, imwq); err != nil {
			t.Fatal(err)
		}
		t.Logf("external: loaded %d B in %s, memory %d B", len(imwq), time.Since(start), ext.MemoryBytes())
		if extRaw, err = ext.ScoreRaw(ctx, tokens); err != nil {
			t.Fatal(err)
		}
		if extBps, err = ext.ScoreHybrid(ctx, tokens, 100, 100, spec); err != nil {
			t.Fatal(err)
		}
		t.Logf("external: raw=%d bps=%d memory after scoring %d B (%.3f GiB); %s", extRaw, extBps, ext.MemoryBytes(), gib(ext.MemoryBytes()), goHeap())
	}()

	var pkg []byte
	{
		imwq, err := os.ReadFile(imwqPath)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		var man Manifest
		pkg, man, err = Package(ctx, hostH, imwq, ClassHybrid, Options{})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("packaged in %s: %d B, keccak %s, model at %d, min %d pages, budget need %.3f GiB headroom %.3f GiB (parse %.3f, act %.3f)",
			time.Since(start), man.BundleBytes, man.BundleKeccak256, man.Budget.ModelPtr, man.MinPages,
			gib(man.Budget.NeedBytes), float64(man.Budget.HeadroomBytes)/(1<<30), gib(man.Budget.ParsedBytes), gib(man.Budget.ActivationBytes))
		t.Logf("imwq info: ops=%d tensors=%d slots=%d maxCols=%d config=%v", man.IMWQ.Ops, man.IMWQ.Tensors, man.IMWQ.Slots, man.IMWQ.MaxCols, man.IMWQ.Config)
		if out := os.Getenv("WIDE_EMBED_OUT"); out != "" {
			if err := os.WriteFile(out, pkg, 0o644); err != nil {
				t.Fatal(err)
			}
			t.Logf("wrote %s", out)
		}
	}

	start := time.Now()
	emb, err := wasmrt.NewWideHybridEmbedded(ctx, pkg)
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close(context.Background()) //nolint:errcheck
	pkg = nil                             // the module holds its own copy now
	t.Logf("embedded: loaded in %s, memory %d B", time.Since(start), emb.MemoryBytes())
	embRaw, err := emb.ScoreRaw(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	embBps, err := emb.ScoreHybrid(ctx, tokens, 100, 100, spec)
	if err != nil {
		t.Fatal(err)
	}
	mem := emb.MemoryBytes()
	t.Logf("embedded: raw=%d bps=%d memory after scoring %d B (%.3f GiB) -> headroom to 4 GiB %.3f GiB; %s",
		embRaw, embBps, mem, gib(mem), gib(Wasm32Limit-mem), goHeap())
	if embRaw != extRaw || embBps != extBps {
		t.Fatalf("PARITY FAILURE: external raw=%d bps=%d, embedded raw=%d bps=%d", extRaw, extBps, embRaw, embBps)
	}
	t.Logf("PARITY OK: raw=%d bps=%d on both paths", embRaw, embBps)
}
