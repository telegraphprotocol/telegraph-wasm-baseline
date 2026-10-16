// Command wideembed builds ONE self-contained wide Q32.32 scoring package per
// intent: a pinned embed host (wide_wasm built with --features embedded or
// embedded-hybrid) with that intent's .imwq injected as a resident data
// segment. The output .wasm is what gets registered on-chain — its keccak256
// (printed, and written to the manifest) covers the model and the interpreter
// together.
//
//	go run ./cmd/wideembed \
//	  --host  wide_runtime/wide_wasm/target/.../wide_embed_hybrid.wasm \
//	  --imwq  local/builds/wide/VULNERABILITY_TRIAGE.imwq \
//	  --class hybrid \
//	  --out   local/builds/wide/VULNERABILITY_TRIAGE.wasm
//
// This is the DEV packager: it accepts any embed host. What gets registered
// should come from cmd/adapterbuild, which only packages into the committed
// pinned hosts (hosts.lock.json), derives the class from the intent catalog,
// and builds twice. wideembed warns when --host is not a pinned host.
//
// The package is verified by loading it through the node's own embedded
// constructor before it is written; any mismatch (wrong host for the class, a
// truncated/malformed .imwq, a model that would overrun the wasm32 4 GiB
// memory) exits non-zero without writing anything.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wideembed"
)

func main() {
	host := flag.String("host", "", "embed host .wasm (--features embedded for judgment, embedded-hybrid for hybrid)")
	imwq := flag.String("imwq", "", "the intent's .imwq model")
	class := flag.String("class", "", "judgment | hybrid (must match the host build)")
	out := flag.String("out", "", "output package .wasm")
	manifest := flag.String("manifest", "", "manifest JSON path (default <out>.manifest.json)")
	limit := flag.Uint64("limit-bytes", wideembed.Wasm32Limit, "linear-memory ceiling for the budget check")
	margin := flag.Uint64("margin-bytes", wideembed.DefaultMargin, "headroom reserved below the ceiling")
	maxTokens := flag.Int("max-tokens", wideembed.DefaultMaxTokens, "longest sequence the node scores (activation bound)")
	flag.Parse()

	if *host == "" || *imwq == "" || *class == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *manifest == "" {
		*manifest = *out + ".manifest.json"
	}

	hostBytes, err := os.ReadFile(*host)
	fail(err, "read host")
	warnIfUnpinned(hostBytes, wideembed.Class(*class))
	model, err := os.ReadFile(*imwq)
	fail(err, "read imwq")

	start := time.Now()
	pkg, man, err := wideembed.Package(context.Background(), hostBytes, model, wideembed.Class(*class), wideembed.Options{
		LimitBytes: *limit, MarginBytes: *margin, MaxTokens: *maxTokens,
	})
	if err != nil {
		if man.Budget.NeedBytes != 0 {
			b, _ := json.MarshalIndent(man.Budget, "", "  ")
			fmt.Fprintf(os.Stderr, "budget: %s\n", b)
		}
		fail(err, "package")
	}
	model = nil //nolint:ineffassign // release before the write

	fail(os.WriteFile(*out, pkg, 0o644), "write package")
	mj, err := json.MarshalIndent(man, "", "  ")
	fail(err, "encode manifest")
	fail(os.WriteFile(*manifest, mj, 0o644), "write manifest")

	fmt.Printf("package:      %s (%d bytes, %.3f GiB)\n", *out, man.BundleBytes, float64(man.BundleBytes)/(1<<30))
	fmt.Printf("class:        %s\n", man.Class)
	fmt.Printf("keccak256:    %s   (registerWasm wasmHash)\n", man.BundleKeccak256)
	fmt.Printf("sha256:       %s\n", man.BundleSHA256)
	fmt.Printf("host sha256:  %s\n", man.HostSHA256)
	fmt.Printf("imwq sha256:  %s\n", man.IMWQSHA256)
	fmt.Printf("model:        at %d, %d bytes; memory min %d pages (host %d)\n", man.Budget.ModelPtr, man.Budget.ModelLen, man.MinPages, man.HostMinPages)
	fmt.Printf("budget:       need %.3f GiB of %.3f GiB (headroom %.3f GiB) at %d tokens\n",
		float64(man.Budget.NeedBytes)/(1<<30), float64(man.Budget.LimitBytes)/(1<<30),
		float64(man.Budget.HeadroomBytes)/(1<<30), man.MaxTokens)
	fmt.Printf("verified:     %v (linear memory after load %d bytes)\n", man.Verified, man.LoadedMemoryBytes)
	fmt.Printf("manifest:     %s\n", *manifest)
	fmt.Printf("elapsed:      %s\n", time.Since(start).Round(time.Millisecond))
}

func fail(err error, what string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "wideembed: %s: %v\n", what, err)
		os.Exit(1)
	}
}

// warnIfUnpinned prints a warning (never fails) when the host is not the pinned
// host for the class in hosts.lock.json — such a package's hash cannot be
// reproduced from the committed hosts, so it should not be registered.
func warnIfUnpinned(host []byte, class wideembed.Class) {
	sum := sha256.Sum256(host)
	sha := hex.EncodeToString(sum[:])
	dir, err := adapterbuild.FindHostsDir("")
	if err != nil {
		fmt.Fprintf(os.Stderr, "wideembed: WARNING: cannot locate hosts.lock.json (%v); host sha256 %s is unchecked — "+
			"use cmd/adapterbuild for anything you register\n", err, sha)
		return
	}
	h, err := adapterbuild.LoadHosts(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "wideembed: WARNING: %v; host sha256 %s is unchecked\n", err, sha)
		return
	}
	if c, ok := h.Lock.Pins(sha); !ok || c != class {
		fmt.Fprintf(os.Stderr, "wideembed: WARNING: host sha256 %s is NOT the pinned %s host in %s — this package's "+
			"hash is not reproducible from the committed hosts; do not register it (use cmd/adapterbuild)\n", sha, class, dir)
	}
}
