// Command adapterbuild packages and optionally publishes a Telegraph scoring
// adapter. The supported V2 path takes an ONNX model already baked into a WASI
// command and produces one ABI-v2 float adapter per intent. The --imwq path is
// retained for the retired Q32.32 format.
//
//	go build -o adapterbuild ./cmd/adapterbuild
//
//	# build only: <out>/<INTENT>.wasm + <INTENT>.wasm.manifest.json
//	adapterbuild --intent CODE_REVIEW_GENERAL --baked-wasm runs/CODE_REVIEW_GENERAL/model.baked.wasm \
//	  --out runs/CODE_REVIEW_GENERAL [--class judgment] [--adapter ... --base-model ...]
//
//	# build + publish to wasm/<INTENT>/<INTENT>.wasm (OBJECT_STORE_* env)
//	adapterbuild --intent CODE_REVIEW_GENERAL --baked-wasm ... --out ... --publish [--overwrite]
//
//	# publish an already-built artifact (re-hashed against its manifest + the host lock)
//	adapterbuild --intent CODE_REVIEW_GENERAL --artifact runs/CODE_REVIEW_GENERAL/CODE_REVIEW_GENERAL.wasm --publish
//
//	# a family build carrying its training record (our builds; see scoring-adapter-standard.md §9)
//	adapterbuild --intent WEATHER --imwq ... --out ... --training-record runs/WEATHER/training.json \
//	  --require-training-record [--families WEATHER_FORECAST,...]
//
// --training-record embeds the record in the package's telegraph.adapter
// section and echoes it as manifest.training; its family must equal --intent.
// --require-training-record also refuses unless the record's children are
// exactly the catalog intents --intent serves (same class, "_" prefix, minus
// those a longer family in --families owns) — adapterbuild.ExpectedChildren.
// Nodes never act on the record; it is a check on our own builds only.
//
// The class comes from the intent catalog (an explicit --class must agree);
// deterministic intents are refused. V2 builds validate the supplied WASI
// command and add metadata. Legacy builds use a pinned embed host and are
// packaged twice for byte identity.
//
// stdout carries exactly one JSON object on success:
//
//	{intent, class, keccak256, sha256, bytes, wasm_path, manifest_path, manifest, url?, publish?}
//
// (manifest.training echoes the training record when one was embedded.)
//
// Human logs go to stderr. Any failure — including a publish failure — exits
// non-zero with nothing on stdout. Publishing does not activate an adapter.
//
// Publishing reads OBJECT_STORE_ENDPOINT, OBJECT_STORE_BUCKET,
// OBJECT_STORE_ACCESS_KEY, OBJECT_STORE_SECRET_KEY, OBJECT_STORE_PUBLIC_BASE_URL
// (optional; default <scheme>://<endpoint>/<bucket>), OBJECT_STORE_SECURE and
// OBJECT_STORE_REGION (optional).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// Output is the single JSON object printed on success.
type Output struct {
	Intent       string                      `json:"intent"`
	Class        string                      `json:"class"`
	Keccak256    string                      `json:"keccak256"`
	SHA256       string                      `json:"sha256"`
	Bytes        int64                       `json:"bytes"`
	WasmPath     string                      `json:"wasm_path"`
	ManifestPath string                      `json:"manifest_path"`
	Manifest     adapterbuild.Manifest       `json:"manifest"`
	URL          string                      `json:"url,omitempty"`
	Publish      *adapterbuild.PublishResult `json:"publish,omitempty"`
}

// newStore is swapped by tests for a fake.
var newStore = func(ctx context.Context) (adapterbuild.ObjectStore, error) {
	cfg, err := adapterbuild.StoreConfigFromEnv()
	if err != nil {
		return nil, err
	}
	return adapterbuild.NewMinioStore(ctx, cfg)
}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("adapterbuild", flag.ContinueOnError)
	fs.SetOutput(stderr)
	intent := fs.String("intent", "", "intent ID (required), e.g. VULNERABILITY_TRIAGE")
	imwq := fs.String("imwq", "", "the intent's converted .imwq (build mode)")
	bakedWasm := fs.String("baked-wasm", "", "an ONNX model baked into a WASI command (ABI-v2 float build mode)")
	artifact := fs.String("artifact", "", "an already-built <INTENT>.wasm to publish (publish-only mode; needs --publish)")
	class := fs.String("class", "", "judgment | hybrid (optional; must agree with the intent catalog)")
	out := fs.String("out", ".", "output directory for <INTENT>.wasm + manifest")
	hostsDir := fs.String("hosts-dir", "", "pinned hosts dir (default: $WIDE_HOSTS_DIR, $TELEGRAPH_REPO/"+adapterbuild.HostsRelDir+", or found upward)")
	maxTokens := fs.Int("max-tokens", 0, "sequence length for the 4 GiB activation bound (default: the node's tokenizer max)")
	limitBytes := fs.Uint64("limit-bytes", 0, "override the 4 GiB budget ceiling (tests only)")
	publish := fs.Bool("publish", false, "upload to wasm/<INTENT>/ in the object store (OBJECT_STORE_* env)")
	overwrite := fs.Bool("overwrite", false, "allow replacing DIFFERENT bytes already at the per-intent key")
	trainingRecord := fs.String("training-record", "", "training record JSON to embed in the package's telegraph.adapter section (family must equal --intent)")
	requireTraining := fs.Bool("require-training-record", false, "refuse to build unless --training-record is given and its children are exactly the children --intent serves")
	families := fs.String("families", "", "comma-separated OTHER installed family intents, for --require-training-record's longer-family exclusion")
	var prov adapterbuild.Provenance
	fs.StringVar(&prov.Adapter, "adapter", "", "provenance: adapter (HF path or local dir)")
	fs.StringVar(&prov.BaseModel, "base-model", "", "provenance: base model")
	fs.StringVar(&prov.BackboneVersion, "backbone-version", "", "provenance: backbone version")
	fs.StringVar(&prov.LoRAHash, "lora-hash", "", "provenance: LoRA weights hash")
	fs.StringVar(&prov.DatasetVersion, "dataset-version", "", "provenance: dataset version")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	logf := func(format string, a ...any) { fmt.Fprintf(stderr, "adapterbuild: "+format+"\n", a...) }
	fail := func(err error) int { fmt.Fprintf(stderr, "adapterbuild: FAILED: %v\n", err); return 1 }

	switch {
	case *intent == "":
		return fail(errors.New("--intent is required"))
	case boolCount(*imwq != "", *bakedWasm != "", *artifact != "") != 1:
		return fail(errors.New("exactly one of --imwq, --baked-wasm, or --artifact is required"))
	case *artifact != "" && !*publish:
		return fail(errors.New("--artifact is the publish-only mode; it needs --publish"))
	case *overwrite && !*publish:
		return fail(errors.New("--overwrite only applies with --publish"))
	case *artifact != "" && (*trainingRecord != "" || *requireTraining):
		return fail(errors.New("--training-record / --require-training-record apply to a build (--imwq), not to publishing an existing artifact"))
	}

	ctx := context.Background()
	// Connect to the store BEFORE the ~minute of packaging, so a bad config
	// fails fast rather than after the work.
	var store adapterbuild.ObjectStore
	if *publish {
		s, err := newStore(ctx)
		if err != nil {
			return fail(err)
		}
		store = s
		logf("object store: %s", store.Describe())
	}

	var (
		art adapterbuild.Artifact
		man adapterbuild.Manifest
		err error
	)
	start := time.Now()
	if *imwq != "" || *bakedWasm != "" {
		var rec *wasmrt.TrainingRecord
		if *trainingRecord != "" {
			if rec, err = adapterbuild.LoadTrainingRecord(*trainingRecord); err != nil {
				return fail(err)
			}
		}
		req := adapterbuild.Req{
			Intent: *intent, IMWQPath: *imwq, BakedWasmPath: *bakedWasm, Class: *class, Provenance: prov,
			OutDir: *out, HostsDir: *hostsDir, MaxTokens: *maxTokens, LimitBytes: *limitBytes,
			Training: rec, RequireTraining: *requireTraining, Families: splitList(*families),
		}
		if *bakedWasm != "" {
			art, man, err = adapterbuild.BuildFloat(ctx, req, logf)
		} else {
			art, man, err = adapterbuild.Build(ctx, req, logf)
		}
		if err != nil {
			return fail(err)
		}
		logf("built %s (%d B, keccak256 %s) in %s", art.Path, art.Bytes, art.Keccak256, time.Since(start).Round(time.Millisecond))
	} else {
		art, man, err = adapterbuild.LoadArtifact(*artifact, *intent, *hostsDir)
		if err != nil {
			return fail(err)
		}
		if *class != "" && string(man.Class) != *class {
			return fail(fmt.Errorf("--class %s but the artifact was built as %s", *class, man.Class))
		}
		logf("re-verified %s against its manifest (sha256 %s)", art.Path, art.SHA256)
	}

	o := Output{
		Intent: *intent, Class: string(man.Class), Keccak256: art.Keccak256, SHA256: art.SHA256,
		Bytes: art.Bytes, WasmPath: art.Path, ManifestPath: art.ManifestPath, Manifest: man,
	}
	if store != nil {
		res, err := adapterbuild.Publish(ctx, store, art, *intent, *overwrite, logf)
		if err != nil {
			logf("the built artifact is still on disk at %s", art.Path)
			return fail(err)
		}
		o.URL, o.Publish = res.URL, &res
	}
	b, err := json.Marshal(o)
	if err != nil {
		return fail(err)
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}

func boolCount(vs ...bool) int {
	n := 0
	for _, v := range vs {
		if v {
			n++
		}
	}
	return n
}

// splitList parses a comma-separated flag value, dropping blanks.
func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
