// Command adapterexec serves one Evaluator adapter (a safetensors delta package)
// on the shared Foundation Model over HTTP: the endpoint you register as
// wasmRunnableUrl. A validator scores a new registration there before it
// downloads the package, then re-scores the same texts itself and rejects the
// registration unless every score matches bit for bit, so the endpoint has to
// run the real foundation, the real engine and the registered package. This is
// the float half of the Telegraph node's scripts/local-dev/wasm-remote-exec
// (`--kind float`), on the node's own runtime (pkg/wasm/runtime) and wire
// types (pkg/wasm/remote).
//
//	FOUNDATION_MODEL_PATH=foundation/foundation_lora_fp32.onnx FOUNDATION_MODEL_HASH=<sha256> \
//	FOUNDATION_ENGINE_WASM_PATH=foundation_engine.wasm FOUNDATION_ENGINE_WASM_HASH=<sha256> \
//	  adapterexec --package runs/CODE_REVIEW/CODE_REVIEW.safetensors [--listen 127.0.0.1:8080]
//
// Request (POST, any path):
//
//	{"batch": [{"tokens": [50281, ...]}, ...],
//	 "foundation_hash": F, "adapter_hash": A, "tokenizer_hash": T, "input_root": R}
//
// Reply 200, one result per item, in order, plus the identity this executor
// actually scored with (never an echo of the request):
//
//	{"results": [{"score": <f32 bits as uint32>} | {"error": "..."}, ...],
//	 "foundation_hash": F, "adapter_hash": A, "tokenizer_hash": T, "input_root": R}
//
// A body without a "batch" array gets 400. Put it behind a TLS reverse proxy to
// serve it publicly; validators refuse private and loopback addresses.
package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/crypto"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// maxRequestBytes caps a request body: the largest batch a node sends is 100
// token lists of up to 512 tokens.
const maxRequestBytes = 1 << 20

// serverReadTimeout bounds reading a request; scoring time is not counted.
const serverReadTimeout = 30 * time.Second

type tokensItem struct {
	Tokens []int64 `json:"tokens"`
}

type batchRequest struct {
	Batch          []json.RawMessage `json:"batch"`
	FoundationHash string            `json:"foundation_hash"`
	AdapterHash    string            `json:"adapter_hash"`
	TokenizerHash  string            `json:"tokenizer_hash"`
	InputRoot      string            `json:"input_root"`
}

// scorer is what the handler needs from a loaded adapter.
type scorer interface {
	ScoreTokens(ctx context.Context, ids []int64) (wasmrt.ModelScore, error)
	Identity() wasmrt.ScoreIdentity
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// handler answers the batch request. Scores run one after another under mu:
// the order does not change the bits, and one worker keeps memory bounded.
func handler(rt scorer) http.HandlerFunc {
	var mu sync.Mutex
	id := rt.Identity()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "POST a batch request"})
			return
		}
		var req batchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		if req.Batch == nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": `this executor implements the batch request only: POST {"batch": [...]}`})
			return
		}
		results := make([]map[string]any, len(req.Batch))
		tokens := make([][]int64, len(req.Batch))
		start := time.Now()
		mu.Lock()
		for i, raw := range req.Batch {
			var it tokensItem
			if err := json.Unmarshal(raw, &it); err != nil {
				results[i] = map[string]any{"error": "bad item: " + err.Error()}
				continue
			}
			tokens[i] = it.Tokens
			s, err := rt.ScoreTokens(r.Context(), it.Tokens)
			if err != nil {
				log.Printf("item %d: tokens_len=%d -> error: %v", i, len(it.Tokens), err)
				results[i] = map[string]any{"error": err.Error()}
				continue
			}
			log.Printf("item %d: tokens_len=%d -> score=%d (%s)", i, len(it.Tokens), s.Bits(), s.Hex())
			results[i] = map[string]any{"score": s.Bits()}
		}
		mu.Unlock()
		body := map[string]any{"results": results}
		if req.FoundationHash != "" || req.AdapterHash != "" || req.TokenizerHash != "" {
			body["foundation_hash"], body["adapter_hash"], body["tokenizer_hash"] = id.FoundationHash, id.AdapterHash, id.TokenizerHash
			if req.FoundationHash != id.FoundationHash || req.AdapterHash != id.AdapterHash || req.TokenizerHash != id.TokenizerHash {
				log.Printf("WARNING: the node asked for foundation %q adapter %q tokenizer %q; this executor runs %q %q %q (the node will reject the registration)",
					req.FoundationHash, req.AdapterHash, req.TokenizerHash, id.FoundationHash, id.AdapterHash, id.TokenizerHash)
			}
		}
		if req.InputRoot != "" {
			body["input_root"] = remote.BatchInputRoot(tokens)
		}
		log.Printf("batch of %d scored in %s", len(req.Batch), time.Since(start).Round(time.Millisecond))
		writeJSON(w, http.StatusOK, body)
	}
}

// load reads the package and the foundation named by the FOUNDATION_* env and
// loads the package on it with the node's runtime.
func load(ctx context.Context, pkgPath string) (*wasmrt.LoRARuntime, error) {
	pkg, err := os.ReadFile(pkgPath)
	if err != nil {
		return nil, err
	}
	cfg, err := wasmrt.FoundationConfigFromEnv("")
	if err != nil {
		return nil, err
	}
	if !cfg.Enabled() {
		return nil, fmt.Errorf("set FOUNDATION_MODEL_PATH/HASH and FOUNDATION_ENGINE_WASM_PATH/HASH to the foundation validators run")
	}
	h, _, err := wasmrt.ParseLoRAHeader(pkg)
	if err != nil {
		return nil, fmt.Errorf("%s is not a safetensors delta package: %w", pkgPath, err)
	}
	// The node tokenises; this executor has no tokenizer, so it takes the
	// package's digest. A package built for another tokenizer echoes it and is
	// rejected by the node.
	cfg.TokenizerSH = h.TokenizerHash
	cfg.PoolSize = 1
	f, err := wasmrt.LoadFoundation(ctx, cfg)
	if err != nil {
		return nil, err
	}
	rt, err := wasmrt.NewLoRARuntime(ctx, f, pkg, "")
	if err != nil {
		return nil, err
	}
	log.Printf("loaded %s: intent %s, class %s, keccak256 0x%s, foundation %s, tokenizer %s",
		pkgPath, h.Intent, h.Class, hex.EncodeToString(crypto.Keccak256(pkg)), f.Hash(), h.TokenizerHash)
	return rt, nil
}

func main() {
	pkgPath := flag.String("package", "", "the safetensors delta package you registered (required)")
	listen := flag.String("listen", "127.0.0.1:8080", "listen address")
	flag.Parse()
	if *pkgPath == "" {
		log.Fatal("--package is required")
	}
	rt, err := load(context.Background(), *pkgPath)
	if err != nil {
		log.Fatalf("adapterexec: %v", err)
	}
	srv := &http.Server{
		Addr:        *listen,
		Handler:     http.MaxBytesHandler(handler(rt), maxRequestBytes),
		ReadTimeout: serverReadTimeout,
	}
	log.Printf("adapterexec listening on %s", *listen)
	log.Fatal(srv.ListenAndServe())
}
