// Command adapterrunner exposes one packaged model adapter through the public
// Telegraph runnable-endpoint protocol. It is a reference deployment tool for
// an adapter author; it is not part of a validator or a Telegraph node.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	defaultListen  = ":8080"
	defaultTimeout = 110 * time.Second
	maxBatchItems  = 100
	maxTokens      = 512
	maxBodyBytes   = 1 << 20
)

var hexScore = regexp.MustCompile(`(?m)^HEX\s*:\s*0x([0-9a-fA-F]{8})\b`)

type batchRequest struct {
	Batch []tokenRequest `json:"batch"`
}

type tokenRequest struct {
	Tokens []int64 `json:"tokens"`
}

type batchResponse struct {
	Results []result `json:"results"`
}

type result struct {
	Score *uint32 `json:"score,omitempty"`
	Error string  `json:"error,omitempty"`
}

type runner struct {
	wasmPath string
	wasmtime string
	timeout  time.Duration
	sem      chan struct{}
}

func main() {
	wasmPath := os.Getenv("ADAPTER_WASM")
	if wasmPath == "" {
		log.Fatal("ADAPTER_WASM is required")
	}
	if info, err := os.Stat(wasmPath); err != nil || info.IsDir() {
		log.Fatalf("ADAPTER_WASM is not a readable file: %v", err)
	}

	concurrency, err := positiveIntEnv("ADAPTER_RUNNER_MAX_CONCURRENT", 1)
	if err != nil {
		log.Fatal(err)
	}
	timeout, err := durationEnv("ADAPTER_RUNNER_TIMEOUT", defaultTimeout)
	if err != nil {
		log.Fatal(err)
	}
	wasmtime := os.Getenv("WASMTIME_BIN")
	if wasmtime == "" {
		wasmtime = "wasmtime"
	}
	if _, err := exec.LookPath(wasmtime); err != nil {
		log.Fatalf("WASMTIME_BIN %q is not executable: %v", wasmtime, err)
	}

	r := &runner{wasmPath: wasmPath, wasmtime: wasmtime, timeout: timeout, sem: make(chan struct{}, concurrency)}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", r.health)
	mux.HandleFunc("/runnable", r.run)

	addr := os.Getenv("ADAPTER_RUNNER_LISTEN")
	if addr == "" {
		addr = defaultListen
	}
	log.Printf("adapterrunner listening on %s; adapter=%s; max_concurrent=%d; timeout=%s", addr, wasmPath, concurrency, timeout)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func (r *runner) health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"ok":true}`))
}

func (r *runner) run(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	defer req.Body.Close()

	var in batchRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, req.Body, maxBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		http.Error(w, "invalid batch request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(in.Batch) == 0 || len(in.Batch) > maxBatchItems {
		http.Error(w, fmt.Sprintf("batch must contain 1 through %d items", maxBatchItems), http.StatusBadRequest)
		return
	}

	out := batchResponse{Results: make([]result, len(in.Batch))}
	for i, item := range in.Batch {
		if len(item.Tokens) == 0 || len(item.Tokens) > maxTokens {
			out.Results[i].Error = fmt.Sprintf("tokens must contain 1 through %d values", maxTokens)
			continue
		}
		bits, err := r.score(req.Context(), item.Tokens)
		if err != nil {
			out.Results[i].Error = err.Error()
			continue
		}
		out.Results[i].Score = &bits
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(out); err != nil {
		log.Printf("write response: %v", err)
	}
}

func (r *runner) score(parent context.Context, tokens []int64) (uint32, error) {
	select {
	case r.sem <- struct{}{}:
		defer func() { <-r.sem }()
	case <-parent.Done():
		return 0, parent.Err()
	}

	dir, err := os.MkdirTemp("", "telegraph-adapterrunner-")
	if err != nil {
		return 0, fmt.Errorf("create input directory: %w", err)
	}
	defer os.RemoveAll(dir)

	mask := make([]int64, len(tokens))
	for i := range mask {
		mask[i] = 1
	}
	b, err := json.Marshal(struct {
		InputIDs      []int64 `json:"input_ids"`
		AttentionMask []int64 `json:"attention_mask"`
	}{tokens, mask})
	if err != nil {
		return 0, fmt.Errorf("encode adapter input: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "input.json"), b, 0o600); err != nil {
		return 0, fmt.Errorf("write adapter input: %w", err)
	}

	ctx, cancel := context.WithTimeout(parent, r.timeout)
	defer cancel()
	// The module receives only /in/input.json. It has no network capability;
	// wasmtime exposes no host directory except this temporary input directory.
	cmd := exec.CommandContext(ctx, r.wasmtime, "run", "--dir", dir+"::/in", r.wasmPath, "/in/input.json", "1")
	cmd.Env = []string{}
	output, err := cmd.CombinedOutput()
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return 0, fmt.Errorf("adapter timed out after %s", r.timeout)
		}
		return 0, fmt.Errorf("adapter failed: %s", strings.TrimSpace(string(output)))
	}
	match := hexScore.FindStringSubmatch(string(output))
	if len(match) != 2 {
		return 0, fmt.Errorf("adapter returned no FP32 score")
	}
	n, err := strconv.ParseUint(match[1], 16, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid adapter score: %w", err)
	}
	bits := uint32(n)
	if !isFinite(bits) {
		return 0, fmt.Errorf("adapter returned a non-finite score")
	}
	return bits, nil
}

func isFinite(bits uint32) bool {
	return !math.IsNaN(float64(math.Float32frombits(bits))) && !math.IsInf(float64(math.Float32frombits(bits)), 0)
}

func positiveIntEnv(name string, fallback int) (int, error) {
	v := os.Getenv(name)
	if v == "" {
		return fallback, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return 0, fmt.Errorf("%s must be a positive integer", name)
	}
	return n, nil
}

func durationEnv(name string, fallback time.Duration) (time.Duration, error) {
	v := os.Getenv(name)
	if v == "" {
		return fallback, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("%s must be a positive Go duration", name)
	}
	return d, nil
}
