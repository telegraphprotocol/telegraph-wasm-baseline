package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/ethereum/go-ethereum/crypto"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/adapterbuild"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/remote"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

const fixtures = "../../float_runtime/testdata"

// tinyExecutor loads the stub foundation fixture and a tiny delta behind the
// handler, and returns the server and the identity a node would send.
func tinyExecutor(t *testing.T) (*httptest.Server, remote.BatchIdentity) {
	t.Helper()
	fnd, err := adapterbuild.LoadFoundationFile(filepath.Join(fixtures, "tiny_foundation.json"))
	if err != nil {
		t.Skipf("fixture: %v", err)
	}
	engine := filepath.Join(fixtures, "foundation_engine.wasm")
	eh, _, err := wasmrt.SHA256File(engine)
	if err != nil {
		t.Skipf("fixture: %v", err)
	}
	t.Setenv("FOUNDATION_MODEL_PATH", filepath.Join(fixtures, "tiny_foundation.onnx"))
	t.Setenv("FOUNDATION_MODEL_HASH", fnd.FoundationHash)
	t.Setenv("FOUNDATION_ENGINE_WASM_PATH", engine)
	t.Setenv("FOUNDATION_ENGINE_WASM_HASH", eh)
	pkgPath := filepath.Join(fixtures, "tiny_judgment.safetensors")
	pkg, err := os.ReadFile(pkgPath)
	if err != nil {
		t.Skipf("fixture: %v", err)
	}
	rt, err := load(context.Background(), pkgPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { rt.Close(context.Background()); rt.Foundation().Close(context.Background()) })
	srv := httptest.NewServer(handler(rt))
	t.Cleanup(srv.Close)
	return srv, remote.BatchIdentity{FoundationHash: fnd.FoundationHash,
		AdapterHash: hex.EncodeToString(crypto.Keccak256(pkg)), TokenizerHash: fnd.TokenizerHash}
}

// The node's own client gets one bit-exact score per item and the identity echo.
func TestExecutor_NodeClient(t *testing.T) {
	srv, ident := tinyExecutor(t)
	batch := [][]int64{{1, 5, 9, 2}, {1, 7, 2}, {1, 5, 9, 2}}
	ident.InputRoot = remote.BatchInputRoot(batch)
	exec := &remote.RemoteExecutor{URL: srv.URL}
	res, echoed, err := exec.ScoreBatchWithIdentity(context.Background(), batch, &ident)
	if err != nil {
		t.Fatal(err)
	}
	if !echoed || len(res) != len(batch) {
		t.Fatalf("echoed %v, %d results", echoed, len(res))
	}
	if res[0].Error != "" || res[0].Score.Bits() != res[2].Score.Bits() {
		t.Fatalf("same tokens, different results: %+v %+v", res[0], res[2])
	}
}

// A node that names another adapter gets this executor's real identity back,
// which its client rejects.
func TestExecutor_WrongAdapterIsRejectedByTheNode(t *testing.T) {
	srv, ident := tinyExecutor(t)
	ident.AdapterHash = "11" + ident.AdapterHash[2:]
	exec := &remote.RemoteExecutor{URL: srv.URL}
	_, _, err := exec.ScoreBatchWithIdentity(context.Background(), [][]int64{{1, 5, 2}}, &ident)
	var rej *remote.RemoteRejectedError
	if !errors.As(err, &rej) {
		t.Fatalf("want an identity echo mismatch, got %v", err)
	}
}

// A body without a batch array is refused with 400.
func TestExecutor_NoBatch(t *testing.T) {
	srv, _ := tinyExecutor(t)
	resp, err := http.Post(srv.URL, "application/json", bytes.NewReader([]byte(`{"tokens":[1,2]}`)))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d", resp.StatusCode)
	}
}
