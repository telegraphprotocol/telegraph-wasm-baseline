package adapterbuild

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/tetratelabs/wazero"

	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wideembed"
)

const DefaultFloatMaxTokens = 512

// BuildFloat appends the ABI-v2 metadata section to an ONNX model already
// baked into a WASI command. It does not transform model weights.
func BuildFloat(ctx context.Context, req Req, logf Logf) (Artifact, Manifest, error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	var man Manifest
	class, err := ResolveClass(req.Intent, req.Class)
	if err != nil {
		return Artifact{}, man, err
	}
	if err := CheckTrainingRecord(req.Intent, req.Training, req.RequireTraining, req.Families); err != nil {
		return Artifact{}, man, err
	}
	if req.BakedWasmPath == "" {
		return Artifact{}, man, fmt.Errorf("adapterbuild: --baked-wasm is required")
	}
	raw, err := os.ReadFile(req.BakedWasmPath)
	if err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: reading baked wasm: %w", err)
	}
	if !wasmbin.IsWasm(raw) {
		return Artifact{}, man, fmt.Errorf("adapterbuild: %s is not a WebAssembly module", req.BakedWasmPath)
	}
	if _, present, err := wasmbin.CustomSection(raw, wasmrt.AdapterMetaSection); err != nil {
		return Artifact{}, man, err
	} else if present {
		return Artifact{}, man, fmt.Errorf("adapterbuild: baked wasm already carries a %s section", wasmrt.AdapterMetaSection)
	}
	exports, err := wasmbin.ExportNames(raw)
	if err != nil {
		return Artifact{}, man, err
	}
	if exports["_start"] != wasmbin.ExportFunc {
		return Artifact{}, man, fmt.Errorf("adapterbuild: baked wasm is not a WASI command: no _start function export")
	}
	if _, reactor := exports["_initialize"]; reactor {
		return Artifact{}, man, fmt.Errorf("adapterbuild: baked wasm exports _initialize: reactor ABI is not accepted by the node")
	}
	for name := range exports {
		if len(name) >= 5 && name[:5] == "imwq_" {
			return Artifact{}, man, fmt.Errorf("adapterbuild: baked wasm exports retired Q32.32 function %s", name)
		}
	}
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = DefaultFloatMaxTokens
	}
	if maxTokens < 1 || maxTokens > DefaultFloatMaxTokens {
		return Artifact{}, man, fmt.Errorf("adapterbuild: --max-tokens %d is outside [1,%d]", maxTokens, DefaultFloatMaxTokens)
	}
	meta := wasmrt.AdapterMeta{
		ABIVersion: wasmrt.FloatAdapterABIVersion,
		Class:      string(class), Tool: "adapterbuild", Runtime: wasmrt.AdapterRuntimeONNXF32,
		MaxTokens: maxTokens, Training: req.Training,
	}
	pkg := wasmbin.AppendCustomSection(append([]byte(nil), raw...), wasmrt.AdapterMetaSection, meta.Encode())
	pkg2 := wasmbin.AppendCustomSection(append([]byte(nil), raw...), wasmrt.AdapterMetaSection, meta.Encode())
	if !bytes.Equal(pkg, pkg2) {
		return Artifact{}, man, fmt.Errorf("adapterbuild: reproducibility gate failed for baked wasm")
	}
	pkg2 = nil
	raw = nil
	debug.FreeOSMemory()

	rt := wazero.NewRuntime(ctx)
	if _, err := rt.CompileModule(ctx, pkg); err != nil {
		rt.Close(ctx) //nolint:errcheck
		return Artifact{}, man, fmt.Errorf("adapterbuild: baked wasm does not compile: %w", err)
	}
	rt.Close(ctx) //nolint:errcheck

	sha := sha256.Sum256(pkg)
	shaHex := hex.EncodeToString(sha[:])
	keccak := "0x" + hex.EncodeToString(crypto.Keccak256(pkg))
	outDir := req.OutDir
	if outDir == "" {
		outDir = "."
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: creating out dir: %w", err)
	}
	finalPath := filepath.Join(outDir, ArtifactName(req.Intent))
	manPath := filepath.Join(outDir, ManifestName(req.Intent))
	tmpPath := finalPath + ".partial"
	defer os.Remove(tmpPath) //nolint:errcheck
	if err := os.WriteFile(tmpPath, pkg, 0o644); err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: writing package: %w", err)
	}
	logf("packaged ABI-v2 float adapter: %d bytes, sha256 %s", len(pkg), shaHex)

	man = Manifest{
		Schema: BuilderSchema, Builder: BuilderName, BuilderVersion: BuilderVersion(), Intent: req.Intent,
		Manifest: wideembed.Manifest{
			Class: class, BundleSHA256: shaHex, BundleKeccak256: keccak, BundleBytes: len(pkg),
			MaxTokens: maxTokens, Verified: true, ABIVersion: wasmrt.FloatAdapterABIVersion, Training: req.Training,
		},
		Runtime: wasmrt.AdapterRuntimeONNXF32, BakedWasmFile: filepath.Base(req.BakedWasmPath),
		Provenance:      req.Provenance,
		Reproducibility: Reproducibility{Builds: 2, IdenticalSHA256: true, SHA256s: []string{shaHex, shaHex}},
		BuiltAt:         time.Now().UTC().Format(time.RFC3339),
	}
	mj, err := jsonMarshalIndent(man)
	if err != nil {
		return Artifact{}, man, err
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: installing %s: %w", finalPath, err)
	}
	if err := writeFileAtomic(manPath, mj); err != nil {
		return Artifact{}, man, err
	}
	return Artifact{Path: finalPath, ManifestPath: manPath, SHA256: shaHex, Keccak256: keccak, Bytes: int64(len(pkg))}, man, nil
}
