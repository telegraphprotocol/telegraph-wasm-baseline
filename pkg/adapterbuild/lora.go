package adapterbuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum/crypto"

	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// The `lora` mode: check and describe a safetensors delta package (an Evaluator
// adapter for the shared Foundation Model, written by
// float_runtime/export_adapter.py) with the node's own parser, and print the
// keccak256 that is registered. Nothing is rewritten: the package IS the
// registrable artifact, so the hash the author registers is the hash of the
// file export_adapter wrote.

// FoundationFile is foundation.json as export_foundation.py writes it.
type FoundationFile struct {
	FoundationVersion string            `json:"foundation_version"`
	FoundationHash    string            `json:"foundation_hash"`
	TokenizerHash     string            `json:"tokenizer_hash"`
	RMax              int               `json:"r_max"`
	NumLayers         int               `json:"num_layers"`
	HiddenSize        int               `json:"hidden_size"`
	Targets           map[string][2]int `json:"targets"`
}

// Spec is the foundation's LoRA layout in the node's form.
func (f FoundationFile) Spec() wasmrt.FoundationSpec {
	return wasmrt.FoundationSpec{RMax: f.RMax, HiddenSize: f.HiddenSize, NumLayers: f.NumLayers, Targets: f.Targets}
}

// LoadFoundationFile reads foundation.json.
func LoadFoundationFile(path string) (FoundationFile, error) {
	var f FoundationFile
	b, err := os.ReadFile(path)
	if err != nil {
		return f, fmt.Errorf("adapterbuild: foundation: %w", err)
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return f, fmt.Errorf("adapterbuild: foundation %s: %w", path, err)
	}
	if f.FoundationHash == "" || f.TokenizerHash == "" {
		return f, fmt.Errorf("adapterbuild: foundation %s names no foundation_hash / tokenizer_hash", path)
	}
	if err := f.Spec().Check(); err != nil {
		return f, fmt.Errorf("adapterbuild: foundation %s: %w", path, err)
	}
	return f, nil
}

// LoRAReq is one `lora` check.
type LoRAReq struct {
	Intent          string
	PackagePath     string
	Class           string // optional; must agree with the catalog when set
	FoundationPath  string // foundation.json (required)
	OutDir          string
	Training        *wasmrt.TrainingRecord // optional: must equal the package's header record
	RequireTraining bool
	Families        []string
	// ModelPath / EnginePath / EngineHash: when set, the package is loaded on
	// the foundation with the node's runtime and scored twice on the default
	// input (identical bits), like the legacy build's verification.
	ModelPath  string
	EnginePath string
	EngineHash string
	// Input is the token sequence scored (default DefaultVerifyInput).
	Input []int64
}

// LoRAManifest describes a checked package.
type LoRAManifest struct {
	Schema         int                    `json:"schema"`
	Builder        string                 `json:"builder"`
	BuilderVersion string                 `json:"builder_version"`
	Intent         string                 `json:"intent"`
	Class          string                 `json:"class"`
	ABI            string                 `json:"abi"`
	Foundation     FoundationRef          `json:"foundation"`
	TokenizerHash  string                 `json:"tokenizer_hash"`
	Rank           int                    `json:"rank"`
	Alpha          float64                `json:"alpha"`
	TargetModules  []string               `json:"target_modules"`
	MaxTokens      int                    `json:"max_tokens"`
	Tensors        int                    `json:"tensors"`
	DeltaBytes     int64                  `json:"delta_bytes"`
	Package        FileRef                `json:"package"`
	Keccak256      string                 `json:"keccak256"`
	Training       *wasmrt.TrainingRecord `json:"training,omitempty"`
	Verification   LoRAVerification       `json:"verification"`
	BuiltAt        string                 `json:"built_at"`
}

// FoundationRef names the foundation a package was built for.
type FoundationRef struct {
	Hash    string `json:"hash"`
	Version string `json:"version"`
}

// LoRAVerification records what was checked.
type LoRAVerification struct {
	Stage1    bool   `json:"stage1"`
	Loaded    bool   `json:"loaded"` // on the foundation, through the node's runtime
	ScoreHex  string `json:"score_hex,omitempty"`
	Repeats   int    `json:"repeats,omitempty"`
	Seconds   string `json:"seconds,omitempty"`
	InputName string `json:"input,omitempty"`
}

// LoRAArtifact is the checked package on disk.
type LoRAArtifact struct {
	Path         string
	ManifestPath string
	SHA256       string
	Keccak256    string
	Bytes        int64
}

// CheckLoRA runs every check the node runs at Stage 1 on req.PackagePath
// against req.FoundationPath, plus the kit's own (class from the catalog,
// intent equals the header's, training record vs the children the family
// serves), optionally loads and scores it, and writes <out>/<INTENT>.safetensors.manifest.json.
func CheckLoRA(ctx context.Context, req LoRAReq, logf Logf) (LoRAArtifact, LoRAManifest, error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	var man LoRAManifest
	if req.Intent == "" || req.PackagePath == "" || req.FoundationPath == "" {
		return LoRAArtifact{}, man, errors.New("adapterbuild lora: --intent, --package and --foundation are required")
	}
	class, err := ResolveClass(req.Intent, req.Class)
	if err != nil {
		return LoRAArtifact{}, man, err
	}
	fnd, err := LoadFoundationFile(req.FoundationPath)
	if err != nil {
		return LoRAArtifact{}, man, err
	}
	pkg, err := os.ReadFile(req.PackagePath)
	if err != nil {
		return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: %w", err)
	}
	if !wasmrt.IsLoRAPackage(pkg) {
		return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: %s is not a safetensors delta package (did you mean the legacy --wasm build?)", req.PackagePath)
	}
	want := wasmrt.LoRAExpect{FoundationHash: fnd.FoundationHash, TokenizerHash: fnd.TokenizerHash, Class: class, Spec: fnd.Spec()}
	a, err := wasmrt.ParseLoRAAdapter(pkg, want)
	if err != nil {
		return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: Stage 1: %w", err)
	}
	h := a.Header
	if h.Intent != req.Intent {
		return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: the package names intent %q, --intent is %q", h.Intent, req.Intent)
	}
	if req.Training != nil {
		got, _ := json.Marshal(h.Training)
		exp, _ := json.Marshal(req.Training)
		if string(got) != string(exp) {
			return LoRAArtifact{}, man, errors.New("adapterbuild lora: --training-record differs from the record embedded in the package (export_adapter --training-record)")
		}
	}
	if err := CheckTrainingRecord(req.Intent, h.Training, req.RequireTraining, req.Families); err != nil {
		return LoRAArtifact{}, man, err
	}
	logf("intent %s → class %s; foundation %s (%s); rank %d alpha %v; %d tensors, %d delta bytes",
		req.Intent, class, short(fnd.FoundationHash), fnd.FoundationVersion, h.Rank, h.Alpha, len(h.Tensors), a.DeltaBytes)

	ver := LoRAVerification{Stage1: true}
	if req.ModelPath != "" || req.EnginePath != "" {
		if req.ModelPath == "" || req.EnginePath == "" {
			return LoRAArtifact{}, man, errors.New("adapterbuild lora: --foundation-onnx and --engine go together")
		}
		eh := req.EngineHash
		if eh == "" {
			s, _, err := wasmrt.SHA256File(req.EnginePath)
			if err != nil {
				return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: %w", err)
			}
			eh = s
		}
		start := time.Now()
		f, err := wasmrt.LoadFoundation(ctx, wasmrt.FoundationConfig{ModelPath: req.ModelPath, ModelHash: fnd.FoundationHash,
			Version: fnd.FoundationVersion, EnginePath: req.EnginePath, EngineHash: eh, PoolSize: 1, TokenizerSH: fnd.TokenizerHash})
		if err != nil {
			return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: %w", err)
		}
		defer f.Close(context.Background())
		rt, err := wasmrt.NewLoRARuntime(ctx, f, pkg, class)
		if err != nil {
			return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: load: %w", err)
		}
		defer rt.Close(context.Background())
		ids := req.Input
		if len(ids) == 0 {
			ids = DefaultVerifyInput
		}
		var first wasmrt.ModelScore
		for i := 0; i < 2; i++ {
			s, err := rt.ScoreTokens(ctx, ids)
			if err != nil {
				return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: score %d: %w", i+1, err)
			}
			if i == 0 {
				first = s
			} else if s.Bits() != first.Bits() {
				return LoRAArtifact{}, man, fmt.Errorf("adapterbuild lora: not deterministic: %s then %s", first, s)
			}
		}
		ver = LoRAVerification{Stage1: true, Loaded: true, ScoreHex: first.Hex(), Repeats: 2,
			Seconds: fmt.Sprintf("%.3f", time.Since(start).Seconds()), InputName: "default"}
		logf("loaded on the foundation and scored twice: %s", first)
	}

	sum := sha256.Sum256(pkg)
	art := LoRAArtifact{Path: req.PackagePath, SHA256: hex.EncodeToString(sum[:]),
		Keccak256: "0x" + hex.EncodeToString(crypto.Keccak256(pkg)), Bytes: int64(len(pkg))}
	outDir := req.OutDir
	if outDir == "" {
		outDir = filepath.Dir(req.PackagePath)
	}
	man = LoRAManifest{
		Schema: 1, Builder: "github.com/telegraphprotocol/telegraph-wasm-baseline/cmd/adapterbuild", BuilderVersion: BuilderVersion(),
		Intent: req.Intent, Class: class, ABI: h.ABI, Foundation: FoundationRef{Hash: h.FoundationHash, Version: h.FoundationVersion},
		TokenizerHash: h.TokenizerHash, Rank: h.Rank, Alpha: h.Alpha, TargetModules: h.TargetModules, MaxTokens: h.MaxTokens,
		Tensors: len(h.Tensors), DeltaBytes: a.DeltaBytes,
		Package: FileRef{File: filepath.Base(req.PackagePath), SHA256: art.SHA256, Bytes: art.Bytes},
		Keccak256: art.Keccak256, Training: h.Training, Verification: ver, BuiltAt: time.Now().UTC().Format(time.RFC3339),
	}
	mb, _ := json.MarshalIndent(man, "", "  ")
	art.ManifestPath = filepath.Join(outDir, filepath.Base(req.PackagePath)+".manifest.json")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return art, man, fmt.Errorf("adapterbuild lora: %w", err)
	}
	if err := writeFileAtomic(art.ManifestPath, append(mb, '\n')); err != nil {
		return art, man, err
	}
	return art, man, nil
}

func short(h string) string {
	if len(h) > 12 {
		return h[:12] + "…"
	}
	return strings.TrimSpace(h)
}
