// Package adapterbuild is the automated packaging step for wide Q32.32 adapters:
// it turns an intent's converted .imwq into the ONE self-contained .wasm that is
// registered on-chain for that intent, and (optionally) publishes it to the
// S3-compatible object store the node fetches from.
//
//	Build:   catalog class -> pinned embed host (sha256 in hosts.lock.json)
//	         -> wideembed.Package TWICE (byte-identical or fail) -> verified load
//	         -> <out>/<INTENT>.wasm + <INTENT>.wasm.manifest.json
//	Publish: wasm/<INTENT>/<INTENT>.wasm (+ .manifest.json), refusing to overwrite
//	         different bytes at that key unless explicitly told to.
//
// Registration (registerWasm with the returned URL + keccak256) stays manual.
//
// Every failure is loud: a host not in the lock, a class that disagrees with the
// catalog, a deterministic intent, a non-reproducible package, a failed verify,
// a failed upload — each returns an error and nothing is left looking finished.
package adapterbuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"time"

	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wideembed"
)

// BuilderName / BuilderSchema identify this builder in every manifest.
const (
	BuilderName   = "github.com/telegraphprotocol/telegraph-wasm-baseline/cmd/adapterbuild"
	BuilderSchema = 1
	builderMajor  = "adapterbuild/1"
)

// BuilderVersion is "adapterbuild/1" plus the VCS revision the binary was built
// from, when the Go toolchain recorded one ("+dirty" for a modified tree).
func BuilderVersion() string {
	v := builderMajor
	if bi, ok := debug.ReadBuildInfo(); ok {
		var rev, mod string
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				rev = s.Value
			case "vcs.modified":
				mod = s.Value
			}
		}
		if rev != "" {
			if len(rev) > 12 {
				rev = rev[:12]
			}
			v += " " + rev
			if mod == "true" {
				v += "+dirty"
			}
		}
	}
	return v
}

// Provenance is where the model came from. Recorded, not verified — the
// package's keccak256 is the identity; these fields let a reader trace it.
type Provenance struct {
	Adapter         string `json:"adapter,omitempty"`
	BaseModel       string `json:"base_model,omitempty"`
	BackboneVersion string `json:"backbone_version,omitempty"`
	LoRAHash        string `json:"lora_hash,omitempty"`
	DatasetVersion  string `json:"dataset_version,omitempty"`
}

// Req is one build.
type Req struct {
	Intent     string
	IMWQPath   string
	Class      string // optional; must agree with the catalog when set
	Provenance Provenance
	OutDir     string // default "."
	HostsDir   string // default: FindHostsDir("")
	MaxTokens  int    // default wideembed.DefaultMaxTokens
	// LimitBytes overrides the 4 GiB budget ceiling (tests only).
	LimitBytes uint64
	// Training, when set, is embedded in the package's telegraph.adapter
	// section and echoed in the manifest ("training"). Its family must equal
	// Intent. See CheckTrainingRecord.
	Training *wasmrt.TrainingRecord
	// RequireTraining refuses the build unless Training is set and its children
	// are exactly ExpectedChildren(Intent, Families).
	RequireTraining bool
	// Families are the OTHER family intents installed (for the longer-family
	// exclusion of RequireTraining); the builder cannot see a node.
	Families []string
}

// HostsLockRef ties a manifest to the exact lock (and host) used.
type HostsLockRef struct {
	Path        string `json:"path"`
	SHA256      string `json:"sha256"`
	HostFile    string `json:"host_file"`
	Toolchain   string `json:"toolchain"`
	Image       string `json:"image"`
	ImageDigest string `json:"image_digest"`
	RustFlags   string `json:"rustflags"`
}

// Reproducibility records the twice-package gate.
type Reproducibility struct {
	Builds          int      `json:"builds"`
	IdenticalSHA256 bool     `json:"identical_sha256"`
	SHA256s         []string `json:"sha256s"`
}

// Manifest is the extended manifest: everything wideembed records (flattened)
// plus intent, provenance, the lock reference and the builder.
type Manifest struct {
	Schema         int    `json:"schema"`
	Builder        string `json:"builder"`
	BuilderVersion string `json:"builder_version"`
	Intent         string `json:"intent"`
	wideembed.Manifest
	IMWQFile        string          `json:"imwq_file"`
	Provenance      Provenance      `json:"provenance"`
	HostsLock       HostsLockRef    `json:"hosts_lock"`
	Reproducibility Reproducibility `json:"reproducibility"`
	BuiltAt         string          `json:"built_at"`
}

// Artifact is what Build leaves on disk.
type Artifact struct {
	Path         string
	ManifestPath string
	SHA256       string
	Keccak256    string
	Bytes        int64
}

// ArtifactName is the per-intent file/object name: <INTENT>.wasm.
func ArtifactName(intent string) string { return intent + ".wasm" }

// ManifestName is the manifest sidecar name: <INTENT>.wasm.manifest.json.
func ManifestName(intent string) string { return ArtifactName(intent) + ".manifest.json" }

// Logf receives human progress lines (the CLI sends them to stderr).
type Logf func(format string, args ...any)

// Build packages req.IMWQPath for req.Intent into the pinned host for the
// intent's class, twice, and writes the artifact + manifest only if both builds
// are byte-identical and the package verified by loading.
func Build(ctx context.Context, req Req, logf Logf) (Artifact, Manifest, error) {
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
	hostsDir, err := FindHostsDir(req.HostsDir)
	if err != nil {
		return Artifact{}, man, err
	}
	hosts, err := LoadHosts(hostsDir)
	if err != nil {
		return Artifact{}, man, err
	}
	host, entry, err := hosts.Host(class)
	if err != nil {
		return Artifact{}, man, err
	}
	logf("intent %s → class %s; pinned host %s (sha256 %s)", req.Intent, class, entry.File, entry.SHA256)

	if req.IMWQPath == "" {
		return Artifact{}, man, fmt.Errorf("adapterbuild: --imwq is required")
	}
	imwq, err := os.ReadFile(req.IMWQPath)
	if err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: reading .imwq: %w", err)
	}
	logf("read %s (%d bytes)", req.IMWQPath, len(imwq))

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
	defer os.Remove(tmpPath) //nolint:errcheck // no-op after a successful rename

	opt := wideembed.Options{MaxTokens: req.MaxTokens, LimitBytes: req.LimitBytes, Training: req.Training}

	// Build 1: verified (instantiated through the node's own embedded constructor).
	start := time.Now()
	pkg, wm, err := wideembed.Package(ctx, host, imwq, class, opt)
	if err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: package build 1: %w", err)
	}
	logf("build 1: %d bytes, sha256 %s, verified=%v (%s)", wm.BundleBytes, wm.BundleSHA256, wm.Verified, time.Since(start).Round(time.Millisecond))
	if !wm.Verified {
		return Artifact{}, man, fmt.Errorf("adapterbuild: build 1 was not verified by load")
	}
	if err := os.WriteFile(tmpPath, pkg, 0o644); err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: writing package: %w", err)
	}
	pkg = nil //nolint:ineffassign,wastedassign // release before build 2
	debug.FreeOSMemory()

	// Build 2: the reproducibility gate. Same inputs must give the same bytes.
	start = time.Now()
	opt2 := opt
	opt2.SkipVerify = true
	pkg2, wm2, err := wideembed.Package(ctx, host, imwq, class, opt2)
	if err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: package build 2: %w", err)
	}
	sum2 := sha256.Sum256(pkg2)
	pkg2 = nil //nolint:ineffassign,wastedassign
	imwq = nil //nolint:ineffassign,wastedassign
	debug.FreeOSMemory()
	sha2 := hex.EncodeToString(sum2[:])
	logf("build 2: sha256 %s (%s)", sha2, time.Since(start).Round(time.Millisecond))
	if sha2 != wm.BundleSHA256 || wm2.BundleSHA256 != wm.BundleSHA256 || wm2.BundleKeccak256 != wm.BundleKeccak256 {
		return Artifact{}, man, fmt.Errorf("adapterbuild: REPRODUCIBILITY GATE FAILED — two packages of the same "+
			"(host, .imwq, class) differ: build 1 sha256 %s, build 2 sha256 %s; nothing written", wm.BundleSHA256, sha2)
	}

	// Read back what hit the disk, so a short or corrupted write cannot be published.
	diskSHA, diskLen, err := sha256File(tmpPath)
	if err != nil {
		return Artifact{}, man, err
	}
	if diskSHA != wm.BundleSHA256 || diskLen != int64(wm.BundleBytes) {
		return Artifact{}, man, fmt.Errorf("adapterbuild: written package reads back as sha256 %s (%d B), built %s (%d B)",
			diskSHA, diskLen, wm.BundleSHA256, wm.BundleBytes)
	}

	man = Manifest{
		Schema:         BuilderSchema,
		Builder:        BuilderName,
		BuilderVersion: BuilderVersion(),
		Intent:         req.Intent,
		Manifest:       wm,
		IMWQFile:       filepath.Base(req.IMWQPath),
		Provenance:     req.Provenance,
		HostsLock: HostsLockRef{
			Path:        HostsRelDir + "/" + LockFileName,
			SHA256:      hosts.LockSHA256,
			HostFile:    entry.File,
			Toolchain:   hosts.Lock.Toolchain,
			Image:       hosts.Lock.Image,
			ImageDigest: hosts.Lock.ImageDigest,
			RustFlags:   hosts.Lock.RustFlags,
		},
		Reproducibility: Reproducibility{Builds: 2, IdenticalSHA256: true, SHA256s: []string{wm.BundleSHA256, sha2}},
		BuiltAt:         time.Now().UTC().Format(time.RFC3339),
	}
	mj, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: encoding manifest: %w", err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: installing package: %w", err)
	}
	if err := writeFileAtomic(manPath, append(mj, '\n')); err != nil {
		return Artifact{}, man, err
	}
	return Artifact{
		Path: finalPath, ManifestPath: manPath,
		SHA256: wm.BundleSHA256, Keccak256: wm.BundleKeccak256, Bytes: int64(wm.BundleBytes),
	}, man, nil
}

// LoadArtifact re-opens a built artifact for publishing: reads its manifest,
// re-hashes the bytes, and checks them against the manifest and the CURRENT host
// lock. Nothing is re-packaged; this is the publish-only path the training
// orchestrator's publish stage uses after its wasm stage built the package.
func LoadArtifact(path, intent, hostsDir string) (Artifact, Manifest, error) {
	var man Manifest
	if intent == "" {
		return Artifact{}, man, fmt.Errorf("adapterbuild: intent is required")
	}
	if filepath.Base(path) != ArtifactName(intent) {
		return Artifact{}, man, fmt.Errorf("adapterbuild: artifact %s is not named %s — per-intent keys are derived "+
			"from the intent, refusing an ambiguous file name", path, ArtifactName(intent))
	}
	manPath := path + ".manifest.json"
	raw, err := os.ReadFile(manPath)
	if err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: reading manifest: %w", err)
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		return Artifact{}, man, fmt.Errorf("adapterbuild: %s: %w", manPath, err)
	}
	if man.Builder != BuilderName || man.Schema != BuilderSchema {
		return Artifact{}, man, fmt.Errorf("adapterbuild: %s was not written by %s schema %d", manPath, BuilderName, BuilderSchema)
	}
	if man.Intent != intent {
		return Artifact{}, man, fmt.Errorf("adapterbuild: manifest is for %s, not %s", man.Intent, intent)
	}
	if !man.Verified || !man.Reproducibility.IdenticalSHA256 || man.Reproducibility.Builds < 2 {
		return Artifact{}, man, fmt.Errorf("adapterbuild: manifest does not record a verified, twice-reproduced build")
	}
	class, err := ResolveClass(intent, string(man.Class))
	if err != nil {
		return Artifact{}, man, err
	}
	dir, err := FindHostsDir(hostsDir)
	if err != nil {
		return Artifact{}, man, err
	}
	hosts, err := LoadHosts(dir)
	if err != nil {
		return Artifact{}, man, err
	}
	if c, ok := hosts.Lock.Pins(man.HostSHA256); !ok || c != class {
		return Artifact{}, man, fmt.Errorf("adapterbuild: artifact was built on host %s, which is not the pinned %s host "+
			"in the current hosts.lock.json — rebuild it", man.HostSHA256, class)
	}
	sha, n, err := sha256File(path)
	if err != nil {
		return Artifact{}, man, err
	}
	if sha != man.BundleSHA256 || n != int64(man.BundleBytes) {
		return Artifact{}, man, fmt.Errorf("adapterbuild: %s is sha256 %s (%d B) but its manifest says %s (%d B)",
			path, sha, n, man.BundleSHA256, man.BundleBytes)
	}
	// keccak256 needs the bytes; the sha256 match above already binds them to the
	// manifest, whose keccak was computed over the same bytes at build time.
	return Artifact{Path: path, ManifestPath: manPath, SHA256: sha, Keccak256: man.BundleKeccak256, Bytes: n}, man, nil
}

func sha256File(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, fmt.Errorf("adapterbuild: hashing %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func writeFileAtomic(path string, b []byte) error {
	tmp := path + ".partial"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return fmt.Errorf("adapterbuild: writing %s: %w", path, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp) //nolint:errcheck
		return fmt.Errorf("adapterbuild: installing %s: %w", path, err)
	}
	return nil
}
