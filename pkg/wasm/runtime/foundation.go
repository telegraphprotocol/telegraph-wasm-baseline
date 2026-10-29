package runtime

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
	"github.com/tetratelabs/wazero/sys"
)

// The shared Foundation Model: ONE immutable ONNX graph (base weights as
// initializers, LoRA branches and reward head as graph inputs) run by ONE
// engine module (a WASI reactor built from the kit's float_runtime/engine),
// loaded once per validator at startup and shared by every Evaluator adapter
// and every score. An adapter is a safetensors delta package (lora_adapter.go) fed
// into the graph at score time; the foundation is never mutated.
//
// Engine ABI (FoundationABIVersion = 2; every function takes/returns i32
// unless noted):
//
//	telegraph_abi() -> 2
//	tg_alloc(len) -> ptr / tg_free(ptr, len)   host copies in and out
//	load_foundation(ptr, len) -> 0 | <0        once per instance
//	foundation_spec() -> len                   JSON FoundationSpec in the message buffer
//	load_adapter(ptr, len) -> handle>0 | <0    the safetensors delta package, re-validated in the guest
//	score(handle, ids_ptr, n) -> i64           f32 bits in 0..2^32, or <0
//	unload_adapter(handle) -> 0 | <0
//	message_ptr() / message_len()              the last error text or the spec JSON
//
// Host: a fixed WASI host (no FS, no
// args, fixed clocks and random source, capped stdio), the same memory cap,
// and the per-score deadline (ADAPTER_SCORE_TIMEOUT). Scores are serialised
// per instance; FOUNDATION_POOL_SIZE instances (default 1) exist per process.
const (
	FoundationABIVersion = 2

	// Env: the Foundation Model files and their pinned hashes. Both hashes
	// are required when the path is set; a mismatch refuses to start.
	FoundationModelPathEnv    = "FOUNDATION_MODEL_PATH"
	FoundationModelHashEnv    = "FOUNDATION_MODEL_HASH"
	FoundationModelVersionEnv = "FOUNDATION_MODEL_VERSION"
	FoundationEngineWasmEnv   = "FOUNDATION_ENGINE_WASM_PATH"
	FoundationEngineHashEnv   = "FOUNDATION_ENGINE_WASM_HASH"
	// FoundationPoolSizeEnv is how many engine instances (workers) hold the
	// foundation. Every instance is a FULL copy of the graph in its own
	// linear memory (wazero gives each instance its own memory; there is no
	// read-only sharing of one module memory between instances), so RAM
	// grows with this number, never with the adapter count. 1 is the serial
	// baseline. Unset: DefaultPoolSize (CPU count and available memory).
	FoundationPoolSizeEnv = "FOUNDATION_POOL_SIZE"
	// DefaultFoundationPool is the serial baseline and the fallback when the
	// machine's memory cannot be read.
	DefaultFoundationPool = 1
	MaxFoundationPool     = 8
	// FoundationInstanceBytes is what one loaded instance of the real
	// foundation costs (measured: 1.8 GB RSS for the 599 MB ModernBERT-base
	// graph with adapters loaded and scoring); DefaultPoolSize sizes by it.
	FoundationInstanceBytes = int64(2) << 30

	// The bounded delta cache: how many Evaluator adapters one instance keeps
	// loaded in the guest at once, and their padded bytes. Past either bound
	// the least recently scored delta is evicted (unload_adapter), logged, and
	// reloaded from its validated package on its next score. A delta is never
	// retained past the bound.
	FoundationDeltaCacheMaxEnv   = "FOUNDATION_DELTA_CACHE_MAX"
	FoundationDeltaCacheBytesEnv = "FOUNDATION_DELTA_CACHE_BYTES"
	DefaultDeltaCacheMax         = 16
	DefaultDeltaCacheBytes       = int64(512) << 20
	foundationStdioCap           = 16 << 10
	foundationMaxModelSize       = int64(hardFoundationBytes)
	// hardFoundationBytes bounds the graph file: it is copied into guest
	// memory and parsed there, so it must fit the memory cap twice over.
	hardFoundationBytes = 1 << 30
)

// FoundationMemoryLimitPages bounds one engine instance's linear memory, the
// 3 GiB (FloatMemoryLimitPages): the spike measured 1.8 GB for the
// 599 MB ModernBERT-base foundation with two adapters loaded and scoring.
const FoundationMemoryLimitPages = FloatMemoryLimitPages

var (
	fnABI            = "telegraph_abi"
	fnTgAlloc        = "tg_alloc"
	fnTgFree         = "tg_free"
	fnLoadFoundation = "load_foundation"
	fnFoundationSpec = "foundation_spec"
	fnLoadAdapter    = "load_adapter"
	fnScore          = "score"
	fnUnloadAdapter  = "unload_adapter"
	fnMessagePtr     = "message_ptr"
	fnMessageLen     = "message_len"
)

// foundationExports is every export the engine must have.
var foundationExports = []string{fnABI, fnTgAlloc, fnTgFree, fnLoadFoundation, fnFoundationSpec, fnLoadAdapter, fnScore, fnUnloadAdapter, fnMessagePtr, fnMessageLen}

// ErrFoundationMismatch marks a configured file whose sha256 is not the
// configured one. The node refuses to start on it.
var ErrFoundationMismatch = errors.New("foundation file hash mismatch")

// FoundationConfig locates the Foundation Model and its engine.
type FoundationConfig struct {
	ModelPath   string
	ModelHash   string // sha256, lowercase hex
	Version     string
	EnginePath  string
	EngineHash  string // sha256, lowercase hex
	PoolSize    int
	TokenizerSH string // the node's pinned tokenizer digest (tokenizer.ModernBERTSHA256 by default)
	// DeltaCacheMax / DeltaCacheBytes bound the per-instance delta cache
	// (0 = the defaults).
	DeltaCacheMax   int
	DeltaCacheBytes int64
	// MemoryLimitPages caps one instance's linear memory (0 =
	// FoundationMemoryLimitPages). The whole cap is committed up front per
	// instance (wazero reserves it so memory.grow never copies), so it is
	// also what a worker costs the OS; tests with the stub foundation use a
	// small one.
	MemoryLimitPages uint32
}

// DefaultPoolSize is the worker count when FOUNDATION_POOL_SIZE is unset:
//
//	clamp(min(cpus/4, (availableBytes - 2 GiB) / FoundationInstanceBytes), 1, MaxFoundationPool)
//
// so a worker has a few cores to itself and its own full copy of the
// foundation fits with headroom; unknown memory (0) means 1.
func DefaultPoolSize(cpus int, availableBytes int64) int {
	if availableBytes <= 0 {
		return DefaultFoundationPool
	}
	byCPU := cpus / 4
	byMem := int((availableBytes - (2 << 30)) / FoundationInstanceBytes)
	n := byCPU
	if byMem < n {
		n = byMem
	}
	if n < 1 {
		n = 1
	}
	if n > MaxFoundationPool {
		n = MaxFoundationPool
	}
	return n
}

// availableMemoryBytes is the machine's memory on Linux (/proc/meminfo
// MemTotal); 0 elsewhere, which keeps the derived pool at 1.
func availableMemoryBytes() int64 {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					return kb * 1024
				}
			}
		}
	}
	return 0
}

// Enabled reports whether a foundation is configured at all.
func (c FoundationConfig) Enabled() bool { return strings.TrimSpace(c.ModelPath) != "" }

// FoundationConfigFromEnv reads the FOUNDATION_* variables. tokenizerHash is
// the digest of the tokenizer the node loaded. It returns an error for a
// half-configured foundation (a path without its hash).
func FoundationConfigFromEnv(tokenizerHash string) (FoundationConfig, error) {
	c := FoundationConfig{
		ModelPath:       strings.TrimSpace(os.Getenv(FoundationModelPathEnv)),
		ModelHash:       normalizeSHA(os.Getenv(FoundationModelHashEnv)),
		Version:         strings.TrimSpace(os.Getenv(FoundationModelVersionEnv)),
		EnginePath:      strings.TrimSpace(os.Getenv(FoundationEngineWasmEnv)),
		EngineHash:      normalizeSHA(os.Getenv(FoundationEngineHashEnv)),
		PoolSize:        DefaultPoolSize(runtime.NumCPU(), availableMemoryBytes()),
		TokenizerSH:     tokenizerHash,
		DeltaCacheMax:   DefaultDeltaCacheMax,
		DeltaCacheBytes: DefaultDeltaCacheBytes,
	}
	if !c.Enabled() {
		return c, nil
	}
	if c.ModelHash == "" {
		return c, fmt.Errorf("%s is set but %s is not: the foundation hash is required", FoundationModelPathEnv, FoundationModelHashEnv)
	}
	if c.EnginePath == "" {
		return c, fmt.Errorf("%s is set but %s is not: the engine module is required", FoundationModelPathEnv, FoundationEngineWasmEnv)
	}
	if c.EngineHash == "" {
		return c, fmt.Errorf("%s is set but %s is not: the engine hash is required", FoundationEngineWasmEnv, FoundationEngineHashEnv)
	}
	if v := strings.TrimSpace(os.Getenv(FoundationPoolSizeEnv)); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > MaxFoundationPool {
			return c, fmt.Errorf("%s=%q: want a whole number in 1..%d", FoundationPoolSizeEnv, v, MaxFoundationPool)
		}
		c.PoolSize = n
	}
	if v := strings.TrimSpace(os.Getenv(FoundationDeltaCacheMaxEnv)); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return c, fmt.Errorf("%s=%q: want a whole number >= 1", FoundationDeltaCacheMaxEnv, v)
		}
		c.DeltaCacheMax = n
	}
	if v := strings.TrimSpace(os.Getenv(FoundationDeltaCacheBytesEnv)); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n < 1 {
			return c, fmt.Errorf("%s=%q: want bytes >= 1", FoundationDeltaCacheBytesEnv, v)
		}
		c.DeltaCacheBytes = n
	}
	return c, nil
}

func normalizeSHA(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimPrefix(s, "0x")
	if len(s) != 64 {
		return ""
	}
	if _, err := hex.DecodeString(s); err != nil {
		return ""
	}
	return s
}

// SHA256File is the lowercase hex sha256 of a file.
func SHA256File(path string) (string, int64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", 0, err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), int64(len(b)), nil
}

// Foundation is the loaded, shared Foundation Model: the compiled engine and
// PoolSize instances, each holding the parsed graph. Safe for concurrent use;
// scores are serialised per instance.
type Foundation struct {
	cfg      FoundationConfig
	spec     FoundationSpec
	wrt      wazero.Runtime
	compiled wazero.CompiledModule
	free     chan *foundationInstance
	all      []*foundationInstance
	mu       sync.Mutex // guards all / closed
	closed   bool

	instancesBuilt atomic.Int64

	// beforeScore, when set (tests only), runs after the per-score deadline is
	// armed and before the engine is called: a stand-in for a slow engine.
	beforeScore    func()
	peakMemory     atomic.Uint64
	cacheLoads     atomic.Int64
	cacheEvictions atomic.Int64
}

// foundationInstance is one engine instance with the foundation loaded and
// its bounded delta cache.
type foundationInstance struct {
	mod     api.Module
	handles map[*LoRARuntime]*cachedDelta // the adapters loaded in this guest
	bytes   uint64                        // their padded bytes in the guest
	tick    uint64                        // LRU clock
	dead    bool
	stdout  *cappedBuffer
	stderr  *cappedBuffer
}

// cachedDelta is one adapter resident in an instance's guest.
type cachedDelta struct {
	handle  int32
	bytes   uint64
	lastUse uint64
}

// DeltaCacheStats is the state of the delta caches across the instances.
type DeltaCacheStats struct {
	MaxEntries    int    `json:"max_entries"`
	MaxBytes      int64  `json:"max_bytes"`
	Instances     int    `json:"instances"`
	Resident      int    `json:"resident"`       // deltas loaded in a guest, summed over instances
	ResidentBytes uint64 `json:"resident_bytes"` // their padded bytes, summed
	Loads         int64  `json:"loads"`          // load_adapter calls (first loads and reloads after eviction)
	Evictions     int64  `json:"evictions"`
}

// LoadFoundation hashes and loads the engine and the graph, builds
// cfg.PoolSize instances and returns the shared Foundation. It refuses (and
// loads nothing) when a file's sha256 differs from its configured hash.
func LoadFoundation(ctx context.Context, cfg FoundationConfig) (*Foundation, error) {
	if !cfg.Enabled() {
		return nil, fmt.Errorf("wasm/runtime: no foundation configured (%s)", FoundationModelPathEnv)
	}
	if cfg.PoolSize < 1 {
		cfg.PoolSize = DefaultFoundationPool
	}
	if cfg.PoolSize > MaxFoundationPool {
		cfg.PoolSize = MaxFoundationPool
	}
	if cfg.TokenizerSH == "" {
		return nil, fmt.Errorf("wasm/runtime: foundation needs the node's tokenizer hash")
	}
	if cfg.DeltaCacheMax < 1 {
		cfg.DeltaCacheMax = DefaultDeltaCacheMax
	}
	if cfg.DeltaCacheBytes < 1 {
		cfg.DeltaCacheBytes = DefaultDeltaCacheBytes
	}
	engine, err := os.ReadFile(cfg.EnginePath)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: foundation engine: %w", err)
	}
	if got := sha256Hex(engine); got != cfg.EngineHash {
		return nil, fmt.Errorf("%w: engine %s has sha256 %s, %s=%s", ErrFoundationMismatch, cfg.EnginePath, got, FoundationEngineHashEnv, cfg.EngineHash)
	}
	model, err := readFoundationModel(cfg)
	if err != nil {
		return nil, err
	}
	if _, err := CheckFoundationEngine(engine); err != nil {
		return nil, fmt.Errorf("wasm/runtime: foundation engine: %w", err)
	}
	limit := cfg.MemoryLimitPages
	if limit == 0 {
		limit = FoundationMemoryLimitPages
	}
	wrt := wazero.NewRuntimeWithConfig(ctx, foundationRuntimeConfig(limit))
	if _, err := wasi_snapshot_preview1.Instantiate(ctx, wrt); err != nil {
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: instantiating the WASI host: %w", err)
	}
	compiled, err := wrt.CompileModule(ctx, engine)
	if err != nil {
		wrt.Close(ctx) //nolint:errcheck
		return nil, fmt.Errorf("wasm/runtime: compiling foundation engine: %w", err)
	}
	for _, f := range compiled.ImportedFunctions() {
		mod, name, _ := f.Import()
		if mod != WASIModule || !allowedWASIImports[name] {
			wrt.Close(ctx) //nolint:errcheck
			return nil, fmt.Errorf("wasm/runtime: foundation engine imports %s.%s, which this host does not provide", mod, name)
		}
	}
	f := &Foundation{cfg: cfg, wrt: wrt, compiled: compiled, free: make(chan *foundationInstance, cfg.PoolSize)}
	for i := 0; i < cfg.PoolSize; i++ {
		inst, spec, err := f.newInstance(ctx, model)
		if err != nil {
			f.Close(ctx) //nolint:errcheck
			return nil, err
		}
		if i == 0 {
			f.spec = spec
		} else if !sameSpec(f.spec, spec) {
			f.Close(ctx) //nolint:errcheck
			return nil, fmt.Errorf("wasm/runtime: foundation instances report different specs")
		}
		f.all = append(f.all, inst)
		f.free <- inst
	}
	return f, nil
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// readFoundationModel reads the graph and checks its hash (every time it is
// read: a file that changed on disk is never loaded silently).
func readFoundationModel(cfg FoundationConfig) ([]byte, error) {
	st, err := os.Stat(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: foundation model: %w", err)
	}
	if st.Size() > foundationMaxModelSize {
		return nil, fmt.Errorf("wasm/runtime: foundation model %s is %d bytes, above the %d byte limit", cfg.ModelPath, st.Size(), foundationMaxModelSize)
	}
	model, err := os.ReadFile(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("wasm/runtime: foundation model: %w", err)
	}
	if got := sha256Hex(model); got != cfg.ModelHash {
		return nil, fmt.Errorf("%w: model %s has sha256 %s, %s=%s", ErrFoundationMismatch, cfg.ModelPath, got, FoundationModelHashEnv, cfg.ModelHash)
	}
	return model, nil
}

// CheckFoundationEngine is the structural check of an engine module: a
// well-formed reactor (exports _initialize, not _start) with every ABI export
// and a memory, importing only allowed WASI functions.
func CheckFoundationEngine(b []byte) (FloatModuleInfo, error) {
	info, err := checkModuleImports(b)
	if err != nil {
		return info, err
	}
	exps, err := exportNames(b)
	if err != nil {
		return info, err
	}
	if _, ok := exps[fnStart]; ok {
		return info, fmt.Errorf("exports %q: a WASI command, not the reactor engine", fnStart)
	}
	for _, name := range foundationExports {
		if _, ok := exps[name]; !ok {
			return info, fmt.Errorf("missing export %q (ABI %d)", name, FoundationABIVersion)
		}
	}
	return info, nil
}

func foundationRuntimeConfig(limitPages uint32) wazero.RuntimeConfig {
	return cachedRuntimeConfig().
		WithCloseOnContextDone(true).
		WithMemoryLimitPages(limitPages).
		WithMemoryCapacityFromMax(true)
}

// foundationModuleConfig is the fixed host: nothing to read, fixed clocks and
// random source, capped stdio, the reactor's _initialize as the start function.
func foundationModuleConfig(stdout, stderr *cappedBuffer) wazero.ModuleConfig {
	return wazero.NewModuleConfig().
		WithName("").
		WithArgs("foundation_engine").
		WithStdin(bytes.NewReader(nil)).
		WithStdout(stdout).
		WithStderr(stderr).
		WithRandSource(newFixedRandSource()).
		WithWalltime(fixedWalltime, sys.ClockResolution(1)).
		WithNanotime(fixedNanotime, sys.ClockResolution(1)).
		WithNanosleep(func(int64) {}).
		WithOsyield(func() {}).
		WithStartFunctions(fnInitialize)
}

// newInstance instantiates the engine and loads model into it.
func (f *Foundation) newInstance(ctx context.Context, model []byte) (*foundationInstance, FoundationSpec, error) {
	stdout, stderr := newCappedBuffer(foundationStdioCap), newCappedBuffer(foundationStdioCap)
	mod, err := f.wrt.InstantiateModule(ctx, f.compiled, foundationModuleConfig(stdout, stderr))
	if err != nil {
		return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: instantiating foundation engine: %w%s", err, tail(stderr))
	}
	inst := &foundationInstance{mod: mod, handles: map[*LoRARuntime]*cachedDelta{}, stdout: stdout, stderr: stderr}
	abi, err := call1(ctx, mod, fnABI)
	if err != nil || abi != FoundationABIVersion {
		mod.Close(ctx) //nolint:errcheck
		if err != nil {
			return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: foundation engine %s: %w", fnABI, err)
		}
		return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: foundation engine declares ABI %d, this node runs %d", abi, FoundationABIVersion)
	}
	status, err := inst.callWithBytes(ctx, fnLoadFoundation, model)
	if err != nil {
		mod.Close(ctx) //nolint:errcheck
		return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: load_foundation: %w", err)
	}
	if status != 0 {
		msg := inst.message(ctx)
		mod.Close(ctx) //nolint:errcheck
		return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: the engine refused the foundation graph (status %d): %s", status, msg)
	}
	n, err := call1(ctx, mod, fnFoundationSpec)
	if err != nil || n <= 0 {
		mod.Close(ctx) //nolint:errcheck
		return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: foundation_spec failed (%d): %v", n, err)
	}
	var spec FoundationSpec
	if err := json.Unmarshal([]byte(inst.message(ctx)), &spec); err != nil {
		mod.Close(ctx) //nolint:errcheck
		return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: foundation_spec is not JSON: %w", err)
	}
	if err := spec.Check(); err != nil {
		mod.Close(ctx) //nolint:errcheck
		return nil, FoundationSpec{}, fmt.Errorf("wasm/runtime: %w", err)
	}
	f.instancesBuilt.Add(1)
	if m := mod.Memory(); m != nil {
		f.notePeak(uint64(m.Size()))
	}
	return inst, spec, nil
}

func sameSpec(a, b FoundationSpec) bool {
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	return bytes.Equal(ja, jb)
}

func (f *Foundation) notePeak(n uint64) {
	for {
		p := f.peakMemory.Load()
		if n <= p || f.peakMemory.CompareAndSwap(p, n) {
			return
		}
	}
}

func call1(ctx context.Context, mod api.Module, name string, args ...uint64) (int32, error) {
	fn := mod.ExportedFunction(name)
	if fn == nil {
		return 0, fmt.Errorf("engine has no export %q", name)
	}
	res, err := fn.Call(ctx, args...)
	if err != nil {
		return 0, err
	}
	if len(res) == 0 {
		return 0, fmt.Errorf("%s returned nothing", name)
	}
	return int32(uint32(res[0])), nil
}

// callWithBytes copies b into guest memory, calls name(ptr, len) and frees
// the copy.
func (inst *foundationInstance) callWithBytes(ctx context.Context, name string, b []byte) (int32, error) {
	ptr, err := call1(ctx, inst.mod, fnTgAlloc, uint64(uint32(len(b))))
	if err != nil {
		return 0, fmt.Errorf("%s: %w", fnTgAlloc, err)
	}
	if ptr == 0 {
		return 0, fmt.Errorf("%s(%d) returned a null pointer", fnTgAlloc, len(b))
	}
	if !inst.mod.Memory().Write(uint32(ptr), b) {
		return 0, fmt.Errorf("writing %d bytes at %d failed", len(b), ptr)
	}
	status, err := call1(ctx, inst.mod, name, uint64(uint32(ptr)), uint64(uint32(len(b))))
	_, _ = call1(ctx, inst.mod, fnTgFree, uint64(uint32(ptr)), uint64(uint32(len(b))))
	return status, err
}

// message reads the engine's message buffer (an error text or the spec).
func (inst *foundationInstance) message(ctx context.Context) string {
	p, err := call1(ctx, inst.mod, fnMessagePtr)
	if err != nil {
		return ""
	}
	n, err := call1(ctx, inst.mod, fnMessageLen)
	if err != nil || n <= 0 {
		return ""
	}
	b, ok := inst.mod.Memory().Read(uint32(p), uint32(n))
	if !ok {
		return ""
	}
	return string(b)
}

func tail(b *cappedBuffer) string {
	s := strings.TrimSpace(b.buf.String())
	if s == "" {
		return ""
	}
	if len(s) > 300 {
		s = "..." + s[len(s)-300:]
	}
	return " (stderr: " + s + ")"
}

// Hash is the foundation graph's sha256 (F).
func (f *Foundation) Hash() string { return f.cfg.ModelHash }

// Version is FOUNDATION_MODEL_VERSION.
func (f *Foundation) Version() string { return f.cfg.Version }

// EngineHash is the engine module's sha256.
func (f *Foundation) EngineHash() string { return f.cfg.EngineHash }

// TokenizerHash is the digest of the tokenizer every adapter must name (T).
func (f *Foundation) TokenizerHash() string { return f.cfg.TokenizerSH }

// Spec is the LoRA input layout the engine reported.
func (f *Foundation) Spec() FoundationSpec { return f.spec }

// PoolSize is how many engine instances hold the foundation.
func (f *Foundation) PoolSize() int { return f.cfg.PoolSize }

// InstancesBuilt counts engine instantiations so far (PoolSize, plus one per
// instance rebuilt after a trap or a timed-out score). Tests pin it.
func (f *Foundation) InstancesBuilt() int64 { return f.instancesBuilt.Load() }

// MemoryBytes is the linear memory the instances hold: the largest measured
// so far per instance, times the pool size.
func (f *Foundation) MemoryBytes() uint64 { return f.peakMemory.Load() * uint64(f.cfg.PoolSize) }

// InstanceMemoryBytes is the largest linear memory one instance has reached.
func (f *Foundation) InstanceMemoryBytes() uint64 { return f.peakMemory.Load() }

// CacheStats reports the delta caches (waits for in-flight scores).
func (f *Foundation) CacheStats() DeltaCacheStats {
	st := DeltaCacheStats{MaxEntries: f.cfg.DeltaCacheMax, MaxBytes: f.cfg.DeltaCacheBytes,
		Loads: f.cacheLoads.Load(), Evictions: f.cacheEvictions.Load()}
	f.mu.Lock()
	closed, n := f.closed, len(f.all)
	f.mu.Unlock()
	if closed {
		return st
	}
	for i := 0; i < n; i++ {
		inst := <-f.free
		if !inst.dead {
			st.Instances++
			st.Resident += len(inst.handles)
			st.ResidentBytes += inst.bytes
		}
		f.free <- inst
	}
	return st
}

// Expect is what a safetensors delta package must match to run on this foundation for an
// intent of class ("" accepts either).
func (f *Foundation) Expect(class string) LoRAExpect {
	return LoRAExpect{FoundationHash: f.cfg.ModelHash, TokenizerHash: f.cfg.TokenizerSH, Class: class, Spec: f.spec}
}

// acquire takes a free instance, rebuilding a dead one first. A context that
// has already ended takes nothing (and kills nothing): that is the caller's
// failure, not an engine call that went wrong.
func (f *Foundation) acquire(ctx context.Context) (*foundationInstance, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var inst *foundationInstance
	select {
	case inst = <-f.free:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if !inst.dead {
		return inst, nil
	}
	model, err := readFoundationModel(f.cfg)
	if err != nil {
		f.free <- inst
		return nil, err
	}
	fresh, spec, err := f.newInstance(context.Background(), model)
	if err != nil {
		f.free <- inst
		return nil, err
	}
	if !sameSpec(spec, f.spec) {
		fresh.mod.Close(context.Background()) //nolint:errcheck
		f.free <- inst
		return nil, fmt.Errorf("wasm/runtime: rebuilt foundation instance reports a different spec")
	}
	f.mu.Lock()
	for i, old := range f.all {
		if old == inst {
			f.all[i] = fresh
		}
	}
	f.mu.Unlock()
	return fresh, nil
}

func (f *Foundation) release(inst *foundationInstance) { f.free <- inst }

// kill closes a dead instance's module; the next acquire rebuilds it.
func (inst *foundationInstance) kill() {
	inst.dead = true
	inst.handles = map[*LoRARuntime]*cachedDelta{}
	inst.bytes = 0
	inst.mod.Close(context.Background()) //nolint:errcheck
}

// ensureAdapter returns rt's guest handle in inst, loading its validated
// package first when it is not resident. Before a load, the least recently
// scored deltas are evicted until the new one fits the cache bound (count
// and bytes); every eviction is logged and counted. The package re-enters the
// guest exactly as it was validated (the same bytes, re-checked by the
// engine), so the foundation and tokenizer gate holds on the cache path too.
func (inst *foundationInstance) ensureAdapter(ctx context.Context, f *Foundation, rt *LoRARuntime) (int32, error) {
	inst.tick++
	if d, ok := inst.handles[rt]; ok {
		d.lastUse = inst.tick
		return d.handle, nil
	}
	need := rt.GuestBytes()
	for len(inst.handles) > 0 && (len(inst.handles)+1 > f.cfg.DeltaCacheMax || inst.bytes+need > uint64(f.cfg.DeltaCacheBytes)) {
		inst.evictLRU(ctx, f, rt)
	}
	h, err := inst.callWithBytes(ctx, fnLoadAdapter, rt.adapter.Raw)
	if err != nil {
		return 0, err
	}
	if h <= 0 {
		return 0, &LoRARejectError{Reason: loraGuestReason(inst.message(ctx)), Detail: fmt.Sprintf("the engine refused the adapter (status %d): %s", h, inst.message(ctx))}
	}
	inst.handles[rt] = &cachedDelta{handle: h, bytes: need, lastUse: inst.tick}
	inst.bytes += need
	f.cacheLoads.Add(1)
	return h, nil
}

// evictLRU unloads the least recently scored delta from the guest.
func (inst *foundationInstance) evictLRU(ctx context.Context, f *Foundation, forRT *LoRARuntime) {
	var victim *LoRARuntime
	var oldest uint64 = ^uint64(0)
	for rt, d := range inst.handles {
		if d.lastUse < oldest {
			victim, oldest = rt, d.lastUse
		}
	}
	if victim == nil {
		return
	}
	d := inst.handles[victim]
	_, _ = call1(ctx, inst.mod, fnUnloadAdapter, uint64(uint32(d.handle)))
	delete(inst.handles, victim)
	inst.bytes -= d.bytes
	f.cacheEvictions.Add(1)
	log.Info().Str("evicted_adapter", short(victim.hash)).Str("for_adapter", short(forRT.hash)).
		Uint64("delta_bytes", d.bytes).Int("resident", len(inst.handles)).Uint64("resident_bytes", inst.bytes).
		Int("max_entries", f.cfg.DeltaCacheMax).Int64("max_bytes", f.cfg.DeltaCacheBytes).
		Msg("[foundation] delta cache bound reached: evicted the least recently scored adapter from the guest (reloaded from its validated package on its next score)")
}

// loraGuestReason maps the engine's "reason: detail" message to a reason.
func loraGuestReason(msg string) string {
	if i := strings.IndexByte(msg, ':'); i > 0 {
		return strings.TrimSpace(msg[:i])
	}
	return LoRAReasonBadHeader
}

// score runs one score of rt on ids, in one instance, under the per-score
// deadline. A trap or a timed-out call kills the instance (rebuilt lazily).
func (f *Foundation) score(ctx context.Context, rt *LoRARuntime, ids []int64) (ModelScore, error) {
	inst, err := f.acquire(ctx)
	if err != nil {
		return ModelScore{}, fmt.Errorf("wasm/runtime: foundation: %w", err)
	}
	defer f.release(inst)
	limit := FloatScoreTimeout()
	callCtx, cancel := context.WithTimeout(ctx, limit)
	defer cancel()
	if f.beforeScore != nil {
		f.beforeScore()
	}

	h, err := inst.ensureAdapter(callCtx, f, rt)
	if err != nil {
		if inst.mod.IsClosed() {
			inst.kill()
		}
		var re *LoRARejectError
		if errors.As(err, &re) {
			return ModelScore{}, fmt.Errorf("%w: %w", ErrFloatGuest, err)
		}
		return ModelScore{}, f.classify(ctx, callCtx, limit, inst, "load_adapter", err)
	}
	buf := make([]byte, 4*len(ids))
	for i, id := range ids {
		binary.LittleEndian.PutUint32(buf[4*i:], uint32(id))
	}
	ptr, err := call1(callCtx, inst.mod, fnTgAlloc, uint64(uint32(len(buf))))
	if err != nil || ptr == 0 {
		return ModelScore{}, f.classify(ctx, callCtx, limit, inst, fnTgAlloc, err)
	}
	if !inst.mod.Memory().Write(uint32(ptr), buf) {
		return ModelScore{}, fmt.Errorf("wasm/runtime: foundation: writing %d token bytes failed", len(buf))
	}
	fn := inst.mod.ExportedFunction(fnScore)
	res, err := fn.Call(callCtx, uint64(uint32(h)), uint64(uint32(ptr)), uint64(uint32(len(ids))))
	_, _ = call1(callCtx, inst.mod, fnTgFree, uint64(uint32(ptr)), uint64(uint32(len(buf))))
	if m := inst.mod.Memory(); m != nil {
		f.notePeak(uint64(m.Size()))
	}
	if err != nil {
		return ModelScore{}, f.classify(ctx, callCtx, limit, inst, fnScore, err)
	}
	if len(res) == 0 {
		return ModelScore{}, fmt.Errorf("%w: score returned nothing", ErrFloatGuest)
	}
	v := int64(res[0])
	if v < 0 {
		return ModelScore{}, fmt.Errorf("%w: score status %d: %s", ErrFloatGuest, v, inst.message(callCtx))
	}
	return NewModelScoreFromBits(uint32(v))
}

// classify turns a failed engine call into the error classes the scorers
// read: the caller's context ending is ours; a timed-out score is the
// adapter's (ErrFloatGuest + ErrScoreTimeout); a trap is the adapter's. All
// three leave the instance unusable, so it is killed and rebuilt lazily.
func (f *Foundation) classify(ctx, callCtx context.Context, limit time.Duration, inst *foundationInstance, what string, err error) error {
	inst.kill()
	if ctxErr := ctx.Err(); ctxErr != nil {
		return fmt.Errorf("wasm/runtime: foundation score stopped: %w", ctxErr)
	}
	if callCtx.Err() != nil {
		return fmt.Errorf("%w: %w: %s did not finish within %s (%s)", ErrFloatGuest, ErrScoreTimeout, what, limit, ScoreTimeoutEnv)
	}
	var exit *sys.ExitError
	if errors.As(err, &exit) {
		return fmt.Errorf("%w: engine exited with code %d%s", ErrFloatGuest, exit.ExitCode(), tail(inst.stderr))
	}
	return fmt.Errorf("%w: %s: %v%s", ErrFloatGuest, what, err, tail(inst.stderr))
}

// unload drops rt from every instance (waiting for in-flight scores).
func (f *Foundation) unload(rt *LoRARuntime) {
	f.mu.Lock()
	closed := f.closed
	n := len(f.all)
	f.mu.Unlock()
	if closed {
		return
	}
	for i := 0; i < n; i++ {
		inst := <-f.free
		if !inst.dead {
			if d, ok := inst.handles[rt]; ok {
				_, _ = call1(context.Background(), inst.mod, fnUnloadAdapter, uint64(uint32(d.handle)))
				delete(inst.handles, rt)
				inst.bytes -= d.bytes
			}
		}
		f.free <- inst
	}
}

// Close releases every instance and the runtime. Idempotent.
func (f *Foundation) Close(ctx context.Context) error {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	if f.closed {
		f.mu.Unlock()
		return nil
	}
	f.closed = true
	n := len(f.all)
	f.mu.Unlock()
	for i := 0; i < n; i++ {
		inst := <-f.free
		inst.mod.Close(ctx) //nolint:errcheck
	}
	if f.wrt != nil {
		return f.wrt.Close(ctx)
	}
	return nil
}

// The process-wide active foundation: loaded once at startup (main.go) and
// read by Stage 1, the installer and the loader. Nil when the node runs none.
var (
	activeFoundationMu sync.RWMutex
	activeFoundation   *Foundation
)

// SetActiveFoundation installs f as the process's foundation (nil clears it).
func SetActiveFoundation(f *Foundation) {
	activeFoundationMu.Lock()
	activeFoundation = f
	activeFoundationMu.Unlock()
}

// ActiveFoundation returns the process's foundation, or nil.
func ActiveFoundation() *Foundation {
	activeFoundationMu.RLock()
	defer activeFoundationMu.RUnlock()
	return activeFoundation
}

// ActiveFoundationHash is the active foundation's hash, or "".
func ActiveFoundationHash() string {
	if f := ActiveFoundation(); f != nil {
		return f.Hash()
	}
	return ""
}
