package runtime

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/tetratelabs/wazero"
)

// CompileCacheDirEnv names the directory wazero persists compiled machine code
// in. Unset means an in-memory cache, still shared by every runtime in the
// process but gone on restart.
const CompileCacheDirEnv = "WASM_COMPILE_CACHE_DIR"

// The process-wide compilation cache.
//
// Why it exists: production adapters are loaded and closed once per epoch
// batch (see pkg/scoring/adapter_budget.go) instead of staying resident for
// the life of the process. Every load would otherwise re-compile the module.
// With a directory cache the compiled code is written to disk the first time
// and read back on every later load of the same bytes, including after a
// restart. The key is wazero's own: sha256 of the module bytes plus the wazero
// version, the CPU features and the runtime config that changes codegen (for
// example WithCloseOnContextDone), so a cache entry can never be reused for a
// different module or an incompatible build.
//
// What is shared is ONLY compiled code. Every load still gets its own
// wazero.Runtime and its own linear memory; nothing a module computes is
// carried from one load to the next.
var (
	compileCacheMu  sync.Mutex
	compileCache    wazero.CompilationCache
	compileCacheDir string
)

// InitCompilationCache installs the process-wide compilation cache. dir is the
// on-disk location; empty means in-memory. If the directory cannot be used the
// cache falls back to in-memory and the error is returned so the caller can
// log it (scoring still works, it just re-compiles after a restart).
//
// Calling it again replaces the cache; the old one is closed. Runtimes already
// built keep working: wazero keeps their compiled code alive until they close.
func InitCompilationCache(dir string) error {
	compileCacheMu.Lock()
	defer compileCacheMu.Unlock()
	old := compileCache
	dir = strings.TrimSpace(dir)
	var initErr error
	var c wazero.CompilationCache
	if dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			initErr = fmt.Errorf("wasm/runtime: compile cache dir %s: %w", dir, err)
		} else if dc, err := wazero.NewCompilationCacheWithDir(dir); err != nil {
			initErr = fmt.Errorf("wasm/runtime: compile cache dir %s: %w", dir, err)
		} else {
			c = dc
		}
	}
	if c == nil {
		c = wazero.NewCompilationCache()
		dir = ""
	}
	compileCache, compileCacheDir = c, dir
	if old != nil {
		old.Close(context.Background()) //nolint:errcheck
	}
	return initErr
}

// SharedCompilationCache returns the process-wide cache, creating it from
// WASM_COMPILE_CACHE_DIR on first use when InitCompilationCache was never
// called (tests, tools).
func SharedCompilationCache() wazero.CompilationCache {
	compileCacheMu.Lock()
	c := compileCache
	compileCacheMu.Unlock()
	if c != nil {
		return c
	}
	_ = InitCompilationCache(os.Getenv(CompileCacheDirEnv))
	compileCacheMu.Lock()
	defer compileCacheMu.Unlock()
	return compileCache
}

// CompilationCacheDir reports the directory the shared cache persists to, or
// "" for an in-memory cache.
func CompilationCacheDir() string {
	compileCacheMu.Lock()
	defer compileCacheMu.Unlock()
	return compileCacheDir
}

// CloseCompilationCache closes the shared cache on shutdown. A later runtime
// build lazily creates a fresh one, so a straggling load after shutdown starts
// fails safe (re-compiles) rather than using a closed cache.
func CloseCompilationCache(ctx context.Context) error {
	compileCacheMu.Lock()
	c := compileCache
	compileCache, compileCacheDir = nil, ""
	compileCacheMu.Unlock()
	if c == nil {
		return nil
	}
	return c.Close(ctx)
}

// cachedRuntimeConfig is the runtime config every wide (and comparator)
// runtime is built with: wazero defaults plus the shared compilation cache.
func cachedRuntimeConfig() wazero.RuntimeConfig {
	return wazero.NewRuntimeConfig().WithCompilationCache(SharedCompilationCache())
}
