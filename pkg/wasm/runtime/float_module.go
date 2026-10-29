package runtime

import (
	"fmt"
	"sort"
	"strings"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
)

// Structural checks of a WASM module against the fixed host: the import
// allow-list the foundation engine (CheckFoundationEngine) must satisfy, and
// the export probe that recognises a LEGACY full-model float adapter (a WASI
// command exporting `_start`) so it can be rejected by name.

const (
	// WASIModule is the only import module a float adapter may use.
	WASIModule = "wasi_snapshot_preview1"

	// fnStart is the WASI command entry point.
	fnStart = "_start"
	// fnInitialize marks a WASI reactor, which this host does not run.
	fnInitialize = "_initialize"
)

// allowedWASIImports is every wasi_snapshot_preview1 function the engine
// may import. It is what the fixed host is built to serve: args, environ
// (always empty), the fixed clocks, the fixed random source, stdio and a
// read-only filesystem holding only the input. Anything else — sockets,
// filesystem writes, signals, renumbering descriptors — is refused at Stage 1
// and again at load, so an adapter cannot depend on a capability the host
// pretends not to have.
//
// The foundation engine (like the retired baked adapters) imports args_sizes_get, args_get,
// random_get, environ_get, environ_sizes_get, clock_time_get, fd_close,
// fd_fdstat_get, fd_filestat_get, fd_prestat_get, fd_prestat_dir_name,
// fd_read, fd_write, path_open and proc_exit; the rest are read-only calls
// Rust's std can emit for the same code.
var allowedWASIImports = map[string]bool{
	"args_get":            true,
	"args_sizes_get":      true,
	"environ_get":         true,
	"environ_sizes_get":   true,
	"clock_res_get":       true,
	"clock_time_get":      true,
	"random_get":          true,
	"fd_close":            true,
	"fd_fdstat_get":       true,
	"fd_filestat_get":     true,
	"fd_prestat_get":      true,
	"fd_prestat_dir_name": true,
	"fd_read":             true,
	"fd_pread":            true,
	"fd_readdir":          true,
	"fd_seek":             true,
	"fd_tell":             true,
	"fd_write":            true,
	"path_open":           true,
	"path_filestat_get":   true,
	"poll_oneoff":         true,
	"proc_exit":           true,
	"sched_yield":         true,
}

// AllowedWASIImports lists the allowed import names, sorted (docs, errors).
func AllowedWASIImports() []string {
	out := make([]string, 0, len(allowedWASIImports))
	for k := range allowedWASIImports {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// FloatModuleInfo is what the structural check learns about a module.
type FloatModuleInfo struct {
	// MinMemoryPages is the module's declared initial memory (64 KiB pages).
	MinMemoryPages uint32
	// DataBytes is the total size of the data section payload (the weights).
	DataBytes int64
	// Imports are the module's (allowed) WASI imports.
	Imports []string

	hasMemory bool
}

// checkModuleImports is the import allow-list check of the foundation engine
// (CheckFoundationEngine), plus the module's memory and data sizes.
func checkModuleImports(b []byte) (FloatModuleInfo, error) {
	var info FloatModuleInfo
	secs, err := wasmbin.Sections(b)
	if err != nil {
		return info, fmt.Errorf("malformed module: %w", err)
	}
	imps, err := wasmbin.Imports(b)
	if err != nil {
		return info, fmt.Errorf("reading imports: %w", err)
	}
	var bad []string
	for _, im := range imps {
		switch {
		case im.Module != WASIModule:
			bad = append(bad, fmt.Sprintf("%s.%s (only %s is provided)", im.Module, im.Name, WASIModule))
		case im.Kind != wasmbin.ImportFunc:
			bad = append(bad, fmt.Sprintf("%s.%s (a %s import; only functions are provided)", im.Module, im.Name, importKindName(im.Kind)))
		case !allowedWASIImports[im.Name]:
			bad = append(bad, fmt.Sprintf("%s.%s (not provided by this host)", im.Module, im.Name))
		default:
			info.Imports = append(info.Imports, im.Name)
		}
	}
	if len(bad) > 0 {
		return info, fmt.Errorf("imports the host does not provide: %s", strings.Join(bad, ", "))
	}
	for _, s := range secs {
		switch s.ID {
		case wasmbin.SecMemory:
			lim, err := wasmbin.ParseMemory(s.Payload(b))
			if err != nil {
				return info, fmt.Errorf("memory section: %w", err)
			}
			info.MinMemoryPages, info.hasMemory = lim.Min, true
		case wasmbin.SecData:
			info.DataBytes += int64(s.End - s.PayloadStart)
		}
	}
	return info, nil
}

func exportNames(b []byte) (map[string]byte, error) {
	exps, err := wasmbin.ExportNames(b)
	if err != nil {
		return nil, fmt.Errorf("reading exports: %w", err)
	}
	return exps, nil
}

func hasMemoryExport(exps map[string]byte) bool {
	for _, k := range exps {
		if k == wasmbin.ExportMemory {
			return true
		}
	}
	return false
}

func importKindName(k byte) string {
	switch k {
	case wasmbin.ImportFunc:
		return "function"
	case wasmbin.ImportTable:
		return "table"
	case wasmbin.ImportMemory:
		return "memory"
	case wasmbin.ImportGlobal:
		return "global"
	default:
		return fmt.Sprintf("kind-0x%02x", k)
	}
}
