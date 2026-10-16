// Package wideembed builds a SELF-CONTAINED wide Q32.32 scoring package: one
// .wasm per intent carrying the wide interpreter AND that intent's .imwq model,
// so registration and verification are a single artifact and the registered
// keccak256 covers model and code together ("the hash is the model").
//
// Mechanism: a post-link rewrite of a pinned EMBED HOST (wide_wasm built with
// --features embedded / embedded-hybrid). The host is never recompiled per
// intent; Package appends two active data segments to it —
//
//  1. the 56-byte embed slot ("IMWQEMB1", version, ptr, len, sha256) written
//     over the host's IMWQ_EMBED_DESC static, whose address the host reports via
//     imwq_embed_desc_addr; and
//  2. the .imwq bytes, placed at the host's original initial-memory size
//     (== __heap_end, the only pre-existing region dlmalloc ever hands out —
//     all other allocations come from memory.grow, past the raised minimum),
//
// and raises the memory section's minimum to cover the model. Every other
// section is copied verbatim. The interpreter code is byte-identical to the
// pinned host, and at run time the host passes the self-reported (ptr, len) to
// the SAME imwq_score / imwq_score_hybrid exports the split path uses, so an
// embedded model scores bit-identically to the same model uploaded by
// LoadArtifact.
//
// Package is a pure function of (host, imwq, class): output is deterministic
// (minimal LEB128 throughout), so anyone holding the pinned host and the .imwq
// reproduces the registered hash. It fails LOUDLY — never truncates, never
// guesses — on a host/class mismatch, a malformed or truncated .imwq, or a model
// whose resident size plus parse copy plus activation peak would exceed the
// wasm32 4 GiB linear-memory ceiling.
package wideembed

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/tetratelabs/wazero"

	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wasmbin"
)

// Class is which scoring path a package serves. It must match the host build.
type Class string

const (
	// ClassJudgment: model only, exports imwq_score (host: --features embedded).
	ClassJudgment Class = "judgment"
	// ClassHybrid: model + comparator/fusion, exports imwq_score_hybrid too
	// (host: --features embedded-hybrid).
	ClassHybrid Class = "hybrid"
)

const (
	pageSize = 65536

	// Wasm32Limit is the wasm32 linear-memory ceiling (65536 pages).
	Wasm32Limit uint64 = 1 << 32

	// DefaultMargin is headroom kept below the limit for the token/spec buffers,
	// allocator overhead and fragmentation that the estimate does not model.
	DefaultMargin uint64 = 64 << 20

	// DefaultMaxTokens is the longest sequence the node scores: the tokenizer's
	// DefaultMaxLen (pkg/wasm/tokenizer). The activation bound is computed for
	// this length; a node configured for longer sequences must package with a
	// matching MaxTokens.
	DefaultMaxTokens = 256
)

// Exports a package host must (or must not) carry.
const (
	expEmbedded      = "imwq_embedded"
	expEmbedDescAddr = "imwq_embed_desc_addr"
	expScore         = "imwq_score"
	expScoreHybrid   = "imwq_score_hybrid"
	expDebugCounts   = "imwq_debug_counts"
	expAlloc         = "imwq_alloc"
	expFree          = "imwq_free"
	expDeterministic = "imw_score_deterministic"
)

// Options tunes Package. The zero value means the defaults.
type Options struct {
	// LimitBytes is the memory ceiling the budget is checked against
	// (default Wasm32Limit). Lowering it is how tests exercise the loud
	// over-budget failure without a multi-GB file.
	LimitBytes uint64
	// MarginBytes is reserved headroom (default DefaultMargin).
	MarginBytes uint64
	// MaxTokens is the sequence length the activation bound assumes
	// (default DefaultMaxTokens).
	MaxTokens int
	// SkipVerify skips instantiating the finished package. Only for callers that
	// verify separately; the CLI always verifies.
	SkipVerify bool
	// Training, when set, is embedded as the "training" field of the package's
	// telegraph.adapter section (the same section, not a second one) and echoed
	// in the Manifest. A record that fails TrainingRecord.Validate is refused. It is
	// self-reported and the node never acts on it. nil leaves the section
	// byte-identical to a build without it.
	Training *wasmrt.TrainingRecord
}

func (o Options) withDefaults() Options {
	if o.LimitBytes == 0 {
		o.LimitBytes = Wasm32Limit
	}
	if o.MarginBytes == 0 {
		o.MarginBytes = DefaultMargin
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = DefaultMaxTokens
	}
	return o
}

// Budget is the 4 GiB check, itemised so a failure says which term dominates.
type Budget struct {
	ModelPtr        uint64 `json:"model_ptr"`
	ModelLen        uint64 `json:"model_len"`
	ModelEnd        uint64 `json:"model_end"`
	ParsedBytes     uint64 `json:"parsed_bytes"`
	ActivationBytes uint64 `json:"activation_bytes"`
	MarginBytes     uint64 `json:"margin_bytes"`
	NeedBytes       uint64 `json:"need_bytes"`
	LimitBytes      uint64 `json:"limit_bytes"`
	HeadroomBytes   int64  `json:"headroom_bytes"`
}

// CheckBudget computes the worst-case linear-memory peak of a package —
// resident model, plus the copy WideModel::parse makes on every score, plus the
// activation bound, plus margin — and errors if it exceeds limit. Pure, so it is
// unit-tested directly with real-model numbers.
func CheckBudget(modelPtr, modelLen, parsed, activation, margin, limit uint64) (Budget, error) {
	b := Budget{
		ModelPtr: modelPtr, ModelLen: modelLen, ModelEnd: modelPtr + modelLen,
		ParsedBytes: parsed, ActivationBytes: activation, MarginBytes: margin, LimitBytes: limit,
	}
	b.NeedBytes = b.ModelEnd + parsed + activation + margin
	b.HeadroomBytes = int64(limit) - int64(b.NeedBytes)
	if b.NeedBytes > limit {
		return b, fmt.Errorf("wideembed: embedded model + parse copy + activation peak exceed the wasm32 memory ceiling: "+
			"need %d bytes (%.2f GiB = model end %.2f + parse %.2f + activations %.2f + margin %.2f) > limit %d (%.2f GiB); "+
			"refusing to build a package that would trap out of memory at score time",
			b.NeedBytes, gib(b.NeedBytes), gib(b.ModelEnd), gib(parsed), gib(activation), gib(margin), limit, gib(limit))
	}
	if b.ModelEnd > Wasm32Limit {
		return b, fmt.Errorf("wideembed: model end %d is past the 4 GiB wasm32 address space", b.ModelEnd)
	}
	return b, nil
}

func gib(v uint64) float64 { return float64(v) / (1 << 30) }

// Manifest records what went into a package and what came out.
type Manifest struct {
	Class           Class    `json:"class"`
	HostSHA256      string   `json:"host_sha256"`
	IMWQSHA256      string   `json:"imwq_sha256"`
	BundleSHA256    string   `json:"bundle_sha256"`
	BundleKeccak256 string   `json:"bundle_keccak256"` // what registerWasm takes as wasmHash
	BundleBytes     int      `json:"bundle_bytes"`
	HostMinPages    uint32   `json:"host_min_pages"`
	MinPages        uint32   `json:"min_pages"`
	DescAddr        uint32   `json:"desc_addr"`
	MaxTokens       int      `json:"max_tokens"`
	Budget          Budget   `json:"budget"`
	IMWQ            IMWQInfo `json:"imwq"`
	// Verified is true when the finished package was instantiated through
	// wasmrt.NewWideEmbedded/NewWideHybridEmbedded and adopted cleanly.
	Verified bool `json:"verified"`
	// LoadedMemoryBytes is linear memory right after that verification load
	// (resident model + one parse copy from DebugCounts).
	LoadedMemoryBytes uint64 `json:"loaded_memory_bytes,omitempty"`
	// ABIVersion is the scoring ABI version written into the package's
	// telegraph.adapter section (wasmrt.AdapterMetaSection).
	ABIVersion int `json:"abi_version"`
	// Training echoes the training record embedded in the section, if any.
	Training *wasmrt.TrainingRecord `json:"training,omitempty"`
}

// hostInfo is what instantiating the bare host tells the packager.
type hostInfo struct {
	descAddr uint32
	minPages uint32
}

// Package builds a self-contained package from an embed host and a .imwq.
func Package(ctx context.Context, host, imwq []byte, class Class, opt Options) ([]byte, Manifest, error) {
	opt = opt.withDefaults()
	var man Manifest
	man.Class, man.MaxTokens = class, opt.MaxTokens

	if class != ClassJudgment && class != ClassHybrid {
		return nil, man, fmt.Errorf("wideembed: class must be %q or %q, got %q", ClassJudgment, ClassHybrid, class)
	}
	if err := opt.Training.Validate(); err != nil {
		return nil, man, fmt.Errorf("wideembed: %w", err)
	}
	if err := checkHostExports(host, class); err != nil {
		return nil, man, err
	}
	info, err := ParseIMWQ(imwq)
	if err != nil {
		return nil, man, err
	}
	man.IMWQ = info
	if uint64(len(imwq)) >= Wasm32Limit {
		return nil, man, fmt.Errorf("wideembed: .imwq is %d bytes, larger than wasm32 memory", len(imwq))
	}

	secs, err := wasmbin.Sections(host)
	if err != nil {
		return nil, man, fmt.Errorf("wideembed: host is not a well-formed wasm module: %w", err)
	}
	var memIdx, dataIdx, dcIdx, codeIdx = -1, -1, -1, -1
	for i, s := range secs {
		switch s.ID {
		case wasmbin.SecImport:
			// Rust embed hosts import nothing; an imported memory would make the
			// memory section rewrite meaningless.
			return nil, man, fmt.Errorf("wideembed: host has an import section — not a wide_wasm embed host")
		case wasmbin.SecMemory:
			memIdx = i
		case wasmbin.SecData:
			dataIdx = i
		case wasmbin.SecDataCount:
			dcIdx = i
		case wasmbin.SecCode:
			codeIdx = i
		}
	}
	if memIdx < 0 || codeIdx < 0 {
		return nil, man, fmt.Errorf("wideembed: host has no memory or code section")
	}
	lim, err := wasmbin.ParseMemory(secs[memIdx].Payload(host))
	if err != nil {
		return nil, man, fmt.Errorf("wideembed: host memory section: %w", err)
	}

	hi, err := probeHost(ctx, host)
	if err != nil {
		return nil, man, err
	}
	if hi.minPages != lim.Min {
		return nil, man, fmt.Errorf("wideembed: host instantiated with %d pages but declares min %d", hi.minPages, lim.Min)
	}
	if uint64(hi.descAddr)+wasmrt.EmbedSlotSize > uint64(lim.Min)*pageSize {
		return nil, man, fmt.Errorf("wideembed: embed slot at %d lies outside the host's initial memory", hi.descAddr)
	}
	man.HostMinPages, man.DescAddr = lim.Min, hi.descAddr

	modelPtr := uint64(lim.Min) * pageSize
	budget, err := CheckBudget(modelPtr, uint64(len(imwq)), info.ParsedBytes,
		info.ActivationBytes(uint64(opt.MaxTokens)), opt.MarginBytes, opt.LimitBytes)
	man.Budget = budget
	if err != nil {
		return nil, man, err
	}
	newMin := uint32((budget.ModelEnd + pageSize - 1) / pageSize)
	if lim.HasMax && lim.Max < newMin {
		return nil, man, fmt.Errorf("wideembed: host declares max %d pages, the model needs a minimum of %d", lim.Max, newMin)
	}
	man.MinPages = newMin

	sum := sha256.Sum256(imwq)
	slot := wasmrt.EmbedSlot{
		Magic: wasmrt.EmbedFilledMagic, Version: wasmrt.EmbedSlotVersion,
		Ptr: uint32(modelPtr), Len: uint32(len(imwq)), SHA256: sum,
	}

	// New data section: original segments verbatim, then slot, then model.
	var newData []byte
	if dataIdx >= 0 {
		p := secs[dataIdx].Payload(host)
		n, k, err := wasmbin.ReadU32(p, 0)
		if err != nil {
			return nil, man, fmt.Errorf("wideembed: host data section: %w", err)
		}
		newData = make([]byte, 0, len(p)+len(imwq)+128)
		newData = wasmbin.AppendU32(newData, n+2)
		newData = append(newData, p[k:]...)
	} else {
		newData = make([]byte, 0, len(imwq)+128)
		newData = wasmbin.AppendU32(newData, 2)
	}
	newData = append(newData, wasmbin.ActiveDataSegment(hi.descAddr, slot.Encode())...)
	newData = append(newData, wasmbin.ActiveDataSegment(uint32(modelPtr), imwq)...)

	newLim := lim
	newLim.Min = newMin

	out := make([]byte, 0, len(host)+len(newData)+64)
	out = append(out, wasmbin.Magic...)
	for i, s := range secs {
		switch i {
		case memIdx:
			out = wasmbin.AppendSection(out, wasmbin.SecMemory, wasmbin.EncodeMemory(newLim))
		case dcIdx:
			n, _, err := wasmbin.ReadU32(s.Payload(host), 0)
			if err != nil {
				return nil, man, fmt.Errorf("wideembed: host datacount section: %w", err)
			}
			out = wasmbin.AppendSection(out, wasmbin.SecDataCount, wasmbin.AppendU32(nil, n+2))
		case dataIdx:
			out = wasmbin.AppendSection(out, wasmbin.SecData, newData)
		default:
			out = append(out, host[s.Start:s.End]...)
		}
		if i == codeIdx && dataIdx < 0 {
			out = wasmbin.AppendSection(out, wasmbin.SecData, newData)
		}
	}

	// Version metadata, last so no section index above shifts. It sits inside
	// the module, so the registered hash covers it.
	meta := wasmrt.AdapterMeta{ABIVersion: wasmrt.AdapterABIVersion, Class: string(class), Tool: "wideembed", Training: opt.Training}
	out = wasmbin.AppendCustomSection(out, wasmrt.AdapterMetaSection, meta.Encode())
	man.ABIVersion, man.Training = meta.ABIVersion, meta.Training

	hostSum := sha256.Sum256(host)
	bundleSum := sha256.Sum256(out)
	man.HostSHA256 = hex.EncodeToString(hostSum[:])
	man.IMWQSHA256 = hex.EncodeToString(sum[:])
	man.BundleSHA256 = hex.EncodeToString(bundleSum[:])
	man.BundleKeccak256 = "0x" + hex.EncodeToString(crypto.Keccak256(out))
	man.BundleBytes = len(out)

	if !opt.SkipVerify {
		mem, err := Verify(ctx, out, class, uint32(modelPtr), uint32(len(imwq)))
		if err != nil {
			return nil, man, fmt.Errorf("wideembed: the finished package failed verification: %w", err)
		}
		man.Verified, man.LoadedMemoryBytes = true, mem
	}
	return out, man, nil
}

// Verify loads a finished package through the node's own embedded constructor
// for its class and checks it adopted the model at (wantPtr, wantLen). Returns
// linear memory after the load.
func Verify(ctx context.Context, pkg []byte, class Class, wantPtr, wantLen uint32) (uint64, error) {
	var rt *wasmrt.WideRuntime
	switch class {
	case ClassHybrid:
		h, err := wasmrt.NewWideHybridEmbedded(ctx, pkg)
		if err != nil {
			return 0, err
		}
		rt = h.WideRuntime
	default:
		r, err := wasmrt.NewWideEmbedded(ctx, pkg)
		if err != nil {
			return 0, err
		}
		rt = r
	}
	defer rt.Close(context.Background()) //nolint:errcheck
	if p, l := rt.ResidentModel(); p != wantPtr || l != wantLen {
		return 0, fmt.Errorf("package adopted model at (%d, %d), packaged at (%d, %d)", p, l, wantPtr, wantLen)
	}
	return rt.MemoryBytes(), nil
}

// checkHostExports enforces the host/class contract before anything is built.
func checkHostExports(host []byte, class Class) error {
	names, err := wasmbin.ExportNames(host)
	if err != nil {
		return fmt.Errorf("wideembed: reading host exports: %w", err)
	}
	for _, want := range []string{expEmbedded, expEmbedDescAddr, expScore, expDebugCounts, expAlloc, expFree} {
		if k, ok := names[want]; !ok || k != wasmbin.ExportFunc {
			return fmt.Errorf("wideembed: host does not export %q — it is not a wide_wasm embed host "+
				"(build with --features embedded or embedded-hybrid)", want)
		}
	}
	if _, ok := names[expDeterministic]; ok {
		return fmt.Errorf("wideembed: host exports %q — that is the shared comparator/interpreter build, not an embed host", expDeterministic)
	}
	_, hasHybrid := names[expScoreHybrid]
	switch {
	case class == ClassHybrid && !hasHybrid:
		return fmt.Errorf("wideembed: class hybrid needs a host built with --features embedded-hybrid (no %q export)", expScoreHybrid)
	case class == ClassJudgment && hasHybrid:
		return fmt.Errorf("wideembed: class judgment needs a host built with --features embedded; this host exports %q", expScoreHybrid)
	}
	return nil
}

// probeHost instantiates the bare host to learn the slot address and initial
// memory, and proves the slot still holds the unfilled sentinel — so the
// packager can never write its segment over the wrong 56 bytes, and never
// re-package an already-packaged module.
func probeHost(ctx context.Context, host []byte) (hostInfo, error) {
	wrt := wazero.NewRuntime(ctx)
	defer wrt.Close(ctx) //nolint:errcheck
	mod, err := wrt.Instantiate(ctx, host)
	if err != nil {
		return hostInfo{}, fmt.Errorf("wideembed: instantiating host: %w", err)
	}
	fn := mod.ExportedFunction(expEmbedDescAddr)
	res, err := fn.Call(ctx)
	if err != nil || len(res) == 0 {
		return hostInfo{}, fmt.Errorf("wideembed: %s: %v", expEmbedDescAddr, err)
	}
	addr := uint32(res[0])
	mem := mod.Memory()
	raw, ok := mem.Read(addr, wasmrt.EmbedSlotSize)
	if !ok {
		return hostInfo{}, fmt.Errorf("wideembed: host embed slot at %d is outside memory", addr)
	}
	slot, err := wasmrt.ParseEmbedSlot(raw)
	if err != nil {
		return hostInfo{}, err
	}
	switch slot.Magic {
	case wasmrt.EmbedHostMagic:
	case wasmrt.EmbedFilledMagic:
		return hostInfo{}, fmt.Errorf("wideembed: the host is already a packaged module (slot filled); package from the bare embed host")
	default:
		return hostInfo{}, fmt.Errorf("wideembed: bytes at the reported slot address %d are %q, not the IMWQHOST sentinel — "+
			"refusing to write over unknown data", addr, slot.Magic[:])
	}
	return hostInfo{descAddr: addr, minPages: mem.Size() / pageSize}, nil
}
