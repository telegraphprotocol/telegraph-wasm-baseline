package runtime

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// A LoRA delta package is the Evaluator adapter for the shared Foundation
// Model (abi "lora-v1"): a standard safetensors file (the container PEFT
// writes) holding the LoRA A/B matrices of every target linear and the reward
// head, never base weights. Layout: u64 LE header length | JSON header | raw
// tensor bytes; the header maps tensor names to {dtype, shape, data_offsets}
// and carries the package fields in its "__metadata__" string map. The
// tensors must tile the data exactly (contiguous by offset, nothing left).
// Its identity is its keccak256, like any registered artifact (wasm_hash).
// Writer: the kit's float_runtime/export_adapter.py; the engine re-checks it
// in the guest.
//
// Metadata (every value a string): format_version "1", abi "lora-v1", class,
// intent, foundation_hash, foundation_version, tokenizer_hash, rank, alpha,
// target_modules (a JSON array string), max_tokens, optional training (a JSON
// string), optional "x-*" extension keys (ignored).
//
// Tensors (dtype F32), for a foundation with L layers, hidden size H and rank
// r <= R_MAX:
//
//	lora.<l>.<target>.A   [in, r]      target in LoRATargets, l in 0..L-1
//	lora.<l>.<target>.B   [r, out]     (pre-transposed: x.A.B needs no Transpose)
//	head.weight           [H, 1]
//	head.bias             [1]
const (
	LoRAFormatVersion = 1
	LoRAABI           = "lora-v1"
	// LoRAFileExt is the extension a delta package is stored and served under.
	LoRAFileExt = ".safetensors"
	// LoRADType is the only tensor dtype a package may carry.
	LoRADType = "F32"
	// LoRARMax is the largest rank this node's foundation graphs hold.
	LoRARMax = 16
	// LoRADefaultMaxBytes caps a package (ADAPTER_LORA_MAX_BYTES overrides it
	// at fetch time). A real ModernBERT-base r=8 delta is 6.8 MB.
	LoRADefaultMaxBytes int64 = 64 << 20
	loraPreamble              = 8
	loraMaxHeaderBytes        = 1 << 20
)

// LoRATargetModules are the PEFT module names an adapter may train (the
// metadata's target_modules). "Wo" names both attn.Wo and mlp.Wo, as PEFT
// matches by suffix.
var LoRATargetModules = []string{"Wi", "Wo", "Wqkv"}

// LoRATargets are the target linears of every layer, as the graph names them.
var LoRATargets = []string{"attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"}

// Rejection reasons a delta package can fail with (Stage 1 and load).
const (
	LoRAReasonBadContainer       = "bad_container"
	LoRAReasonMissingMetadata    = "missing_metadata"
	LoRAReasonBadVersion         = "bad_version"
	LoRAReasonBadHeader          = "bad_header"
	LoRAReasonBadABI             = "bad_abi"
	LoRAReasonFoundationMismatch = "foundation_mismatch"
	LoRAReasonTokenizerMismatch  = "tokenizer_mismatch"
	LoRAReasonClassMismatch      = "class_mismatch"
	LoRAReasonRankTooLarge       = "rank_too_large"
	LoRAReasonBadTargetModule    = "bad_target_module"
	LoRAReasonBadDType           = "bad_dtype"
	LoRAReasonBadShape           = "bad_shape"
	LoRAReasonBadOffsets         = "bad_offsets"
	LoRAReasonTrailingBytes      = "trailing_bytes"
	LoRAReasonTruncated          = "truncated"
	LoRAReasonTooLarge           = "too_large"
	LoRAReasonNonFinite          = "non_finite"
	LoRAReasonBadMaxTokens       = "bad_max_tokens"
	LoRAReasonTrainingRecord     = "training_record"
	// LoRAReasonNoFoundation is a delta package met by a node that runs no
	// Foundation Model (FOUNDATION_MODEL_PATH unset).
	LoRAReasonNoFoundation = "no_foundation"
)

// LoRARejectError is a delta package refused with a machine-readable reason.
type LoRARejectError struct {
	Reason string
	Detail string
}

func (e *LoRARejectError) Error() string { return e.Reason + ": " + e.Detail }

// LoRARejectReason returns the reason of a *LoRARejectError in err's chain, or "".
func LoRARejectReason(err error) string {
	var re *LoRARejectError
	if errors.As(err, &re) {
		return re.Reason
	}
	return ""
}

func loraReject(reason, format string, a ...any) error {
	return &LoRARejectError{Reason: reason, Detail: fmt.Sprintf(format, a...)}
}

// LoRATensor is one tensor of the package: a safetensors header entry.
type LoRATensor struct {
	Name   string
	DType  string
	Shape  []int
	Offset int64 // data_offsets[0], relative to the start of the tensor data
	Length int64 // data_offsets[1] - data_offsets[0]
}

// LoRAHeader is the package's metadata plus its tensor table (sorted by offset).
type LoRAHeader struct {
	FormatVersion     int
	ABI               string
	Class             string
	Intent            string
	FoundationHash    string
	FoundationVersion string
	TokenizerHash     string
	Rank              int
	Alpha             float64
	TargetModules     []string
	MaxTokens         int
	Training          *TrainingRecord
	Tensors           []LoRATensor
	// Metadata is the raw __metadata__ map (extension keys included).
	Metadata map[string]string
}

// FoundationSpec is the LoRA input layout of a foundation graph, as the engine
// reports it after load_foundation (and as export_foundation.py writes it).
type FoundationSpec struct {
	ABI        int               `json:"abi,omitempty"`
	RMax       int               `json:"r_max"`
	HiddenSize int               `json:"hidden_size"`
	NumLayers  int               `json:"num_layers"`
	MaxTokens  int               `json:"max_tokens,omitempty"`
	Targets    map[string][2]int `json:"targets"` // target -> [in, out]
}

// Check reports why s is not a usable layout.
func (s FoundationSpec) Check() error {
	if s.RMax <= 0 || s.RMax > LoRARMax || s.HiddenSize <= 0 || s.NumLayers <= 0 {
		return fmt.Errorf("foundation spec: r_max %d (node max %d), hidden %d, layers %d", s.RMax, LoRARMax, s.HiddenSize, s.NumLayers)
	}
	for _, t := range LoRATargets {
		d, ok := s.Targets[t]
		if !ok || d[0] <= 0 || d[1] <= 0 {
			return fmt.Errorf("foundation spec: target %s missing or empty", t)
		}
	}
	return nil
}

// LoRAExpect is what a package must match on this node.
type LoRAExpect struct {
	FoundationHash string
	TokenizerHash  string
	// Class is the intent's class ("judgment" / "hybrid"); "" accepts either.
	Class    string
	Spec     FoundationSpec
	MaxBytes int64
}

// LoRAAdapter is a parsed, validated package. Raw is the whole file (what the
// engine is handed); DeltaBytes is the tensor data size.
type LoRAAdapter struct {
	Header     LoRAHeader
	Raw        []byte
	DeltaBytes int64
}

// Meta is the adapter's metadata in the AdapterMeta shape the scorers read
// (class, token cap, training record).
func (a *LoRAAdapter) Meta() AdapterMeta {
	if a == nil {
		return AdapterMeta{}
	}
	return AdapterMeta{
		ABIVersion: AdapterABIVersion, Class: a.Header.Class, Tool: "lora",
		Runtime: LoRAABI, MaxTokens: a.Header.MaxTokens, Training: a.Header.Training,
	}
}

// safetensorsEntry is one tensor entry of the safetensors header.
type safetensorsEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// readSafetensorsHeader decodes the container: the header length, the JSON
// header object, the __metadata__ map and the tensor entries (sorted by
// offset). It needs only the header bytes present (b may be the head of a
// longer file).
func readSafetensorsHeader(b []byte) (meta map[string]string, tensors []LoRATensor, dataStart int, err error) {
	if len(b) < loraPreamble+1 {
		return nil, nil, 0, loraReject(LoRAReasonBadContainer, "%d bytes, shorter than a safetensors header", len(b))
	}
	hl := binary.LittleEndian.Uint64(b[:loraPreamble])
	if hl == 0 || hl > loraMaxHeaderBytes {
		return nil, nil, 0, loraReject(LoRAReasonBadContainer, "safetensors header length %d", hl)
	}
	if uint64(len(b)) < loraPreamble+hl {
		return nil, nil, 0, loraReject(LoRAReasonTruncated, "safetensors header runs past the end of the file")
	}
	raw := b[loraPreamble : loraPreamble+int(hl)]
	if raw[0] != '{' {
		return nil, nil, 0, loraReject(LoRAReasonBadContainer, "not a safetensors file (the header does not start with '{')")
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, nil, 0, loraReject(LoRAReasonBadContainer, "safetensors header is not a JSON object: %v", err)
	}
	for name, v := range entries {
		if name == "__metadata__" {
			if err := json.Unmarshal(v, &meta); err != nil {
				return nil, nil, 0, loraReject(LoRAReasonMissingMetadata, "__metadata__ is not a string map: %v", err)
			}
			continue
		}
		var e safetensorsEntry
		if err := json.Unmarshal(v, &e); err != nil {
			return nil, nil, 0, loraReject(LoRAReasonBadHeader, "tensor %s: %v", name, err)
		}
		tensors = append(tensors, LoRATensor{Name: name, DType: e.DType, Shape: e.Shape, Offset: e.DataOffsets[0], Length: e.DataOffsets[1] - e.DataOffsets[0]})
	}
	sort.Slice(tensors, func(i, j int) bool {
		if tensors[i].Offset != tensors[j].Offset {
			return tensors[i].Offset < tensors[j].Offset
		}
		return tensors[i].Name < tensors[j].Name
	})
	if meta == nil {
		return nil, nil, 0, loraReject(LoRAReasonMissingMetadata, "no __metadata__ in the safetensors header")
	}
	return meta, tensors, loraPreamble + int(hl), nil
}

// IsLoRAPackage reports whether b is (the head of) a LoRA delta package: a
// safetensors file whose __metadata__ declares a lora-* abi. It parses the
// header rather than trusting a magic, and needs only the header present.
func IsLoRAPackage(b []byte) bool {
	meta, _, _, err := readSafetensorsHeader(b)
	return err == nil && strings.HasPrefix(meta["abi"], "lora-")
}

// ParseLoRAHeader reads the container and the metadata (no tensor checks).
// It returns the header and the offset of the tensor data.
func ParseLoRAHeader(b []byte) (*LoRAHeader, int, error) {
	meta, tensors, dataStart, err := readSafetensorsHeader(b)
	if err != nil {
		return nil, 0, err
	}
	h := &LoRAHeader{Tensors: tensors, Metadata: meta}
	var missing error
	get := func(key string) string {
		v, ok := meta[key]
		if !ok && missing == nil {
			missing = loraReject(LoRAReasonMissingMetadata, "__metadata__ has no %s", key)
		}
		return v
	}
	ver, abi := get("format_version"), get("abi")
	h.Class, h.Intent = get("class"), get("intent")
	h.FoundationHash, h.FoundationVersion, h.TokenizerHash = get("foundation_hash"), get("foundation_version"), get("tokenizer_hash")
	rank, alpha, mods, maxTok := get("rank"), get("alpha"), get("target_modules"), get("max_tokens")
	if missing != nil {
		return nil, 0, missing
	}
	h.ABI = abi
	if h.FormatVersion, err = strconv.Atoi(ver); err != nil {
		return nil, 0, loraReject(LoRAReasonBadVersion, "format_version %q is not a number", ver)
	}
	if h.Rank, err = strconv.Atoi(rank); err != nil {
		return nil, 0, loraReject(LoRAReasonBadHeader, "rank %q is not a number", rank)
	}
	if h.Alpha, err = strconv.ParseFloat(alpha, 64); err != nil {
		return nil, 0, loraReject(LoRAReasonBadHeader, "alpha %q is not a number", alpha)
	}
	if err := json.Unmarshal([]byte(mods), &h.TargetModules); err != nil {
		return nil, 0, loraReject(LoRAReasonBadTargetModule, "target_modules %q is not a JSON array of strings", mods)
	}
	if h.MaxTokens, err = strconv.Atoi(maxTok); err != nil {
		return nil, 0, loraReject(LoRAReasonBadMaxTokens, "max_tokens %q is not a number", maxTok)
	}
	if tr, ok := meta["training"]; ok {
		var rec TrainingRecord
		if err := json.Unmarshal([]byte(tr), &rec); err != nil {
			return nil, 0, loraReject(LoRAReasonTrainingRecord, "%v", err)
		}
		h.Training = &rec
	}
	for k := range meta {
		switch k {
		case "format_version", "abi", "class", "intent", "foundation_hash", "foundation_version",
			"tokenizer_hash", "rank", "alpha", "target_modules", "max_tokens", "training":
		default:
			if !strings.HasPrefix(k, "x-") {
				return nil, 0, loraReject(LoRAReasonBadHeader, "unknown metadata key %q (extension keys start with x-)", k)
			}
		}
	}
	return h, dataStart, nil
}

// ParseLoRAAdapter parses b and checks every rule of the format against want.
// Any mismatch is a *LoRARejectError; there is no fallback.
func ParseLoRAAdapter(b []byte, want LoRAExpect) (*LoRAAdapter, error) {
	maxBytes := want.MaxBytes
	if maxBytes <= 0 {
		maxBytes = LoRADefaultMaxBytes
	}
	if int64(len(b)) > maxBytes {
		return nil, loraReject(LoRAReasonTooLarge, "%d bytes, above the %d byte cap", len(b), maxBytes)
	}
	h, dataStart, err := ParseLoRAHeader(b)
	if err != nil {
		return nil, err
	}
	if err := checkLoRAHeader(h, want); err != nil {
		return nil, err
	}
	data := b[dataStart:]
	if err := checkLoRATensors(h, want.Spec, data); err != nil {
		return nil, err
	}
	return &LoRAAdapter{Header: *h, Raw: b, DeltaBytes: int64(len(data))}, nil
}

func checkLoRAHeader(h *LoRAHeader, want LoRAExpect) error {
	if h.FormatVersion != LoRAFormatVersion {
		return loraReject(LoRAReasonBadVersion, "format_version %d, this node reads %d", h.FormatVersion, LoRAFormatVersion)
	}
	if h.ABI != LoRAABI {
		return loraReject(LoRAReasonBadABI, "abi %q, this node runs %q", h.ABI, LoRAABI)
	}
	if want.FoundationHash == "" {
		return loraReject(LoRAReasonNoFoundation, "this node runs no Foundation Model (FOUNDATION_MODEL_PATH unset)")
	}
	if !strings.EqualFold(h.FoundationHash, want.FoundationHash) {
		return loraReject(LoRAReasonFoundationMismatch, "built for foundation %s (%s), this node runs %s",
			short(h.FoundationHash), h.FoundationVersion, short(want.FoundationHash))
	}
	if want.TokenizerHash == "" || !strings.EqualFold(h.TokenizerHash, want.TokenizerHash) {
		return loraReject(LoRAReasonTokenizerMismatch, "built for tokenizer %s, this node uses %s",
			short(h.TokenizerHash), short(want.TokenizerHash))
	}
	switch h.Class {
	case ClassJudgment, ClassHybrid:
	default:
		return loraReject(LoRAReasonClassMismatch, "class %q; an Evaluator adapter is %q or %q", h.Class, ClassJudgment, ClassHybrid)
	}
	if want.Class != "" && h.Class != want.Class {
		return loraReject(LoRAReasonClassMismatch, "class %q, the intent needs %q", h.Class, want.Class)
	}
	if h.Intent == "" {
		return loraReject(LoRAReasonBadHeader, "intent is empty")
	}
	if err := want.Spec.Check(); err != nil {
		return fmt.Errorf("wasm/runtime: %w", err)
	}
	if h.Rank <= 0 || h.Rank > want.Spec.RMax {
		return loraReject(LoRAReasonRankTooLarge, "rank %d, the foundation takes 1..%d", h.Rank, want.Spec.RMax)
	}
	if !(h.Alpha > 0) || math.IsInf(h.Alpha, 0) {
		return loraReject(LoRAReasonBadHeader, "alpha %v is not a positive finite number", h.Alpha)
	}
	if len(h.TargetModules) == 0 {
		return loraReject(LoRAReasonBadTargetModule, "target_modules is empty")
	}
	seen := map[string]bool{}
	for _, m := range h.TargetModules {
		i := sort.SearchStrings(LoRATargetModules, m)
		if i >= len(LoRATargetModules) || LoRATargetModules[i] != m || seen[m] {
			return loraReject(LoRAReasonBadTargetModule, "target module %q (allowed, once each: %s)", m, strings.Join(LoRATargetModules, ", "))
		}
		seen[m] = true
	}
	if h.MaxTokens < 1 || h.MaxTokens > MaxFloatMaxTokens {
		return loraReject(LoRAReasonBadMaxTokens, "max_tokens %d is outside 1..%d", h.MaxTokens, MaxFloatMaxTokens)
	}
	if err := h.Training.Validate(); err != nil {
		return loraReject(LoRAReasonTrainingRecord, "%v", err)
	}
	return nil
}

// LoRAExpectedShapes is the exact tensor set a package of rank r must carry
// for spec, in canonical (table) order.
func LoRAExpectedShapes(rank int, s FoundationSpec) ([]string, map[string][]int) {
	names := make([]string, 0, 2*len(LoRATargets)*s.NumLayers+2)
	shapes := make(map[string][]int, 2*len(LoRATargets)*s.NumLayers+2)
	for l := 0; l < s.NumLayers; l++ {
		for _, t := range LoRATargets {
			d := s.Targets[t]
			a, b := fmt.Sprintf("lora.%d.%s.A", l, t), fmt.Sprintf("lora.%d.%s.B", l, t)
			names = append(names, a, b)
			shapes[a] = []int{d[0], rank}
			shapes[b] = []int{rank, d[1]}
		}
	}
	names = append(names, "head.weight", "head.bias")
	shapes["head.weight"] = []int{s.HiddenSize, 1}
	shapes["head.bias"] = []int{1}
	return names, shapes
}

func checkLoRATensors(h *LoRAHeader, s FoundationSpec, data []byte) error {
	_, want := LoRAExpectedShapes(h.Rank, s)
	if len(h.Tensors) != len(want) {
		return loraReject(LoRAReasonBadShape, "%d tensors, the foundation needs %d", len(h.Tensors), len(want))
	}
	seen := make(map[string]bool, len(want))
	var off int64
	for _, t := range h.Tensors {
		w, ok := want[t.Name]
		if !ok || seen[t.Name] {
			return loraReject(LoRAReasonBadShape, "unexpected or repeated tensor %q", t.Name)
		}
		seen[t.Name] = true
		if t.DType != LoRADType {
			return loraReject(LoRAReasonBadDType, "tensor %s has dtype %q, want %s", t.Name, t.DType, LoRADType)
		}
		if !equalShape(t.Shape, w) {
			return loraReject(LoRAReasonBadShape, "tensor %s shape %v, want %v", t.Name, t.Shape, w)
		}
		n := int64(4)
		for _, d := range t.Shape {
			n *= int64(d)
		}
		if t.Length != n {
			return loraReject(LoRAReasonBadOffsets, "tensor %s data_offsets span %d bytes, its shape needs %d", t.Name, t.Length, n)
		}
		if t.Offset != off {
			return loraReject(LoRAReasonBadOffsets, "tensor %s at offset %d, expected %d (tensors must tile the data with no gap or overlap)", t.Name, t.Offset, off)
		}
		if t.Offset+t.Length > int64(len(data)) {
			return loraReject(LoRAReasonTruncated, "tensor %s ends at %d, the data is %d bytes", t.Name, t.Offset+t.Length, len(data))
		}
		off += t.Length
	}
	if off != int64(len(data)) {
		return loraReject(LoRAReasonTrailingBytes, "%d bytes after the last tensor", int64(len(data))-off)
	}
	for i := 0; i+4 <= len(data); i += 4 {
		f := math.Float32frombits(binary.LittleEndian.Uint32(data[i:]))
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return loraReject(LoRAReasonNonFinite, "non-finite value at data offset %d", i)
		}
	}
	return nil
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func short(h string) string {
	if h == "" {
		return "(none)"
	}
	if len(h) > 12 {
		return h[:12] + "…"
	}
	return h
}

// ScoreIdentity is what a model score is a function of besides the input:
// the Foundation Model (F), the Evaluator adapter (A) and the tokenizer (T).
// Recorded wherever a score or verdict is persisted.
type ScoreIdentity struct {
	FoundationHash    string `json:"foundation_hash,omitempty"`
	FoundationVersion string `json:"foundation_version,omitempty"`
	EngineHash        string `json:"engine_hash,omitempty"`
	AdapterHash       string `json:"adapter_hash,omitempty"`
	TokenizerHash     string `json:"tokenizer_hash,omitempty"`
	// The pinned execution contract: the engine ABI, the arithmetic precision
	// (f32, no fused multiply-add, the graph's own operation order, one
	// thread) and the token limit the input was truncated to.
	ABI       int    `json:"abi,omitempty"`
	Precision string `json:"precision,omitempty"`
	MaxTokens int    `json:"max_tokens,omitempty"`
}

// ScorePrecision names the pinned arithmetic of every model score: IEEE
// binary32 in WebAssembly (no FMA, no relaxed SIMD), the graph's operation
// order, single-threaded.
const ScorePrecision = "wasm-f32"

// InputHash is the sha256 of the exact input the guest sees for ids (the
// legacy input JSON encoding, so legacy and foundation scores hash alike).
func InputHash(ids []int64) string {
	sum := sha256.Sum256(EncodeFloatInput(ids))
	return hex.EncodeToString(sum[:])
}
