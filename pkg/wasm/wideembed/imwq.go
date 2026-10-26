package wideembed

import (
	"encoding/binary"
	"fmt"
)

// Opcodes the activation estimate needs (format.py Op / wide_wasm interp.rs).
const (
	opEmbed   = 1
	opQMatmul = 2
	opScore   = 13
)

// opBytes is the on-disk size of one BOPS op record (format.py _pack_binary).
const opBytes = 60

// IMWQInfo is what the packager learns from walking a .imwq's framing.
type IMWQInfo struct {
	Ops     int
	Tensors int
	// Config is the 10-int config block: arch, hidden, layers, heads, kv_heads,
	// head_dim, inter, max_seq, rope_theta_milli, rope.
	Config [10]int32
	// TensorBytes is Σ n·8 over every blob — the weight payload.
	TensorBytes uint64
	// ParsedBytes estimates what WideModel::parse allocates inside the module on
	// every parse (it COPIES every tensor into a Vec<i64>): tensor data plus the
	// op vector plus per-tensor bookkeeping. Rounded up, never down.
	ParsedBytes uint64
	// Slots is the number of distinct op output slots — the interpreter keeps
	// one live buffer per slot id for the whole run.
	Slots int
	// MaxCols is the widest activation row any op can produce (EMBED hidden or a
	// QMATMUL/SCORE output dimension). Other ops never widen a row.
	MaxCols uint64
}

// MaxSeq returns the config's max_seq (the RoPE table length).
func (i IMWQInfo) MaxSeq() uint64 { return uint64(max(i.Config[7], 0)) }

// HeadDim returns the config's head_dim.
func (i IMWQInfo) HeadDim() uint64 { return uint64(max(i.Config[5], 0)) }

// ActivationBytes is a conservative upper bound on the interpreter's activation
// peak for a sequence of seq tokens: every slot alive at the widest row, plus
// the attention op's temporaries (clones of q/k/v, its output and per-head
// buffers), plus the RoPE cos/sin tables built at max_seq. Over-estimating only
// makes the 4 GiB gate stricter; under-estimating would let a package through
// that later traps with an out-of-memory at score time.
func (i IMWQInfo) ActivationBytes(seq uint64) uint64 {
	row := seq * i.MaxCols * 8
	live := uint64(i.Slots+8) * row
	rope := 2 * i.MaxSeq() * i.HeadDim() * 8
	return live + rope
}

// ParseIMWQ walks a .imwq's framing exactly as format.py writes it:
//
//	'IMWQ' | mjson_len:u32 | mjson | 'BOPS' | num_ops:u32 | ops[num_ops]×60 |
//	config 10×i32 | num_blobs:u32 | per blob: ndim:u32, dims[ndim]:u64, n:u64, i64[n]
//
// It must end EXACTLY at len(b): a truncated (or padded) artifact is a build
// error here, not an i64::MIN sentinel at score time. Weights are not read —
// only lengths — so this is fast on a 1.2 GB file.
func ParseIMWQ(b []byte) (IMWQInfo, error) {
	var info IMWQInfo
	off := 0
	need := func(n uint64, what string) error {
		if uint64(off)+n > uint64(len(b)) {
			return fmt.Errorf("wideembed: .imwq truncated reading %s at offset %d (need %d bytes, file is %d)", what, off, n, len(b))
		}
		return nil
	}
	u32 := func(what string) (uint32, error) {
		if err := need(4, what); err != nil {
			return 0, err
		}
		v := binary.LittleEndian.Uint32(b[off:])
		off += 4
		return v, nil
	}
	u64 := func(what string) (uint64, error) {
		if err := need(8, what); err != nil {
			return 0, err
		}
		v := binary.LittleEndian.Uint64(b[off:])
		off += 8
		return v, nil
	}

	if len(b) < 4 || string(b[:4]) != "IMWQ" {
		return info, fmt.Errorf("wideembed: not a .imwq (missing IMWQ magic)")
	}
	off = 4
	mlen, err := u32("mjson_len")
	if err != nil {
		return info, err
	}
	if err := need(uint64(mlen)+4, "mjson + BOPS tag"); err != nil {
		return info, err
	}
	off += int(mlen)
	if string(b[off:off+4]) != "BOPS" {
		return info, fmt.Errorf("wideembed: .imwq has no BOPS block after the JSON meta (offset %d)", off)
	}
	off += 4
	nOps, err := u32("num_ops")
	if err != nil {
		return info, err
	}
	if err := need(uint64(nOps)*opBytes+40, "ops + config"); err != nil {
		return info, err
	}
	info.Ops = int(nOps)

	// Pass 1 over ops: remember output slots and which weights produce rows.
	slots := map[int32]struct{}{}
	type wref struct {
		op  uint16
		idx int32
	}
	var widthRefs []wref
	for i := 0; i < int(nOps); i++ {
		rec := b[off : off+opBytes]
		op := binary.LittleEndian.Uint16(rec[0:])
		out := int32(binary.LittleEndian.Uint32(rec[12:]))
		widx := int32(binary.LittleEndian.Uint32(rec[16:]))
		slots[out] = struct{}{}
		if op == opEmbed || op == opQMatmul || op == opScore {
			widthRefs = append(widthRefs, wref{op, widx})
		}
		off += opBytes
	}
	info.Slots = len(slots)
	for i := range info.Config {
		info.Config[i] = int32(binary.LittleEndian.Uint32(b[off:]))
		off += 4
	}

	nb, err := u32("num_blobs")
	if err != nil {
		return info, err
	}
	info.Tensors = int(nb)
	dims := make([][]uint64, nb)
	for t := 0; t < int(nb); t++ {
		ndim, err := u32("blob ndim")
		if err != nil {
			return info, err
		}
		if ndim > 8 {
			return info, fmt.Errorf("wideembed: blob %d claims %d dims — not a .imwq this converter wrote", t, ndim)
		}
		d := make([]uint64, ndim)
		for k := range d {
			if d[k], err = u64("blob dim"); err != nil {
				return info, err
			}
		}
		dims[t] = d
		n, err := u64("blob n")
		if err != nil {
			return info, err
		}
		if n > uint64(len(b))/8 {
			return info, fmt.Errorf("wideembed: blob %d claims %d elements, more than the file holds", t, n)
		}
		if err := need(n*8, "blob data"); err != nil {
			return info, err
		}
		off += int(n * 8)
		info.TensorBytes += n * 8
	}
	if off != len(b) {
		return info, fmt.Errorf("wideembed: .imwq has %d trailing bytes after the last blob — not what the converter writes", len(b)-off)
	}

	for _, r := range widthRefs {
		if r.idx < 0 || int(r.idx) >= len(dims) || len(dims[r.idx]) == 0 {
			return info, fmt.Errorf("wideembed: op %d references weight %d, which does not exist", r.op, r.idx)
		}
		d := dims[r.idx]
		var cols uint64
		if r.op == opEmbed {
			if len(d) == 2 {
				cols = d[1]
			} else {
				cols = 1
			}
		} else {
			cols = d[0]
		}
		info.MaxCols = max(info.MaxCols, cols)
	}
	if h := uint64(max(info.Config[1], 0)); h > info.MaxCols {
		info.MaxCols = h
	}

	// Rust WideModel: Vec<WideOp> (64 B each after padding) + per tensor a
	// Tensor{dims: Vec<u64>, data: Vec<i64>} (48 B + dims) + the data itself.
	info.ParsedBytes = info.TensorBytes + uint64(nOps)*64 + uint64(nb)*(48+8*8)
	return info, nil
}
