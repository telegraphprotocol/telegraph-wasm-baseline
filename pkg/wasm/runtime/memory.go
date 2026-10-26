package runtime

import (
	"context"
	"fmt"
	"math"

	"github.com/tetratelabs/wazero/api"
)

// writeStr allocates WASM heap memory via the module's alloc export, copies
// the string bytes into it, and returns:
//   - ptr  : byte offset in WASM linear memory
//   - size : byte length of the string
//   - free : a deferred cleanup function the caller must invoke
//
// Pattern:
//
//	ptr, size, free, err := r.writeStr(ctx, s)
//	if err != nil { ... }
//	defer free(ctx)
func (r *Runtime) writeStr(ctx context.Context, s string) (ptr, size uint32, free func(context.Context), err error) {
	if len(s) == 0 {
		// Zero-length string — return a null ptr, no-op free
		return 0, 0, func(context.Context) {}, nil
	}

	sz := uint32(len(s))

	res, callErr := r.fnAlloc.Call(ctx, uint64(sz))
	if callErr != nil {
		return 0, 0, nil, fmt.Errorf("alloc(%d): %w", sz, callErr)
	}
	if len(res) == 0 {
		return 0, 0, nil, fmt.Errorf("alloc returned no pointer")
	}

	p := uint32(res[0])
	if !r.mem.Write(p, []byte(s)) {
		// If write fails the memory may have moved — free what we can
		r.fnDealloc.Call(ctx, uint64(p), uint64(sz)) //nolint:errcheck
		return 0, 0, nil, fmt.Errorf("mem.Write at ptr=%d len=%d failed", p, sz)
	}

	freeFn := func(ctx context.Context) {
		r.fnDealloc.Call(ctx, uint64(p), uint64(sz)) //nolint:errcheck
	}

	return p, sz, freeFn, nil
}

// growIfNeeded grows WASM linear memory so it is at least `needed` bytes.
// wazero memories start at 1 page (64 KiB); for large vector inputs they
// may need to grow.
func (r *Runtime) growIfNeeded(needed uint32) error {
	if r.mem.Size() >= needed {
		return nil
	}
	pages := (needed-r.mem.Size()+65535)/65536 + 1
	if _, ok := r.mem.Grow(pages); !ok {
		return fmt.Errorf("wasm/runtime: failed to grow memory by %d pages", pages)
	}
	return nil
}

// decodeF32 converts a raw WASM u64 result to float32, guarding against
// NaN/Inf and clamping to [0, 1].
func decodeF32(raw uint64) float32 {
	v := api.DecodeF32(raw)
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		return 0
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// checkText validates that a string is non-empty and within the allowed byte limit.
func checkText(field, s string) error {
	if s == "" {
		return fmt.Errorf("wasm/runtime: %s must not be empty", field)
	}
	if len(s) > MaxTextBytes {
		return fmt.Errorf("wasm/runtime: %s too large (%d > %d bytes)", field, len(s), MaxTextBytes)
	}
	return nil
}
