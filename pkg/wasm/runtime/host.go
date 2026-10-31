package runtime

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// The fixed host every model score runs in (the foundation engine; before
// 2026-10-30 also the full-model float adapters, which are no longer
// accepted). Every value here is part of the determinism contract: two
// validators scoring the same adapter on the same text must hand the guest
// byte-identical inputs, or the score can fork. Change any of them only
// together with every validator.
const (
	// FloatFixedWalltimeSec / FloatFixedNanotime are the guest's clocks. They
	// never move: a guest that times itself sees 0 s, and nothing it computes
	// can depend on when or how fast it ran.
	FloatFixedWalltimeSec int64 = 1640995200 // 2022-01-01T00:00:00Z
	FloatFixedNanotime    int64 = 1_000_000_000

	// DefaultFloatMaxTokens is the token cap when the adapter sets no
	// max_tokens; MaxFloatMaxTokens is the most it may ask for. The node's
	// tokenizer is loaded with MaxLen = MaxFloatMaxTokens.
	DefaultFloatMaxTokens = 512
	MaxFloatMaxTokens     = 512

	// FloatMemoryLimitPages bounds a guest's linear memory: 3 GiB (49152
	// pages). Measured on the real foundation: ~1.7 GB with adapters loaded
	// and scoring, so the limit leaves headroom while keeping a hostile
	// module from taking the 4 GiB wasm32 maximum.
	FloatMemoryLimitPages uint32 = 49152

	// FloatDefaultScoreTimeout bounds every single score, whatever the
	// caller's own deadline: ADAPTER_SCORE_TIMEOUT overrides it.
	FloatDefaultScoreTimeout = 5 * time.Minute
)

// ScoreTimeoutEnv names the per-call scoring timeout (a Go duration).
const ScoreTimeoutEnv = "ADAPTER_SCORE_TIMEOUT"

// ErrScoreTimeout marks a guest stopped because one score ran past the
// per-call timeout. It is the adapter's failure (it also matches
// ErrFloatGuest), never the caller's context ending.
var ErrScoreTimeout = errors.New("score timed out")

// ErrFloatGuest marks a failure the adapter is responsible for (a trap, a
// refused package, a timed-out score), as opposed to the caller's context
// ending or the node's own foundation failing.
var ErrFloatGuest = errors.New("float adapter failed")

// FloatScoreTimeout is the per-call scoring timeout: ADAPTER_SCORE_TIMEOUT
// when it is a positive Go duration, else FloatDefaultScoreTimeout.
func FloatScoreTimeout() time.Duration {
	if v := strings.TrimSpace(os.Getenv(ScoreTimeoutEnv)); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}
	return FloatDefaultScoreTimeout
}

// floatRandSeed seeds the guest's random source (wasi random_get). Rust's std
// seeds HashMap from it, so it must be the same on every validator.
var floatRandSeed = []byte("telegraph/float-adapter/random/v1")

// fixedRandSource is a deterministic byte stream: SHA-256(seed || counter_be64)
// blocks. A fresh one per instance, so every instance sees the same bytes.
type fixedRandSource struct {
	ctr uint64
	buf []byte
}

func newFixedRandSource() *fixedRandSource { return &fixedRandSource{} }

func (r *fixedRandSource) Read(p []byte) (int, error) {
	n := 0
	for n < len(p) {
		if len(r.buf) == 0 {
			var c [8]byte
			binary.BigEndian.PutUint64(c[:], r.ctr)
			r.ctr++
			h := sha256.Sum256(append(append([]byte{}, floatRandSeed...), c[:]...))
			r.buf = h[:]
		}
		k := copy(p[n:], r.buf)
		r.buf = r.buf[k:]
		n += k
	}
	return n, nil
}

func fixedWalltime() (int64, int32) { return FloatFixedWalltimeSec, 0 }
func fixedNanotime() int64          { return FloatFixedNanotime }

// cappedBuffer keeps the first limit bytes written to it and drops the rest,
// remembering that it did. It never returns an error to the guest: a failed
// write would change what the guest does (Rust's println! panics on it), and
// the host's verdict on an over-long output is made after the run.
type cappedBuffer struct {
	limit     int
	buf       bytes.Buffer
	truncated bool
}

func newCappedBuffer(limit int) *cappedBuffer { return &cappedBuffer{limit: limit} }

func (c *cappedBuffer) Write(p []byte) (int, error) {
	room := c.limit - c.buf.Len()
	if room <= 0 {
		if len(p) > 0 {
			c.truncated = true
		}
		return len(p), nil
	}
	if len(p) > room {
		c.buf.Write(p[:room])
		c.truncated = true
		return len(p), nil
	}
	c.buf.Write(p)
	return len(p), nil
}

// TruncateTokens caps ids at max, keeping the last id (the closing special
// token the tokenizer appends) in the last slot. Shorter input is returned
// unchanged.
func TruncateTokens(ids []int64, max int) []int64 {
	if max <= 0 || len(ids) <= max {
		return ids
	}
	out := make([]int64, 0, max)
	out = append(out, ids[:max-1]...)
	return append(out, ids[len(ids)-1])
}

// EncodeFloatInput is the canonical JSON form of one input (input_ids and an
// all-ones attention_mask of the same length): what InputHash digests, and
// what the kit's tools print as a sample input.
func EncodeFloatInput(ids []int64) []byte {
	var b bytes.Buffer
	b.Grow(len(ids)*8 + 48)
	b.WriteString(`{"input_ids":[`)
	for i, id := range ids {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatInt(id, 10))
	}
	b.WriteString(`],"attention_mask":[`)
	for i := range ids {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteByte('1')
	}
	b.WriteString(`]}`)
	return b.Bytes()
}

// ParseScoreHex parses "0x" + exactly 8 hex digits into the f32 bits (the
// form scores are printed and pinned in).
func ParseScoreHex(tok string) (uint32, error) {
	if len(tok) != 10 || (tok[:2] != "0x" && tok[:2] != "0X") {
		return 0, fmt.Errorf("malformed HEX score %q (want 0x and 8 hex digits)", tok)
	}
	v, err := strconv.ParseUint(tok[2:], 16, 32)
	if err != nil {
		return 0, fmt.Errorf("malformed HEX score %q: %v", tok, err)
	}
	return uint32(v), nil
}
