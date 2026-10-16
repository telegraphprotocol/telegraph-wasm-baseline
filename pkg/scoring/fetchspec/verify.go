package fetchspec

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// Result records what happened when a spec was tried against its upstream.
//
// Stored rather than only logged, because the whole reason this exists is that
// the previous gate recorded PASS for a 21MB Polish blocklist registered as a
// fraud-risk miner and for an aircraft feed registered as a maritime one. A
// verdict nobody can query is a verdict nobody acts on.
type Result struct {
	OK bool

	// Value is what the spec produced, when OK.
	Value float64

	// Reason explains a failure in terms the miner's author can act on.
	Reason string
}

func (r Result) String() string {
	if r.OK {
		return fmt.Sprintf("ok (value=%g)", r.Value)
	}
	return "failed: " + r.Reason
}

// Verify calls the upstream once and reports whether the spec actually works.
//
// This is the gate that was missing. Registration previously accepted a miner
// on an HTTP 200 alone, which is why Validate=PASS never meant the intent was
// answerable: of three miners checked against their own declarations, three
// declared a field that was missing, lossy, or double-scaled.
//
// It is deliberately a REAL call. A spec can be syntactically perfect and
// still name a field the upstream does not serve, and the only way to know is
// to ask. One call at registration is far cheaper than discovering it during
// an epoch, when the miner is already being dispatched to and the failure
// looks like a bad miner rather than a bad declaration.
func Verify(ctx context.Context, spec Spec, params map[string]string) Result {
	if err := spec.Validate(); err != nil {
		return Result{Reason: err.Error()}
	}

	value, err := Fetch(ctx, spec, params)
	if err != nil {
		return Result{Reason: err.Error()}
	}

	// A non-finite value passes every structural check and then poisons the
	// comparator, which does integer arithmetic.
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return Result{Reason: "the spec produced a non-finite value"}
	}

	return Result{OK: true, Value: value}
}

// VerifyAgainst additionally checks the spec against a known-good value.
//
// Used when onboarding a miner for an intent that already has a truth source:
// a spec can fetch a perfectly valid number from the wrong FIELD and look
// healthy on its own. The GRID miner reports lmp_usd_per_mwh and
// lmp_usd_per_mwh_milli side by side, a thousand-fold apart, and picking the
// wrong one is silent — both are numbers, both are present, both are correct
// answers to a question nobody asked.
//
// toleranceBps is the intent's own band. Passing here does not mean the miner
// is good, only that it is answering the same question in the same units.
func VerifyAgainst(ctx context.Context, spec Spec, params map[string]string, want float64, toleranceBps int64) Result {
	res := Verify(ctx, spec, params)
	if !res.OK {
		return res
	}
	if want == 0 {
		return res
	}

	errBps := math.Abs(res.Value-want) / math.Abs(want) * 10000
	if errBps > float64(toleranceBps) {
		return Result{
			Reason: fmt.Sprintf(
				"value %g is %.0f bps from the reference %g (tolerance %d bps) — "+
					"usually a wrong field or a missing rescale rather than a wrong answer",
				res.Value, errBps, want, toleranceBps),
			Value: res.Value,
		}
	}
	return res
}

// IsOracle reports whether a miner spec reads from the same host as the truth
// spec it will be scored against.
//
// Such a miner cannot fail: it and the comparator are reading the same number
// from the same place, so it scores a perfect ten thousand by construction and
// tells you nothing about its quality. Nineteen of our fifty-one registered
// miners are in this position today.
//
// Not an error — one of these is still a useful availability check, and the
// operator may accept it knowingly. But it must be recorded, because a
// leaderboard that silently contains its own answer key is worse than no
// leaderboard.
func IsOracle(minerSpec, truthSpec Spec) bool {
	mh, th := hostOf(minerSpec.URL), hostOf(truthSpec.URL)
	return mh != "" && mh == th
}

func hostOf(rawURL string) string {
	s := rawURL
	if i := strings.Index(s, "://"); i >= 0 {
		s = s[i+3:]
	}
	if i := strings.IndexAny(s, "/?#"); i >= 0 {
		s = s[:i]
	}
	if i := strings.Index(s, "@"); i >= 0 {
		s = s[i+1:]
	}
	// Drop the port: the same host on two ports is still the same host, and an
	// oracle that moved to :8080 is no less an oracle.
	if i := strings.LastIndex(s, ":"); i >= 0 && !strings.Contains(s[i:], "]") {
		s = s[:i]
	}
	return strings.ToLower(s)
}
