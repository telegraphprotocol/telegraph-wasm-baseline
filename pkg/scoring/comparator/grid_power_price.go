package comparator

import (
	"fmt"
)

// GRID_POWER_PRICE: the CAISO Day-Ahead Market Locational Marginal Price for a
// pinned pricing node and a pinned, already-SETTLED hourly interval, in
// milli-dollars per MWh.
//
// This intent looks like the most volatile of the three wired here — an
// electricity price moves every five minutes — and the pinned-past-hour design
// is precisely what removes that volatility. The Day-Ahead Market clears once
// per operating day, and OASIS's PRC_LMP report for a past settled hour is
// never revised: re-querying it returns the same bytes forever. That was
// verified rather than assumed, twice and a week apart. The build spec records
// two independent fetches seconds apart returning LMP_PRC=38.09459 for the
// pinned hour; an independent fetch made for this port, eight days later,
// returned the identical 38.09459.
//
// The contrast with ROUTE_ETA is worth holding onto, because the two reach
// determinism from opposite directions. OSRM is deterministic because it has no
// clock at all. CAISO is deterministic because we refuse to ask it about now —
// "latest" would be a real-time price that changes under the scorer. Hence the
// pin is a hard requirement below, not a default.

// ─────────────────────────────── pinned intervals ──────────────────────────────

// LMPNode is the CAISO pricing node GRID_POWER_PRICE is scored at.
//
// TH_NP15_GEN-APND is the NP15 default load aggregation point, one of CAISO's
// main trading hubs. It is chosen for liquidity: a hub node is quoted in every
// DAM run, whereas an obscure generator node can be absent from a given report
// and would turn a settled-market question into an intermittent one.
const LMPNode = "TH_NP15_GEN-APND"

// LMPMarket is the market run scored. DAM (day-ahead), never RTM.
//
// This is the load-bearing half of the determinism argument. The same node and
// hour queried against RTM returns a five-minute real-time price that is
// revised and re-published; only the day-ahead run is a single cleared,
// settled, immutable figure.
const LMPMarket = "DAM"

// GridInterval is one pinned settlement hour.
//
// Start is the UTC instant the DAM hour begins, and it doubles as the question
// key. Keying on the instant rather than a short label like "H14" is deliberate
// — the key is what a miner is asked for and what the truth fetcher resolves,
// so an ambiguous label would let the two drift onto different hours while both
// look correct. Adjacent DAM hours differ by thousands of bps, so that drift
// would read as "every miner is wrong", not as a bug here.
type GridInterval struct {
	Start string // RFC3339 UTC instant for the start of the settled hour
	Name  string // human description for logs and the audit trail
}

// GridIntervals are the settlement hours GRID_POWER_PRICE is scored on.
//
// One interval, not three. ROUTE_ETA asks three routes specifically to catch a
// miner that ignores its arguments and replays a registered default — a real
// risk there, because every ROUTE_ETA miner carries a different baked-in route.
// The same risk exists here and was tested directly rather than designed
// around: the single registered miner carries a DIFFERENT default hour
// (20260920T07:00, which returns ~$49.518/MWh) from the pinned one, and when
// asked for the pinned hour it returned $38.09459 — the settled CAISO figure,
// not its default. It honours the interval it is given.
//
// The hour below is the one the build spec and the reference host
// grid_power_price_price.mjs pin, so it is the one with an independently known
// expected value (38.09459 raw, 38095 milli). Adding further hours is cheap and
// needs no code change, but each one costs an OASIS round trip, and OASIS
// throttles hard — an adjacent-hour probe during this port was answered with a
// rate-limit page rather than a ZIP. A second interval should therefore be
// added with a retry/backoff policy, not on its own.
var GridIntervals = []GridInterval{
	{
		Start: "2026-09-13T14:00:00Z",
		Name:  "CAISO NP15 DAM hour beginning 2026-09-13T14:00Z",
	},
}

// GridKeys is the fixed question set, in the order rounds are run.
var GridKeys = gridKeys()

func gridKeys() []string {
	keys := make([]string, 0, len(GridIntervals))
	for _, iv := range GridIntervals {
		keys = append(keys, iv.Start)
	}
	return keys
}

// GridIntervalFor resolves a question key to its pinned interval.
func GridIntervalFor(key string) (GridInterval, bool) {
	for _, iv := range GridIntervals {
		if iv.Start == key {
			return iv, true
		}
	}
	return GridInterval{}, false
}

// gridQuestion is the question put to every miner for one settlement hour.
//
// It names the node, the market run and the interval explicitly. All three have
// to survive translation into the miner's own OASIS-shaped call, and each one
// silently changes the answer if it is dropped: omitting the market gets a
// real-time price instead of a day-ahead one, and omitting the interval gets
// the miner's registered default hour — which for the one registered miner is a
// genuinely different price (~$49.5 against ~$38.1), well outside any
// tolerance.
func gridQuestion(key string) string {
	iv, ok := GridIntervalFor(key)
	if !ok {
		return fmt.Sprintf("What was the CAISO DAM LMP at %s?", key)
	}
	return fmt.Sprintf(
		"What was the CAISO Day-Ahead Market locational marginal price in US dollars per MWh "+
			"at pricing node %s for the settlement hour beginning %s UTC (%s)?",
		LMPNode, iv.Start, iv.Name)
}

// describeGridInterval renders a key for logs and the stored audit trail.
func describeGridInterval(key string) string {
	return LMPNode + "@" + key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// ────────────────────────────────── coverage ───────────────────────────────────
