package comparator

import (
	"fmt"
)

// MACRO_ECONOMIC_INDICATOR: a released macroeconomic statistic for a pinned
// (country, indicator, year) observation, in milli-percent.
//
// The intent looks like the least deterministic of the three wired so far —
// "macro indicator" suggests something that moves — and the pin is what removes
// that. A World Bank annual series for a SETTLED year is a single published
// number: two validators querying USA/SL.UEM.TOTL.ZS/2019 minutes or months
// apart both get 3.669. Nothing here is "latest" or MRV, and the miner's own
// registration YAML says so in as many words ("do NOT send MRV (it overrides
// date)").
//
// ─────────────────── build.json says .rate_pct; the miner does not ───────────
//
// The published artifact MACRO_ECONOMIC_INDICATOR.build.json declares
// value_path ".rate_pct", which describes an INTEREST-RATE miner returning a
// flat {"rate_pct": ...} envelope. No such miner is registered. The one active
// miner for this intent — macro-wb-unemp-us, registration 2753 — is the World
// Bank Indicators API, and its payload is a two-element array:
//
//	[ {page,pages,per_page,total,lastupdated}, [ {indicator,country,date,value,...} ] ]
//
// so the scalar sits at [1][0].value, not at .rate_pct. This port follows the
// MINER, not the value_path, and that is a deliberate choice rather than a
// silent divergence:
//
//   - The same build.json's ground_truth block is entirely about unemployment —
//     it names SL.UEM.TOTL.ZS, country USA, year 2019, method
//     "response[1][0].value", and even records the live-observed 3.669. Only the
//     one-line value_path was left describing a rate. The spec contradicts
//     itself; the unemployment reading is the one the rest of the document,
//     the tolerance argument and the registered miner all agree on.
//   - The intent catalog's description covers all three quantities ("central
//     bank interest rates, consumer price index (CPI) updates, and unemployment
//     figures"), so unemployment is in scope, not a substitution.
//   - Critically, following the miner changes NOTHING the WASM sees. The spec
//     buffer [2,150] and the x1000 milli-percent scale are taken verbatim from
//     the build spec and are unchanged here; only the host-side JSON path
//     differs, and the host side is exactly the part the artifact does not
//     commit to. Had the scale or tolerance been the thing in dispute, this
//     would be a blocker rather than a comment.
//
// The same deviation is flagged in the session note; the build reference's
// value_path for this intent should be corrected to match the shipped miner.
//
// ─────────────────── the miner's label_field is a container ──────────────────
//
// macro-wb-unemp-us registers semantics.signal_mapping.label_field: "1". That
// resolves — "1" is an array index into the top-level response and selects the
// observations array — but it yields the CONTAINER, not a scalar. Anything
// generically following that mapping lands on [{...observation...}] and has to
// already know to descend to [0].value to get 3.669. That is fragile in the
// worst way: it does not error, it just hands the next stage a list where a
// number was expected, and a downstream reader that coerces rather than fails
// would produce a plausible non-answer. The descent is hardcoded in
// parseWorldBankIndicator below precisely so the knowledge is written down in
// one place with a test, instead of living implicitly in a "1".
//
// ────────────────────── truth and the miner share a source ───────────────────
//
// Recorded rather than solved, because it cannot be solved from here: this
// miner's registered base_url IS api.worldbank.org — the upstream itself, not
// the omni-chat proxy the slug's YAML is served from. The truth fetch below
// calls the same host, the same path and the same pinned query. The comparator
// therefore CANNOT fail macro-wb-unemp-us on value: it is being compared
// against itself, and will score 10000 by construction. That number measures
// "we called the World Bank twice", not miner quality. Genuine separation needs
// a second, independent feed for the same observation — FRED's UNRATE is the
// obvious candidate and the build spec already describes wiring it with an
// ALFRED vintage pin, but it is key-gated (FRED_API_KEY) and out of scope here.
// Until then, read a MACRO_ECONOMIC_INDICATOR leaderboard as a liveness and
// parse check, not a ranking.

// MacroObservation is one pinned macro statistic: a country, a World Bank
// indicator code, and a settled year.
//
// All three dimensions are carried explicitly rather than baked into the truth
// fetcher, because the pin is the entire determinism argument — a fetch that
// defaulted any of them (especially to "latest" or MRV) would give two
// validators different answers for the same key and there would be no error to
// notice.
type MacroObservation struct {
	Key       string // short stable ID, used as the question key
	Country   string // ISO code as the World Bank accepts it in the path
	Indicator string // World Bank indicator code, e.g. SL.UEM.TOTL.ZS
	Year      int    // the pinned reference year
	Name      string // human description for logs and the audit trail
}

// MacroObservations are the statistics MACRO_ECONOMIC_INDICATOR is scored on
// each epoch.
//
// USA/2019 first: it is the observation the build spec pins, the miner's
// registered default, and the one with an independently recorded expected value
// (3.669%), so it is the key that can be checked against a run of the reference
// host. The other two are not padding — they exist to catch a miner that
// ignores the (country, year) it is asked for and replays its registered
// default. US/2010 is deep in the post-crisis peak at 9.633% and DE/2019 is
// 3.163%, both hundreds to thousands of bps away from 3.669%, so a miner
// replaying USA/2019 scores zero on them rather than sneaking inside the 150
// bps band.
//
// Every key is a pre-pandemic or otherwise settled year on purpose. A
// just-released period is still undergoing heavy revision, which would turn the
// revision caveat below from a small residual risk into the dominant source of
// disagreement.
var MacroObservations = []MacroObservation{
	{
		Key: "US-UNEMP-2019", Country: "US", Indicator: "SL.UEM.TOTL.ZS", Year: 2019,
		Name: "United States unemployment rate, 2019 (ILO modelled)",
	},
	{
		Key: "US-UNEMP-2010", Country: "US", Indicator: "SL.UEM.TOTL.ZS", Year: 2010,
		Name: "United States unemployment rate, 2010 (ILO modelled)",
	},
	{
		Key: "DE-UNEMP-2019", Country: "DE", Indicator: "SL.UEM.TOTL.ZS", Year: 2019,
		Name: "Germany unemployment rate, 2019 (ILO modelled)",
	},
}

// MacroKeys is the fixed question set, in the order rounds are run.
var MacroKeys = macroKeys()

func macroKeys() []string {
	keys := make([]string, 0, len(MacroObservations))
	for _, o := range MacroObservations {
		keys = append(keys, o.Key)
	}
	return keys
}

// MacroObservationFor resolves a question key to its pinned observation.
func MacroObservationFor(key string) (MacroObservation, bool) {
	for _, o := range MacroObservations {
		if o.Key == key {
			return o, true
		}
	}
	return MacroObservation{}, false
}

// macroQuestion is the question put to every miner for one observation.
//
// It names the indicator CODE alongside the prose, for the same reason
// routeQuestion names coordinates: the request builder has to turn this into
// the miner's own World Bank call, and the (country, indicator, year) triple is
// the part that must survive that translation. A question that said only "US
// unemployment in 2019" would leave the builder to guess an indicator, and the
// World Bank publishes several unemployment series that disagree by
// hundreds of bps.
func macroQuestion(key string) string {
	o, ok := MacroObservationFor(key)
	if !ok {
		return fmt.Sprintf("What is the value of macro observation %s?", key)
	}
	return fmt.Sprintf(
		"What is the World Bank indicator %s (%s) for country %s in year %d, as a percentage?",
		o.Indicator, o.Name, o.Country, o.Year)
}

// describeMacro renders a key for logs and the stored audit trail.
func describeMacro(key string) string {
	if o, ok := MacroObservationFor(key); ok {
		return fmt.Sprintf("%s/%s/%d", o.Country, o.Indicator, o.Year)
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// ──────────────────────────────── macro coverage ───────────────────────────────
