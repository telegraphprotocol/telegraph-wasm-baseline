// Package comparator is the host side of the deterministic scoring path: the
// code that turns a miner's raw response and an off-chain reference value into
// the two integers the comparator WASM is called with.
//
// The division of labour is fixed by the WASM's design (see "Deterministic
// Intents - WASM Build Reference.md" at C:\Work\Telegraph Docs). The WASM does
// only pure-integer arithmetic:
//
//	imw_score_deterministic(ground_truth_value i64, miner_value i64, spec_ptr) -> [0,10000]
//
// Truth comes FIRST. Note that the Go wrapper
// DeterministicRuntime.ScoreDeterministic deliberately takes (minerValue,
// truthValue) in the host-natural order and swaps them at the call; do not
// "fix" one to match the other. A swap is silent under RelTol — it only
// changes the denominator, so it still returns a plausible score.
//
// It performs no I/O and holds no per-intent knowledge — all twenty-four
// deterministic intents ship the *same* binary and differ only in the two-int
// spec buffer [comparator_code, tolerance] and in this host-side code. That
// prohibition is the point: with no network, no clock and no floats inside the
// WASM, every validator that derives the same two integers gets a bit-identical
// score. Anything that fetches, parses or rounds therefore has to live here.
//
// Scope today is FX_NOW, ROUTE_ETA and ONCHAIN_METRIC_VERIFY. Everything an
// intent needs to be scored
// — its comparator spec, its fixed-point scale, the fixed question set, the
// question text, and its per-miner parsers — is registered in one place
// (intentConfigs below), so adding a third intent is a data change here plus a
// truth source, not a change to the scoring pass.
//
// The per-miner parsers remain deliberate scaffolding. Parser count grows with
// *miners* (1,340 registered), not intents (134), and ROUTE_ETA makes the cost
// concrete: five of its six scoreable miners are the same OSRM endpoint and
// differ only in a registered default, so they need one parser repeated five
// times. This should move to a declarative response mapping carried in the
// miner's registration YAML — miner_registry.yaml_raw already holds the whole
// YAML, so no schema change is needed — the way semantics.signal_mapping
// already describes the request side.
package comparator

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
)

// Comparator codes, matching comparator_core() in
// pipelines/int8_runtime/imw_wasm/src/hybrid.rs. Only RelTol and PassFail are
// used by the twenty-four deterministic intents; Exact and AbsTol exist in the
// WASM but are unused.
const (
	CmpExact    int64 = 0
	CmpAbsTol   int64 = 1
	CmpRelTol   int64 = 2
	CmpPassFail int64 = 3
)

// Scale is the fixed-point multiplier FX_NOW values are expressed in before
// they reach the WASM, which is integer-only.
//
// 10000 — "basis points of rate", so 0.871 becomes 8710 — is NOT a free choice.
// It is the scale the intent's build spec declares
// (build_pipeline/specs/FX_NOW.json: "rate * 10000 (fixed integer scale =
// basis points of rate; NEVER float)"), the scale the reference host
// fx_price.mjs emits as rateBps, and therefore the scale the published artifact
// is documented against. Every deterministic intent declares its own — Gwei for
// ONCHAIN_METRIC_VERIFY, cents for LIVE_SHELF_PRICE — so this constant is
// FX_NOW's, not a global one.
//
// The comparator itself is scale-agnostic: RelTol divides by the truth, so any
// consistent scale yields the same score. What a mismatch costs is agreement —
// a validator scaling by 1e8 and one scaling by 1e4 both compute correct scores
// in isolation, and the artifact hash they publish against says 1e4.
//
// The tradeoff this scale makes is precision: at 1e4 a rate of 0.871 has four
// significant digits, so quantisation is ~5.7 bps against a 50 bps tolerance —
// roughly a tenth of the band, and enough to move a near-tie. That is the
// spec's tradeoff to revisit, not one to silently diverge from.
const Scale = 10000

// ScaleRouteETA is ROUTE_ETA's fixed-point multiplier: 1, because the value is
// already an integer.
//
// ROUTE_ETA scores whole seconds. Its build spec is explicit — "eta_seconds =
// round(duration_in_seconds) -> i64 (integer seconds, NEVER float)" — and the
// reference host route_eta_price.mjs emits Math.round(duration), turning OSRM's
// 733.2 into 733. A multiplier of 1 is therefore not "no scale chosen"; it is
// the scale the published artifact is documented against, and it is why the
// rounding in ScaleValue matters more here than for FX_NOW: at scale 1 a
// truncation loses a whole second, whereas at FX_NOW's 1e4 it loses 1/10000th
// of a rate unit.
const ScaleRouteETA = 1

// ScaleMacroIndicator is MACRO_ECONOMIC_INDICATOR's fixed-point multiplier:
// 1000, "milli-percent", so an unemployment rate of 3.669% becomes 3669.
//
// It is the scale the intent's build spec declares ("rate_pct * 1000, rounded
// to the nearest integer -> i64 (milli-percent ...; NEVER float)") and is
// chosen to match the SOURCE's own precision exactly rather than to be generous:
// World Bank's SL.UEM.TOTL.ZS publishes three decimal digits on a percentage
// (3.669, 5.28, 9.633), so x1000 preserves every published digit and invents
// none. A larger scale would manufacture precision the feed does not have.
//
// Note the interaction with the 150 bps tolerance, which is the opposite of
// FX_NOW's: quantisation here is negligible (0.0005pp, well under 1 bps at any
// realistic rate), but the RELATIVE band is wide in absolute terms precisely
// because the magnitude is small — 150 bps of a 3.669% rate is only 0.055pp.
// That is deliberate, and the spec argues it at length: one decimal digit of
// honest rounding (3.7 vs 3.669) is already ~85 bps of relative error on a
// low-magnitude percentage, so an FX-style tight band would fail an honest
// miner. It was tried at 30 bps and did exactly that.
const ScaleMacroIndicator = 1000

// ScaleLiquidityDepth is LIQUIDITY_DEPTH_VERIFY's fixed-point multiplier: 100,
// because the value is USD carried as integer cents.
//
// From the build spec's parse_rule: "USD liquidity value * 100, rounded to the
// nearest integer cent -> i64, and NEVER float". The reference host
// liquidity_depth_price.mjs applies the same USD_CENTS = 100.
//
// Cents are far finer than this intent needs — a hundred-million-dollar pool
// measured to the cent is fifteen significant digits against a 300 bps band —
// but precision is not what the scale is for. It is what the published artifact
// is documented against, and the headroom is real: at cents, i64 does not come
// close to overflowing even for the deepest pool on any chain.
const ScaleLiquidityDepth = 100

// ScaleGridPowerPrice is GRID_POWER_PRICE's fixed-point multiplier: 1000,
// turning $/MWh into milli-dollars per MWh, so $38.09459 becomes 38095.
//
// The build spec argues the choice rather than asserting it, and the argument
// is not the usual overflow one: RelTol computes diff*10000 in i128, so even a
// micro-dollar scale could not overflow at any realistic LMP. 1000 is chosen
// for precision hygiene. CAISO publishes LMP_PRC to five decimals, and the last
// two of those are re-derivation noise — summing the four published components
// of the pinned hour gives 38.09458 against a published 38.09459. Keeping three
// decimals preserves more precision than any miner honestly reports a $/MWh
// price to, while discarding digits that would otherwise make two correct
// readers disagree.
//
// Quantisation at this scale is negligible against the 30 bps band: half a
// milli-dollar on a ~$38 price is ~0.013 bps, about 1/2300th of the tolerance.
// Contrast FX_NOW, where quantisation eats a tenth of its band.
const ScaleGridPowerPrice = 1000

// ScaleOnchainMetric is ONCHAIN_METRIC_VERIFY's fixed-point multiplier: 1,
// because the rescale the spec asks for has already happened by the time a
// value reaches ScaleValue.
//
// The build spec's parse_rule is "wei / 1e9, truncated to an integer (Gwei)".
// That is a real rescale, but it deliberately does NOT live here, and the
// reason is the word "truncated": ScaleValue rounds to nearest, which is
// correct for a rate or a duration and wrong for this. Rounding a wei balance
// up at the Gwei boundary would put the host half a Gwei above every honest
// miner that truncated — a bias applied to truth and miner alike only if both
// go through the same code, which they cannot here because the miner's value
// arrives already in Gwei.
//
// So the division happens once, in big.Int, in parseETHBalance and in the truth
// fetcher (see onchain_metric_verify.go), and the multiplier registered here is
// the identity. A 1 in this table means "already in the spec's integer domain",
// not "no scale was chosen".
const ScaleOnchainMetric = 1

// ScaleAssetReserve is ASSET_RESERVE_ATTESTATION's fixed-point multiplier: 1.
//
// This is the one intent here whose scale is NOT a float multiplier, and the
// reason is that the rescale needs a value this constant cannot see. The build
// spec's parse_rule says "reserve_amount_in_underlying_units * 1e6" — but the
// multiplicand is the reserve in WHOLE UNITS of the underlying asset, and the
// Chainlink PoR feed does not report whole units. It reports at its own
// decimals() (8 for the WBTC feed, verified live at the pinned block), so the
// raw answer 11614464922334 is 116144.64922334 BTC and the micro-unit value is
// that divided by 10^(8-6) = 100, giving 116144649223.
//
// Applying 1e6 to the raw integer instead would give 1.16e19: positive,
// non-overflowing, entirely plausible on inspection, and eight orders of
// magnitude wrong — which under RelTol is a flat 0 for every honest miner
// rather than an error anyone would see. So the rescale is done exactly, in
// integer arithmetic, by toMicroUnits, which reads the feed's decimals at the
// same pinned block as the answer. By the time a value reaches ScaleValue it is
// already in micro-units, and 1 is the multiplier that leaves it there.
const ScaleAssetReserve = 1

// ScaleFor returns the fixed-point multiplier for an intent.
//
// Scales are per-intent by design: each deterministic intent's build spec
// declares its own parse_rule, and the published artifact is documented against
// it. Never reuse one intent's scale for another — a wrong scale does not show
// up as a wrong score (RelTol divides by the truth, so it is scale-agnostic),
// it shows up as disagreement with the artifact's documented domain.
func ScaleFor(intentID string) (float64, bool) {
	c, ok := intentConfigs[intentID]
	if !ok {
		return 0, false
	}
	return c.Scale, true
}

// ScaleValue converts a parsed value into an intent's fixed-point integer
// domain.
//
// It rejects non-finite and non-positive values rather than letting them
// through as 0 or a saturated integer. A NaN that reached the WASM as some
// arbitrary i64 would produce a plausible-looking score from a meaningless
// value, which is worse than a miner that visibly failed to answer.
//
// Zero is admitted for a PASS_FAIL intent, on the same reasoning as
// checkMinerValue: its domain is {0, 1} and 0 is an answer, not a failure.
// EMAIL_SECURITY reads DNS rcode 0 (NOERROR) as "the record exists", so
// rejecting zero here excluded every correctly-answering miner even after the
// parse succeeded. The two checks must agree — a value admitted by one and
// rejected by the other fails late, with a message about scaling rather than
// about the value.
//
// Negative values remain rejected for every intent. That is NOT universal and
// is known debt: an intent whose quantity can legitimately go negative — an
// electricity price during oversupply — must not route through this function
// unchanged, or a negative truth is silently excluded rather than scored.
func ScaleValue(intentID string, v float64) (int64, error) {
	scale, ok := ScaleFor(intentID)
	if !ok {
		return 0, fmt.Errorf("intent %q has no registered scale", intentID)
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("value is not a finite number: %v", v)
	}
	if v == 0 && allowsZero(intentID) {
		return 0, nil
	}
	if v <= 0 {
		return 0, fmt.Errorf("value must be positive, got %v", v)
	}
	scaled := v * scale
	if scaled >= math.MaxInt64 {
		return 0, fmt.Errorf("value %v overflows the fixed-point domain", v)
	}
	// Rounded, per each spec's "rounded to the nearest integer"; truncation
	// would bias every miner's value downwards by up to half a unit — ~5.7 bps
	// of FX_NOW's 50 bps tolerance, and a whole second for ROUTE_ETA.
	return int64(math.Round(scaled)), nil
}

// ScaleFloat converts a rate to FX_NOW's fixed-point integer domain.
//
// Retained as FX_NOW's named entry point; ScaleValue is the general form.
func ScaleFloat(rate float64) (int64, error) { return ScaleValue("FX_NOW", rate) }

// Spec is the two-integer buffer passed to the WASM alongside the values.
type Spec struct {
	Code      int64 // comparator kind (CmpRelTol for FX_NOW)
	Tolerance int64 // half-score point; basis points when Code is CmpRelTol
}

// Buffer renders the spec in the order the WASM reads it.
func (s Spec) Buffer() [2]int64 { return [2]int64{s.Code, s.Tolerance} }

// IntentConfig is everything the scoring pass needs to run one deterministic
// intent: what to compare, on what scale, and which fixed questions to ask.
//
// Gathering it in one value is what keeps the pass itself intent-agnostic. The
// pass iterates Keys, calls Question for each, and hands the parsed values to
// the WASM with Spec — it never names an intent.
type IntentConfig struct {
	Spec  Spec    // comparator kind + tolerance, from the build spec
	Scale float64 // fixed-point multiplier, from the build spec's parse_rule
	Label string  // short description stored on each epoch_scores row

	// Keys is the fixed question set asked identically of every miner in a
	// round. A ranking built from miners answering *different* questions is not
	// a ranking, so these are fixed rather than generated per epoch.
	Keys []string

	// Question renders the text put to a miner for one key. The request builder
	// turns it into that miner's own API call from its registration YAML.
	Question func(key string) string

	// Describe renders a key for logs and the stored audit trail.
	Describe func(key string) string

	// Fusion is set only for a HYBRID intent, whose score comes from a
	// model-bearing WASM rather than the shared comparator. Nil means the
	// intent is scored by pure comparison, which is the common case.
	Fusion *Fusion

	// ResponseFormat is intent_specs.response_format, the LEGACY source of the
	// intent's response format. The comparator reads the format through
	// FormatFor, which prefers the node's intent format registry
	// (intent_response_formats, then this column, then the family's) once it
	// is loaded. Nil (NULL column) means not defined here. See
	// response_format.go.
	ResponseFormat *responseparser.Format

	// KeyParams is truth_params.key_params by question key, used to fill
	// {<param>} placeholders in ResponseFormat descriptions. May be nil.
	KeyParams map[string]map[string]string
}

// IsHybrid reports whether an intent is scored by the model-bearing path.
func IsHybrid(intentID string) bool {
	c, ok := intentConfigs[intentID]
	return ok && c.Fusion != nil
}

// intentConfigs maps an intent to its full comparator configuration.
//
// This is protocol-level data, not logic: every validator must agree on the
// tolerance or they will disagree on scores. It is a Go map today only because
// two intents are wired; the spec+tolerance half belongs in config or the
// on-chain intent registry, so that retuning a tolerance does not require a
// binary release and cannot silently differ between validators running
// different builds.
var intentConfigs = map[string]IntentConfig{
	// FX mid-rates are a near-canonical quote, so the band is tight: 50 bps
	// (0.5%). Under RelTol the tolerance is the *half-score* point, and decay
	// is linear to zero at 2x — so a miner is scored 5000 at 50 bps of error
	// and 0 at 100 bps or worse.
	"FX_NOW": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 50},
		Scale:    Scale,
		Label:    "FX_NOW mid-market rate",
		Keys:     FXPairs,
		Question: fxQuestion,
		Describe: func(q string) string { return "USD/" + q },
	},

	// ROUTE_ETA is the loosest band of the two at 800 bps (8%), and that is
	// deliberate rather than sloppy: routing engines legitimately disagree on
	// the ETA for an identical origin/destination because they use different
	// road-speed profiles and turn-cost models. The build spec argues the
	// number at length — cross-engine divergence of 5-15% is ordinary, so a
	// tighter band would fail a miner for using a different-but-reasonable
	// engine rather than for being wrong.
	//
	// Under RelTol that means a miner is scored 5000 at 8% error and 0 at 16%
	// or worse.
	"ROUTE_ETA": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 800},
		Scale:    ScaleRouteETA,
		Label:    "ROUTE_ETA driving transit time",
		Keys:     RouteKeys,
		Question: routeQuestion,
		Describe: describeRoute,
	},

	// MACRO_ECONOMIC_INDICATOR is 150 bps (1.5%), and the reason it is looser
	// than FX_NOW's 50 bps is the opposite of ROUTE_ETA's: there is no
	// cross-source disagreement to absorb here at all. A pinned-year World Bank
	// statistic is a single published number, and two honest queriers of
	// USA/SL.UEM.TOTL.ZS/2019 get the byte-identical 3.669. What the band
	// absorbs is MAGNITUDE. This is a relative tolerance on a low-magnitude
	// percentage: an unemployment rate honestly rounded to one decimal — 3.7
	// against the source's 3.669 — is a mere 0.031pp absolute difference but
	// already ~85 bps of relative error, and a worst-case one-decimal rounding
	// runs to ~140 bps at these levels. The build spec records trying an
	// FX-style 30 bps band first and zero-scoring exactly that honest miner.
	//
	// 150 bps still fails a miner decisively where it matters: adjacent years
	// differ by hundreds of bps (2018->2019 moved ~230 bps, 2019->2020 over
	// 1000), so a miner replaying the wrong period cannot hide inside the band.
	// Under RelTol a miner is scored 5000 at 1.5% error and 0 at 3% or worse.
	//
	// Two things about this intent are not what the paperwork says, both
	// argued in full at the top of macro_economic_indicator.go. First, the
	// build.json's value_path is ".rate_pct" — an interest-rate envelope no
	// registered miner serves — while the same document's ground_truth block,
	// the tolerance argument and the one active miner are all unemployment;
	// this port follows the miner, and the spec buffer and scale are unchanged.
	// Second, the sole miner's base_url IS api.worldbank.org, the same host the
	// truth fetch calls, so the comparator cannot fail it on value. Its perfect
	// score is arithmetic, not evidence of quality.
	"MACRO_ECONOMIC_INDICATOR": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 150},
		Scale:    ScaleMacroIndicator,
		Label:    "MACRO_ECONOMIC_INDICATOR pinned macro statistic",
		Keys:     MacroKeys,
		Question: macroQuestion,
		Describe: describeMacro,
	},

	// LIQUIDITY_DEPTH_VERIFY sits between the two at 300 bps (3%), and the
	// argument for it is that the quantity has no instant. A DEX pool's
	// reserves change with EVERY SWAP: the pinned WETH/USDC 0.05% pool clears
	// thousands of trades an hour, so "the liquidity of that pool" is only
	// well-defined at a block height, and neither indexer exposes one. The
	// reference fetch and a miner's own fetch therefore land on different
	// blocks by construction, and no amount of care on our side closes that —
	// unlike FX_NOW, where a dated pin exists, or ROUTE_ETA, where the engine
	// routes over a frozen snapshot.
	//
	// Two independent errors then stack. Timing: a handful of blocks of swaps
	// and LP movement between the two reads. Indexer method: GeckoTerminal's
	// reserve_in_usd and DexScreener's liquidity.usd value the same reserves
	// with their own token prices, so they differ even at an identical block.
	// The live check behind this number measured both — the build spec records
	// ~5 bps between indexers, and the two registered miners probed together
	// on 2026-09-21 differed by ~9 bps. 300 bps is roughly thirty times the
	// observed spread, which is the right shape for a band that has to absorb
	// an occasional large LP withdrawal without failing an honest miner, while
	// still catching the failures that actually matter here: a stale cache, a
	// wrong fee tier, or the wrong pool entirely — all of which miss by tens of
	// percent, not tenths.
	//
	// Under RelTol a miner is scored 5000 at 3% error and 0 at 6% or worse.
	//
	// Caveat on WHAT is scored, carried from the intent file: the catalog
	// describes order-book depth and slippage bands; this scores total pool
	// liquidity, which is related but not the same quantity.
	"LIQUIDITY_DEPTH_VERIFY": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 300},
		Scale:    ScaleLiquidityDepth,
		Label:    "LIQUIDITY_DEPTH_VERIFY DEX pool liquidity",
		Keys:     PoolKeys,
		Question: liquidityQuestion,
		Describe: describePool,
	},

	// MINING_HASHPRICE_VERIFY sits at 400 bps (4%), and the arithmetic says why.
	// Hashprice is DERIVED, not quoted: subsidy_sats * (1e15 * 86400) /
	// (difficulty * 2^32). Both inputs are exact integers pinned by the block
	// height — difficulty decodes from the header's committed `bits`, the subsidy
	// from the halving schedule — so a correct implementation has no numerical
	// slack at all and should land on the truth to the sat. The band is not there
	// for noise; it is there for one legitimate modelling disagreement, whether
	// transaction fees are counted. The spec excludes them, and a fees-inclusive
	// miner quotes a few hundred bps richer under ordinary mempool conditions,
	// which 400 bps absorbs. It is deliberately too narrow to absorb the errors
	// that matter: a wrong halving epoch is off by a factor of two (10000 bps)
	// and a wrong retarget epoch by hundreds to thousands, so both still score 0.
	//
	// Under RelTol that means 5000 at 4% error and 0 at 8% or worse.
	//
	// Scale is 1: the parse_rule defines the value as integer satoshis, the
	// atomic BTC unit, so there is nothing left to rescale.
	"MINING_HASHPRICE_VERIFY": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 400},
		Scale:    ScaleHashprice,
		Label:    "MINING_HASHPRICE_VERIFY sats per PH/s per day",
		Keys:     HashpriceKeys,
		Question: hashpriceQuestion,
		Describe: describeHashprice,
	},

	// GRID_POWER_PRICE carries the tightest band here at 30 bps (0.30%),
	// tighter even than FX_NOW's 50, and the reason is that the scored quantity
	// is not an observation at all — it is a settled figure with a single
	// correct value.
	//
	// The other intents each have irreducible spread to absorb. FX rates move
	// continuously, so two honest sources read seconds apart genuinely differ.
	// Routing engines model road speeds differently. DEX liquidity has no
	// canonical instant. A CAISO Day-Ahead Market LMP for a PINNED (node, DAM,
	// hour) has none of that: the day-ahead market clears once per operating
	// day, the OASIS report for a past hour is never revised, and any two
	// readers of that report get the same number. Verified twice — the build
	// spec records two fetches seconds apart returning LMP_PRC=38.09459, and an
	// independent fetch for this port eight days later returned the same
	// 38.09459 to the digit. There is nothing here for a wide band to absorb.
	//
	// So 30 bps is not sized for source disagreement; it is sized for
	// re-derivation noise, and that is measurable from the same report. A miner
	// reconstructing the LMP from its published components gets 27.70032 +
	// (-0.26689) + 0.20224 + 10.45891 = 38.09458 against the published
	// 38.09459 — off by a thousandth of a dollar, which at this magnitude is
	// under 1 bp. The band is roughly thirty times that, leaving room for a
	// miner that rounds to cents (worst case ~1.3 bps here) while staying far
	// too tight for anything that is actually wrong.
	//
	// What it must catch, it catches by orders of magnitude. The failure that
	// matters is answering for the wrong hour, and adjacent DAM hours in the
	// live pull spanned $21.60 to $50.50 — thousands of bps. The one registered
	// miner's own default hour prices at ~$49.5 against the pinned ~$38.1, a
	// ~3000 bps miss. A band ten times wider would still fail those, so the
	// tight one costs nothing and refuses fabricated or stale values too.
	//
	// Under RelTol a miner is scored 5000 at 30 bps of error and 0 at 60 bps or
	// worse.
	"GRID_POWER_PRICE": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 30},
		Scale:    ScaleGridPowerPrice,
		Label:    "GRID_POWER_PRICE CAISO day-ahead nodal LMP",
		Keys:     GridKeys,
		Question: gridQuestion,
		Describe: describeGridInterval,
	},

	// LIVE_SHELF_PRICE sits at 300 bps (3%), and the reason is the nature of
	// the quantity rather than of the source. A retail price has no canonical
	// value at the comparison instant the way an FX mid-rate does: the same
	// item is repriced by promotions, and two scrapers reading the same listing
	// minutes apart legitimately differ. The build spec cites measured spread
	// in this very dataset — 4.99 against 5.09 EUR for one barcode at one chain
	// within days, ~200 bps, with same-store repricing windows reaching 400-500
	// bps — so a band much under 300 would fail a miner for the market moving
	// rather than for being wrong. Much over 300 and a genuinely stale or
	// wrong-listing price starts passing: at 600 bps a 5.00 EUR item can be
	// answered 5.30 and still score.
	//
	// Under RelTol a miner is scored 5000 at 300 bps of error and 0 at 600 bps
	// or worse.
	//
	// The band is sized for the LIVE mode this intent is named for, not for the
	// mode actually wired. Because the keys pin immutable Open Prices records,
	// a correct miner scores a flat 10000 with zero spread — the tolerance is
	// doing no work today. See live_shelf_price.go.
	"LIVE_SHELF_PRICE": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 300},
		Scale:    ScaleShelfPrice,
		Label:    "LIVE_SHELF_PRICE pinned retail observation",
		Keys:     ShelfKeys,
		Question: shelfQuestion,
		Describe: describeShelf,
	},

	// ONCHAIN_METRIC_VERIFY is the tightest band of any wired intent at 5 bps
	// (0.05%), and unlike the others that is not a judgement about how much
	// disagreement is reasonable — it is a statement that there should be NONE.
	// An eth_getBalance at a pinned block height is canonical consensus state:
	// the reference and the miner read the same number out of the same chain,
	// so a correct answer is bit-identical, not merely close. The build spec
	// says so outright — the band exists only to absorb "the sub-Gwei
	// truncation from the wei->Gwei rescale and benign client-level
	// differences, not real disagreement".
	//
	// Under RelTol a miner is therefore scored 5000 at 5 bps of error and 0 at
	// 10 bps or worse, which makes this intent near pass/fail in practice: any
	// real divergence — a different block, a different address, a stale cache
	// — is orders of magnitude past 10 bps and scores 0. That is the intended
	// behaviour for a verification intent rather than an accidental harshness.
	//
	// The band is only usable because the question is pinned to a fixed block.
	// Scored against "latest" this tolerance would be unworkable, not strict:
	// one ordinary transfer landing between two validators' reads is already
	// far past 10 bps. See PinnedBlock.
	"ONCHAIN_METRIC_VERIFY": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 5},
		Scale:    ScaleOnchainMetric,
		Label:    "ONCHAIN_METRIC_VERIFY wallet balance at a pinned block",
		Keys:     BalanceKeyIDs,
		Question: onchainQuestion,
		Describe: describeOnchain,
	},

	// ASSET_RESERVE_ATTESTATION sits between FX_NOW and ROUTE_ETA at 150 bps
	// (1.5%), and the number follows from what the value IS rather than from how
	// volatile it looks. It is an attestation on a heartbeat, not a live quote: a
	// Chainlink Proof-of-Reserve aggregator writes on-chain only when the
	// custodian's balance moves past the feed's 1% deviation threshold or the
	// heartbeat elapses. The series is therefore a step function, not a continuous
	// one, and a miner reading one round while the truth fetch moments later reads
	// the next can differ by close to that 1% trigger with neither side wrong.
	//
	// 150 bps is sized to absorb exactly one such update-boundary straddle and no
	// more. An FX-style 50 bps band would fail honest miners for the feed's own
	// update semantics rather than for being wrong; a 300 bps band would swallow a
	// real shortfall, which for an asset that is supposed to be fully backed is the
	// only error that actually matters. Under RelTol a miner is scored 5000 at 1.5%
	// error and 0 at 3% or worse.
	//
	// The pin does most of the work the tolerance would otherwise have to do:
	// because the scored block is fixed (see ReserveFeed.Block), a miner that
	// honours it reads the same round as the truth fetch and the straddle never
	// arises. The band is headroom for miners that only serve latest — which, as
	// noted at reserveParsers, is the one registered miner today.
	"ASSET_RESERVE_ATTESTATION": {
		Spec:     Spec{Code: CmpRelTol, Tolerance: 150},
		Scale:    ScaleAssetReserve,
		Label:    "ASSET_RESERVE_ATTESTATION custodied reserve balance",
		Keys:     ReserveKeys,
		Question: reserveQuestion,
		Describe: describeReserve,
	},
}

// ConfigFor returns an intent's full comparator configuration.
func ConfigFor(intentID string) (IntentConfig, bool) {
	c, ok := intentConfigs[intentID]
	return c, ok
}

// SpecFor returns the comparator spec for an intent.
func SpecFor(intentID string) (Spec, bool) {
	c, ok := intentConfigs[intentID]
	if !ok {
		return Spec{}, false
	}
	return c.Spec, true
}

// KeysFor returns the fixed question set an intent is scored on.
func KeysFor(intentID string) ([]string, bool) {
	c, ok := intentConfigs[intentID]
	if !ok {
		return nil, false
	}
	return c.Keys, true
}

// IsDeterministic reports whether an intent is scored by the comparator path.
//
// This is narrower than "not judgment/hybrid": catalog.IsJudgmentIntent tells
// you an intent takes the model path, whereas this tells you the comparator
// path is actually *wired* — a deterministic intent with no spec and no parsers
// here would otherwise fall through to a comparison of two zero values.
func IsDeterministic(intentID string) bool {
	_, ok := intentConfigs[intentID]
	return ok
}

// DeterministicIntents lists every intent wired to the comparator path.
//
// Exported so that tests can assert invariants across the whole registry
// rather than against a hand-maintained list of names: a gate that has to be
// edited every time an intent is ported is one that gets edited rather than
// heeded.
func DeterministicIntents() []string {
	out := make([]string, 0, len(intentConfigs))
	for id := range intentConfigs {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// fxQuestion is the question put to every miner for one currency pair.
//
// It is phrased as a plain question because the request builder turns it into
// each miner's own API call from that miner's registration YAML. Every miner in
// the round gets the identical text, which is what makes their answers
// comparable.
func fxQuestion(quote string) string {
	return fmt.Sprintf("What is the current mid-market exchange rate from USD to %s?", quote)
}

// âââââââââââââââââââââââââââ miner response parsing âââââââââââââââââââââââââââ

// ParseMinerValue extracts the scored quantity from one miner's raw response.
//
// The response arrives as the JSON string that extractAnswerText produced: for
// these non-LLM miners it is the upstream body marshalled back to text, so the
// parsers below read the provider's own schema directly rather than trying to
// read a number out of prose.
//
// Parsing deliberately happens per miner. The ten FX_NOW providers share no
// response shape — keys differ in case and nesting, values arrive as both JSON
// numbers and strings, one uses a comma decimal separator, and three quote a
// base other than USD. A single configured value_path cannot express that,
// which is why the reference doc's ".rate" matches none of them. ROUTE_ETA is
// the opposite shape and no better served by one path: five of its six miners
// return the identical OSRM envelope and the sixth returns a duration *matrix*.
//
// Parsers are scoped per intent rather than keyed by slug alone. Slugs are not
// guaranteed unique across intents — the registry already contains
// byte-identical registrations under two different intents — so a flat map
// would let one intent's parser silently claim another's miner.
//
// Order of the parse paths:
//
//  1. the intent's response_format, read by the parser model — only when a
//     parser is installed (PARSER_LLM_ENABLED=true) and the intent has a
//     format; see response_format.go
//  2. the miner's declared mapping (registration YAML)
//  3. the hand-written parser for the slug
//
// A failure on path 1 is final (not scored, never 0), except when the model
// never answered, which falls through to 2 and 3.
func ParseMinerValue(intentID, slug, key, body string) (float64, error) {
	return ParseMinerValueCtx(context.Background(), intentID, slug, key, body)
}

// ParseMinerValueCtx is ParseMinerValue with a context for the parser-model
// call. The scoring passes use it; the audit pass (parsedValues) does not need
// it because its parses are served from the parser's cache.
func ParseMinerValueCtx(ctx context.Context, intentID, slug, key, body string) (float64, error) {
	if strings.TrimSpace(body) == "" {
		return 0, fmt.Errorf("miner %q returned an empty response", slug)
	}
	if v, handled, err := parseViaResponseFormat(ctx, intentID, slug, key, body); handled {
		if err != nil {
			return 0, err
		}
		return checkMinerValue(intentID, slug, v)
	}
	return parseLegacy(intentID, slug, key, body)
}

// parseLegacy is the per-miner path: declared mapping, then hardcoded parser.
func parseLegacy(intentID, slug, key, body string) (float64, error) {
	if strings.TrimSpace(body) == "" {
		return 0, fmt.Errorf("miner %q returned an empty response", slug)
	}

	// A miner's OWN declaration wins over a parser we wrote for it.
	//
	// This is the direction the code is moving: the mapping is in the
	// registration YAML the miner already ships, so a new miner is a
	// registration rather than a rebuild. The hardcoded parsers remain as a
	// fallback only while the two paths are being proved equal (see
	// TestDeclarativeParityWithHardcodedParsers); once parity holds they go.
	if m, ok := MinerMappingFor(slug); ok {
		v, err := ParseWithMapping(m, body)
		if err != nil {
			return 0, fmt.Errorf("parsing %s response via its declared mapping: %w", slug, err)
		}
		return checkMinerValue(intentID, slug, v)
	}

	byslug, ok := parsersByIntent[intentID]
	if !ok {
		return 0, fmt.Errorf("no parsers registered for intent %q", intentID)
	}
	p, ok := byslug[slug]
	if !ok {
		return 0, fmt.Errorf("no %s parser registered for miner %q", intentID, slug)
	}
	v, err := p(key, body)
	if err != nil {
		return 0, fmt.Errorf("parsing %s response: %w", slug, err)
	}
	return checkMinerValue(intentID, slug, v)
}

// checkMinerValue rejects a value the comparator cannot use.
//
// Shared by both the declarative and hardcoded paths so the two cannot drift
// on what counts as usable — a difference there would show up as one path
// scoring a miner and the other excluding it, for the same response.
//
// Zero is admitted for a PASS_FAIL intent and rejected everywhere else, which
// is the per-intent rule this function's earlier comment called for. For a
// pass_fail comparator the whole domain is {0, 1}: EMAIL_SECURITY asks whether
// a domain publishes an SPF record and reads DNS rcode 0 (NOERROR) as the
// answer, so rejecting 0 excluded every correctly-answering miner and left the
// intent with nothing to rank. For a rel_tol intent a zero rate or a zero
// price is still a broken response, so the rule stands there.
//
// Negative values remain rejected for every intent. That is still real debt:
// CAISO LMP legitimately goes negative during oversupply and such an hour is
// silently excluded rather than scored. Admitting negatives needs a signed
// domain per intent, which the comparator kind alone does not imply — unlike
// zero, which pass_fail does imply — so it is left documented rather than
// quietly widened.
func checkMinerValue(intentID, slug string, v float64) (float64, error) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("miner %q produced an unusable value: %v", slug, v)
	}
	if v == 0 && allowsZero(intentID) {
		return 0, nil
	}
	if v <= 0 {
		return 0, fmt.Errorf("miner %q produced an unusable value: %v", slug, v)
	}
	return v, nil
}

// allowsZero reports whether zero is a legitimate answer for an intent.
//
// Driven by the comparator kind rather than a separate column: pass_fail means
// the answer is a flag, and a flag that can never be 0 is not a flag. Deriving
// it keeps the two from disagreeing, which a hand-maintained list would
// eventually do.
func allowsZero(intentID string) bool {
	cfg, ok := ConfigFor(intentID)
	return ok && cfg.Spec.Code == CmpPassFail
}

// ParseMinerRate extracts a USD->quote rate from one FX_NOW miner's response.
//
// Retained as FX_NOW's named entry point; ParseMinerValue is the general form.
// The quote is upper-cased here because FX_NOW's parsers key on currency codes.
func ParseMinerRate(slug, quote, body string) (float64, error) {
	return ParseMinerValue("FX_NOW", slug, strings.ToUpper(quote), body)
}

// parserFunc reads the scored quantity out of one provider's response body.
// The key is the intent's question key: a currency code for FX_NOW, a route ID
// for ROUTE_ETA.
type parserFunc func(key, body string) (float64, error)

// parsersByIntent holds one parser per registered miner, per intent.
var parsersByIntent = map[string]map[string]parserFunc{
	"ASSET_RESERVE_ATTESTATION": reserveParsers,
	"FX_NOW":                    parsers,
	"ROUTE_ETA":                 routeParsers,
	"ONCHAIN_METRIC_VERIFY":     onchainParsers,
	"LIQUIDITY_DEPTH_VERIFY":    liquidityParsers,
	"MINING_HASHPRICE_VERIFY":   hashpriceParsers,
	"GRID_POWER_PRICE":          gridParsers,
	"MACRO_ECONOMIC_INDICATOR":  macroParsers,
	"LIVE_SHELF_PRICE":          shelfParsers,
}

// parsers holds one entry per registered FX_NOW miner, keyed by the slug in
// miner_registry. Registration IDs 1397-1406 as listed in ALL_INTENT_MINERS.xlsx.
//
// Note that four of these providers cannot answer an arbitrary USD pair at all:
// fx-boc quotes only USD/CAD, fx-riksbank only SEK crosses, and fx-cbr and
// fx-hnb quote against RUB and EUR respectively and are converted below. Which
// miners are eligible for a given pair is therefore a property of the pair, not
// a fixed list — see EligibleMiners.
var parsers = map[string]parserFunc{
	"fx-frankfurter": parseRatesUpper, // {"rates":{"EUR":0.871}}
	"fx-vatcomply":   parseRatesUpper, // {"rates":{"EUR":0.860437}}
	"fx-er-api":      parseRatesUpper, // {"rates":{"EUR":...}}
	"fx-erapi-v4":    parseRatesUpper, // {"rates":{"EUR":...}} (2dp only)
	"fx-fawaz":       parseFawaz,      // {"usd":{"eur":0.871}} lowercase
	"fx-coinbase":    parseCoinbase,   // {"data":{"rates":{"EUR":"0.871"}}} strings
	"fx-hnb":         parseHNB,        // [{"srednji_tecaj":"1,148100"}] EUR-based, comma decimal
	"fx-cbr":         parseCBR,        // {"Valute":{"USD":{"Value":...}}} RUB-based
	"fx-boc":         parseBOC,        // {"observations":[{"FXUSDCAD":{"v":"1.38"}}]} CAD only
	"fx-riksbank":    parseRiksbank,   // [{"date":...,"value":9.81}] SEK only
}

// parseRatesUpper handles the common {"base":"USD","rates":{"EUR":0.871}} shape
// shared by Frankfurter, VATComply and both ExchangeRate-API versions.
func parseRatesUpper(quote, body string) (float64, error) {
	var resp struct {
		Base  string             `json:"base"`
		Rates map[string]float64 `json:"rates"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding response: %w", err)
	}
	// base_code is er-api's spelling of the same field; an absent base is
	// treated as USD because every one of these endpoints is called with USD.
	if resp.Base != "" && resp.Base != "USD" {
		return 0, fmt.Errorf("expected a USD base, got %q", resp.Base)
	}
	rate, ok := resp.Rates[quote]
	if !ok {
		return 0, fmt.Errorf("no %s rate in response", quote)
	}
	return rate, nil
}

// parseFawaz handles the currency-api shape, which nests rates under the
// lowercased base currency and lowercases every quote key.
func parseFawaz(quote, body string) (float64, error) {
	var resp map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding response: %w", err)
	}
	raw, ok := resp["usd"]
	if !ok {
		return 0, fmt.Errorf("no usd block in response")
	}
	var rates map[string]float64
	if err := json.Unmarshal(raw, &rates); err != nil {
		return 0, fmt.Errorf("decoding usd rates: %w", err)
	}
	rate, ok := rates[strings.ToLower(quote)]
	if !ok {
		return 0, fmt.Errorf("no %s rate in response", quote)
	}
	return rate, nil
}

// parseCoinbase handles Coinbase's exchange-rates shape. Its rates are JSON
// *strings* at sixteen decimal places, not numbers, so they are parsed rather
// than decoded straight into a float.
func parseCoinbase(quote, body string) (float64, error) {
	var resp struct {
		Data struct {
			Currency string            `json:"currency"`
			Rates    map[string]string `json:"rates"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding response: %w", err)
	}
	if resp.Data.Currency != "" && resp.Data.Currency != "USD" {
		return 0, fmt.Errorf("expected a USD base, got %q", resp.Data.Currency)
	}
	s, ok := resp.Data.Rates[quote]
	if !ok {
		return 0, fmt.Errorf("no %s rate in response", quote)
	}
	rate, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing %s rate %q: %w", quote, s, err)
	}
	return rate, nil
}

// parseHNB handles the Croatian National Bank feed, which is the most distant
// from the others in two ways: its rates are EUR-based (srednji_tecaj is how
// many units of the listed currency make one EUR) and it formats decimals with
// a comma, per Croatian convention.
//
// Only the USD row is meaningful here: it gives EUR->USD, which inverts to the
// USD->EUR rate the other miners quote directly. Any other quote currency would
// need a cross-rate through two HNB rows, which this scaffolding does not do.
func parseHNB(quote, body string) (float64, error) {
	if quote != "EUR" {
		return 0, fmt.Errorf("fx-hnb quotes EUR crosses only, cannot serve %s", quote)
	}
	var rows []struct {
		Valuta  string `json:"valuta"`
		Srednji string `json:"srednji_tecaj"`
	}
	if err := json.Unmarshal([]byte(body), &rows); err != nil {
		return 0, fmt.Errorf("decoding response: %w", err)
	}
	for _, r := range rows {
		if r.Valuta != "USD" {
			continue
		}
		// Croatian decimal comma: "1,148100" is 1.1481, not 1148100.
		v, err := strconv.ParseFloat(strings.Replace(r.Srednji, ",", ".", 1), 64)
		if err != nil {
			return 0, fmt.Errorf("parsing srednji_tecaj %q: %w", r.Srednji, err)
		}
		if v <= 0 {
			return 0, fmt.Errorf("non-positive EUR/USD rate %v", v)
		}
		return 1 / v, nil // EUR->USD inverted to USD->EUR
	}
	return 0, fmt.Errorf("no USD row in response")
}

// parseCBR handles the Central Bank of Russia daily feed, which quotes every
// currency against RUB with an explicit Nominal (some currencies are quoted per
// 10 or per 100 units). USD->quote is derived as a cross-rate through RUB.
func parseCBR(quote, body string) (float64, error) {
	var resp struct {
		Valute map[string]struct {
			Nominal float64 `json:"Nominal"`
			Value   float64 `json:"Value"`
		} `json:"Valute"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding response: %w", err)
	}
	perUnit := func(code string) (float64, error) {
		v, ok := resp.Valute[code]
		if !ok {
			return 0, fmt.Errorf("no %s entry in response", code)
		}
		if v.Nominal <= 0 || v.Value <= 0 {
			return 0, fmt.Errorf("unusable %s entry (nominal %v, value %v)", code, v.Nominal, v.Value)
		}
		return v.Value / v.Nominal, nil // RUB per one unit
	}
	usdRUB, err := perUnit("USD")
	if err != nil {
		return 0, err
	}
	quoteRUB, err := perUnit(quote)
	if err != nil {
		return 0, err
	}
	return usdRUB / quoteRUB, nil
}

// parseBOC handles the Bank of Canada Valet series, which serves one observed
// series per request and carries its values as strings under the series key.
//
// The series is fixed at FXUSDCAD, so this miner can only answer USD/CAD.
func parseBOC(quote, body string) (float64, error) {
	if quote != "CAD" {
		return 0, fmt.Errorf("fx-boc serves the FXUSDCAD series only, cannot serve %s", quote)
	}
	var resp struct {
		Observations []map[string]json.RawMessage `json:"observations"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding response: %w", err)
	}
	if len(resp.Observations) == 0 {
		return 0, fmt.Errorf("no observations in response")
	}
	// Valet returns oldest-first; the pinned request asks for one observation,
	// but take the last defensively so a wider window still reads the latest.
	last := resp.Observations[len(resp.Observations)-1]
	raw, ok := last["FXUSDCAD"]
	if !ok {
		return 0, fmt.Errorf("no FXUSDCAD observation in response")
	}
	var cell struct {
		V string `json:"v"`
	}
	if err := json.Unmarshal(raw, &cell); err != nil {
		return 0, fmt.Errorf("decoding FXUSDCAD observation: %w", err)
	}
	rate, err := strconv.ParseFloat(cell.V, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing FXUSDCAD value %q: %w", cell.V, err)
	}
	return rate, nil
}

// parseRiksbank handles the Swedish Riksbank SWEA series, an array of dated
// observations for one pre-selected series. The series used for FX_NOW is
// SEKUSDPMI, which is SEK per USD — already USD->SEK, so no inversion.
func parseRiksbank(quote, body string) (float64, error) {
	if quote != "SEK" {
		return 0, fmt.Errorf("fx-riksbank serves the SEKUSDPMI series only, cannot serve %s", quote)
	}
	var rows []struct {
		Date  string  `json:"date"`
		Value float64 `json:"value"`
	}
	if err := json.Unmarshal([]byte(body), &rows); err != nil {
		return 0, fmt.Errorf("decoding response: %w", err)
	}
	if len(rows) == 0 {
		return 0, fmt.Errorf("no observations in response")
	}
	return rows[len(rows)-1].Value, nil // latest in the pinned window
}

// âââââââââââââââââââââââââââââââ pair eligibility ââââââââââââââââââââââââââââââ

// FXPairs are the currency pairs FX_NOW is scored on each epoch.
//
// They are fixed rather than derived from a live question because every miner
// in a round must be asked the *same* pair — a ranking built from miners
// answering different questions is not a ranking. Fixing them also keeps the
// LLM request-builder out of the deterministic path.
//
// EUR, CAD and SEK specifically: these three are the only combination that
// leaves no registered FX_NOW miner unscoreable. Three of the ten quote a
// single currency each (fx-hnb EUR, fx-boc CAD, fx-riksbank SEK), so dropping
// any one of these pairs would structurally exclude that miner from the intent
// rather than merely rank it low. The other seven quote all three.
var FXPairs = []string{"EUR", "CAD", "SEK"}

// quoteCoverage records which quote currencies each miner can actually serve.
// A miner that cannot quote the pair is not a bad miner — it is not a
// participant in that round, and scoring it 0 would be a false verdict about
// its quality rather than a statement about its coverage.
var quoteCoverage = map[string]func(quote string) bool{
	"fx-frankfurter": anyQuote,
	"fx-vatcomply":   anyQuote,
	"fx-er-api":      anyQuote,
	"fx-erapi-v4":    anyQuote,
	"fx-fawaz":       anyQuote,
	"fx-coinbase":    anyQuote,
	"fx-cbr":         anyQuote, // cross-rate through RUB
	"fx-hnb":         onlyQuote("EUR"),
	"fx-boc":         onlyQuote("CAD"),
	"fx-riksbank":    onlyQuote("SEK"),
}

func anyQuote(string) bool { return true }

func onlyQuote(supported string) func(string) bool {
	return func(q string) bool { return q == supported }
}

// coverageByIntent records, per intent, which question keys each miner can
// actually serve.
//
// The two wired intents sit at opposite ends of this. FX_NOW's coverage is
// genuinely uneven — three of its ten miners serve exactly one currency each —
// so the map does real work. ROUTE_ETA's is uniform: every scoreable miner
// accepts arbitrary coordinates, and the route baked into a slug like
// "route-osrm-sf-la" is only a registered *default*. That was verified rather
// than assumed: three differently-slugged miners asked the pinned London route
// all returned the same duration as the reference fetcher.
var coverageByIntent = map[string]map[string]func(key string) bool{
	"ASSET_RESERVE_ATTESTATION": reserveCoverage,
	"FX_NOW":                    quoteCoverage,
	"ROUTE_ETA":                 routeCoverage,
	"ONCHAIN_METRIC_VERIFY":     onchainCoverage,
	"LIQUIDITY_DEPTH_VERIFY":    liquidityCoverage,
	"MINING_HASHPRICE_VERIFY":   hashpriceCoverage,
	"GRID_POWER_PRICE":          gridCoverage,
	"MACRO_ECONOMIC_INDICATOR":  macroCoverage,
	"LIVE_SHELF_PRICE":          shelfCoverage,
}

// CanServeKey reports whether a miner can answer one of an intent's questions.
func CanServeKey(intentID, slug, key string) bool {
	byslug, ok := coverageByIntent[intentID]
	if !ok {
		// An intent with NO coverage table declares no restriction, so every
		// miner is eligible and a wrong answer is scored rather than skipped.
		//
		// This used to return false, which silently excluded every miner of
		// any intent not listed in the compiled-in map above — and the map
		// lists only the nine deterministic intents. EMAIL_SECURITY and
		// VULNERABILITY_TRIAGE are wired, have miners, and were reaching the
		// hybrid pass to find nothing eligible: an epoch where the intent is
		// simply absent, which reads exactly like every miner failing.
		//
		// Excluding by default is also the wrong direction on its own terms.
		// Coverage exists to avoid punishing a miner for a question it
		// structurally cannot answer (three FX_NOW miners serve one currency
		// each). Absent evidence of such a limit, the safe assumption is that
		// the miner can answer and should be held to its answer.
		return true
	}
	if f, ok := byslug[slug]; ok {
		return f(key)
	}

	// A SUFFIXED VARIANT of a listed miner inherits its coverage.
	//
	// The tables were written when each intent had one miner, and the
	// registration waves then added variants of it: reserveCoverage lists
	// "res-wbtc-por" while the nine registered miners are
	// "res-wbtc-por-1rpc", "-drpc", "-flashbots" and so on. Every one was
	// excluded, so five wired intents fetched ground truth, asked nineteen
	// miners and ranked nobody — silently, because an intent with no eligible
	// miners looks exactly like an intent whose miners all failed.
	//
	// Matching on "<listed>-" rather than a bare prefix is deliberate: it
	// admits "res-wbtc-por-drpc" without admitting an unrelated miner that
	// merely starts with the same letters.
	for known, f := range byslug {
		if strings.HasPrefix(slug, known+"-") {
			return f(key)
		}
	}

	// Within an intent that HAS a coverage table the table is otherwise
	// exhaustive, so an unrelated slug is excluded rather than assumed
	// capable. These omissions are deliberate and tested: deregistered
	// miners, and ones known not to answer the question at all (ROUTE_ETA's
	// Photon miners geocode place names and return no duration).
	return false
}

// CanServe reports whether an FX_NOW miner can quote USD against a currency.
//
// Retained as FX_NOW's named entry point; CanServeKey is the general form.
func CanServe(slug, quote string) bool {
	return CanServeKey("FX_NOW", slug, strings.ToUpper(quote))
}

// EligibleForKey filters miner slugs to those that can answer this question.
//
// Callers should rank only these. Including a miner that structurally cannot
// answer would score it 0 and push it to the bottom of the leaderboard for a
// reason that has nothing to do with the quality of the data it does serve.
func EligibleForKey(intentID string, slugs []string, key string) []string {
	eligible := make([]string, 0, len(slugs))
	for _, s := range slugs {
		if CanServeKey(intentID, s, key) {
			eligible = append(eligible, s)
		}
	}
	return eligible
}

// EligibleMiners filters FX_NOW miners to those that can quote the pair.
func EligibleMiners(slugs []string, quote string) []string {
	return EligibleForKey("FX_NOW", slugs, strings.ToUpper(quote))
}
