package comparator

import (
	"fmt"
)

// MINING_HASHPRICE_VERIFY: Bitcoin mining revenue per PH/s per day, in whole
// satoshis, subsidy-only.
//
// This intent differs from FX_NOW and ROUTE_ETA in a way that shapes every
// decision below: hashprice is not a quantity anybody publishes, it is a
// quantity you DERIVE. There is no upstream to read it off. Two honest
// implementations can therefore land on different numbers for the same instant
// without either being wrong, and the three axes on which they diverge are
// exactly what the build spec engineers away or budgets for:
//
//   - Fees. Including mempool fee revenue in the per-block reward moves the
//     answer by roughly 1-3% under normal conditions and much more during a
//     congestion spike. Worse, fee revenue is a function of WHEN you look, so a
//     fees-inclusive figure is not reproducible across validators at all. The
//     spec excludes fees; the 400 bps band is sized to still absorb a miner who
//     includes them.
//   - The hashrate estimate. "Network hashrate" is not observable; every
//     explorer estimates it over a window (1d/3d/7d) of its own choosing, and
//     those windows disagree continuously. The spec sidesteps this entirely by
//     going through DIFFICULTY, which is an exact integer committed on-chain in
//     the block header, rather than through an estimated hashrate.
//   - The USD price feed. Avoided by denominating in sats rather than dollars.
//
// Two of the three largest divergence sources are thus designed out, which is
// why 400 bps is defensible where ROUTE_ETA needs 800: what remains is the fee
// question plus epoch staleness, not open-ended model disagreement.
//
// Because the value is derived, the key set pins specific block HEIGHTS. A
// "current" hashprice is not deterministic — it moves every retarget, and a
// miner read minutes before a validator would legitimately disagree. A
// height-pinned one is: both inputs to the formula are constant for the whole
// epoch containing that height (difficulty for the ~2016-block retarget epoch,
// subsidy for the ~210,000-block halving epoch), so the answer is stable
// forever and every validator asking the same height derives the same integer.

// ScaleHashprice is MINING_HASHPRICE_VERIFY's fixed-point multiplier: 1.
//
// Unlike FX_NOW's 1e4, this is not a scale we chose for precision — it is that
// the value is ALREADY in the atomic unit. The build spec is explicit:
// "satoshis are the atomic BTC unit, no further rescale needed". A satoshi is
// 1e-8 BTC, so the fixed-point work was done by Bitcoin's own denomination and
// multiplying again would put the published artifact's domain out of step with
// every other implementation of this spec for no gained resolution.
//
// Quantisation at this scale is negligible against the 400 bps band: one sat in
// ~233,000 is about 0.004 bps, four orders of magnitude finer than the
// tolerance, where ROUTE_ETA's scale-1 second is ~14 bps of its band.
const ScaleHashprice = 1

// ────────────────────────────── the pinned keys ───────────────────────────────

// HashpriceBlock is one pinned Bitcoin block height the intent is scored on.
type HashpriceBlock struct {
	Key    string // the question key; the height as a decimal string
	Height int64  //
	Name   string // human description for logs and the audit trail
}

// HashpriceBlocks are the block heights MINING_HASHPRICE_VERIFY is scored on.
//
// 800000 first because it is the registered miner's own declared default (its
// input_schema carries `height: default "800000"`), so it is the one height
// with an independently known expected value: the miner returned 233214 sats
// when probed, which the build spec's formula also derives from the block
// header alone.
//
// The other two are not padding. They sit in different HALVING epochs — 800000
// is the 6.25 BTC era, 840000 is the first block of the 3.125 BTC era, 900000
// is well inside it — so a miner that hardcodes a subsidy, or that computes
// floor(height/210000) with an off-by-one at the boundary, answers 800000
// correctly and misses the other two by a factor of two. 840000 is chosen
// precisely because it is the halving block itself, which is where that
// off-by-one lives. A single-height question set could not distinguish a
// correct implementation from one that replays a memorised number.
var HashpriceBlocks = []HashpriceBlock{
	{Key: "800000", Height: 800000, Name: "BTC block 800000 (6.25 BTC subsidy epoch)"},
	{Key: "840000", Height: 840000, Name: "BTC block 840000 (fourth halving, first 3.125 BTC block)"},
	{Key: "900000", Height: 900000, Name: "BTC block 900000 (3.125 BTC subsidy epoch)"},
}

// HashpriceKeys is the fixed question set, in the order rounds are run.
var HashpriceKeys = hashpriceKeys()

func hashpriceKeys() []string {
	keys := make([]string, 0, len(HashpriceBlocks))
	for _, b := range HashpriceBlocks {
		keys = append(keys, b.Key)
	}
	return keys
}

// HashpriceBlockFor resolves a question key to its pinned block.
func HashpriceBlockFor(key string) (HashpriceBlock, bool) {
	for _, b := range HashpriceBlocks {
		if b.Key == key {
			return b, true
		}
	}
	return HashpriceBlock{}, false
}

// hashpriceQuestion is the question put to every miner for one pinned height.
//
// It states the height, the unit and the fee exclusion, because all three are
// load-bearing and only the height survives into the miner's URL. A question
// that asked for "the current hashprice" would be unscoreable — the answer
// would depend on when the miner read it — and one that omitted "excluding
// transaction fees" would invite a fees-inclusive answer that is a few hundred
// bps rich against a truth that excludes them, eating most of the band for a
// disagreement about the question rather than about the data.
func hashpriceQuestion(key string) string {
	b, ok := HashpriceBlockFor(key)
	if !ok {
		return fmt.Sprintf("What is the Bitcoin hashprice in sats per PH/s per day at block height %s?", key)
	}
	return fmt.Sprintf(
		"What is the Bitcoin mining hashprice at block height %d, in satoshis per PH/s per day, "+
			"computed from the block's difficulty and subsidy and excluding transaction fees?",
		b.Height)
}

// describeHashprice renders a key for logs and the stored audit trail.
func describeHashprice(key string) string {
	if b, ok := HashpriceBlockFor(key); ok {
		return "btc:" + b.Key
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// ──────────────────────────────── block coverage ───────────────────────────────
