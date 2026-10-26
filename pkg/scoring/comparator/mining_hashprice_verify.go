package comparator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"net/http"
	"strconv"
	"strings"
	"time"
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

// ─────────────────────────────── the formula ──────────────────────────────────

// diff1Target is the difficulty-1 target, the numerator of the difficulty
// definition: difficulty = diff1_target / target.
//
// It is a big.Int rather than a float because the value is a 224-bit quantity
// and float64 carries 53 bits of mantissa. Computing difficulty as a ratio of
// two big.Ints and only then converting keeps the full precision of the
// division; building either operand as a float first would throw away ~170 bits
// before the divide.
var diff1Target, _ = new(big.Int).SetString(
	"00000000FFFF0000000000000000000000000000000000000000000000000000", 16)

// DifficultyFromBits decodes a block header's compact 'bits' field and returns
// the network difficulty it commits to.
//
// bits is Bitcoin's compact encoding of a 256-bit target: the top byte is a
// base-256 exponent and the low three bytes are the mantissa, so
// target = mantissa * 2^(8*(exponent-3)).
//
// Deriving difficulty here rather than reading mempool.space's own precomputed
// `difficulty` field is the build spec's explicit instruction, and it is not
// pedantry: the derived number is reproducible from raw chain data by any
// validator holding the block header, from any explorer, or from a full node —
// whereas trusting an aggregator's float would make the score depend on one
// third party's rounding. The two were cross-checked live and agree: bits
// 386218132 at height 800000 decodes to 53911173001054.586, which is exactly
// what mempool.space reports for that block.
func DifficultyFromBits(bits uint32) (float64, error) {
	exponent := bits >> 24
	mantissa := bits & 0x007fffff
	if mantissa == 0 {
		return 0, fmt.Errorf("compact bits %d has a zero mantissa", bits)
	}
	// Exponents below 3 mean a right-shifted mantissa. No mainnet block header
	// has ever carried one, but decoding it as a left shift would silently
	// produce a target billions of times too large and a difficulty near zero,
	// which would then make every miner's honest answer look wildly wrong.
	if exponent < 3 || exponent > 32 {
		return 0, fmt.Errorf("compact bits %d has an out-of-range exponent %d", bits, exponent)
	}
	target := new(big.Int).Lsh(big.NewInt(int64(mantissa)), uint(8*(exponent-3)))
	if target.Sign() <= 0 {
		return 0, fmt.Errorf("compact bits %d decodes to a non-positive target", bits)
	}
	diff, _ := new(big.Rat).SetFrac(diff1Target, target).Float64()
	if diff <= 0 || math.IsInf(diff, 0) || math.IsNaN(diff) {
		return 0, fmt.Errorf("compact bits %d decodes to an unusable difficulty %v", bits, diff)
	}
	return diff, nil
}

// SubsidySatsForHeight returns the block subsidy at a height, in satoshis.
//
// No fetch is involved: Bitcoin's issuance schedule is fixed, 50 BTC halving
// every 210,000 blocks. It is computed in integer satoshis (5_000_000_000 >>
// halvings) rather than as 50/2^n BTC in floating point because the halved
// values are exact integers in sats and the float form would start accumulating
// representation error at the deeper halvings for no benefit.
//
// After 33 halvings the subsidy is zero, and an intent scoring a zero-subsidy
// height would have a zero ground truth — RelTol divides by the truth, so that
// is an error rather than a score of nothing.
func SubsidySatsForHeight(height int64) (int64, error) {
	if height < 0 {
		return 0, fmt.Errorf("negative block height %d", height)
	}
	halvings := height / 210000
	if halvings >= 33 {
		return 0, fmt.Errorf("height %d is past the final halving; subsidy is zero", height)
	}
	return int64(5_000_000_000) >> uint(halvings), nil
}

// HashpriceSats computes the scored quantity from the two pinned primitives.
//
// The identity is the build spec's, verbatim:
//
//	expected_blocks_per_day_at_1PHs = (1e15 H/s * 86400 s) / (difficulty * 2^32)
//	hashprice_sats                  = round(expected_blocks_per_day * subsidy_sats)
//
// The 2^32 is the number of hashes per unit of difficulty — a difficulty-1
// target is hit once per 2^32 hashes on average — so difficulty * 2^32 is the
// expected hash count per block. Dividing a day of 1 PH/s hashing by it gives
// the expected number of blocks that hashrate finds per day, and multiplying by
// the subsidy gives the sats earned. Fees are deliberately absent: see the
// package-level note above.
//
// Rounding happens here rather than in ScaleValue because the parse_rule
// defines hashprice as already-integer sats; ScaleValue's multiplier for this
// intent is 1, so the two roundings agree and neither double-rounds.
func HashpriceSats(difficulty float64, subsidySats int64) (float64, error) {
	if difficulty <= 0 || math.IsNaN(difficulty) || math.IsInf(difficulty, 0) {
		return 0, fmt.Errorf("unusable difficulty %v", difficulty)
	}
	if subsidySats <= 0 {
		return 0, fmt.Errorf("unusable subsidy %d sats", subsidySats)
	}
	const hashesPerDayAt1PHs = 1e15 * 86400
	const hashesPerDifficulty = 4294967296 // 2^32
	blocksPerDay := hashesPerDayAt1PHs / (difficulty * hashesPerDifficulty)
	return math.Round(blocksPerDay * float64(subsidySats)), nil
}

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
// when probed, and the formula below derives 233214 from the block header
// alone.
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

// hashpriceParsers holds one entry per registered MINING_HASHPRICE_VERIFY
// miner, keyed by the slug in miner_registry.
//
// There is exactly one today (registration 2764), which is the honest state of
// this intent's field rather than an omission: the registry query for
// MINING_HASHPRICE_VERIFY returns a single active miner. A one-miner round is
// scoreable — the comparator checks an answer against truth, not against the
// field — but it produces a leaderboard of one, so it verifies rather than
// ranks.
var hashpriceParsers = map[string]parserFunc{
	"mine-mempool-hashprice": parseHashpriceField, // {"hashprice_sats_per_ph_s_day":233214,...}
}

// parseHashpriceField reads hashprice_sats_per_ph_s_day from a miner response.
//
// This is the one intent so far where the build spec's value_path matches the
// registered miner exactly — the miner's YAML declares the same field name in
// semantics.signal_mapping.label_field — so the parser is a straight read.
//
// It is decoded as json.Number rather than float64 so that a value delivered as
// a JSON *string* ("233214", which several explorer-style APIs do for large
// integers) is accepted instead of failing as a type error. json.Number keeps
// the digits exactly; a float64 decode of a large sats figure would be lossy
// above 2^53, which hashprice at a low difficulty could approach.
//
// The height is deliberately NOT cross-checked against the key here. A miner
// echoing the wrong height is a wrong ANSWER, not an unparseable one, and it
// must be scored badly rather than excluded — silently dropping it would let a
// miner that ignores the question avoid being caught by the very multi-height
// key set chosen to catch it. The cross-check belongs on the truth side, where
// it guards our own fetch, and that is where it is.
func parseHashpriceField(_ string, body string) (float64, error) {
	var resp map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding hashprice response: %w", err)
	}
	raw, ok := resp["hashprice_sats_per_ph_s_day"]
	if !ok {
		return 0, fmt.Errorf("no hashprice_sats_per_ph_s_day field in response")
	}
	s := strings.Trim(strings.TrimSpace(string(raw)), `"`)
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing hashprice_sats_per_ph_s_day %q: %w", s, err)
	}
	return v, nil
}

// ──────────────────────────────── block coverage ───────────────────────────────

// hashpriceCoverage records which pinned heights each miner can serve.
//
// The single registered miner takes the height as a query parameter with 800000
// only as a registered DEFAULT, so it can answer any height on the chain — the
// same shape as ROUTE_ETA's coordinates, and verified the same way rather than
// assumed. The map is kept rather than collapsed so a miner later registered
// against a bounded source (a single-epoch snapshot, say) can be given real
// coverage without reshaping the scoring pass.
var hashpriceCoverage = map[string]func(key string) bool{
	"mine-mempool-hashprice": anyQuote,
}

// ───────────────────────────────── ground truth ────────────────────────────────

// MempoolSource derives the reference hashprice from raw Bitcoin block header
// data served by mempool.space's keyless public API.
//
// It fetches PRIMITIVES and computes the answer here, which is the whole point.
// There is no endpoint anywhere that serves "the subsidy-only hashprice at
// height 800000"; anything claiming to would be some provider's own choice
// about fees and windows. Fetching only `bits` and `height` — two values
// committed in the block header and therefore identical from any explorer or
// full node — and applying the spec's formula makes the truth auditable: a
// validator can re-derive it from the header alone.
//
// It deliberately does NOT read mempool.space's precomputed `difficulty` field,
// per the build spec, and deliberately does not use the omni-chat /truth/ proxy
// that serves the registered miner. Truth fetched from the same host as the
// miner cannot ever fail that miner — the comparator would be comparing a value
// to itself, and would report a perfect score for a host that had started
// returning nonsense.
type MempoolSource struct {
	// BaseURL is the mempool.space-compatible REST API base.
	BaseURL string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewMempoolSource builds a MempoolSource with defaults suitable for epoch
// scoring.
func NewMempoolSource() *MempoolSource {
	return &MempoolSource{
		BaseURL: "https://mempool.space/api",
		Client:  &http.Client{Timeout: 15 * time.Second},
	}
}

// FetchReference derives the pinned hashprice for one block height.
//
// The pin argument is accepted and ignored, as for ROUTE_ETA but for a stronger
// reason: the question key IS the pin. A block height names an immutable point
// in the chain, so there is no "as of when" left to specify — that is precisely
// the property that makes this intent deterministic, and a date pin could only
// contradict it. The parameter stays because it is the TruthSource contract.
func (s *MempoolSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	block, ok := HashpriceBlockFor(key)
	if !ok {
		return Reference{}, fmt.Errorf("no pinned block for key %q", key)
	}
	base := strings.TrimRight(s.BaseURL, "/")

	// Two hops, because the API is addressed by hash: height -> hash -> header.
	hashBody, err := s.get(ctx, fmt.Sprintf("%s/block-height/%d", base, block.Height))
	if err != nil {
		return Reference{}, fmt.Errorf("resolving block %d to a hash: %w", block.Height, err)
	}
	hash := strings.TrimSpace(string(hashBody))
	// This endpoint returns a bare hash as text/plain, not JSON. A 64-hex-char
	// check is what distinguishes it from an error page served with a 200,
	// which would otherwise be fed to the next request as a "hash".
	if len(hash) != 64 {
		return Reference{}, fmt.Errorf("block-height %d returned %q, which is not a block hash", block.Height, hash)
	}

	raw, err := s.get(ctx, fmt.Sprintf("%s/block/%s", base, hash))
	if err != nil {
		return Reference{}, fmt.Errorf("fetching block %s: %w", hash, err)
	}

	bits, gotHeight, err := parseBlockHeader(string(raw))
	if err != nil {
		return Reference{}, fmt.Errorf("reading block %d header: %w", block.Height, err)
	}
	// Guard our own fetch, not the miner's answer. A hash/height mismatch here
	// would mean we computed truth for a different block than the one every
	// miner was asked about, which would fail the entire field at once for our
	// bug — the least visible and most damaging failure this path has.
	if gotHeight != block.Height {
		return Reference{}, fmt.Errorf("asked for block %d but got height %d", block.Height, gotHeight)
	}

	difficulty, err := DifficultyFromBits(bits)
	if err != nil {
		return Reference{}, err
	}
	subsidy, err := SubsidySatsForHeight(block.Height)
	if err != nil {
		return Reference{}, err
	}
	hashprice, err := HashpriceSats(difficulty, subsidy)
	if err != nil {
		return Reference{}, err
	}

	sats, err := ScaleValue("MINING_HASHPRICE_VERIFY", hashprice)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the reference hashprice: %w", err)
	}

	return Reference{
		RateBps: sats,
		RawRate: hashprice,
		Pair:    block.Name,
		// The block's own timestamp, not time.Now(): the fetch has no bearing
		// on the value, and dating the audit trail by when we happened to ask
		// would suggest the truth could have been different on another day.
		Date:   blockDate(string(raw)),
		Source: fmt.Sprintf("mempool.space:block/%d/bits=%d", block.Height, bits),
	}, nil
}

// get performs one keyless GET and returns the body.
func (s *MempoolSource) get(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("building mempool.space request: %w", err)
	}
	req.Header.Set("accept", "application/json")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s returned %s", url, resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", url, err)
	}
	return body, nil
}

// parseBlockHeader reads the two header fields the formula needs.
//
// Only `bits` and `height` are read. The response also carries `difficulty`,
// `nTx`, `extras.reward` and much else; none of it is used, because every
// additional field taken from the aggregator is another value a validator
// cannot independently reproduce from a block header.
//
// bits is validated as a uint32 rather than decoded into a float: it is a
// 32-bit header field, and a JSON number decoded into float64 and cast back
// would silently round the low mantissa bits, moving difficulty by a hair and
// the hashprice with it.
func parseBlockHeader(body string) (uint32, int64, error) {
	var parsed struct {
		Bits   json.Number `json:"bits"`
		Height json.Number `json:"height"`
	}
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		return 0, 0, fmt.Errorf("decoding block: %w", err)
	}
	if parsed.Bits == "" {
		return 0, 0, fmt.Errorf("block has no bits field")
	}
	bits, err := strconv.ParseUint(parsed.Bits.String(), 10, 32)
	if err != nil {
		return 0, 0, fmt.Errorf("parsing bits %q: %w", parsed.Bits.String(), err)
	}
	height, err := strconv.ParseInt(parsed.Height.String(), 10, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("parsing height %q: %w", parsed.Height.String(), err)
	}
	return uint32(bits), height, nil
}

// blockDate renders the block's own mined timestamp as an ISO date, for the
// audit trail. An unreadable timestamp is not fatal — it is provenance
// decoration, not an input to the score — so it degrades to empty.
func blockDate(body string) string {
	var parsed struct {
		Timestamp int64 `json:"timestamp"`
	}
	if err := json.Unmarshal([]byte(body), &parsed); err != nil || parsed.Timestamp <= 0 {
		return ""
	}
	return time.Unix(parsed.Timestamp, 0).UTC().Format("2006-01-02")
}

// Register this intent's ground-truth fetcher so a spec row can name it.
// Lives here rather than in the registry file so the factory sits beside the
// parsing code that understands this upstream's quirks.
//
// Parameters are unused today: block heights are the question keys.
func init() {
	RegisterTruthFactory("mempool", func(map[string]any) (TruthSource, error) {
		return NewMempoolSource(), nil
	})
}
