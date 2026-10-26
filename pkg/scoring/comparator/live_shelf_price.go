package comparator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// LIVE_SHELF_PRICE: the retail price of one pinned product observation, in
// integer currency cents.
//
// The name oversells what is actually scored, and the gap matters enough to
// state up front. An Open Prices `price_id` is a crowd-sourced HISTORICAL
// RECORD — a photographed price tag at one store on one date — and the API is
// append-only, so a record never changes once submitted. Pinning one is what
// makes this intent scoreable at all: it gives every validator the same answer
// no matter when they ask, exactly as pinning a block height does for the
// on-chain intents. What it does NOT give is a live shelf price. Nothing here
// moves between epochs, so a ranking produced by this intent measures whether a
// miner can faithfully read a fixed record, not whether it tracks a market.
//
// The build spec is explicit about the same thing ("only the pinned OBSERVATION
// is canonical, not the live market price it captured"), and sized the
// tolerance for the LIVE mode this intent is eventually meant for rather than
// for the pinned mode that is actually wired. Read a 10000 here as "parsed the
// record correctly", not as "priced the shelf correctly".

// ─────────────────────────────── the currency trap ─────────────────────────────
//
// Open Prices records carry their own currency: the same endpoint serves EUR,
// USD, CHF and others, and price_id 28372 really is 1.35 USD while 331088 is
// 5.23 EUR. The comparator sees only an integer, and `.price_cents` carries no
// currency at all — so scoring a EUR miner value against a USD truth produces a
// perfectly plausible integer that is simply the wrong fact. Under RelTol that
// lands as "this miner is 20% off" rather than as an error, which is the worst
// kind of failure: silent and wrong-looking-right.
//
// Two defences, both deliberate:
//
//  1. The pinned key set below is EUR-only, verified record by record against
//     the upstream. A single currency across the round means no cross-currency
//     comparison is even expressible.
//  2. Both the truth parse and the miner parse ASSERT the currency and fail
//     loudly if it is not the expected one. Belt and braces, because (1) is a
//     property of a list a future editor can extend carelessly, whereas (2)
//     catches the mistake at the point it would do damage. A record whose
//     currency changed under us — or a key added without checking — becomes a
//     parse error, and a parse error excludes the miner from the round rather
//     than scoring it zero.
//
// The alternative — converting currencies to a common unit — was rejected: it
// would put an FX rate, and therefore a second ground-truth source with its own
// staleness, inside a price comparison. ShelfKeys stays single-currency instead.

// shelfCurrency is the one currency LIVE_SHELF_PRICE is scored in.
//
// EUR because the pinned records are French supermarket observations and
// because the registered miner's own default (price_id=1) is EUR. It is a
// constant rather than a per-key field so that adding a key of another currency
// is a visible, deliberate change rather than a quiet one.
const shelfCurrency = "EUR"

// ScaleShelfPrice is LIVE_SHELF_PRICE's fixed-point multiplier: 100, i.e.
// currency cents.
//
// From the build spec's parse_rule, verbatim: "price_in_major_units * 100,
// rounded to the nearest integer -> i64 (currency cents, NEVER float). E.g.
// price=5.23 -> price_cents=523." The reference host live_shelf_price_price.mjs
// emits the same thing (SCALE_CENTS = 100, Math.round(price * 100)), and the
// registered miner already reports `price_cents` on exactly this scale — 27.7
// arrives as 2770 — so the host, the miner and the spec agree.
//
// Cents is also the natural precision of the underlying fact: retail prices are
// quoted to two decimals, so this scale is lossless rather than a truncation
// compromise. That is unlike FX_NOW, where 1e4 quantises a rate to ~5.7 bps of
// a 50 bps band.
const ScaleShelfPrice = 100

// ShelfObservation is one pinned Open Prices record.
//
// Currency and the expected price are recorded here rather than only fetched,
// so that a record silently changing (or a key pointing at the wrong record)
// shows up as a mismatch in the live test instead of as a shifted truth every
// miner is then scored against.
type ShelfObservation struct {
	Key         string  // question key: the Open Prices price_id, as text
	Currency    string  // the record's own currency — must be shelfCurrency
	Price       float64 // the observed price in major units, as verified live
	ProductCode string  // barcode, for the question text and the audit trail
	ProductName string  // human label for logs; empty for older records
	Store       string  // merchant, for the audit trail
	Date        string  // the observation date the record carries
}

// ShelfObservations are the pinned product observations LIVE_SHELF_PRICE is
// scored on each epoch.
//
// Every one was fetched from prices.openfoodfacts.org and confirmed EUR before
// being listed; the Price field is the value observed at that fetch. They span
// two stores and two eras (2023 L'Éléfàn, 2026 E.Leclerc) and two orders of
// magnitude of price (0.99 to 27.70) on purpose: a miner that returns a
// hardcoded number, or that drops the cents scaling, cannot fake all four.
//
// Four keys rather than one because the sole registered miner takes price_id as
// a query parameter with a registered DEFAULT of "1". Asking only the default
// would not distinguish a miner that honours the parameter from one that
// replays its default — the same failure ROUTE_ETA's three routes exist to
// catch.
var ShelfObservations = []ShelfObservation{
	{
		Key: "1", Currency: "EUR", Price: 27.70,
		ProductCode: "1541513213246", Store: "L'Éléfàn", Date: "2023-11-27",
	},
	{
		Key: "100", Currency: "EUR", Price: 3.73,
		ProductCode: "3267110008242", Store: "L'Éléfàn", Date: "2023-11-27",
	},
	{
		Key: "331088", Currency: "EUR", Price: 5.23,
		ProductCode: "3564700514047", ProductName: "CHATAIGNES ENT PELEES NJ 420G",
		Store: "E.Leclerc", Date: "2026-09-15",
	},
	{
		Key: "331090", Currency: "EUR", Price: 0.99,
		ProductCode: "3564700512562", Store: "E.Leclerc", Date: "2026-09-15",
	},
}

// ShelfKeys is the fixed question set, in the order rounds are run.
var ShelfKeys = shelfKeys()

func shelfKeys() []string {
	keys := make([]string, 0, len(ShelfObservations))
	for _, o := range ShelfObservations {
		keys = append(keys, o.Key)
	}
	return keys
}

// ShelfObservationFor resolves a question key to its pinned observation.
func ShelfObservationFor(key string) (ShelfObservation, bool) {
	for _, o := range ShelfObservations {
		if o.Key == key {
			return o, true
		}
	}
	return ShelfObservation{}, false
}

// shelfQuestion is the question put to every miner for one pinned observation.
//
// It names the price_id explicitly, because that — not the product name — is
// what identifies the record. A question phrased as "what does this chestnut
// tin cost" has many true answers across stores and dates; a question naming
// the record has exactly one, which is the whole basis for scoring it.
//
// The currency is stated too. The miner's response carries a currency and this
// path rejects anything that is not EUR, so a question that left it implicit
// would be asking for a value that the parser is entitled to throw away.
func shelfQuestion(key string) string {
	o, ok := ShelfObservationFor(key)
	if !ok {
		return fmt.Sprintf("What is the recorded retail price in %s cents for Open Prices price_id %s?", shelfCurrency, key)
	}
	label := o.ProductName
	if label == "" {
		label = "barcode " + o.ProductCode
	}
	return fmt.Sprintf(
		"What is the recorded retail price, in %s cents, for Open Prices price_id %s (%s at %s on %s)?",
		shelfCurrency, o.Key, label, o.Store, o.Date)
}

// describeShelf renders a key for logs and the stored audit trail.
func describeShelf(key string) string {
	if o, ok := ShelfObservationFor(key); ok {
		return o.Store + ":" + o.Key
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// shelfParsers holds one entry per registered LIVE_SHELF_PRICE miner, keyed by
// the slug in miner_registry.
//
// One miner, shelf-open-prices (registration 2759). That is the entire field,
// which is worth saying plainly: a one-miner intent cannot produce a ranking,
// only a pass/fail verdict on the single participant. The scoring path runs
// regardless — it is the same code either way — but a leaderboard of one is not
// evidence about anything.
//
// It is also a proxy over the same upstream this package fetches truth from, so
// a correct answer here is a round trip, not an independent measurement. Truth
// is fetched from prices.openfoodfacts.org DIRECTLY rather than through the
// miner's own omni-chat host precisely so that the two paths can at least
// diverge; if truth came from the proxy, no miner could ever fail.
var shelfParsers = map[string]parserFunc{
	"shelf-open-prices": parseOpenPricesMiner, // {"price_cents":2770,"currency":"EUR",...}
}

// parseOpenPricesMiner reads the shelf price out of the miner's response.
//
// It returns MAJOR UNITS, not cents, even though the miner reports cents. Every
// parserFunc in this package returns an unscaled value and lets ScaleValue
// apply the intent's multiplier once, so that the miner's value and the
// reference are scaled and rounded by the same code. Dividing by 100 here and
// multiplying by 100 there looks redundant, and is: the redundancy is what
// keeps this parser from being the one place a scale can silently diverge.
//
// Three checks that each guard a silent wrong answer rather than a crash:
//
//   - currency must be EUR. See the currency-trap note above; this is the check
//     that makes a cross-currency comparison impossible rather than merely
//     unlikely.
//   - price_id must match the key asked. The miner takes price_id as a query
//     parameter with a registered default of "1", so a request that lost the
//     parameter comes back as a valid, well-formed answer to a DIFFERENT
//     question. Without this check that scores as a wrong price.
//   - price_cents and price must agree, when both are present. They are two
//     spellings of one number, so a disagreement means the proxy's own scaling
//     is broken, and the intent's whole scale rests on it.
func parseOpenPricesMiner(key, body string) (float64, error) {
	// price_cents is decoded as json.Number rather than int64 so a proxy that
	// ever starts emitting "2770" as a string, or 2770.0, is a clear parse
	// failure here instead of a silent zero from a type mismatch.
	var resp struct {
		PriceID    json.Number  `json:"price_id"`
		Price      *json.Number `json:"price"`
		PriceCents *json.Number `json:"price_cents"`
		Currency   string       `json:"currency"`
	}
	dec := json.NewDecoder(strings.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&resp); err != nil {
		return 0, fmt.Errorf("decoding Open Prices miner response: %w", err)
	}

	if resp.Currency == "" {
		return 0, fmt.Errorf("response carries no currency; a price_cents with no currency is not a scoreable value")
	}
	if !strings.EqualFold(resp.Currency, shelfCurrency) {
		return 0, fmt.Errorf(
			"record is priced in %s, but LIVE_SHELF_PRICE is scored in %s; comparing them would be a silent wrong answer",
			resp.Currency, shelfCurrency)
	}

	// An answer to the wrong price_id is not a bad price, it is a bad answer.
	if got := strings.TrimSpace(resp.PriceID.String()); got != "" && got != key {
		return 0, fmt.Errorf("response is for price_id %s, but %s was asked", got, key)
	}

	if resp.PriceCents == nil {
		return 0, fmt.Errorf("no price_cents in response")
	}
	cents, err := strconv.ParseFloat(resp.PriceCents.String(), 64)
	if err != nil {
		return 0, fmt.Errorf("parsing price_cents %q: %w", resp.PriceCents.String(), err)
	}

	// Cross-check the two spellings. Tolerance is half a cent because
	// price_cents is a rounding of price, so 0.005 of drift is legitimate and
	// anything larger means they describe different numbers.
	if resp.Price != nil {
		major, err := strconv.ParseFloat(resp.Price.String(), 64)
		if err != nil {
			return 0, fmt.Errorf("parsing price %q: %w", resp.Price.String(), err)
		}
		if diff := major*ScaleShelfPrice - cents; diff > 0.5 || diff < -0.5 {
			return 0, fmt.Errorf("price %v and price_cents %v disagree; the miner's own scaling is inconsistent", major, cents)
		}
	}

	return cents / ScaleShelfPrice, nil
}

// ──────────────────────────────── shelf coverage ───────────────────────────────

// shelfCoverage records which pinned observations each miner can serve.
//
// Uniform, and for a stronger reason than ROUTE_ETA's: price_id is a declared
// query parameter in the miner's registration YAML rather than a value baked
// into its slug, and all four pinned ids were probed against the live miner and
// answered correctly (1 -> 2770, 100 -> 373, 331088 -> 523, 331090 -> 99).
//
// The map is kept rather than collapsed so that a second miner registered
// against a bounded catalog — a single merchant's own API, say — can be given
// real coverage without reshaping the scoring pass.
var shelfCoverage = map[string]func(key string) bool{
	"shelf-open-prices": anyQuote,
}

// ───────────────────────────────── ground truth ────────────────────────────────

// OpenPricesSource reads the reference shelf price from Open Prices directly.
//
// DIRECTLY is the load-bearing word. The one registered miner is a proxy served
// from omni-chat over this same upstream, and fetching truth through that proxy
// would make the comparator structurally incapable of failing it: any error the
// proxy introduced would appear identically on both sides of the comparison and
// cancel. Going to prices.openfoodfacts.org means the proxy's scaling, its
// currency handling and its price_id routing are all things the score can
// actually catch — and the price_cents/price cross-check in the miner parser
// exists for the same reason.
//
// What this still does not buy is independence of the FACT. Miner and truth
// read the same crowd-sourced record, so a wrong record is wrong for both. That
// is inherent to a pinned-observation intent and is not something a second
// endpoint would fix.
//
// The endpoint matches the reference host live_shelf_price_price.mjs
// (GET /api/v1/prices/{id}), so a score computed here can be checked against a
// run of live_shelf_price_score.mjs on the same inputs.
type OpenPricesSource struct {
	// BaseURL is the Open Prices API base.
	BaseURL string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewOpenPricesSource builds an OpenPricesSource with defaults suitable for
// epoch scoring. The API is fully keyless, so there is no credential to wire.
func NewOpenPricesSource() *OpenPricesSource {
	return &OpenPricesSource{
		BaseURL: "https://prices.openfoodfacts.org/api/v1",
		Client:  &http.Client{Timeout: 15 * time.Second},
	}
}

// FetchReference resolves the pinned observation for one key, already scaled to
// integer currency cents.
//
// The pin argument is accepted and ignored, as for OSRM but for a different
// reason: an Open Prices record is append-only and immutable, so there is no
// "as of when" to express — the price_id IS the pin. The parameter stays
// because it is the TruthSource contract.
func (s *OpenPricesSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	obs, ok := ShelfObservationFor(key)
	if !ok {
		// Refusing an unpinned id is a determinism requirement, not tidiness:
		// scoring against an arbitrary caller-supplied record would let the
		// question differ between validators.
		return Reference{}, fmt.Errorf("no pinned shelf observation for price_id %q", key)
	}

	endpoint := fmt.Sprintf("%s/prices/%s", strings.TrimRight(s.BaseURL, "/"), url.PathEscape(obs.Key))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return Reference{}, fmt.Errorf("building Open Prices request: %w", err)
	}
	req.Header.Set("accept", "application/json")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return Reference{}, fmt.Errorf("fetching the Open Prices reference observation: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Reference{}, fmt.Errorf("Open Prices request for price_id %s returned %s", obs.Key, resp.Status)
	}
	// The full record embeds the whole product and location objects and runs to
	// a few KB; the limit is generous rather than tight for that reason.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return Reference{}, fmt.Errorf("reading the Open Prices response: %w", err)
	}

	price, currency, servedDate, err := parseOpenPricesRecord(obs.Key, string(body))
	if err != nil {
		return Reference{}, fmt.Errorf("reading the Open Prices reference observation: %w", err)
	}

	cents, err := ScaleValue("LIVE_SHELF_PRICE", price)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the reference shelf price: %w", err)
	}

	return Reference{
		RateBps: cents,
		RawRate: price,
		Pair:    fmt.Sprintf("price_id=%s %s %s", obs.Key, obs.ProductCode, currency),
		Date:    servedDate,
		Source:  "openprices:price_id=" + obs.Key,
	}, nil
}

// parseOpenPricesRecord reads price, currency and date from the upstream
// record.
//
// It is a separate function from parseOpenPricesMiner rather than shared with
// it, which breaks the pattern ROUTE_ETA follows (where truth and miner run
// through the identical parser). The envelopes genuinely differ — upstream
// returns `price` as a float in major units with no cents field, the proxy
// returns both — so one function would have to accept either shape and would
// therefore stop catching a proxy that dropped price_cents. The cost is that
// the two readings could drift; the currency and id assertions are duplicated
// in both for that reason.
func parseOpenPricesRecord(key, body string) (price float64, currency string, date string, err error) {
	var rec struct {
		ID       json.Number `json:"id"`
		Price    *float64    `json:"price"`
		Currency string      `json:"currency"`
		Date     string      `json:"date"`
		// Recorded but not acted on: a discounted price is still the price on
		// the shelf, which is the fact this intent scores.
		Discounted bool `json:"price_is_discounted"`
	}
	if err := json.Unmarshal([]byte(body), &rec); err != nil {
		return 0, "", "", fmt.Errorf("decoding Open Prices record: %w", err)
	}

	if got := strings.TrimSpace(rec.ID.String()); got != "" && got != key {
		return 0, "", "", fmt.Errorf("Open Prices returned record %s, but price_id %s was requested", got, key)
	}
	if rec.Currency == "" {
		return 0, "", "", fmt.Errorf("record %s carries no currency", key)
	}
	// The same assertion as in the miner parser, deliberately duplicated: this
	// is the one that stops a truth value in the wrong currency from being the
	// yardstick every miner is then measured against.
	if !strings.EqualFold(rec.Currency, shelfCurrency) {
		return 0, "", "", fmt.Errorf(
			"record %s is priced in %s, but LIVE_SHELF_PRICE is scored in %s; it must not be pinned as a key",
			key, rec.Currency, shelfCurrency)
	}
	// A nil price is distinguished from a zero price on purpose. Open Prices
	// can carry a record with the field absent, and decoding that into a plain
	// float64 would yield 0 — a free product, scored as a real answer.
	if rec.Price == nil {
		return 0, "", "", fmt.Errorf("record %s carries no price", key)
	}
	if *rec.Price <= 0 {
		return 0, "", "", fmt.Errorf("record %s has an unusable price: %v", key, *rec.Price)
	}
	return *rec.Price, strings.ToUpper(rec.Currency), rec.Date, nil
}

// Register this intent's ground-truth fetcher so a spec row can name it.
// Lives here rather than in the registry file so the factory sits beside the
// parsing code that understands this upstream's quirks.
//
// Parameters are unused today: price ids are the question keys.
func init() {
	RegisterTruthFactory("open_prices", func(map[string]any) (TruthSource, error) {
		return NewOpenPricesSource(), nil
	})
}
