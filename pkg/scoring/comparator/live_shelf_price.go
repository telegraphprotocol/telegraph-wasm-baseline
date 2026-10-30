package comparator

import (
	"fmt"
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

// ──────────────────────────────── shelf coverage ───────────────────────────────
