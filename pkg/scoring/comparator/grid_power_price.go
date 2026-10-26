package comparator

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
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

// gridParsers holds one entry per registered GRID_POWER_PRICE miner, keyed by
// the slug in miner_registry (registration ID 2758).
//
// There is exactly one. That is the honest state of this intent, and it has a
// consequence worth stating at the point of definition rather than burying in a
// report: with a single miner there is no ranking. A one-entry leaderboard
// reports whether that miner agrees with CAISO, which is a liveness check, not
// a quality signal.
var gridParsers = map[string]parserFunc{
	"grid-caiso-np15-dam": parseCAISOMiner, // {"lmp_usd_per_mwh":38.09459,"lmp_usd_per_mwh_milli":38095,...}
}

// parseCAISOMiner reads the $/MWh price out of the registered miner's response.
//
// It reads lmp_usd_per_mwh — the UNSCALED dollar figure — and not the
// lmp_usd_per_mwh_milli field that the miner's registration YAML names as its
// label_field and that the build spec's value_path points at. That looks like a
// contradiction and is a deliberate choice, for one reason: scaling has to
// happen in exactly one place. ScaleValue multiplies by 1000 for this intent,
// so returning the already-multiplied milli field here would scale it twice and
// hand the WASM a value a thousandfold too large. Reading dollars keeps this
// parser's contract identical to every other parser in the package — return the
// natural unit, let ScaleValue produce the integer domain.
//
// The milli field is still used, as a cross-check. The miner computes it
// itself, so if its own rounding disagrees with ours by more than half a
// milli-dollar the two fields are not describing the same number and the
// response is rejected rather than silently preferred one way. This is cheap
// and it catches the exact failure the two-field schema invites.
//
// The interval is verified too. The upstream echoes the startdatetime it was
// actually asked for, so a miner that ignored the requested hour and served its
// registered default can be caught here instead of being scored — adjacent DAM
// hours are thousands of bps apart, so an unchecked mismatch would show up as a
// catastrophic score for what is really a request-building bug on our side.
func parseCAISOMiner(key, body string) (float64, error) {
	var resp struct {
		OK    *bool  `json:"ok"`
		Error string `json:"error"`

		Node          string `json:"node"`
		Market        string `json:"market"`
		StartDateTime string `json:"startdatetime"`

		LMP      *float64 `json:"lmp_usd_per_mwh"`
		LMPMilli *float64 `json:"lmp_usd_per_mwh_milli"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding CAISO miner response: %w", err)
	}

	// The miner proxies OASIS, so an upstream failure comes back as HTTP 502
	// with {"ok":false,"error":"HTTP Error 429: Too Many Requests"} and no price
	// field at all. Captured live during this port. It must be an error, not a
	// zero — OASIS throttles, so this is a routine occurrence, and scoring it 0
	// would rank the miner last for our upstream's rate limit.
	if resp.OK != nil && !*resp.OK {
		if resp.Error != "" {
			return 0, fmt.Errorf("CAISO miner reported upstream failure: %s", resp.Error)
		}
		return 0, fmt.Errorf("CAISO miner reported ok=false with no error detail")
	}
	if resp.LMP == nil {
		return 0, fmt.Errorf("no lmp_usd_per_mwh in CAISO miner response")
	}

	if resp.Market != "" && !strings.EqualFold(resp.Market, LMPMarket) {
		return 0, fmt.Errorf("expected the %s market run, got %q", LMPMarket, resp.Market)
	}
	if resp.Node != "" && resp.Node != LMPNode {
		return 0, fmt.Errorf("expected node %s, got %q", LMPNode, resp.Node)
	}
	if err := checkEchoedInterval(key, resp.StartDateTime); err != nil {
		return 0, err
	}

	// Cross-check the miner's own scaling against ours. Half a milli-dollar is
	// the widest honest disagreement two correct rounders can have on the same
	// input; anything beyond it means the two fields disagree about the price.
	if resp.LMPMilli != nil {
		want := roundHalfAwayFromZero(*resp.LMP * ScaleGridPowerPrice)
		if diff := *resp.LMPMilli - want; diff > 0.5 || diff < -0.5 {
			return 0, fmt.Errorf(
				"CAISO miner's lmp_usd_per_mwh_milli (%v) disagrees with its lmp_usd_per_mwh (%v, scales to %v)",
				*resp.LMPMilli, *resp.LMP, want)
		}
	}
	return *resp.LMP, nil
}

// checkEchoedInterval confirms a response describes the hour that was asked
// for.
//
// OASIS's own wire format for an instant is "20260913T14:00-0000", the miner
// echoes that same string back, and the CSV carries a third spelling
// ("2026-09-13T14:00:00-00:00"), so the comparison is made on parsed instants
// rather than on text. Comparing the strings would reject three correct
// renderings of the same moment.
//
// An empty echo is accepted: not every response shape carries the request back,
// and refusing one that omits it would exclude a miner for a field that is not
// part of the scored value.
func checkEchoedInterval(key, echoed string) error {
	if strings.TrimSpace(echoed) == "" {
		return nil
	}
	want, err := time.Parse(time.RFC3339, key)
	if err != nil {
		return nil // an unpinned key cannot be checked against; not the miner's fault
	}
	got, err := parseOASISInstant(echoed)
	if err != nil {
		return fmt.Errorf("unreadable interval %q in response: %w", echoed, err)
	}
	if !got.Equal(want) {
		return fmt.Errorf(
			"response is for interval %s, not the pinned %s (a different DAM hour is a different price)",
			got.UTC().Format(time.RFC3339), want.UTC().Format(time.RFC3339))
	}
	return nil
}

// parseOASISInstant reads the three spellings of an instant this intent meets:
// OASIS's compact request form ("20260913T14:00-0000"), the CSV's offset form
// ("2026-09-13T14:00:00-00:00") and plain RFC3339.
func parseOASISInstant(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	for _, layout := range []string{
		"20060102T15:04-0700",
		"2006-01-02T15:04:05-07:00",
		time.RFC3339,
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised instant format %q", s)
}

// roundHalfAwayFromZero matches the reference host's Math.round semantics for
// the sign-preserving case.
//
// math.Round in Go is already half-away-from-zero, so this is a named wrapper
// rather than arithmetic; it exists so the cross-check above reads as the same
// rule the build spec states ("rounded to the nearest integer", sign
// preserved), instead of leaving a reader to confirm Go's rounding mode.
func roundHalfAwayFromZero(v float64) float64 {
	if v < 0 {
		return -float64(int64(-v + 0.5))
	}
	return float64(int64(v + 0.5))
}

// ────────────────────────────────── coverage ───────────────────────────────────

// gridCoverage records which settlement hours each miner can serve.
//
// The one registered miner takes node, market and interval as query parameters
// with no server-side restriction, so it can serve any settled hour — verified
// by asking it for the pinned hour, which is NOT its registered default, and
// getting the pinned hour's price back.
//
// The map is kept rather than collapsed into "everyone serves everything" for
// the same reason ROUTE_ETA's is: a miner later registered against a single
// fixed hour, or against a different ISO, needs real coverage here and should
// not require reshaping the scoring pass to get it.
var gridCoverage = map[string]func(key string) bool{
	"grid-caiso-np15-dam": anyQuote,
}

// ───────────────────────────────── ground truth ────────────────────────────────

// CAISOSource reads the reference DAM LMP from CAISO OASIS.
//
// It goes to oasis.caiso.com directly, matching the endpoint and query the
// reference host grid_power_price_price.mjs uses. Fetching it through the
// omni-chat host that serves the miner would be the more convenient path and is
// specifically avoided: if truth and miner come from the same process, the
// comparator can never fail anyone, and the score would measure only that one
// server is self-consistent. Here the miner and the truth fetch are independent
// round trips to CAISO, so a miner that fabricates, caches staleness or reads
// the wrong hour is actually detectable.
//
// Read the resulting ranking with the field size in mind, though: one miner
// proxying the same OASIS report the truth fetch reads will agree to the last
// digit, and it did. That is evidence the miner is honestly proxying CAISO. It
// is not evidence about quality, because there is nothing to compare it to.
type CAISOSource struct {
	// BaseURL is the OASIS SingleZip endpoint.
	BaseURL string
	// Node is the pricing node queried. Defaults to LMPNode.
	Node string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewCAISOSource builds a CAISOSource with defaults suitable for epoch scoring.
//
// HTTPS, deliberately: the reference host grid_power_price_price.mjs and the
// build spec's documented method both use plain http, but OASIS serves the same
// SingleZip endpoint over TLS — checked during this port, HTTP 200 with the
// identical zip content type — and there is no reason for the node's own truth
// fetch to travel in clear text. The scheme is not part of the comparator
// contract; only the value is, and it is the same value. ROUTE_ETA's source
// makes the same upgrade for the same reason.
//
// The timeout is 30s against ROUTE_ETA's 15s because the response is ZIP
// wrapped: the build spec measures a redirect plus ZIP fetch plus unzip at
// 1-2s, and OASIS is markedly slower under load.
func NewCAISOSource() *CAISOSource {
	return &CAISOSource{
		BaseURL: "https://oasis.caiso.com/oasisapi/SingleZip",
		Node:    LMPNode,
		Client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// FetchReference resolves the settled DAM LMP for one pinned hour, already
// scaled to milli-dollars per MWh.
//
// The pin argument is accepted and ignored, as it is for OSRM, but for the
// opposite reason: ROUTE_ETA's source has no notion of time, whereas this one
// is entirely about time and the time is carried by the question KEY. The key
// IS the pin. A Pin.Date here would be a second, conflicting way to say the
// same thing, and the one that lost would do so silently.
func (s *CAISOSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	iv, ok := GridIntervalFor(key)
	if !ok {
		return Reference{}, fmt.Errorf("no pinned settlement interval for key %q", key)
	}
	start, err := time.Parse(time.RFC3339, iv.Start)
	if err != nil {
		return Reference{}, fmt.Errorf("pinned interval %q is not a valid instant: %w", iv.Start, err)
	}
	start = start.UTC()

	// Refuse an unsettled hour, matching the reference host. A DAM hour that
	// has not cleared yet either returns nothing or returns a figure that will
	// later be replaced, and a ground truth that changes after the fact is
	// worse than no score: it makes two validators scoring the same epoch at
	// different moments disagree, which is precisely what the deterministic
	// path exists to prevent.
	if !start.Before(time.Now().UTC().Add(-time.Hour)) {
		return Reference{}, fmt.Errorf(
			"pinned interval %s is not a past, settled DAM hour", iv.Start)
	}

	node := s.Node
	if node == "" {
		node = LMPNode
	}
	end := start.Add(time.Hour)

	// OASIS wants its own compact instant format; RFC3339 is rejected.
	//
	// The zone suffix is hardcoded as the literal "-0000" and NOT produced by a
	// Go layout, which is the single sharpest trap in this file. Formatting a
	// UTC time with the "-0700" layout yields "+0000", and OASIS answers that
	// with HTTP 200 and a ZIP whose single entry is an ERROR document rather
	// than the CSV — no status code, no exception, just a report that quietly
	// does not contain a price. It cost a debugging session here, and it is
	// precisely the plausible-looking silent failure the deterministic path is
	// supposed to make impossible. Both instants are converted to UTC above, so
	// the fixed suffix is always correct.
	const oasisLayout = "20060102T15:04"
	url := fmt.Sprintf(
		"%s?resultformat=6&queryname=PRC_LMP&version=12&market_run_id=%s&node=%s&startdatetime=%s-0000&enddatetime=%s-0000",
		strings.TrimRight(s.BaseURL, "/"), LMPMarket, node,
		start.Format(oasisLayout), end.Format(oasisLayout))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Reference{}, fmt.Errorf("building CAISO OASIS request: %w", err)
	}
	req.Header.Set("accept", "application/zip")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return Reference{}, fmt.Errorf("fetching the CAISO OASIS report: %w", err)
	}
	defer resp.Body.Close()

	// OASIS rate-limits on BURST, not volume, and it is far stricter than its
	// documentation suggests. Measured during this port: back-to-back requests
	// return HTTP 429 from the second onward, while eight requests spaced 15s
	// apart all succeeded with byte-identical payloads. The practical rule for
	// a caller is therefore to leave >=15s between fetches, which the scoring
	// pass gets for free today because this intent has a single question key.
	//
	// A 429 is surfaced as its own error rather than folded into the generic
	// status case because it is the difference between "CAISO disagrees" and
	// "we asked too fast" — and only the latter is fixable by retrying.
	if resp.StatusCode == http.StatusTooManyRequests {
		return Reference{}, fmt.Errorf(
			"CAISO OASIS rate-limited the truth fetch (429); it throttles on burst, so space fetches >=15s apart")
	}
	if resp.StatusCode != http.StatusOK {
		return Reference{}, fmt.Errorf("CAISO OASIS request returned %s", resp.Status)
	}
	// 8 MiB: one settled hour for one node is ~500 bytes compressed, so this is
	// a bound against a pathological response rather than a real expectation.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return Reference{}, fmt.Errorf("reading the CAISO OASIS response: %w", err)
	}

	lmp, csvRow, err := parseOASISZip(body, node, start)
	if err != nil {
		return Reference{}, fmt.Errorf("reading the CAISO OASIS reference price: %w", err)
	}

	milli, err := ScaleValue("GRID_POWER_PRICE", lmp)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the reference LMP: %w", err)
	}

	return Reference{
		RateBps: milli,
		RawRate: lmp,
		Pair:    fmt.Sprintf("%s %s %s", node, LMPMarket, csvRow),
		Date:    start.Format("2006-01-02"),
		Source:  "caiso-oasis:PRC_LMP:" + LMPMarket,
	}, nil
}

// parseOASISZip pulls the settled LMP_PRC price out of an OASIS SingleZip
// response, returning the $/MWh value and the interval the CSV reported.
//
// OASIS has no unwrapped JSON endpoint — the response is always a ZIP holding
// one CSV — so the unwrapping lives here. It is the awkward part of this
// intent, and it is awkward in a specific, checkable way rather than a vague
// one: three details in the CSV are each capable of producing a
// plausible-but-wrong number.
//
//   - The price is in a column named MW. OASIS reuses one value column across
//     its reports, so for PRC_LMP rows the "MW" header carries $/MWh. Reading
//     a column literally named for megawatts as a price looks like a bug and is
//     correct; the live row for the pinned hour is MW=38.09459, which is a
//     price and not a quantity.
//   - Five rows come back for one hour, not one. Only XML_DATA_ITEM=LMP_PRC is
//     the settled LMP; the other four (LMP_CONG_PRC, LMP_ENE_PRC, LMP_LOSS_PRC,
//     LMP_GHG_PRC) are its components. Taking the first row, or summing, would
//     return a number of the right magnitude and the wrong meaning —
//     LMP_ENE_PRC alone is 27.70032 against the true 38.09459.
//   - The rows are not filtered to the requested hour by position. The interval
//     is matched explicitly, so a wider window or a reordered report cannot
//     silently score against the wrong hour.
func parseOASISZip(body []byte, node string, start time.Time) (float64, string, error) {
	zr, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		// A rate-limit or maintenance page arrives as HTML with HTTP 200, so a
		// ZIP that will not open usually means throttling rather than
		// corruption. Say so — this was hit live during the port.
		return 0, "", fmt.Errorf(
			"response is not a ZIP (OASIS serves throttling notices as HTML with a 200 status): %w", err)
	}

	var csvFile *zip.File
	for _, f := range zr.File {
		if strings.HasSuffix(strings.ToLower(f.Name), ".csv") {
			csvFile = f
			break
		}
	}
	if csvFile == nil {
		// Name the entries. OASIS signals a malformed query by returning HTTP
		// 200 with a ZIP whose single entry is an error document instead of the
		// report, so the entry name is the only thing that distinguishes "bad
		// request" from "no data" — and without it the failure reads as an
		// inexplicable empty ZIP.
		names := make([]string, 0, len(zr.File))
		for _, f := range zr.File {
			names = append(names, f.Name)
		}
		return 0, "", fmt.Errorf(
			"no CSV entry in the OASIS ZIP (%d entries: %s) — OASIS returns an error document with HTTP 200 for a malformed query",
			len(zr.File), strings.Join(names, ", "))
	}

	rc, err := csvFile.Open()
	if err != nil {
		return 0, "", fmt.Errorf("opening %s in the OASIS ZIP: %w", csvFile.Name, err)
	}
	defer rc.Close()

	records, err := csv.NewReader(io.LimitReader(rc, 16<<20)).ReadAll()
	if err != nil {
		return 0, "", fmt.Errorf("parsing %s: %w", csvFile.Name, err)
	}
	if len(records) < 2 {
		return 0, "", fmt.Errorf("OASIS CSV has no data rows (%d lines)", len(records))
	}

	idx := make(map[string]int, len(records[0]))
	for i, h := range records[0] {
		idx[strings.TrimSpace(h)] = i
	}
	col := func(row []string, name string) string {
		i, ok := idx[name]
		if !ok || i >= len(row) {
			return ""
		}
		return strings.TrimSpace(row[i])
	}

	for _, row := range records[1:] {
		if col(row, "XML_DATA_ITEM") != "LMP_PRC" {
			continue // a component, not the settled LMP
		}
		if n := col(row, "NODE_ID"); n != "" && n != node {
			continue
		}
		got, err := parseOASISInstant(col(row, "INTERVALSTARTTIME_GMT"))
		if err != nil || !got.Equal(start) {
			continue
		}
		raw := col(row, "MW") // OASIS's price column for LMP_PRC rows
		lmp, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return 0, "", fmt.Errorf("non-numeric LMP_PRC value %q: %w", raw, err)
		}
		return lmp, col(row, "INTERVALSTARTTIME_GMT"), nil
	}

	return 0, "", fmt.Errorf(
		"no settled LMP_PRC row for node %s at %s in the OASIS report (%d data rows)",
		node, start.Format(time.RFC3339), len(records)-1)
}

// Register this intent's ground-truth fetcher so a spec row can name it.
// Lives here rather than in the registry file so the factory sits beside the
// parsing code that understands this upstream's quirks.
//
// Parameters are unused today: node / market are pinned in GridIntervalFor; params reserved for a second CAISO node.
func init() {
	RegisterTruthFactory("caiso", func(map[string]any) (TruthSource, error) {
		return NewCAISOSource(), nil
	})
}
