package comparator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
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

// DateParam renders the year as the World Bank's `date` query value.
//
// The range form "2019:2019" rather than the bare "2019" is what the miner's
// registration YAML carries as its default, and both return the identical
// single observation. Matching the miner's own spelling keeps the truth fetch
// and the miner's call byte-comparable when debugging a disagreement.
func (o MacroObservation) DateParam() string {
	return fmt.Sprintf("%d:%d", o.Year, o.Year)
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

// macroParsers holds one entry per registered MACRO_ECONOMIC_INDICATOR miner
// that can actually be scored, keyed by the slug in miner_registry.
//
// One entry, because one miner is active: macro-wb-unemp-us (registration
// 2753). The ten macro-* registrations in the 1483-1492 range are all
// deregistered and are deliberately absent rather than carried with a parser —
// a deregistered miner is not a participant, and giving it a parser would
// invite it back into a round the moment someone iterated the map instead of
// the registry.
var macroParsers = map[string]parserFunc{
	"macro-wb-unemp-us": parseWorldBankIndicator, // [{page...},[{...,"value":3.669}]]
}

// parseWorldBankIndicator reads response[1][0].value from a World Bank
// Indicators API payload.
//
// The envelope is the unusual part and the reason this is a hand-written parser
// rather than a JSON path: the top level is a two-element ARRAY whose first
// element is pagination metadata and whose second is the observations list. The
// miner's registered label_field "1" points at that second element — the list,
// not a number — so the descent to [0].value is done here explicitly.
//
// Three failure shapes are distinguished rather than collapsed, because they
// mean different things:
//
//   - a `message` block: the World Bank answers a bad indicator or country with
//     HTTP 200 and an error document, so the status line cannot be trusted;
//   - a null or empty observations list: this is what an out-of-range year
//     returns ([{...,"total":0},null]) — a coverage gap, not a wrong answer;
//   - a null `value` on a present observation: the country/year exists in the
//     series but has no published figure.
//
// All three return an error, which excludes the miner from the round. That is
// the correct verdict: ParseMinerValue's caller must not turn "no such
// observation" into a scored 0, which would rank a working miner last for a gap
// in the source.
//
// The key is accepted and ignored. The pinned (country, indicator, year) live
// in the REQUEST, not the response, so there is nothing to select on here —
// but see validateWorldBankKey, which the truth fetch uses to confirm the
// response actually describes the observation that was asked for.
func parseWorldBankIndicator(_ string, body string) (float64, error) {
	obs, err := worldBankObservation(body)
	if err != nil {
		return 0, err
	}
	if obs.Value == nil {
		return 0, fmt.Errorf("observation for %s/%s has a null value (no published figure)",
			obs.Country.ID, obs.Date)
	}
	return *obs.Value, nil
}

// wbObservation is one row of the World Bank observations list.
//
// Value is a *float64 and not a float64 because the API uses JSON null for "no
// figure published", and a plain float64 would decode that to 0 — a value that
// is both in-domain for a percentage and catastrophically wrong, scoring the
// miner as claiming zero unemployment rather than as unable to answer.
type wbObservation struct {
	Indicator struct {
		ID    string `json:"id"`
		Value string `json:"value"`
	} `json:"indicator"`
	Country struct {
		ID    string `json:"id"`
		Value string `json:"value"`
	} `json:"country"`
	CountryISO3 string   `json:"countryiso3code"`
	Date        string   `json:"date"`
	Value       *float64 `json:"value"`
}

// worldBankObservation decodes the envelope and returns the single observation.
func worldBankObservation(body string) (wbObservation, error) {
	// Decoded into RawMessage first because the two elements have different
	// types — an object then an array — which no single Go struct describes.
	var envelope []json.RawMessage
	if err := json.Unmarshal([]byte(body), &envelope); err != nil {
		// A World Bank error document is a bare OBJECT, not the usual array, so
		// a failed array decode is the expected path into this branch. Surface
		// the API's own message when there is one; it names the bad dimension.
		if msg := worldBankErrorMessage(body); msg != "" {
			return wbObservation{}, fmt.Errorf("World Bank API error: %s", msg)
		}
		return wbObservation{}, fmt.Errorf("decoding World Bank response: %w", err)
	}
	if len(envelope) < 2 {
		return wbObservation{}, fmt.Errorf(
			"World Bank response has %d element(s), want the [metadata, observations] pair", len(envelope))
	}

	var rows []wbObservation
	// A JSON null second element decodes into a nil slice without error, which
	// is exactly what an out-of-range year returns; it is caught by the length
	// check immediately below rather than by the decode.
	if err := json.Unmarshal(envelope[1], &rows); err != nil {
		return wbObservation{}, fmt.Errorf("decoding World Bank observations: %w", err)
	}
	if len(rows) == 0 {
		return wbObservation{}, fmt.Errorf("World Bank returned no observations for the pinned query")
	}
	// The pinned query asks for exactly one year, so exactly one row is
	// expected. Take the first rather than the last: the API returns newest
	// first, so on an accidentally widened date range the first row is the most
	// recent year — but the truth path also calls validateWorldBankKey, which
	// rejects a row that is not the year asked for outright.
	return rows[0], nil
}

// worldBankErrorMessage pulls the human-readable text out of a World Bank error
// document, which is returned with HTTP 200 and looks like
// {"message":[{"id":"120","key":"Invalid value","value":"..."}]}.
func worldBankErrorMessage(body string) string {
	var doc struct {
		Message []struct {
			Key   string `json:"key"`
			Value string `json:"value"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(body), &doc); err != nil || len(doc.Message) == 0 {
		return ""
	}
	m := doc.Message[0]
	return strings.TrimSpace(m.Key + ": " + m.Value)
}

// validateWorldBankKey checks that a decoded observation is the one that was
// asked for.
//
// This is the guard against the failure the whole pin exists to prevent: a
// request that silently lost its `date` parameter still returns HTTP 200 with a
// perfectly well-formed observation for some OTHER year, which parses fine and
// scores as a plausible number. Comparing the response's own country/date
// against the pin turns that into an error instead of a wrong ranking.
//
// Country is matched against both the two-letter id and the iso3 code because
// the API echoes "US" in country.id while accepting either "US" or "USA" in the
// path, and the miner registers the two-letter form.
func validateWorldBankKey(o MacroObservation, obs wbObservation) error {
	if obs.Date != fmt.Sprintf("%d", o.Year) {
		return fmt.Errorf("World Bank returned year %q, want %d", obs.Date, o.Year)
	}
	want := strings.ToUpper(o.Country)
	if !strings.EqualFold(obs.Country.ID, want) && !strings.EqualFold(obs.CountryISO3, want) {
		return fmt.Errorf("World Bank returned country %q/%q, want %q",
			obs.Country.ID, obs.CountryISO3, want)
	}
	if obs.Indicator.ID != "" && !strings.EqualFold(obs.Indicator.ID, o.Indicator) {
		return fmt.Errorf("World Bank returned indicator %q, want %q", obs.Indicator.ID, o.Indicator)
	}
	return nil
}

// ──────────────────────────────── macro coverage ───────────────────────────────

// macroCoverage records which observations each miner can serve.
//
// macro-wb-unemp-us serves all of them. Its slug says "unemp-us" but the
// country is a registered path PARAMETER with a default of "US", not a
// hardcoded value — the same shape as ROUTE_ETA's slug-baked routes — so
// DE-UNEMP-2019 is answerable by the same endpoint. The indicator, by contrast,
// IS hardcoded in the miner's external_path
// (/v2/country/{country}/indicator/SL.UEM.TOTL.ZS), so a CPI or interest-rate
// key added later would need this map to stop returning true unconditionally.
// It is kept as a map rather than collapsed for exactly that reason.
var macroCoverage = map[string]func(key string) bool{
	"macro-wb-unemp-us": servesUnemploymentSeries,
}

// servesUnemploymentSeries reports whether a key is on the SL.UEM.TOTL.ZS
// series, which is the only indicator macro-wb-unemp-us's path can reach.
func servesUnemploymentSeries(key string) bool {
	o, ok := MacroObservationFor(key)
	return ok && o.Indicator == "SL.UEM.TOTL.ZS"
}

// ───────────────────────────────── ground truth ────────────────────────────────

// WorldBankSource reads the reference macro statistic from the World Bank
// Indicators API.
//
// It matches the endpoint the build spec's ground_truth.method records
// (GET /v2/country/{c}/indicator/{i}?format=json&date={y} -> response[1][0].value)
// and is keyless, so it needs no secret in the validator's environment.
//
// Two caveats belong with this type rather than in a design doc:
//
//  1. The miner IS this source. See the package-level note above — a perfect
//     score for macro-wb-unemp-us is arithmetic, not evidence.
//  2. Revisions. The API serves the CURRENT revision of a series for the
//     requested year and offers no vintage parameter, so pinning (country,
//     indicator, year) makes the fetch identical across validators AT THE SAME
//     TIME but not across time: if the World Bank re-estimates 2019 next year,
//     a score computed then differs from one computed now. For a multi-year
//     settled figure that drift is small and is part of what the 150 bps band
//     absorbs. FRED's ALFRED vintage pin is the real fix and is key-gated.
type WorldBankSource struct {
	// BaseURL is the World Bank Indicators API base.
	BaseURL string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewWorldBankSource builds a WorldBankSource with defaults suitable for epoch
// scoring.
func NewWorldBankSource() *WorldBankSource {
	return &WorldBankSource{
		BaseURL: "https://api.worldbank.org/v2",
		Client:  &http.Client{Timeout: 15 * time.Second},
	}
}

// FetchReference resolves the pinned statistic for one key, already scaled to
// milli-percent.
//
// The pin argument is accepted and ignored, as it is for OSRM, but for a
// different reason worth distinguishing: OSRM has no notion of "as of when",
// whereas the World Bank does have vintages and simply exposes no parameter to
// select one. The observation's own year is carried in the KEY, not in Pin —
// Pin.Date is an ISO day, which cannot express "the 2019 annual figure". The
// parameter stays because it is the TruthSource contract, and a FRED-backed
// source wired later would use it to carry an ALFRED realtime date.
func (s *WorldBankSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	obs, ok := MacroObservationFor(key)
	if !ok {
		return Reference{}, fmt.Errorf("no pinned macro observation for key %q", key)
	}

	// per_page is sent to match the miner's registered call exactly; with a
	// single-year date range it changes nothing, which is the point — the truth
	// fetch should differ from the miner's request in no dimension the response
	// depends on.
	endpoint := fmt.Sprintf("%s/country/%s/indicator/%s?format=json&per_page=5&date=%s",
		strings.TrimRight(s.BaseURL, "/"),
		url.PathEscape(obs.Country),
		url.PathEscape(obs.Indicator),
		url.QueryEscape(obs.DateParam()))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return Reference{}, fmt.Errorf("building World Bank request: %w", err)
	}
	req.Header.Set("accept", "application/json")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return Reference{}, fmt.Errorf("fetching World Bank reference observation: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Reference{}, fmt.Errorf("World Bank request returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return Reference{}, fmt.Errorf("reading World Bank response: %w", err)
	}

	// Decoded by the same envelope reader the miner's response goes through, so
	// truth and miner values cannot diverge through two subtly different
	// readings of one payload — the failure that would present as every miner
	// being slightly wrong rather than as a bug here.
	row, err := worldBankObservation(string(body))
	if err != nil {
		return Reference{}, fmt.Errorf("reading the World Bank reference observation: %w", err)
	}
	// Only the truth path validates the pin. A miner is scored on the value it
	// returns, including when that value is for the wrong year — that is a
	// miner being wrong, and the comparator's job. The validator lying to
	// itself about which observation it fetched is not.
	if err := validateWorldBankKey(obs, row); err != nil {
		return Reference{}, fmt.Errorf("World Bank reference does not match the pin: %w", err)
	}
	if row.Value == nil {
		return Reference{}, fmt.Errorf("World Bank has no published figure for %s", describeMacro(key))
	}

	milliPct, err := ScaleValue("MACRO_ECONOMIC_INDICATOR", *row.Value)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the reference statistic: %w", err)
	}

	return Reference{
		RateBps: milliPct,
		RawRate: *row.Value,
		Pair:    obs.Name,
		// The observation's own pinned year, not today's date: this field is
		// what makes a score re-derivable, and "when we happened to call" is
		// not the dimension a reader needs to reproduce an annual statistic.
		Date:   row.Date,
		Source: "worldbank:" + describeMacro(key),
	}, nil
}

// Register this intent's ground-truth fetcher so a spec row can name it.
// Lives here rather than in the registry file so the factory sits beside the
// parsing code that understands this upstream's quirks.
//
// Parameters are unused today: country / indicator / year are encoded in the key.
func init() {
	RegisterTruthFactory("worldbank", func(map[string]any) (TruthSource, error) {
		return NewWorldBankSource(), nil
	})
}
