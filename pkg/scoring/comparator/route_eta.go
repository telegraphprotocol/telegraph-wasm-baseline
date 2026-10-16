package comparator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ROUTE_ETA: the driving transit time between a pinned origin and destination,
// in whole seconds.
//
// The intent looks volatile — an ETA "depends on when you ask" — and that
// instinct is what the spec's choice of source is designed around. The OSRM
// public demo routes over a FIXED OSM snapshot with no live-traffic term, so
// duration is a pure function of the coordinate pair: two validators calling it
// seconds or days apart get the same answer. That was checked rather than
// assumed. The reference host route_eta_price.mjs records 733.2s for the pinned
// London route, and repeated fetches a year later still return 733.2s.
//
// The honest cost of that choice: this scores a traffic-free routing time, not
// the "live traffic transit time" the intent catalog's prose describes. The
// 800 bps band is sized for cross-ENGINE disagreement (if ORS or Google were
// ever wired as a second engine), not for temporal drift within OSRM.

// Route is one pinned origin/destination pair.
//
// Coordinates are stored in OSRM's own order — longitude first — because that
// is the order they are serialised into the request path, and silently
// transposing lat/lon is the single easiest way to ask for a route across the
// wrong continent while still getting a plausible-looking duration back.
type Route struct {
	Key       string  // short stable ID, used as the question key
	Name      string  // human description for logs and the audit trail
	OriginLon float64 //
	OriginLat float64 //
	DestLon   float64 //
	DestLat   float64 //
	City      string  // for the question text put to miners
}

// Coords renders the pair as OSRM's `lon,lat;lon,lat` path segment.
func (r Route) Coords() string {
	return fmt.Sprintf("%g,%g;%g,%g", r.OriginLon, r.OriginLat, r.DestLon, r.DestLat)
}

// Routes are the journeys ROUTE_ETA is scored on each epoch.
//
// LON is the route the build spec pins and the reference host defaults to, so
// it is the one with an independently known expected value (733.2s raw,
// 733 after rounding). The other two exist for a specific reason rather than
// for volume: every ROUTE_ETA miner carries a *different* route as its
// registered default, so asking only one route cannot distinguish a miner that
// honours the coordinates it is given from one that ignores them and replays
// its own default. Asking three catches that — a miner replaying a baked-in
// route scores near zero on the two it was not built around.
//
// All three are short, dense-urban drives on well-mapped road networks, chosen
// so OSRM's public demo can certainly route them.
var Routes = []Route{
	{
		Key: "LON", Name: "Trafalgar Square to Liverpool Street, London", City: "London",
		OriginLon: -0.1278, OriginLat: 51.5074, DestLon: -0.0770, DestLat: 51.5155,
	},
	{
		Key: "BER", Name: "Brandenburg Gate to Alexanderplatz, Berlin", City: "Berlin",
		OriginLon: 13.3777, OriginLat: 52.5163, DestLon: 13.4132, DestLat: 52.5219,
	},
	{
		Key: "NYC", Name: "Times Square to Wall Street, New York", City: "New York",
		OriginLon: -73.9855, OriginLat: 40.7580, DestLon: -74.0089, DestLat: 40.7061,
	},
}

// RouteKeys is the fixed question set, in the order rounds are run.
var RouteKeys = routeKeys()

func routeKeys() []string {
	keys := make([]string, 0, len(Routes))
	for _, r := range Routes {
		keys = append(keys, r.Key)
	}
	return keys
}

// RouteFor resolves a question key to its pinned route.
func RouteFor(key string) (Route, bool) {
	for _, r := range Routes {
		if r.Key == key {
			return r, true
		}
	}
	return Route{}, false
}

// routeQuestion is the question put to every miner for one route.
//
// It names the coordinates explicitly rather than only the landmarks. The
// request builder has to turn this into each miner's own OSRM call, and the
// coordinate pair is the part that must survive that translation — a question
// that said only "Trafalgar Square to Liverpool Street" would leave the builder
// to geocode, which is exactly the step the four unscoreable route-photon-*
// miners perform and this intent does not score.
func routeQuestion(key string) string {
	r, ok := RouteFor(key)
	if !ok {
		return fmt.Sprintf("What is the driving time in seconds for route %s?", key)
	}
	return fmt.Sprintf(
		"What is the driving-route travel time in seconds from coordinates %g,%g to %g,%g (%s)?",
		r.OriginLon, r.OriginLat, r.DestLon, r.DestLat, r.Name)
}

// describeRoute renders a key for logs and the stored audit trail.
func describeRoute(key string) string {
	if r, ok := RouteFor(key); ok {
		return r.City + ":" + r.Key
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// routeParsers holds one entry per registered ROUTE_ETA miner that can actually
// be scored, keyed by the slug in miner_registry (registration IDs 1663-1672).
//
// Six of the ten are here. The four absent ones — route-photon-berlin,
// -london, -miami, -tokyo — are deliberately omitted rather than given a
// failing parser: photon.komoot.io is a GEOCODER, not a routing engine. It
// takes a place name and returns a point, so it carries no duration at any
// path and cannot be asked an origin/destination pair. A miner that cannot be
// asked the question is excluded from the round, not scored zero.
//
// Note how little these differ: five of the six are the identical OSRM route
// envelope and share one parser. That repetition is the argument for the
// declarative response mappings described in the package doc.
var routeParsers = map[string]parserFunc{
	"route-osrm-berlin":  parseOSRMRoute, // {"routes":[{"duration":733.2}]}
	"route-osrm-chi-det": parseOSRMRoute,
	"route-osrm-lon-par": parseOSRMRoute,
	"route-osrm-mia-nyc": parseOSRMRoute,
	"route-osrm-sf-la":   parseOSRMRoute,
	"route-osrm-table":   parseOSRMTable, // {"durations":[[0,733.2],[779.2,0]]}
}

// parseOSRMRoute reads routes[0].duration from a standard OSRM /route response.
//
// OSRM reports a `code` field that is "Ok" only on success; a failed route
// still returns HTTP 200 with a code like "NoRoute", so the code is checked
// rather than trusting the status line. Duration is a float in seconds and is
// returned unrounded — rounding happens once, in ScaleValue, so the miner and
// the reference are rounded by the same code.
func parseOSRMRoute(_ string, body string) (float64, error) {
	var resp struct {
		Code   string `json:"code"`
		Routes []struct {
			Duration float64 `json:"duration"`
			Distance float64 `json:"distance"`
		} `json:"routes"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding OSRM response: %w", err)
	}
	if resp.Code != "" && resp.Code != "Ok" {
		return 0, fmt.Errorf("OSRM returned code %q", resp.Code)
	}
	if len(resp.Routes) == 0 {
		return 0, fmt.Errorf("no route in OSRM response")
	}
	return resp.Routes[0].Duration, nil
}

// parseOSRMTable reads the A->B cell from an OSRM /table duration matrix.
//
// The matrix for a two-coordinate request is [[0, A->B], [B->A, 0]], so the
// wanted cell is durations[0][1]. Taking the transpose instead returns the
// REVERSE journey, which for the pinned London route is 779.2s against 733.2s
// — a ~6% error that lands inside the 800 bps tolerance and would therefore be
// scored as a good answer rather than caught as a bug. That is why this is a
// separate parser with an explicit test rather than a tolerant path lookup.
func parseOSRMTable(_ string, body string) (float64, error) {
	var resp struct {
		Code      string      `json:"code"`
		Durations [][]float64 `json:"durations"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding OSRM table response: %w", err)
	}
	if resp.Code != "" && resp.Code != "Ok" {
		return 0, fmt.Errorf("OSRM returned code %q", resp.Code)
	}
	if len(resp.Durations) < 1 || len(resp.Durations[0]) < 2 {
		return 0, fmt.Errorf("OSRM table has no origin->destination cell (got %d rows)", len(resp.Durations))
	}
	return resp.Durations[0][1], nil
}

// ──────────────────────────────── route coverage ───────────────────────────────

// routeCoverage records which routes each miner can serve.
//
// Every scoreable ROUTE_ETA miner can serve every route, which is the opposite
// of FX_NOW's uneven coverage. The reason is worth recording because it was the
// question that decided whether this intent was portable at all: the route in a
// slug like "route-osrm-sf-la" is only the `coords` path parameter's registered
// DEFAULT, not a hardcoded value. Probing three differently-slugged miners with
// the pinned London coordinates returned London's duration from all three.
//
// The map is kept — rather than collapsed into "all miners serve all routes" —
// so that a miner later registered against a bounded routing engine can be
// given real coverage without reshaping the scoring pass.
var routeCoverage = map[string]func(key string) bool{
	"route-osrm-berlin":  anyQuote,
	"route-osrm-chi-det": anyQuote,
	"route-osrm-lon-par": anyQuote,
	"route-osrm-mia-nyc": anyQuote,
	"route-osrm-sf-la":   anyQuote,
	"route-osrm-table":   anyQuote,
}

// ───────────────────────────────── ground truth ────────────────────────────────

// OSRMSource reads the reference driving time from the OSRM public demo.
//
// It matches the endpoint and query the reference host route_eta_price.mjs
// uses, so a score computed here can be checked against a run of
// route_eta_score.mjs on the same inputs.
//
// Read the ranking this produces with one caveat firmly in mind: the truth
// source and five of the six scoreable miners are the SAME OSRM demo server.
// Those five will score at or near 10000 by construction, which measures "we
// called one server twice", not miner quality. route-osrm-table is the only
// mild independence in the field — a different endpoint on the same engine and
// the same road graph. A ROUTE_ETA leaderboard is not a quality signal until a
// genuinely independent engine (OpenRouteService, Google) is wired, and both
// are key-gated.
type OSRMSource struct {
	// BaseURL is the OSRM-compatible driving-route API base.
	BaseURL string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewOSRMSource builds an OSRMSource with defaults suitable for epoch scoring.
//
// HTTPS, deliberately: the registered miners carry an http:// base_url, but the
// demo serves both and there is no reason for the node's own truth fetch to
// travel in clear text.
func NewOSRMSource() *OSRMSource {
	return &OSRMSource{
		BaseURL: "https://router.project-osrm.org/route/v1/driving",
		Client:  &http.Client{Timeout: 15 * time.Second},
	}
}

// FetchReference resolves the pinned driving time for one route, already scaled
// to whole seconds.
//
// The pin argument is accepted and ignored. That is not an oversight: OSRM's
// demo has no notion of "as of when" — it routes over a fixed OSM snapshot, so
// there is nothing to pin to. The parameter stays in the signature because it
// is the TruthSource contract, and because an engine wired later (one with a
// live-traffic term) would need it.
func (s *OSRMSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	route, ok := RouteFor(key)
	if !ok {
		return Reference{}, fmt.Errorf("no pinned route for key %q", key)
	}

	url := fmt.Sprintf("%s/%s?overview=false", strings.TrimRight(s.BaseURL, "/"), route.Coords())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Reference{}, fmt.Errorf("building OSRM request: %w", err)
	}
	req.Header.Set("accept", "application/json")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return Reference{}, fmt.Errorf("fetching OSRM reference route: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Reference{}, fmt.Errorf("OSRM route request returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return Reference{}, fmt.Errorf("reading OSRM response: %w", err)
	}

	// The reference is parsed by the SAME function that parses a miner's
	// response. Truth and miner values therefore cannot diverge through two
	// subtly different readings of one envelope — the failure mode that would
	// show up as every miner being slightly wrong rather than as a bug here.
	duration, err := parseOSRMRoute(key, string(body))
	if err != nil {
		return Reference{}, fmt.Errorf("reading the OSRM reference route: %w", err)
	}

	seconds, err := ScaleValue("ROUTE_ETA", duration)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the reference duration: %w", err)
	}

	return Reference{
		RateBps: seconds,
		RawRate: duration,
		Pair:    route.Name,
		Date:    time.Now().UTC().Format("2006-01-02"),
		Source:  "osrm:router.project-osrm.org/route/v1/driving",
	}, nil
}
