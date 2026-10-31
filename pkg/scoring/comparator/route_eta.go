package comparator

import (
	"fmt"
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

// ──────────────────────────────── route coverage ───────────────────────────────
