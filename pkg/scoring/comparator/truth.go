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

// TruthSource fetches the reference value an intent's miners are scored
// against, already scaled to the intent's fixed-point domain.
//
// FetchReference rather than a raw rate is the interface boundary on purpose:
// scaling belongs with the fetch, so a caller cannot pair a correctly-fetched
// rate with the wrong multiplier.
type TruthSource interface {
	FetchReference(ctx context.Context, quote string, pin Pin) (Reference, error)
}

// Pin identifies the point at which ground truth is read.
//
// An empty Date means "latest" — the current published fix, which is what the
// reference host fx_price.mjs fetches and therefore what this path uses by
// default. A non-empty Date reads that specific day's fix instead.
//
// The tradeoff is worth stating plainly, because it is not a settled question.
// The ECB publishes once per business day around 16:00 CET, so /latest returns
// the same value all day and two validators scoring within the same publication
// window agree. They disagree only if their calls straddle a publication — a
// narrow window, but a real one, and the FX_NOW spec's own ttl_note names the
// requirement it breaks ("validators must agree on the reference rate for the
// same tick"). Dating the pin closes that window at the cost of scoring against
// a rate up to ~48h old.
//
// Latest is the default because it matches the reference implementation; the
// dated form exists so the straddle can be closed later without reworking the
// interface.
type Pin struct {
	Date string // ISO yyyy-mm-dd, or "" for the latest published fix
}

// LatestPin reads the current published fix, matching fx_price.mjs.
func LatestPin() Pin { return Pin{} }

// PinForEpoch returns the pin used when scoring an epoch.
//
// It reads the latest published fix, matching fx_price.mjs. The parameter is
// kept so that binding an epoch to a specific date later is a change to this
// one function rather than to every call site.
func PinForEpoch(time.Time) Pin { return LatestPin() }

// DatedPin reads a specific day's ECB fix.
//
// Unused by the scoring path today. It exists because the response carries the
// date the feed actually served, so a score can be re-derived later against the
// exact fix it was computed from.
func DatedPin(at time.Time) Pin {
	return Pin{Date: at.UTC().Format("2006-01-02")}
}

// ECBSource reads the European Central Bank daily reference fix.
//
// The ECB fix is used as ground truth rather than a median of the miners
// because a median is circular: it asks the field being ranked to define the
// answer, and all ten FX_NOW miners are registered to a single wallet, so the
// "consensus" it would produce is one operator's. The ECB rate is published
// independently of every miner.
//
// The known cost is that several miners (Frankfurter, VATComply, ER-API)
// redistribute the ECB fix, so they are scored against their own upstream and
// will cluster near a perfect score. That compresses the leaderboard rather
// than corrupting it — the genuinely independent quotes (Coinbase, and the
// central banks) still separate — but it means a tight FX_NOW ranking should be
// read as "the field largely shares a source", not as "every miner is excellent".
//
// It is fetched via Frankfurter, which serves the ECB fix over a stable dated
// endpoint. That the host reads it through a URL one miner also uses is not a
// conflict of interest in the scoring sense: the value is the ECB's, and the
// same value is fetched regardless of which miners are in the round.
type ECBSource struct {
	// BaseURL is the Frankfurter-compatible endpoint serving the ECB fix.
	BaseURL string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchRate also honours the context deadline.
	Client *http.Client
}

// NewECBSource builds an ECBSource with defaults suitable for epoch scoring.
func NewECBSource() *ECBSource {
	return &ECBSource{
		BaseURL: "https://api.frankfurter.app",
		Client:  &http.Client{Timeout: 15 * time.Second},
	}
}

// FetchRate reads the ECB reference rate for USD->quote, returning the rate and
// the date the feed served it for.
//
// An empty pin.Date reads /latest, as fx_price.mjs does. The returned date is
// the feed's own, not the requested one: it is what makes a score auditable
// after the fact, and over a weekend or holiday it is the preceding business
// day rather than today.
func (s *ECBSource) FetchRate(ctx context.Context, quote string, pin Pin) (float64, string, error) {
	quote = strings.ToUpper(quote)
	if quote == "USD" {
		return 0, "", fmt.Errorf("USD/USD is not a scoreable pair")
	}

	when := pin.Date
	if when == "" {
		when = "latest"
	}
	url := fmt.Sprintf("%s/%s?from=USD&to=%s", strings.TrimRight(s.BaseURL, "/"), when, quote)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, "", fmt.Errorf("building ECB request: %w", err)
	}

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, "", fmt.Errorf("fetching ECB reference rate: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, "", fmt.Errorf("ECB reference rate request returned %s", resp.Status)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, "", fmt.Errorf("reading ECB response: %w", err)
	}

	var parsed struct {
		Date  string             `json:"date"`
		Rates map[string]float64 `json:"rates"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return 0, "", fmt.Errorf("decoding ECB response: %w", err)
	}

	rate, ok := parsed.Rates[quote]
	if !ok {
		return 0, "", fmt.Errorf("ECB fix (%s) has no %s rate", when, quote)
	}
	if rate <= 0 {
		return 0, "", fmt.Errorf("ECB fix (%s) returned an unusable %s rate: %v", when, quote, rate)
	}
	return rate, parsed.Date, nil
}

// Reference is a resolved ground-truth value together with the provenance
// needed to audit or reproduce it.
//
// It mirrors what fetchRateBps returns in the reference host fx_price.mjs
// (rateBps / rawRate / pair / date / source), so a score computed here can be
// checked against a run of fx_score.mjs on the same inputs.
type Reference struct {
	RateBps int64   // the value passed to the WASM as ground_truth_value
	RawRate float64 // the unscaled rate, for logging and diagnostics
	Pair    string  // e.g. "USD/EUR"
	Date    string  // the reference date the feed actually served
	Source  string  // provenance, e.g. "frankfurter:2026-09-17"
}

// FetchReference resolves the pinned ECB fix for USD->quote and returns it
// already scaled to the intent's fixed-point domain.
//
// Callers should prefer this to FetchRate: it applies the scale in one place,
// so a caller cannot accidentally pair a correctly-fetched rate with the wrong
// multiplier, and it carries the resolved date, which can differ from the
// requested pin over a weekend or holiday.
func (s *ECBSource) FetchReference(ctx context.Context, quote string, pin Pin) (Reference, error) {
	quote = strings.ToUpper(quote)
	rate, servedDate, err := s.FetchRate(ctx, quote, pin)
	if err != nil {
		return Reference{}, err
	}
	bps, err := ScaleFloat(rate)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the reference rate: %w", err)
	}
	return Reference{
		RateBps: bps,
		RawRate: rate,
		Pair:    "USD/" + quote,
		Date:    servedDate,
		Source:  "frankfurter:" + servedDate,
	}, nil
}

// ResolvedDate reports the reference date the source actually served.
//
// The ECB publishes on business days only, so a pin that lands on a weekend or
// holiday is answered with the preceding business day's fix. That substitution
// is deterministic — every validator asking for the same date gets the same
// earlier date back — but it is worth surfacing, because it means the pinned
// date and the scored date are not always equal.
func ResolvedDate(body string) string {
	var parsed struct {
		Date string `json:"date"`
	}
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		return ""
	}
	return parsed.Date
}
