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

// LIQUIDITY_DEPTH_VERIFY: the total liquidity held by a named DEX pool, valued
// in USD and scored in integer cents.
//
// What makes this intent scoreable at all is the pool ADDRESS. The quantity is
// live — it moves with every swap — so there is no pinned instant two
// validators can both read. What they can both read is the same pool's reserves
// a few blocks apart, and independent indexers querying one address converge
// because they are all reducing the same observable on-chain state. The
// determinism therefore comes from the address plus a tolerance band wide
// enough to absorb the fetch gap, not from a pin.
//
// The honest caveat, recorded rather than fixed: the intent catalog's
// description asks for "bid-ask order book depth, cumulative slippage bands,
// and market maker liquidity". What is actually scored here is TOTAL POOL
// LIQUIDITY — a single scalar of reserves-in-USD. The two are related (pool
// depth is what produces an AMM's slippage curve) but they are not the same
// quantity, and a miner that reported perfect slippage bands and no total would
// be unscoreable here. Scoring the slippage curve would need a vector
// comparator; the shipped artifact is a scalar RelTol comparator.
//
// This is also the first ported intent whose field contains two genuinely
// INDEPENDENT miners — one reading GeckoTerminal, one reading DexScreener —
// rather than several views of one upstream. That is the only configuration in
// which a comparator can discriminate at all, so read the caveat on
// GeckoTerminalSource below about which of the two shares the truth source.

// Pool is one pinned DEX pool, named on both indexers' vocabularies.
//
// The address is the determinism key and is deliberately duplicated across
// Network and ChainID rather than derived: GeckoTerminal's network slug ("eth")
// and DexScreener's chain id ("ethereum") are different identifiers for the
// same chain, and inventing a mapping would be one more place for a silent
// mismatch. Both are recorded explicitly so the two miners can be asked about
// the same pool in their own terms.
type Pool struct {
	Key     string // short stable ID, used as the question key
	Name    string // human description for logs and the audit trail
	Address string // the on-chain pool/pair address — the determinism key
	Network string // GeckoTerminal network slug, e.g. "eth"
	ChainID string // DexScreener chain id, e.g. "ethereum"
	DEX     string // the AMM, for the question text and the audit trail
}

// Pools are the pools LIQUIDITY_DEPTH_VERIFY is scored on each epoch.
//
// One pool, not three. That differs from ROUTE_ETA's fixed set on purpose: the
// route set exists to catch a miner replaying its registered default instead of
// honouring the coordinates it was given, and there are six ROUTE_ETA miners
// carrying six different defaults. Here there are two miners, both registered
// against this same address as their default, so a second pool would test the
// omni-chat proxy's parameter handling rather than add a scoring signal. A
// second pool becomes worth adding the moment a third miner registers with a
// different default.
//
// WETH/USDC 0.05% on Uniswap v3 specifically: it is the pool the build spec
// pins, the one the reference host liquidity_depth_price.mjs defaults to, and
// one of the deepest pools on Ethereum — so its liquidity is large enough that
// a single swap moves it by a fraction of a basis point rather than jolting it
// across the tolerance band mid-round.
var Pools = []Pool{
	{
		Key:     "ETH-WETH-USDC-005",
		Name:    "WETH / USDC 0.05% (Uniswap v3, Ethereum)",
		Address: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
		Network: "eth",
		ChainID: "ethereum",
		DEX:     "Uniswap v3",
	},
}

// PoolKeys is the fixed question set, in the order rounds are run.
var PoolKeys = poolKeys()

func poolKeys() []string {
	keys := make([]string, 0, len(Pools))
	for _, p := range Pools {
		keys = append(keys, p.Key)
	}
	return keys
}

// PoolFor resolves a question key to its pinned pool.
func PoolFor(key string) (Pool, bool) {
	for _, p := range Pools {
		if p.Key == key {
			return p, true
		}
	}
	return Pool{}, false
}

// liquidityQuestion is the question put to every miner for one pool.
//
// It names the raw address and both chain identifiers rather than the pool's
// friendly name. "The WETH/USDC pool on Ethereum" is ambiguous — Uniswap v3
// alone runs several fee tiers over that token pair, and their liquidity
// differs by an order of magnitude, far outside a 300 bps band. The address is
// the only unambiguous form, and it is the part that must survive the request
// builder's translation into each miner's own call.
func liquidityQuestion(key string) string {
	p, ok := PoolFor(key)
	if !ok {
		return fmt.Sprintf("What is the total USD liquidity of DEX pool %s?", key)
	}
	return fmt.Sprintf(
		"What is the total liquidity in USD of the %s pool at address %s on %s (chain id %s)?",
		p.DEX, p.Address, p.Network, p.ChainID)
}

// describePool renders a key for logs and the stored audit trail.
func describePool(key string) string {
	if p, ok := PoolFor(key); ok {
		return p.Network + ":" + p.Address
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// liquidityParsers holds one entry per registered LIQUIDITY_DEPTH_VERIFY miner,
// keyed by the slug in miner_registry (registration IDs 2765-2766).
//
// Both are served from the same omni-chat host but read DIFFERENT upstream
// indexers, and they return the same envelope shape, so they share a parser.
// The shared parser is not laziness here — the two responses are genuinely the
// same schema, unlike ROUTE_ETA's five-identical-OSRM-miners case, where one
// parser is repeated because five miners are the same endpoint.
var liquidityParsers = map[string]parserFunc{
	// {"liquidity_usd":106046544.1906,"liquidity_usd_cents":10604654419,...}
	"liq-gecko-pool-cents": parseLiquidityUSD,
	"liq-dex-pair-cents":   parseLiquidityUSD,
}

// parseLiquidityUSD reads a miner's claimed pool liquidity as whole USD.
//
// It returns DOLLARS, not cents, even though the miners' own field is named
// liquidity_usd_cents and the WASM is fed cents. Scaling happens exactly once,
// in ScaleValue, so the reference and every miner pass through identical
// rounding; a parser that returned cents would have to be paired with a scale
// of 1 and would silently be 100x wrong the moment someone reused ScaleValue.
//
// liquidity_usd_cents is preferred over liquidity_usd when present because it
// is the field the miner's registration YAML names as its label_field, i.e. the
// value it is actually asserting. liquidity_usd is the float it derived that
// from, and re-deriving dollars from the cents integer costs nothing and keeps
// the scored quantity the one the miner committed to.
//
// Both fields are accepted as either JSON numbers or strings. The two miners
// wired today emit numbers, but the upstreams they proxy both emit these
// figures as STRINGS (GeckoTerminal's reserve_in_usd is "106046544.1906"), so a
// miner registered later against the raw upstream would otherwise decode to
// zero — which ScaleValue rejects, but only after a confusing detour.
func parseLiquidityUSD(_ string, body string) (float64, error) {
	var resp struct {
		LiquidityUSDCents json.RawMessage `json:"liquidity_usd_cents"`
		LiquidityUSD      json.RawMessage `json:"liquidity_usd"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, fmt.Errorf("decoding liquidity response: %w", err)
	}

	if cents, ok, err := numberOrString(resp.LiquidityUSDCents); err != nil {
		return 0, fmt.Errorf("reading liquidity_usd_cents: %w", err)
	} else if ok {
		return cents / 100, nil
	}
	if usd, ok, err := numberOrString(resp.LiquidityUSD); err != nil {
		return 0, fmt.Errorf("reading liquidity_usd: %w", err)
	} else if ok {
		return usd, nil
	}
	return 0, fmt.Errorf("no liquidity_usd_cents or liquidity_usd in response")
}

// numberOrString decodes a JSON value that may be quoted or bare.
//
// The ok result distinguishes "absent" from "present and zero". Collapsing
// those two would turn a missing field into a zero liquidity, which ScaleValue
// rejects as non-positive — the right outcome, but reported as a bad value
// rather than as an unparseable response, which is the more useful diagnosis.
func numberOrString(raw json.RawMessage) (float64, bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, false, nil
	}
	var n float64
	if err := json.Unmarshal(raw, &n); err == nil {
		return n, true, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return 0, false, fmt.Errorf("value %s is neither a number nor a string", string(raw))
	}
	n, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, false, fmt.Errorf("parsing %q: %w", s, err)
	}
	return n, true, nil
}

// ──────────────────────────────── pool coverage ────────────────────────────────

// liquidityCoverage records which pools each miner can serve.
//
// Both accept an arbitrary (chain, address) pair — the pool in each YAML's
// input_schema is a registered DEFAULT, not a hardcoded value — so coverage is
// uniform today. The map is kept rather than collapsed into "everyone serves
// everything" so that a miner later registered against a single-chain indexer
// can be given real coverage without reshaping the scoring pass.
//
// The one real constraint is not expressible here and is worth stating: a pool
// key is only serveable by BOTH miners if the address exists on a chain each
// indexer covers under its own slug. Adding a pool on a chain DexScreener does
// not index would silently drop liq-dex-pair-cents from that round.
var liquidityCoverage = map[string]func(key string) bool{
	"liq-gecko-pool-cents": anyQuote,
	"liq-dex-pair-cents":   anyQuote,
}

// ───────────────────────────────── ground truth ────────────────────────────────

// GeckoTerminalSource reads reference pool liquidity from the GeckoTerminal
// public API, keyless, exactly as the reference host
// liquidity_depth_price.mjs does.
//
// It talks to api.geckoterminal.com DIRECTLY and deliberately does not go
// through the omni-chat /truth/ proxy, even though one of the two miners is
// that proxy's GeckoTerminal endpoint and would be trivially easier to call.
// Both miners are served from omni-chat; if truth came from the same host then
// a host-wide fault, a stale cache or a deliberate skew would move truth and
// both miners together and the comparator could never fail anyone. A truth
// source that cannot disagree with the field is not a verifier.
//
// Read the resulting ranking with the obvious caveat: liq-gecko-pool-cents
// proxies THIS SAME upstream, so it will score at or near 10000 by
// construction. Observed live on 2026-09-21, the miner returned 10604654419
// cents against an upstream 10604654419 — identical to the cent. That is a
// measurement of "we called GeckoTerminal twice", not of miner quality.
// liq-dex-pair-cents is the only independent reading in the field, and its
// score (DexScreener's liquidity.usd against GeckoTerminal's reserve_in_usd) is
// the only number in this intent that carries information.
type GeckoTerminalSource struct {
	// BaseURL is the GeckoTerminal API v2 base.
	BaseURL string
	// Client is the HTTP client used for the fetch. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewGeckoTerminalSource builds a source with defaults suitable for epoch
// scoring.
func NewGeckoTerminalSource() *GeckoTerminalSource {
	return &GeckoTerminalSource{
		BaseURL: "https://api.geckoterminal.com/api/v2",
		Client:  &http.Client{Timeout: 15 * time.Second},
	}
}

// FetchReference resolves the pinned pool's liquidity, already scaled to
// integer USD cents.
//
// The pin argument is accepted and ignored, and that is a real limitation
// rather than an oversight. Unlike FX_NOW's dated ECB fix, GeckoTerminal serves
// only "now" for a pool — there is no as-of parameter — so two validators
// scoring the same round minutes apart read two genuinely different values.
// What keeps them in agreement is the 300 bps band, not the pin, which is why
// the band cannot be tightened toward FX_NOW's 50 bps without breaking
// cross-validator agreement. The parameter stays in the signature because it is
// the TruthSource contract and because an indexer with block-height queries
// (a subgraph pinned to a block) could honour it later.
func (s *GeckoTerminalSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	pool, ok := PoolFor(key)
	if !ok {
		return Reference{}, fmt.Errorf("no pinned pool for key %q", key)
	}

	endpoint := fmt.Sprintf("%s/networks/%s/pools/%s",
		strings.TrimRight(s.BaseURL, "/"),
		url.PathEscape(pool.Network),
		url.PathEscape(pool.Address))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return Reference{}, fmt.Errorf("building GeckoTerminal request: %w", err)
	}
	req.Header.Set("accept", "application/json")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return Reference{}, fmt.Errorf("fetching GeckoTerminal pool reserves: %w", err)
	}
	defer resp.Body.Close()

	// 429 is called out separately because it is the failure this keyless
	// endpoint actually produces under load, and it must surface as an error
	// rather than as a missing field: GeckoTerminal answers a rate-limited
	// request with a JSON error body, which would otherwise parse to "no
	// reserve_in_usd" and read like a schema change.
	if resp.StatusCode == http.StatusTooManyRequests {
		return Reference{}, fmt.Errorf("GeckoTerminal rate-limited the reference fetch (429); truth is unavailable for this round")
	}
	if resp.StatusCode != http.StatusOK {
		return Reference{}, fmt.Errorf("GeckoTerminal pool request returned %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return Reference{}, fmt.Errorf("reading GeckoTerminal response: %w", err)
	}

	usd, poolName, err := parseGeckoTerminalPool(string(body), pool.Address)
	if err != nil {
		return Reference{}, fmt.Errorf("reading the GeckoTerminal reference reserves: %w", err)
	}

	cents, err := ScaleValue("LIQUIDITY_DEPTH_VERIFY", usd)
	if err != nil {
		return Reference{}, fmt.Errorf("scaling the reference liquidity: %w", err)
	}

	if poolName == "" {
		poolName = pool.Name
	}
	return Reference{
		RateBps: cents,
		RawRate: usd,
		Pair:    poolName,
		Date:    time.Now().UTC().Format(time.RFC3339),
		Source:  "geckoterminal:/networks/" + pool.Network + "/pools/" + pool.Address + "#reserve_in_usd",
	}, nil
}

// parseGeckoTerminalPool reads data.attributes.reserve_in_usd as whole USD.
//
// This is a separate function from parseLiquidityUSD rather than a shared one,
// because the upstream envelope and the miners' envelope really are different
// schemas — sharing here would mean a tolerant "find a number somewhere" path,
// and the two must not be able to drift into reading different fields.
//
// The returned address is checked against the one requested. GeckoTerminal
// resolves some inputs leniently, and a response describing a DIFFERENT pool
// would produce a perfectly well-formed number that is simply the wrong pool's
// liquidity — the exact failure the address-as-determinism-key design exists to
// prevent, and one that would show up only as every miner scoring zero.
func parseGeckoTerminalPool(body, wantAddress string) (float64, string, error) {
	var resp struct {
		Data struct {
			Attributes struct {
				Address      string          `json:"address"`
				Name         string          `json:"name"`
				ReserveInUSD json.RawMessage `json:"reserve_in_usd"`
			} `json:"attributes"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return 0, "", fmt.Errorf("decoding GeckoTerminal response: %w", err)
	}
	attrs := resp.Data.Attributes

	if got := strings.ToLower(attrs.Address); got != "" && got != strings.ToLower(wantAddress) {
		return 0, "", fmt.Errorf("GeckoTerminal returned pool %s, not the requested %s", attrs.Address, wantAddress)
	}

	// reserve_in_usd arrives as a quoted decimal string ("106046544.1906"),
	// not a JSON number; numberOrString accepts both so a schema change to a
	// bare number would not break the fetch.
	usd, ok, err := numberOrString(attrs.ReserveInUSD)
	if err != nil {
		return 0, "", fmt.Errorf("reading reserve_in_usd: %w", err)
	}
	if !ok {
		return 0, "", fmt.Errorf("no reserve_in_usd in GeckoTerminal response")
	}
	return usd, attrs.Name, nil
}

// Register this intent's ground-truth fetcher so a spec row can name it.
// Lives here rather than in the registry file so the factory sits beside the
// parsing code that understands this upstream's quirks.
//
// Parameters are unused today: network / pool address are carried by the question key.
func init() {
	RegisterTruthFactory("geckoterminal", func(map[string]any) (TruthSource, error) {
		return NewGeckoTerminalSource(), nil
	})
}
