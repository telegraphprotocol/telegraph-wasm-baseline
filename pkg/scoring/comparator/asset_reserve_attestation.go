package comparator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"strings"
	"time"
)

// ASSET_RESERVE_ATTESTATION: the reserve balance a custodian is attested to
// hold behind a tokenized asset, in micro-units of the underlying.
//
// This is the first intent wired here whose ground truth is on-chain state
// rather than an HTTP quote, and that changes two things relative to FX_NOW and
// ROUTE_ETA.
//
// First, the value is an ATTESTATION, not a live quote. A Chainlink
// Proof-of-Reserve aggregator is not streamed: its nodes watch the custodian's
// reserve wallets off-chain and write on-chain only when the balance moves past
// the feed's deviation threshold (1% for this feed) or the heartbeat elapses.
// So "the reserve right now" is a step function, not a continuous series, and
// two honest readers can legitimately sit on either side of one step.
//
// Second, and more consequential for determinism: on-chain state is addressable
// by block, so — unlike OSRM, which has nothing to pin to — this intent CAN be
// made exactly reproducible, and therefore must be. See ReserveFeed.Block.

// ReserveFeed is one pinned proof-of-reserve reading: a specific aggregator
// contract, read at a specific block.
//
// The Block field is the whole reason this type exists rather than a bare feed
// address. `latest` is not a reproducible question: the feed writes a new round
// every heartbeat, so two validators calling `latest` seconds apart can read
// different rounds and compute different truths for the same epoch — a
// disagreement that looks exactly like a miner being wrong. Chainlink's PoR
// rounds move by ~1% at a time, so a straddled update would consume most of the
// 150 bps band on its own.
//
// Pinning costs nothing here because the feed is heartbeat-driven rather than
// per-block: the same round is returned across a wide span of blocks between
// writes, so a pin does not have to land on the exact block of an on-chain
// write to be stable. That was verified rather than assumed — blocks 0x18d20a9
// and 0x18d20e9 (64 blocks apart) return byte-identical latestRoundData().
//
// The pin is deliberately set ~64 blocks behind head at the time of writing.
// Pinning at head would reintroduce the problem it solves: a block at the tip
// can still be reorged out, and a validator that read a since-orphaned block
// would derive a truth no one else can reproduce.
//
// A pin has a SHELF LIFE, and a much shorter one than is comfortable. To a
// public RPC's keyless tier, any block behind its retention window is an
// "archive" request: publicnode refused this very block ~40 minutes after it
// was verified working, while still answering the same call at `latest`. So a
// pin is not a write-once constant — it must be refreshed (and re-verified
// across providers) on a cadence, or scoring stops. That is the real operating
// cost of determinism here, and the argument for an operator-controlled archive
// endpoint in production. See NewChainlinkPoRSource.
type ReserveFeed struct {
	Key     string // short stable ID, used as the question key
	Name    string // human description for logs and the audit trail
	Asset   string // the underlying custodied asset, e.g. "BTC"
	Token   string // the tokenized claim on it, e.g. "WBTC"
	Address string // the Chainlink PoR aggregator (EACAggregatorProxy) address
	Block   string // PINNED block number, hex. Never "latest" — see above.
}

// ReserveFeeds are the attestations ASSET_RESERVE_ATTESTATION is scored on.
//
// One feed, because one miner is registered against this intent and it serves
// exactly this feed. Adding a second feed today would not broaden the ranking —
// it would create a key that the only participant cannot answer, which
// EligibleForKey would then (correctly) exclude it from, producing a round with
// no participants at all.
//
// The WBTC PoR aggregator is the feed the build spec pins and the reference
// host asset_reserve_price.mjs defaults to, so it is the one with an
// independently known expected value. It reports BitGo's custodied BTC backing
// WBTC on Ethereum mainnet.
var ReserveFeeds = []ReserveFeed{
	{
		Key:     "WBTC",
		Name:    "BitGo custodied BTC reserves backing WBTC",
		Asset:   "BTC",
		Token:   "WBTC",
		Address: "0xa81FE04086865e63E12dD3776978E49DEEa2ea4e",
		// Ethereum mainnet block 26026153 (0x18d20a9), ~64 blocks behind head
		// on 2026-09-21. Verified byte-identical latestRoundData() at this
		// block across three independent keyless providers (publicnode,
		// drpc.org, merkle.io): answer=11614464922334, updatedAt=1789995215.
		Block: "0x18d20a9",
	},
}

// ReserveKeys is the fixed question set, in the order rounds are run.
var ReserveKeys = reserveKeys()

func reserveKeys() []string {
	keys := make([]string, 0, len(ReserveFeeds))
	for _, f := range ReserveFeeds {
		keys = append(keys, f.Key)
	}
	return keys
}

// ReserveFeedFor resolves a question key to its pinned feed.
func ReserveFeedFor(key string) (ReserveFeed, bool) {
	for _, f := range ReserveFeeds {
		if f.Key == key {
			return f, true
		}
	}
	return ReserveFeed{}, false
}

// reserveQuestion is the question put to every miner for one feed.
//
// It names the feed address AND the pinned block explicitly, for the same
// reason routeQuestion names coordinates: the request builder has to turn this
// into the miner's own call, and both parameters must survive that translation.
// A question that said only "how much BTC backs WBTC" would let each miner pick
// its own block, which is precisely the non-determinism the pin exists to
// remove.
func reserveQuestion(key string) string {
	f, ok := ReserveFeedFor(key)
	if !ok {
		return fmt.Sprintf("What is the attested reserve balance for feed %s?", key)
	}
	return fmt.Sprintf(
		"What is the attested reserve balance of %s backing %s, per Chainlink Proof-of-Reserve feed %s at block %s?",
		f.Asset, f.Token, f.Address, f.Block)
}

// describeReserve renders a key for logs and the stored audit trail.
func describeReserve(key string) string {
	if f, ok := ReserveFeedFor(key); ok {
		return f.Token + ":" + f.Key
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// reserveParsers holds one entry per registered ASSET_RESERVE_ATTESTATION
// miner, keyed by the slug in miner_registry.
//
// Exactly one: res-wbtc-por (registration 2763). Ten other res-* slugs exist in
// the registry (res-cg-cat, res-circle, res-llama-* and friends, registrations
// 1905-1914) and are all DEREGISTERED, so they are absent here rather than
// given a failing parser — the same reason the route-photon-* geocoders are
// absent from routeParsers. A miner that is not a participant is excluded, not
// scored zero.
//
// The one-miner field is the honest limit of this port, and it is stated at the
// TruthSource below rather than hidden: a ranking of one participant is not a
// ranking. What this port does establish is the scoring path — truth fetch,
// scale agreement, comparator call — so that a second PoR miner becomes a map
// entry rather than a project.
var reserveParsers = map[string]parserFunc{
	"res-wbtc-por": parseAttestedReserve, // {"attested_reserve":116144649223}
}

// parseAttestedReserve reads attested_reserve from the miner's response.
//
// The value is already in the comparator's micro-unit domain (see
// ScaleAssetReserve), so this returns it unscaled and ScaleValue's multiplier
// of 1 leaves it alone. That is worth being explicit about because the field is
// NOT the only numeric field in the body — res-wbtc-por also returns
// attested_reserve_raw (the raw 8-decimal on-chain integer, 100x larger) and
// attested_reserve_btc (a float in whole BTC, 1e6 smaller). Reading either by
// mistake yields a value that is still positive, still plausible-looking, and
// wrong by two or six orders of magnitude — which under RelTol scores a
// flat 0 rather than erroring, and would read as "this miner is lying" instead
// of "we read the wrong field". Hence the build spec's value_path is
// `.attested_reserve` and this parser reads that field and no other.
//
// json.Number rather than float64: at the micro-unit scale a reserve of
// 116,144.649 BTC is 116144649223, which is under 2^53 today but is one
// thousand-fold asset growth away from losing integer precision in a float64.
// An intent whose entire purpose is verifying a reserve figure should not carry
// a rounding hazard in its own reader.
func parseAttestedReserve(_ string, body string) (float64, error) {
	var resp struct {
		AttestedReserve *json.Number `json:"attested_reserve"`
	}
	dec := json.NewDecoder(strings.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&resp); err != nil {
		return 0, fmt.Errorf("decoding PoR response: %w", err)
	}
	if resp.AttestedReserve == nil {
		return 0, fmt.Errorf("no attested_reserve field in response")
	}
	v, err := resp.AttestedReserve.Float64()
	if err != nil {
		return 0, fmt.Errorf("parsing attested_reserve %q: %w", resp.AttestedReserve.String(), err)
	}
	return v, nil
}

// ────────────────────────────── feed coverage ─────────────────────────────────

// reserveCoverage records which feeds each miner can serve.
//
// res-wbtc-por takes the feed address as a query parameter, so in principle it
// could serve any PoR aggregator; in practice only the WBTC feed is scored, so
// the distinction is untested and anyQuote is the honest entry. The map is kept
// rather than collapsed so that a miner registered against a single fixed feed
// can later be given real coverage without reshaping the scoring pass.
var reserveCoverage = map[string]func(key string) bool{
	"res-wbtc-por": anyQuote,
}

// ───────────────────────────────── ground truth ────────────────────────────────

// Chainlink AggregatorV3Interface selectors. Function selectors rather than an
// ABI binding because the two calls needed here have no arguments and a
// fixed-width return, so hand-decoding is less machinery than a codegen
// dependency for a 32-byte-word read.
const (
	selDecimals        = "0x313ce567" // decimals() -> uint8
	selLatestRoundData = "0xfeaf968c" // latestRoundData() -> (uint80,int256,uint256,uint256,uint80)
)

// ChainlinkPoRSource reads the attested reserve straight off the Chainlink
// Proof-of-Reserve aggregator via a keyless public Ethereum JSON-RPC.
//
// Reading the aggregator directly rather than through an indexer or an API
// wrapper is the point. The registered miner res-wbtc-por is served from
// omni-chat, and it too ultimately reads this same contract — but if the host's
// truth were fetched from that same omni-chat /truth/ endpoint, the comparator
// would be comparing a host to itself and could never fail anyone. Truth has to
// come from a path the miner does not control, and for on-chain state the
// canonical such path is the chain.
//
// Read the resulting ranking with that structure in mind: res-wbtc-por reads
// the same contract the truth source reads, so a near-perfect score for it
// measures "both sides decoded the same 32-byte word correctly", not custodial
// diligence. This intent scores CORRECTNESS OF REPORTING, not quality of
// attestation — the miner's job is to relay the feed faithfully, and that is
// what a 10000 here means.
type ChainlinkPoRSource struct {
	// RPCURL is a keyless Ethereum JSON-RPC endpoint. When set it is tried
	// first; Fallbacks are tried after it.
	RPCURL string
	// Fallbacks are further keyless endpoints tried in order when the one
	// before them fails. See NewChainlinkPoRSource for why more than one is
	// needed.
	Fallbacks []string
	// Client is the HTTP client used for the call. A caller should set a
	// timeout on it; FetchReference also honours the context deadline.
	Client *http.Client
}

// NewChainlinkPoRSource builds a source with defaults suitable for epoch
// scoring.
//
// Several endpoints rather than one, because the pin and the free tier pull in
// opposite directions and that was learned the hard way here.
//
// Reading a PINNED block is, to most public providers, an ARCHIVE request, and
// keyless tiers keep only a short window of recent state. publicnode.com — the
// endpoint the reference host asset_reserve_price.mjs uses — was observed
// serving this feed at `latest` while rejecting the very same call at a block
// roughly 128 behind head, with:
//
//	{"code":-32602,"message":"Archive requests require a personal token..."}
//
// That window was measured in MINUTES, not days: a pin verified working was
// refused by the same provider forty minutes later. So a single default
// endpoint would make this intent unscoreable almost immediately after any pin
// is chosen — while still answering `latest` perfectly, which is the tempting
// wrong fix.
//
// The failover is safe precisely BECAUSE the block is pinned: every provider is
// being asked for the same immutable historical state, so which one answers
// cannot change the value. Verified — publicnode, drpc.org and merkle.io all
// returned byte-identical latestRoundData() at block 0x18d20a9. Failing over
// between providers on a `latest` query would be the opposite: a silent source
// of validator disagreement.
//
// The honest limitation this leaves: free tiers also rate-limit, and all three
// of these were seen returning 429/403 at various points within one session. A
// production deployment should point RPCURL at an endpoint the operator
// controls. These defaults make the path work without a key, not reliably.
func NewChainlinkPoRSource() *ChainlinkPoRSource {
	return &ChainlinkPoRSource{
		RPCURL: "https://eth.drpc.org",
		Fallbacks: []string{
			"https://ethereum-rpc.publicnode.com", // the reference host's default
			"https://eth.merkle.io",
		},
		Client: &http.Client{Timeout: 20 * time.Second},
	}
}

// endpoints returns the RPC endpoints to try, in order.
func (s *ChainlinkPoRSource) endpoints() []string {
	out := make([]string, 0, 1+len(s.Fallbacks))
	if s.RPCURL != "" {
		out = append(out, s.RPCURL)
	}
	out = append(out, s.Fallbacks...)
	return out
}

// ethCall performs one eth_call at a pinned block, trying each configured
// endpoint until one answers.
//
// Every endpoint's error is kept and reported together rather than only the
// last. The failures are not interchangeable — "archive requests require a
// token" means the pin has aged out and must be refreshed, while a 429 means
// try again later — and collapsing them to the final one hides which.
func (s *ChainlinkPoRSource) ethCall(ctx context.Context, to, data, block string) (string, error) {
	eps := s.endpoints()
	if len(eps) == 0 {
		return "", fmt.Errorf("no Ethereum RPC endpoint configured")
	}
	var errs []string
	for _, ep := range eps {
		res, err := s.ethCallOn(ctx, ep, to, data, block)
		if err == nil {
			return res, nil
		}
		errs = append(errs, fmt.Sprintf("%s: %v", ep, err))
		// A cancelled or timed-out context will not be helped by another
		// endpoint, and continuing would hammer each in turn against a deadline
		// that has already passed.
		if ctx.Err() != nil {
			break
		}
	}
	return "", fmt.Errorf("all %d Ethereum RPC endpoints failed: %s", len(eps), strings.Join(errs, "; "))
}

// ethCallOn performs one eth_call against one endpoint.
func (s *ChainlinkPoRSource) ethCallOn(ctx context.Context, rpcURL, to, data, block string) (string, error) {
	payload, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "eth_call",
		"params": []any{
			map[string]string{"to": to, "data": data},
			block,
		},
	})
	if err != nil {
		return "", fmt.Errorf("encoding eth_call request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, rpcURL, strings.NewReader(string(payload)))
	if err != nil {
		return "", fmt.Errorf("building eth_call request: %w", err)
	}
	req.Header.Set("content-type", "application/json")

	client := s.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("calling Ethereum RPC: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("reading RPC response: %w", err)
	}

	// The body is decoded BEFORE the status is checked, because the informative
	// part of a failure is in the body and not on the status line. publicnode
	// answers an out-of-retention pin with HTTP 403 whose body is
	// {"code":-32602,"message":"Archive requests require a personal token"} —
	// "403 Forbidden" alone reads like an auth problem and sends you looking in
	// the wrong place, as it did here.
	//
	// Conversely a JSON-RPC error can arrive with HTTP 200 and an `error`
	// member, so the status line does not tell you the call succeeded either.
	// Both directions have to be checked.
	var parsed struct {
		Result string `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err == nil && parsed.Error != nil {
		return "", fmt.Errorf("RPC error %d: %s", parsed.Error.Code, parsed.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %s: %s", resp.Status, strings.TrimSpace(truncate(string(body), 160)))
	}
	if parsed.Result == "" {
		return "", fmt.Errorf("empty result")
	}
	return parsed.Result, nil
}

// truncate bounds an upstream error body so a provider's HTML error page does
// not become the whole log line.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// RoundData is one decoded Chainlink latestRoundData() return, kept whole for
// the audit trail rather than reduced to the answer alone.
type RoundData struct {
	RoundID   *big.Int
	Answer    *big.Int // the reserve, at the feed's own `decimals` scale
	StartedAt *big.Int
	UpdatedAt *big.Int
	Answered  *big.Int
}

// decodeLatestRoundData splits an ABI-encoded latestRoundData() return into its
// five 32-byte words.
//
// `answer` is declared int256 and is decoded as a SIGNED two's-complement value
// rather than as a plain big unsigned integer. A negative reserve is nonsense
// and is rejected below — but a feed that ever returned one would, read as
// unsigned, become a ~1.2e77 "reserve" that is positive, enormous, and would
// score every honest miner a flat 0. Decoding the declared type is what turns
// that into a visible error.
func decodeLatestRoundData(hexResult string) (RoundData, error) {
	h := strings.TrimPrefix(hexResult, "0x")
	if len(h) < 64*5 {
		return RoundData{}, fmt.Errorf("unexpected latestRoundData() length: %d hex chars", len(h))
	}
	word := func(i int) (*big.Int, bool) {
		v, ok := new(big.Int).SetString(h[i*64:(i+1)*64], 16)
		return v, ok
	}
	var out RoundData
	targets := []**big.Int{&out.RoundID, &out.Answer, &out.StartedAt, &out.UpdatedAt, &out.Answered}
	for i, t := range targets {
		v, ok := word(i)
		if !ok {
			return RoundData{}, fmt.Errorf("word %d of latestRoundData() is not hex", i)
		}
		*t = v
	}
	// int256 two's complement: values with the top bit set are negative.
	if out.Answer.BitLen() == 256 {
		out.Answer.Sub(out.Answer, new(big.Int).Lsh(big.NewInt(1), 256))
	}
	return out, nil
}

// FetchReference resolves the pinned attested reserve for one feed, already
// scaled to the intent's micro-unit domain.
//
// The pin argument is accepted and ignored: this intent carries its own pin, a
// block number, in ReserveFeed.Block. Pin's Date field is an ECB-style
// publication day and has no meaning for a chain, and silently interpreting one
// as the other is exactly the kind of near-miss that produces a reproducible
// but wrong truth. The parameter stays because it is the TruthSource contract.
func (s *ChainlinkPoRSource) FetchReference(ctx context.Context, key string, _ Pin) (Reference, error) {
	feed, ok := ReserveFeedFor(key)
	if !ok {
		return Reference{}, fmt.Errorf("no pinned reserve feed for key %q", key)
	}
	if feed.Block == "" || feed.Block == "latest" {
		// Defensive rather than theoretical: the miner's own registration YAML
		// defaults block to "latest", so "latest" is the value that arrives if
		// a pin is ever dropped, and it must fail rather than score.
		return Reference{}, fmt.Errorf("feed %s has no pinned block; 'latest' is not reproducible", key)
	}

	// decimals() is read at the SAME pinned block as the answer, not at head.
	// The scale and the value it scales have to come from one consistent view
	// of the contract: an aggregator upgraded behind its proxy between the two
	// reads would otherwise be decoded at the wrong power of ten — a silent
	// 100x error, not a crash.
	decHex, err := s.ethCall(ctx, feed.Address, selDecimals, feed.Block)
	if err != nil {
		return Reference{}, fmt.Errorf("reading feed decimals: %w", err)
	}
	decimals, ok := new(big.Int).SetString(strings.TrimPrefix(decHex, "0x"), 16)
	if !ok {
		return Reference{}, fmt.Errorf("feed decimals is not hex: %q", decHex)
	}
	if !decimals.IsInt64() || decimals.Int64() < 0 || decimals.Int64() > 30 {
		return Reference{}, fmt.Errorf("implausible feed decimals: %s", decimals)
	}

	roundHex, err := s.ethCall(ctx, feed.Address, selLatestRoundData, feed.Block)
	if err != nil {
		return Reference{}, fmt.Errorf("reading latestRoundData: %w", err)
	}
	round, err := decodeLatestRoundData(roundHex)
	if err != nil {
		return Reference{}, fmt.Errorf("decoding latestRoundData: %w", err)
	}
	if round.Answer.Sign() < 0 {
		return Reference{}, fmt.Errorf("PoR feed returned a negative reserve: %s", round.Answer)
	}
	// updatedAt == 0 marks an incomplete round. Its answer is not a reserve
	// reading at all, and scoring against it would fail every miner for a
	// defect in the feed.
	if round.UpdatedAt.Sign() == 0 {
		return Reference{}, fmt.Errorf("PoR feed round is incomplete (updatedAt == 0)")
	}

	micro, err := toMicroUnits(round.Answer, int(decimals.Int64()))
	if err != nil {
		return Reference{}, fmt.Errorf("rescaling the reserve answer: %w", err)
	}

	return Reference{
		RateBps: micro,
		// RawRate carries whole units of the underlying (BTC here) rather than
		// the micro-unit integer, because it exists for humans reading logs and
		// "116144.649 BTC" is checkable against a block explorer at a glance.
		// It is diagnostic only — nothing scores off it.
		RawRate: float64(micro) / 1e6,
		Pair:    feed.Token + "/" + feed.Asset,
		Date:    time.Unix(round.UpdatedAt.Int64(), 0).UTC().Format("2006-01-02"),
		Source:  fmt.Sprintf("chainlink-por:%s@%s", feed.Address, feed.Block),
	}, nil
}

// toMicroUnits rescales a raw feed answer from the feed's own `decimals` to the
// comparator's fixed 1e6 micro-unit domain.
//
// THE SCALE QUESTION, since it is the one place a wrong answer is silent.
//
// The build spec's parse_rule reads "reserve_amount_in_underlying_units * 1e6".
// That is NOT "multiply the number the feed returns by 1e6" — the multiplicand
// is the reserve in WHOLE UNITS of the underlying asset, and the feed does not
// report whole units. The WBTC PoR aggregator reports at its own decimals() = 8
// (verified live at the pinned block), so its raw answer 11614464922334 is
// 116144.64922334 BTC, and the micro-unit figure is
//
//	116144.64922334 * 1e6 = 116144649223  ==  11614464922334 / 100
//
// i.e. the operation on the RAW answer is a division by 10^(decimals-6) = 100,
// not a multiplication by 1e6. Applying 1e6 to the raw integer instead would
// yield 1.16e19 — a plausible-looking positive integer that is 1e8 too large,
// would overflow nothing, would crash nothing, and would score every honest
// miner a flat 0.
//
// This is done in big.Int rather than float64 deliberately: an exact integer
// division has no rounding question at all, whereas doing it in floating point
// would reintroduce one at precisely the scale the intent is trying to pin
// down. It is also why the intent's ScaleValue multiplier is 1 — the rescale
// happens here, at the only place that knows the feed's decimals.
//
// The decimals >= 6 branch is the live one for every PoR feed seen so far
// (8 for BTC-denominated, 18 for ETH-denominated); the other branch handles a
// hypothetical sub-micro feed and scales up instead.
func toMicroUnits(answer *big.Int, decimals int) (int64, error) {
	const microExp = 6
	scaled := new(big.Int)
	switch {
	case decimals >= microExp:
		div := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(decimals-microExp)), nil)
		scaled.Quo(answer, div)
	default:
		mul := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(microExp-decimals)), nil)
		scaled.Mul(answer, mul)
	}
	if !scaled.IsInt64() {
		return 0, fmt.Errorf("reserve %s at %d decimals overflows the i64 comparator domain", answer, decimals)
	}
	if scaled.Sign() <= 0 {
		return 0, fmt.Errorf("reserve rescales to a non-positive value: %s", scaled)
	}
	return scaled.Int64(), nil
}

// Register this intent's ground-truth fetcher so a spec row can name it.
// Lives here rather than in the registry file so the factory sits beside the
// parsing code that understands this upstream's quirks.
//
// Parameters are unused today: feed address and pinned block are carried by the key.
func init() {
	RegisterTruthFactory("chainlink_por", func(map[string]any) (TruthSource, error) {
		return NewChainlinkPoRSource(), nil
	})
}
