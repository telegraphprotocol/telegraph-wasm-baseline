package comparator

import (
	"fmt"
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

// ──────────────────────────────── pool coverage ────────────────────────────────
