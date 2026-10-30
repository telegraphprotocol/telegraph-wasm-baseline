package comparator

import (
	"fmt"
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

// ────────────────────────────── feed coverage ─────────────────────────────────
