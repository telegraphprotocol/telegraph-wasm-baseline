package comparator

import (
	"fmt"
)

// ONCHAIN_METRIC_VERIFY: a wallet's ETH balance at a PINNED block height,
// expressed in whole Gwei.
//
// This is the only one of the wired intents whose ground truth is exact rather
// than merely stable. An FX fix is a published opinion and an OSRM duration is
// one engine's model, so both tolerate disagreement; a balance at a fixed block
// is canonical consensus state, and every archive node on Earth returns the
// identical integer for it. That is why the band is 5 bps — the tightest of any
// deterministic intent — and it is the reason the pin below is not a detail but
// the whole design.
//
// Verified live rather than assumed: eth.merkle.io and the registered miner both
// returned 0x4254812b06de0cf3 wei for 0xd8dA…6045 at block 23,000,000, and
// eth.drpc.org matched merkle on the beacon deposit contract at the same height.

// ───────────────────────────────── pinned keys ─────────────────────────────────

// BalanceKey is one pinned (address, block) pair.
//
// The block is stored as a decimal uint64 and rendered to hex at the call, because
// JSON-RPC takes a hex quantity while the miner's registered `block` parameter
// accepts either. Keeping the canonical form decimal means the two renderings
// come from one number rather than from two hand-written strings that can drift.
type BalanceKey struct {
	Key     string // short stable ID, used as the question key
	Name    string // human description for logs and the audit trail
	Address string // 0x-prefixed, EIP-55 mixed case as published
	Block   uint64 // PINNED height; never a tag
}

// BlockHex renders the pinned height as the JSON-RPC quantity, which is
// minimal-width hex with no leading zeroes.
func (b BalanceKey) BlockHex() string { return fmt.Sprintf("0x%x", b.Block) }

// PinnedBlock is the height every ONCHAIN_METRIC_VERIFY key is read at.
//
// Pinning is the single most important decision in this port, and it is forced
// rather than preferred. The registered miner's `block` parameter DEFAULTS to
// "latest", and "latest" is not a reproducible key: Ethereum produces a block
// every ~12s, and a funded address moves between them. Two validators scoring
// the same epoch a few seconds apart would read two different balances, compute
// two different truths, and disagree about every miner in the round — and at a
// 5 bps band a single ordinary transfer is far more than enough to flip a score
// from 10000 to 0. The reference host onchain_metric_price.mjs makes the same
// judgement structurally, by *refusing* to accept blockTag == "latest" at all.
//
// So the tolerance does not absorb chain-tip drift, and it was never meant to:
// the spec's tolerance_meaning says the 5 bps exists only to absorb the sub-Gwei
// truncation of the wei->Gwei rescale, "not real disagreement". Pinning is what
// makes that claim true. Everything else here — exact integer parsing, archive
// RPC selection — is downstream of this line.
//
// 23,000,000 specifically: it is final beyond any reorg depth (a reorg past a
// few blocks has never happened on mainnet), it predates this port by months so
// it cannot be tuned to flatter a miner, and it was confirmed servable by two
// independent keyless archive providers. It is a constant rather than a rolling
// "tip minus N" because a rolling pin reintroduces exactly the disagreement the
// pin removes unless every validator also agrees on when "now" is.
const PinnedBlock uint64 = 23_000_000

// BalanceKeys are the (address, block) pairs ONCHAIN_METRIC_VERIFY is scored on.
//
// Three addresses rather than one, for the same reason ROUTE_ETA asks three
// routes: the miner carries a registered DEFAULT address, so a single question
// cannot distinguish a miner that honours the address it is given from one that
// ignores it and replays its default. Asking three catches that — a miner
// replaying a baked-in address scores 0 on the two it was not built around.
//
// They also span four orders of magnitude on purpose (≈4.8 Gether, ≈30.7
// Tgwei... through ≈6.5e16 Gwei), which is what exercises the integer-exactness
// problem documented on parseETHBalance. A key set clustered at one magnitude
// would let a float-based parser look correct.
var BalanceKeys = []BalanceKey{
	{
		Key: "VITALIK", Name: "vitalik.eth balance", Block: PinnedBlock,
		Address: "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
	},
	{
		// The beacon chain deposit contract: the largest single ETH holder, and
		// the key that proves the parser is exact. Its Gwei value is ~6.5e16,
		// above 2^53, so it is NOT representable at 1 Gwei resolution in a
		// float64 — see parseETHBalance.
		Key: "BEACON", Name: "beacon deposit contract balance", Block: PinnedBlock,
		Address: "0x00000000219ab540356cBB839Cbe05303d7705Fa",
	},
	{
		Key: "EXCHANGE", Name: "exchange cold wallet balance", Block: PinnedBlock,
		Address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
	},
}

// BalanceKeyIDs is the fixed question set, in the order rounds are run.
var BalanceKeyIDs = balanceKeyIDs()

func balanceKeyIDs() []string {
	keys := make([]string, 0, len(BalanceKeys))
	for _, k := range BalanceKeys {
		keys = append(keys, k.Key)
	}
	return keys
}

// BalanceKeyFor resolves a question key to its pinned address and block.
func BalanceKeyFor(key string) (BalanceKey, bool) {
	for _, k := range BalanceKeys {
		if k.Key == key {
			return k, true
		}
	}
	return BalanceKey{}, false
}

// onchainQuestion is the question put to every miner for one pinned key.
//
// It states the block height explicitly, and as a decimal number, because the
// height is the part that must survive translation into the miner's own call.
// A question that said only "the current balance" would be answered at the tip
// — which is precisely the non-reproducible read the pin exists to prevent.
func onchainQuestion(key string) string {
	k, ok := BalanceKeyFor(key)
	if !ok {
		return fmt.Sprintf("What is the on-chain balance for %s?", key)
	}
	return fmt.Sprintf(
		"What is the Ethereum balance in Gwei of address %s at block %d (%s)?",
		k.Address, k.Block, k.Name)
}

// describeOnchain renders a key for logs and the stored audit trail. The block
// is included because a score is only auditable against the height it was
// computed at.
func describeOnchain(key string) string {
	if k, ok := BalanceKeyFor(key); ok {
		return fmt.Sprintf("%s@%d", k.Key, k.Block)
	}
	return key
}

// ───────────────────────────── miner response parsing ─────────────────────────

// ──────────────────────────────── key coverage ─────────────────────────────────
