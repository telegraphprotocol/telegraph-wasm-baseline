// Package catalog is the runtime source of per-intent scoring classification:
// which intents are scored on the model/adapter path (judgment + hybrid) versus
// the deterministic comparator path. It materialises, for the code that needs
// it at runtime, the classification that is otherwise authoritative only in the
// design docs (intent-catalog-final-classification.md at C:\Work\Telegraph Docs).
//
// It exists because the WASM promotion gate must route judgment/hybrid intents
// to the hidden Evaluation-Pool gate, and the epoch loop must route them to the
// judgment scorer; the intent_classes table (classes_db.go) overrides this
// compiled-in map when loaded. Keep this
// map in sync with the classification doc when intents are added or reclassified.
package catalog

// judgmentOrHybrid is the set of intents that carry an evaluation adapter and so
// are ranked against the hidden Evaluation Pool: the 11 Judgment ("adapter
// only") intents plus the 20 Hybrid ("comparator + adapter") intents from the
// final classification. The 32 purely-deterministic ("comparator only") intents
// are absent — they need no model and take the historical/comparator path.
var judgmentOrHybrid = map[string]struct{}{
	// ── Judgment (adapter only) — 11 ──────────────────────────────
	"CODE_REVIEW":                {},
	"LLM_OUTPUT_EVALUATION":      {},
	"CONTRACT_OBLIGATION_AUDIT":  {},
	"TASK_EXECUTION_QUALITY":     {},
	"CUSTOMER_TICKET_RESOLUTION": {},
	"FRAUD_RISK":                 {},
	"TRANSACTION_RISK":           {},
	"MEDIA_FORENSIC_VERIFY":      {},
	"SOCIAL_BOT_DETECTION":       {},
	"SATELLITE_IMAGERY_ANALYSIS": {},
	"SHIPMENT_DELAY_RISK":        {},

	// ── Hybrid (comparator + adapter) — 20 ────────────────────────
	"KYC_BIOMETRIC_LIVENESS":    {},
	"DOCUMENT_AUTHENTICITY":     {},
	"EVENT_OUTCOME_RESOLUTION":  {},
	"WASH_TRADING_DETECTION":    {},
	"OPTIMAL_EXECUTION_ROUTE":   {},
	"SECURITY_REVIEW":           {},
	"VULNERABILITY_TRIAGE":      {},
	"URL_SAFE":                  {},
	"EMAIL_SECURITY":            {},
	"MALWARE_DETECTION":         {},
	"THREAT_INTELLIGENCE":       {},
	"VENDOR_VERIFY":             {},
	"PRODUCT_AUTHENTICITY":      {},
	"SHIP_RATE_ETA":             {},
	"RETURN_POLICY_VERIFY":      {},
	"FARE_RULE_VERIFY":          {},
	"SANCTIONS_SCREENING_MATCH": {},
	"B2B_IDENTITY_ENRICHMENT":   {},
	"INVOICE_LEDGER_RECONCILE":  {},
	"LIVE_PORT_CONGESTION":      {},
}

// IsJudgmentIntent reports whether intentID is scored on the model/adapter path
// (Judgment or Hybrid) and therefore ranked against the hidden Evaluation Pool.
// Deterministic (comparator-only) intents return false and keep the historical
// rank-agreement gate.
func IsJudgmentIntent(intentID string) bool {
	// Prefer the loaded classification; fall back to the map below when the
	// table has not been read, so a node that cannot reach the database keeps
	// the classification it shipped with rather than treating everything as
	// deterministic and scoring judgment intents at zero.
	if c, ok := ClassOf(intentID); ok {
		return c == ClassHybrid || c == ClassJudgment
	}
	_, ok := judgmentOrHybrid[intentID]
	return ok
}
