package catalog

// compiledHybrid is the Hybrid ("comparator + adapter") subset of the
// compiled-in judgmentOrHybrid map, from the same final classification
// (intent-catalog-final-classification.md). Everything in judgmentOrHybrid that
// is not listed here is Judgment ("adapter only").
//
// It exists for OFFLINE tools that must pick a scoring class without a
// database — cmd/adapterbuild decides which pinned embed host (judgment or
// hybrid) an intent's self-contained package is built into, and a package built
// for the wrong class is refused by every node at install time. ClassOf keeps
// its conservative "report hybrid" fallback for the node; this does not change
// it.
var compiledHybrid = map[string]struct{}{
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

// CompiledClassOf returns an intent's class from the compiled-in classification
// only — never the DB-loaded intent_classes table — distinguishing judgment
// from hybrid. Intents outside the model/adapter path (deterministic, or not in
// the catalog at all) return ("", false): the compiled-in data does not list
// deterministic intents, so callers must not read absence as "deterministic".
func CompiledClassOf(intentID string) (Class, bool) {
	if _, ok := judgmentOrHybrid[intentID]; !ok {
		return "", false
	}
	if _, ok := compiledHybrid[intentID]; ok {
		return ClassHybrid, true
	}
	return ClassJudgment, true
}
