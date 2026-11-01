package adapterbuild

import (
	"fmt"
	"strings"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/comparator"
	wasmrt "github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/runtime"
)

// ResolveClass decides which class ("judgment" or "hybrid", the
// telegraph.adapter section's class) an intent's package is built for. The
// intent catalog is the authority, not the caller:
//
//   - the DB-loaded classification when one is loaded in-process
//     (catalog.LoadedClassCount() > 0), else the compiled-in classification
//     (catalog.CompiledClassOf) — the builder normally runs offline;
//   - deterministic / fact-checkable intents are refused: they are scored by the
//     shared comparator module and never get a float adapter;
//   - an intent the catalog does not know is refused;
//   - an explicit class, when given, must agree — a mismatch is an error, never
//     an override, because a package built for the wrong class is refused by
//     every node at Stage 1.
//
// A node decides the class it demands from the intent's loaded comparator
// spec (comparator.IsHybrid: a spec with a fusion block), which comes from its
// database and is not available offline; the catalog classification is the
// offline source of the same fact.
func ResolveClass(intent, explicit string) (string, error) {
	if intent == "" {
		return "", fmt.Errorf("adapterbuild: intent is required")
	}
	classOf, source := classSource()
	c, ok := classOf(intent)
	if !ok || c == catalog.ClassDeterministic {
		if c == catalog.ClassDeterministic || comparator.IsDeterministic(intent) {
			return "", fmt.Errorf("adapterbuild: %s is a deterministic (fact-checkable) intent — it is scored by the "+
				"shared comparator module and does not get a float adapter", intent)
		}
		return "", fmt.Errorf("adapterbuild: %s is not a judgment or hybrid intent in the %s — refusing to guess a class", intent, source)
	}
	var derived string
	switch c {
	case catalog.ClassJudgment:
		derived = wasmrt.ClassJudgment
	case catalog.ClassHybrid:
		derived = wasmrt.ClassHybrid
	default:
		return "", fmt.Errorf("adapterbuild: %s has unknown class %q in the %s", intent, c, source)
	}
	if explicit != "" && explicit != derived {
		return "", fmt.Errorf("adapterbuild: --class %s disagrees with the %s, which classifies %s as %s — "+
			"refusing (a package built for the wrong class is refused by every node)", explicit, source, intent, derived)
	}
	return derived, nil
}

// classSource is the class authority the builder uses everywhere (ResolveClass
// and the training-record check): the DB-loaded classification when one is
// loaded in-process, else the compiled-in classification. Its intent list is
// catalog.ListIntents(), which follows the same precedence.
func classSource() (func(string) (catalog.Class, bool), string) {
	if catalog.LoadedClassCount() > 0 {
		return catalog.ClassOf, "intent_classes (loaded)"
	}
	return compiledClassOf, "compiled-in catalog"
}

// GenericIntent is the reserved prefix of the GENERIC fallback Evaluator
// adapter: always judgment, never in the compiled-in intent catalog.
const GenericIntent = "GENERIC"

// compiledClassOf resolves the live catalog's *_GENERAL IDs to their original
// intent's compiled-in class; a loaded intent_classes table is never overridden.
func compiledClassOf(intent string) (catalog.Class, bool) {
	if intent == GenericIntent {
		return catalog.ClassJudgment, true
	}
	if c, ok := catalog.CompiledClassOf(intent); ok {
		return c, ok
	}
	return catalog.CompiledClassOf(strings.TrimSuffix(intent, "_GENERAL"))
}
