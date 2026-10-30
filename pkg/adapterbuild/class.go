package adapterbuild

import (
	"fmt"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/comparator"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wideembed"
)

// ResolveClass decides which class (and therefore which pinned host) an
// intent's package is built for. The intent catalog is the authority, not the
// caller:
//
//   - the DB-loaded classification when one is loaded in-process
//     (catalog.LoadedClassCount() > 0), else the compiled-in classification
//     (catalog.CompiledClassOf) — the builder normally runs offline;
//   - deterministic / fact-checkable intents are refused: they are scored by the
//     shared comparator module and never get a wide package;
//   - an intent the catalog does not know takes an explicit --class judgment or
//     hybrid: the compiled-in list predates many live judgment/hybrid prefixes
//     (for example WEATHER_FORECAST_VERIFY), and the node still refuses a package
//     whose class does not match the intent's live class. Without --class it is
//     refused, with a pointer to the node's resolver;
//   - for an intent the catalog does know, an explicit class must agree — a
//     mismatch is an error, never an override, because a package built for the
//     wrong class is refused by every node at install time (or, worse, scored by
//     the wrong path).
func ResolveClass(intent, explicit string) (wideembed.Class, error) {
	if intent == "" {
		return "", fmt.Errorf("adapterbuild: intent is required")
	}
	classOf, source := classSource()
	c, ok := classOf(intent)
	if !ok || c == catalog.ClassDeterministic {
		if c == catalog.ClassDeterministic || comparator.IsDeterministic(intent) {
			return "", fmt.Errorf("adapterbuild: %s is a deterministic (fact-checkable) intent — it is scored by the "+
				"shared comparator module and does not get a wide package", intent)
		}
		if explicit == string(wideembed.ClassJudgment) || explicit == string(wideembed.ClassHybrid) {
			return wideembed.Class(explicit), nil
		}
		return "", fmt.Errorf("adapterbuild: %s is not a judgment or hybrid intent in the %s — refusing to guess a class; "+
			"if the node lists it (GET /api/wasm?resolve=%s), pass --class judgment or --class hybrid", intent, source, intent)
	}
	var derived wideembed.Class
	switch c {
	case catalog.ClassJudgment:
		derived = wideembed.ClassJudgment
	case catalog.ClassHybrid:
		derived = wideembed.ClassHybrid
	default:
		return "", fmt.Errorf("adapterbuild: %s has unknown class %q in the %s", intent, c, source)
	}
	if explicit != "" && wideembed.Class(explicit) != derived {
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
	return catalog.CompiledClassOf, "compiled-in catalog"
}
