// Package family resolves which installed adapter serves an intent.
//
// Intents are becoming hyper-specific (WEATHER_FORECAST_LAHORE,
// WEATHER_FORECAST_TOKYO) so that miners answer the same question and can be
// compared fairly. A separate adapter competition per such intent is not
// viable, so an adapter registered on a FAMILY intent is reused by every CHILD
// intent whose name extends it with an underscore. Each child keeps its own
// miners, rankings, eval-pool data, spec and ground truth; only the scoring
// adapter is shared.
//
// The rule is longest-prefix-wins: a child X uses the adapter of the longest
// installed key K where X == K or X starts with K + "_". An exact match (a
// child with its own adapter) therefore always wins, and WEATHER_FORECAST_LAHORE
// prefers WEATHER_FORECAST over WEATHER. The separator is part of the rule:
// WEATHERX never resolves to WEATHER.
//
// There is no configuration: the set of families IS the set of intents with an
// installed adapter, which every validator derives from the same on-chain
// Active registrations, so every validator resolves identically. Rollback is
// evicting the family adapter: its children then fall to the next shorter
// installed prefix, or to whatever path they took before.
//
// Everything here is pure and deterministic. No map iteration order is ever
// involved: callers pass a membership predicate, and the walk is over the
// intent name from longest to shortest.
package family

import (
	"sort"
	"strings"
)

// Separator is the segment delimiter in canonical intent names.
const Separator = "_"

// Resolve returns the longest installed key that is intentID itself or a
// "_"-delimited prefix of it. installed reports whether a key has an adapter.
//
// The walk is from the full name down, cutting at each "_" from the right, so
// the first hit is the longest prefix by construction. Returns ("", false) for
// an empty intentID or when nothing matches.
func Resolve(intentID string, installed func(string) bool) (key string, ok bool) {
	if intentID == "" || installed == nil {
		return "", false
	}
	candidate := intentID
	for {
		if installed(candidate) {
			return candidate, true
		}
		i := strings.LastIndex(candidate, Separator)
		if i <= 0 {
			// No separator left (or a leading one, which cannot bound a
			// non-empty prefix): nothing shorter can match.
			return "", false
		}
		candidate = candidate[:i]
	}
}

// IsChild reports whether intentID equals familyID or extends it with "_".
// It is the single definition of "under the family" that Children and the
// scorers share, so the separator rule cannot drift between them.
func IsChild(familyID, intentID string) bool {
	if familyID == "" || intentID == "" {
		return false
	}
	return intentID == familyID || strings.HasPrefix(intentID, familyID+Separator)
}

// Children returns every intent in all that equals familyID or starts with
// familyID + "_", sorted and de-duplicated. The family itself is included when
// present in all, because the family's own pool and miners are part of what
// its adapter serves.
func Children(familyID string, all []string) []string {
	if familyID == "" {
		return nil
	}
	seen := make(map[string]struct{}, len(all))
	out := make([]string, 0, len(all))
	for _, id := range all {
		if !IsChild(familyID, id) {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// Serving groups every intent in all by the installed key that serves it,
// applying Resolve against the installed set. Intents that resolve to nothing
// are omitted. The returned slices are sorted. Used for the startup/promote
// "which family serves which children" report and the API resolve view.
func Serving(installed []string, all []string) map[string][]string {
	set := make(map[string]struct{}, len(installed))
	for _, k := range installed {
		set[k] = struct{}{}
	}
	isInstalled := func(k string) bool { _, ok := set[k]; return ok }

	out := make(map[string][]string)
	seen := make(map[string]struct{}, len(all))
	for _, id := range all {
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		key, ok := Resolve(id, isInstalled)
		if !ok {
			continue
		}
		out[key] = append(out[key], id)
	}
	for k := range out {
		sort.Strings(out[k])
	}
	return out
}

// ServedChildren returns, sorted, the intents in all that an adapter on
// familyID would serve: those IsChild(familyID, ·) and sameClass(·), minus any
// that a LONGER installed key claims by Resolve (familyID itself counts as
// installed — it is the adapter being asked about). installed may be nil
// (nothing else installed). The family itself is included when it is in all
// and sameClass.
//
// It is the pure form of scoring.CatalogFamilyChildrenServed (which reads the
// node's loaded catalog and live scorers), for callers that must supply their
// own class source and installed set — cmd/adapterbuild checks a training
// record against it offline. A test in pkg/scoring keeps the two equal.
func ServedChildren(familyID string, all []string, sameClass func(string) bool, installed func(string) bool) []string {
	isKey := func(k string) bool { return k == familyID || (installed != nil && installed(k)) }
	out := make([]string, 0, 4)
	for _, id := range Children(familyID, all) {
		if sameClass != nil && !sameClass(id) {
			continue
		}
		if key, ok := Resolve(id, isKey); ok && key == familyID {
			out = append(out, id)
		}
	}
	return out
}
