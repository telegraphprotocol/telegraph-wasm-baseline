package catalog

import (
	"context"
	"database/sql"
	"fmt"
	"sort"
	"sync"
)

// Class is the scoring path an intent takes.
type Class string

const (
	// ClassDeterministic scores by pure integer comparison against a fetched
	// reference. No model, no adapter, and it keeps the historical
	// rank-agreement gate.
	ClassDeterministic Class = "deterministic"

	// ClassHybrid fuses a comparator verdict with an adapter's judgment of the
	// response text.
	ClassHybrid Class = "hybrid"

	// ClassJudgment has no checkable fact at all — the adapter is the whole
	// score.
	ClassJudgment Class = "judgment"
)

// classes holds the loaded classification.
//
// Empty until LoadClasses runs, and that is deliberate: IsJudgmentIntent falls
// back to the compiled-in map while it is empty, so a node that cannot reach
// the database keeps scoring with the classification it shipped with rather
// than silently treating every intent as deterministic — which would route
// judgment intents to the comparator and score them all at zero.
var classes struct {
	sync.RWMutex
	byIntent map[string]Class
}

// LoadClasses reads the intent classification from the database.
//
// This replaces a hand-maintained Go map whose own comment admitted the
// problem: "there is no on-chain or DB column carrying this classification.
// Keep this map in sync with the classification doc when intents are added or
// reclassified." Synchronising a map against a Word document by hand is a
// classification that drifts, and the drift is silent — a misclassified intent
// does not error, it is routed to the wrong gate and scored by the wrong
// machinery until somebody notices.
func LoadClasses(ctx context.Context, db *sql.DB) (map[string]Class, error) {
	rows, err := db.QueryContext(ctx, `SELECT intent_id, class FROM intent_classes`)
	if err != nil {
		return nil, fmt.Errorf("catalog: reading intent_classes: %w", err)
	}
	defer rows.Close()

	loaded := make(map[string]Class)
	for rows.Next() {
		var id, c string
		if err := rows.Scan(&id, &c); err != nil {
			return nil, fmt.Errorf("catalog: scanning intent_classes: %w", err)
		}
		switch Class(c) {
		case ClassDeterministic, ClassHybrid, ClassJudgment:
			loaded[id] = Class(c)
		default:
			// The column has a CHECK constraint, so this means the constraint
			// and this code disagree — worth failing loudly rather than
			// defaulting, because a silent default picks a scoring path.
			return nil, fmt.Errorf("catalog: intent %s has unknown class %q", id, c)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("catalog: iterating intent_classes: %w", err)
	}
	if len(loaded) == 0 {
		return nil, fmt.Errorf("catalog: intent_classes is empty; refusing to replace the " +
			"compiled-in classification with nothing")
	}

	classes.Lock()
	classes.byIntent = loaded
	classes.Unlock()
	return loaded, nil
}

// ClassOf returns an intent's scoring class.
//
// Falls back to the compiled-in map when the table has not been loaded, so
// behaviour is unchanged for a node that has not migrated.
func ClassOf(intentID string) (Class, bool) {
	classes.RLock()
	c, ok := classes.byIntent[intentID]
	classes.RUnlock()
	if ok {
		return c, true
	}
	if _, isJH := judgmentOrHybrid[intentID]; isJH {
		// The compiled-in map does not distinguish the two, and guessing
		// wrong picks a scoring path — so report hybrid, which is the
		// conservative answer: it still routes to the Evaluation Pool gate.
		return ClassHybrid, true
	}
	return "", false
}

// UseClasses installs a classification directly, as LoadClasses would from the
// table. For tests and tools (mirrors comparator.UseSpecs); production loads
// the intent_classes table.
func UseClasses(m map[string]Class) {
	cp := make(map[string]Class, len(m))
	for k, v := range m {
		cp[k] = v
	}
	classes.Lock()
	classes.byIntent = cp
	classes.Unlock()
}

// ClearClasses drops the loaded classification, restoring the compiled-in
// fallback. For tests.
func ClearClasses() {
	classes.Lock()
	classes.byIntent = nil
	classes.Unlock()
}

// LoadedClassCount reports how many intents the loaded classification covers.
func LoadedClassCount() int {
	classes.RLock()
	defer classes.RUnlock()
	return len(classes.byIntent)
}

// ListIntents returns every classified intent ID, sorted.
//
// It reads the loaded intent_classes table when one has been loaded and falls
// back to the compiled-in map otherwise, the same precedence ClassOf applies,
// so the set of intents this reports is exactly the set ClassOf can classify.
// Used by the adapter-family machinery to enumerate a family's children: a
// child that is not classified cannot be served by a family adapter, because
// its class — which must equal the family's — is unknown.
func ListIntents() []string {
	classes.RLock()
	loaded := classes.byIntent
	classes.RUnlock()

	var out []string
	if len(loaded) > 0 {
		out = make([]string, 0, len(loaded))
		for id := range loaded {
			out = append(out, id)
		}
	} else {
		out = make([]string, 0, len(judgmentOrHybrid))
		for id := range judgmentOrHybrid {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}
