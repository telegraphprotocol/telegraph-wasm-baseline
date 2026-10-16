// Package intentformat is the one place a node learns each intent's response
// format: the shape every miner's answer for that intent is read into, both
// for the answer a user receives (POST /engine/v1/ask, WS ask) and for scoring
// (the comparator reads the format's scored field).
//
// Sources, in order of precedence:
//
//  1. intent_response_formats - one row per intent, any class (deterministic,
//     hybrid, judgment). The home for every format.
//  2. intent_specs.response_format - the older column, which only comparator
//     intents have (the table requires a truth source). Used for an intent
//     with no row in (1), so nothing that already works breaks. When both
//     exist and differ, (1) wins and the difference is logged.
//  3. Family inheritance - a child intent with no format of its own
//     (WEATHER_FORECAST_LAHORE) uses the format of its longest "_"-prefix
//     family that has one (WEATHER_FORECAST), by the same rule adapters use
//     (pkg/scoring/family.Resolve), and only when both are classified and of
//     the same class. A child's own format always wins.
//
// Loaded once at startup, like intent_specs and intent_classes. A bad row is
// skipped and logged; it never fails the load.
package intentformat

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/catalog"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/family"
	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
)

// Sources an Entry can come from.
const (
	SourceTable = "intent_response_formats"
	SourceSpecs = "intent_specs"
)

// LegacyVersion is the version reported for a format that comes from
// intent_specs.response_format, which has no version column.
const LegacyVersion = 1

// Entry is one intent's own format.
type Entry struct {
	IntentID string
	Format   *responseparser.Format
	Version  int
	Source   string
}

// Resolved is the format that applies to an intent: its own, or its family's.
type Resolved struct {
	Entry
	// ServedBy is the intent whose format this is: the intent itself, or the
	// family it inherits from.
	ServedBy string
}

// Row is one intent_response_formats row as read from the table.
type Row struct {
	IntentID string
	Format   []byte
	Version  int
}

// ClassFunc reports an intent's scoring class (catalog.ClassOf).
type ClassFunc func(intentID string) (catalog.Class, bool)

// Report is what Build found, for logs and tests.
type Report struct {
	// Skipped holds every row that could not be used, with why.
	Skipped map[string]error
	// Differs lists intents whose table row and intent_specs format both
	// exist and are not the same format (the table wins).
	Differs []string
	// FromTable and FromSpecs count the loaded formats by source.
	FromTable, FromSpecs int
}

// Registry holds every intent's own format. Lookups apply family inheritance.
type Registry struct {
	own     map[string]Entry
	classOf ClassFunc
}

// Build validates rows and merges them with the intent_specs formats. Pure:
// no I/O. classOf decides which intents need a scored field (deterministic and
// hybrid do) and which families a child may inherit from; nil means
// catalog.ClassOf.
func Build(rows []Row, specFormats map[string]*responseparser.Format, classOf ClassFunc) (*Registry, Report) {
	if classOf == nil {
		classOf = catalog.ClassOf
	}
	r := &Registry{own: make(map[string]Entry, len(rows)+len(specFormats)), classOf: classOf}
	rep := Report{Skipped: map[string]error{}}

	for _, row := range rows {
		id := strings.TrimSpace(row.IntentID)
		if id == "" {
			rep.Skipped[row.IntentID] = fmt.Errorf("row has an empty intent_id")
			continue
		}
		f, err := responseparser.ParseIntentFormat(row.Format)
		if err == nil {
			err = CheckClass(id, f, classOf)
		}
		if err == nil && row.Version < 1 {
			err = fmt.Errorf("version %d: must be 1 or more", row.Version)
		}
		if err != nil {
			rep.Skipped[id] = err
			log.Error().Err(err).Str("intent", id).
				Msg("[intent_format] intent_response_formats row is malformed — SKIPPED; the intent falls back to intent_specs.response_format or has no format until it is fixed")
			continue
		}
		r.own[id] = Entry{IntentID: id, Format: f, Version: row.Version, Source: SourceTable}
		rep.FromTable++
	}

	ids := make([]string, 0, len(specFormats))
	for id := range specFormats {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		sf := specFormats[id]
		if sf == nil {
			continue
		}
		if own, ok := r.own[id]; ok {
			if !Same(own.Format, sf) {
				rep.Differs = append(rep.Differs, id)
				log.Warn().Str("intent", id).
					Msg("[intent_format] intent_response_formats and intent_specs.response_format both define this intent's format and DIFFER — using intent_response_formats; align or clear intent_specs.response_format")
			}
			continue
		}
		r.own[id] = Entry{IntentID: id, Format: sf, Version: LegacyVersion, Source: SourceSpecs}
		rep.FromSpecs++
	}
	return r, rep
}

// CheckClass applies the class rule: a deterministic or hybrid intent must
// have a scored field, because its comparator reads exactly that field. An
// unclassified intent and a judgment intent may have one or not.
func CheckClass(intentID string, f *responseparser.Format, classOf ClassFunc) error {
	if classOf == nil {
		classOf = catalog.ClassOf
	}
	c, ok := classOf(intentID)
	if !ok {
		return nil
	}
	if (c == catalog.ClassDeterministic || c == catalog.ClassHybrid) && !f.HasScored() {
		return fmt.Errorf("intent is %s but its format has no \"scored\": true field; the comparator would have nothing to read", c)
	}
	return nil
}

// Same reports whether two formats are the same once decoded (key order and
// whitespace in the stored JSON do not matter).
func Same(a, b *responseparser.Format) bool {
	ja, errA := json.Marshal(a)
	jb, errB := json.Marshal(b)
	return errA == nil && errB == nil && bytes.Equal(ja, jb)
}

// Own returns an intent's own format, without family inheritance.
func (r *Registry) Own(intentID string) (Entry, bool) {
	if r == nil {
		return Entry{}, false
	}
	e, ok := r.own[intentID]
	return e, ok
}

// Resolve returns the format that applies to intentID: its own, else the
// longest "_"-prefix family's of the same class.
func (r *Registry) Resolve(intentID string) (Resolved, bool) {
	if r == nil || intentID == "" {
		return Resolved{}, false
	}
	if e, ok := r.own[intentID]; ok {
		return Resolved{Entry: e, ServedBy: intentID}, true
	}
	cls, clsOK := r.classOf(intentID)
	if !clsOK {
		// An unclassified child cannot be shown to share its family's class,
		// and a format carries a class-dependent rule (the scored field).
		return Resolved{}, false
	}
	key, ok := family.Resolve(intentID, func(k string) bool {
		if k == intentID {
			return false
		}
		if _, has := r.own[k]; !has {
			return false
		}
		fc, fok := r.classOf(k)
		return fok && fc == cls
	})
	if !ok {
		return Resolved{}, false
	}
	return Resolved{Entry: r.own[key], ServedBy: key}, true
}

// Intents lists every intent with its own format, sorted.
func (r *Registry) Intents() []string {
	if r == nil {
		return nil
	}
	out := make([]string, 0, len(r.own))
	for id := range r.own {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// Load reads intent_response_formats and builds the registry over the
// intent_specs formats. A database without the table yet (migration 0010 not
// re-applied) is not an error: the registry then holds the intent_specs
// formats only, so behaviour is what it was. Any other read error is
// returned WITH that same intent_specs-only registry, so the caller can log
// it and still serve every format it had.
func Load(ctx context.Context, db *sql.DB, specFormats map[string]*responseparser.Format) (*Registry, Report, error) {
	rows, err := readRows(ctx, db)
	if err != nil {
		reg, rep := Build(nil, specFormats, nil)
		if isMissingTable(err) {
			log.Warn().Err(err).
				Msg("[intent_format] intent_response_formats does not exist (apply migrations/0023) — only intent_specs.response_format formats are loaded")
			return reg, rep, nil
		}
		return reg, rep, err
	}
	reg, rep := Build(rows, specFormats, nil)
	return reg, rep, nil
}

func readRows(ctx context.Context, db *sql.DB) ([]Row, error) {
	if db == nil {
		return nil, fmt.Errorf("intent_format: no database")
	}
	rs, err := db.QueryContext(ctx, `SELECT intent_id, format, version FROM intent_response_formats ORDER BY intent_id`)
	if err != nil {
		return nil, fmt.Errorf("intent_format: reading intent_response_formats: %w", err)
	}
	defer rs.Close()
	var out []Row
	for rs.Next() {
		var row Row
		if err := rs.Scan(&row.IntentID, &row.Format, &row.Version); err != nil {
			return nil, fmt.Errorf("intent_format: scanning intent_response_formats: %w", err)
		}
		out = append(out, row)
	}
	if err := rs.Err(); err != nil {
		return nil, fmt.Errorf("intent_format: iterating intent_response_formats: %w", err)
	}
	return out, nil
}

func isMissingTable(err error) bool {
	s := err.Error()
	return strings.Contains(s, "intent_response_formats") &&
		(strings.Contains(s, "does not exist") || strings.Contains(s, "42P01"))
}

// ── the node's loaded registry ──────────────────────────────────────────────

var active struct {
	sync.RWMutex
	r *Registry
}

// Use installs the registry every lookup reads; nil uninstalls it.
func Use(r *Registry) {
	active.Lock()
	defer active.Unlock()
	active.r = r
}

// Loaded reports whether a registry is installed. While none is, callers keep
// their pre-registry behaviour (the comparator reads intent_specs formats).
func Loaded() bool {
	active.RLock()
	defer active.RUnlock()
	return active.r != nil
}

// For returns the format that applies to intentID from the installed registry.
func For(intentID string) (Resolved, bool) {
	active.RLock()
	r := active.r
	active.RUnlock()
	return r.Resolve(intentID)
}

// Installed returns the installed registry (nil when none).
func Installed() *Registry {
	active.RLock()
	defer active.RUnlock()
	return active.r
}
