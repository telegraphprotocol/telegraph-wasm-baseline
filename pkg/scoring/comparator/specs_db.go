package comparator

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/scoring/responseparser"
)

// LoadSpecs reads every active intent spec from the database and returns the
// configs the scorer needs (question template and keys, tolerance, scale,
// response format, fusion).
//
// It does NOT build truth sources: deterministic ground truth comes from
// eval-pool truth-source material (scoring.MaterialTruthSource). The
// truth_source column is no longer read by scoring and a row is not refused
// for an unknown or empty one; truth_params is read only for its key_params,
// which fill {<param>} placeholders in the response format.
//
// This replaces the hand-maintained intentConfigs map. The map was fine for two
// intents and became the bottleneck at nine: every new intent, every retuned
// tolerance and every pinned question key was a code change, a rebuild and a
// redeploy of every validator. Tolerance in particular is protocol data — two
// validators on different builds would score the same miner differently and
// disagree on the epoch, silently.
//
// Rows are DATA, not code: an unparseable key set, scale, fusion block or
// response format is rejected loudly here rather than producing an intent
// that loads and then scores every miner identically.
func LoadSpecs(ctx context.Context, db *sql.DB) (map[string]IntentConfig, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT intent_id, comparator_code, tolerance_bps, scale,
		       truth_source, truth_params, question_keys, question_template, label,
		       COALESCE(fusion, 'null'::jsonb),
		       COALESCE(response_format, 'null'::jsonb)
		FROM intent_specs
		WHERE active
		ORDER BY intent_id`)
	if err != nil && strings.Contains(err.Error(), "response_format") {
		// A database that predates the column (migration 0009 not yet applied)
		// must keep scoring exactly as before rather than lose the whole
		// deterministic path over a column only the parser model reads.
		rows, err = db.QueryContext(ctx, `
			SELECT intent_id, comparator_code, tolerance_bps, scale,
			       truth_source, truth_params, question_keys, question_template, label,
			       COALESCE(fusion, 'null'::jsonb), 'null'::jsonb
			FROM intent_specs
			WHERE active
			ORDER BY intent_id`)
	}
	if err != nil {
		return nil, fmt.Errorf("comparator: reading intent_specs: %w", err)
	}
	defer rows.Close()

	configs := make(map[string]IntentConfig)

	for rows.Next() {
		var (
			intentID, truthSource, questionTmpl, label string
			code                                       int64
			tolerance                                  int64
			scale                                      float64
			paramsJSON, keysJSON, fusionJSON           []byte
			formatJSON                                 []byte
		)
		if err := rows.Scan(&intentID, &code, &tolerance, &scale,
			&truthSource, &paramsJSON, &keysJSON, &questionTmpl, &label, &fusionJSON, &formatJSON); err != nil {
			return nil, fmt.Errorf("comparator: scanning intent_specs: %w", err)
		}
		_ = truthSource // no longer read by scoring: truth comes from eval-pool material

		cfg, err := parseSpecRow(intentID, code, tolerance, scale, paramsJSON, keysJSON, questionTmpl, label, fusionJSON, formatJSON)
		if err != nil {
			// One malformed row must not take every other intent down with it:
			// skip it, loudly, and keep loading. The skipped intent is simply not
			// on the comparator/hybrid path until its row is fixed.
			log.Error().Err(err).Str("intent", intentID).
				Msg("[comparator] intent_specs row is malformed — this intent is SKIPPED (not scored on the comparator/hybrid path); fix the row")
			continue
		}
		configs[intentID] = cfg
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("comparator: iterating intent_specs: %w", err)
	}
	return configs, nil
}

// parseSpecRow validates one intent_specs row and builds its config. An error
// means the row is unusable as written. The row's truth_source is not an
// input: it is no longer used by scoring, so an unknown or empty one is not a
// reason to refuse the row.
func parseSpecRow(intentID string, code, tolerance int64, scale float64,
	paramsJSON, keysJSON []byte, questionTmpl, label string, fusionJSON, formatJSON []byte) (IntentConfig, error) {
	var keys []string
	if err := json.Unmarshal(keysJSON, &keys); err != nil {
		return IntentConfig{}, fmt.Errorf("comparator: %s question_keys is not a JSON array: %w", intentID, err)
	}
	if len(keys) == 0 {
		// An intent with no questions cannot be scored. Silently keeping it
		// would produce an epoch where it is simply absent, which looks the
		// same as every miner failing.
		return IntentConfig{}, fmt.Errorf("comparator: %s has no question keys", intentID)
	}
	if scale == 0 {
		return IntentConfig{}, fmt.Errorf("comparator: %s has scale 0; every miner value would collapse to zero", intentID)
	}

	var params map[string]any
	if len(paramsJSON) > 0 {
		if err := json.Unmarshal(paramsJSON, &params); err != nil {
			return IntentConfig{}, fmt.Errorf("comparator: %s truth_params is not a JSON object: %w", intentID, err)
		}
	}

	// A fusion block marks the intent HYBRID: its score comes from a
	// model-bearing WASM that fuses the comparator verdict with a
	// judgment about the response text. Absent means deterministic.
	fusion, err := parseFusion(intentID, fusionJSON)
	if err != nil {
		return IntentConfig{}, err
	}

	// The response format is the intent's own description of the value
	// a miner's answer must yield (read by the parser model). A malformed
	// one is rejected here, loudly, like a malformed fusion block: it would
	// otherwise load and then fail every miner at scoring time.
	format, err := responseparser.ParseFormat(formatJSON)
	if err != nil {
		return IntentConfig{}, fmt.Errorf("comparator: %s: %w", intentID, err)
	}

	tmpl := questionTmpl
	return IntentConfig{
		ResponseFormat: format,
		KeyParams:      keyParams(params),
		Fusion:         fusion,
		Spec:           Spec{Code: code, Tolerance: tolerance},
		Scale:          scale,
		Label:          label,
		Keys:           keys,
		Question:       func(key string) string { return strings.ReplaceAll(tmpl, "{key}", key) },
		Describe:       func(key string) string { return key },
	}, nil
}

// Fusion is the extra configuration a HYBRID intent needs.
//
// A hybrid score is not a comparison, it is a fusion: the comparator's verdict
// on a checkable fact is combined with an embedded model's judgment of the
// response text. These are the thresholds that decide how.
//
// The gate matters more than the weights. Under it a miner that gets the FACT
// wrong cannot be rescued by fluent prose, which is the whole reason the fact
// check is not simply averaged with the judgment.
type Fusion struct {
	// Mode selects the fusion rule. 0 = gate, the only one built.
	Mode int64 `json:"mode"`

	// Pass is the comparator score at or above which the fact counts as
	// answered correctly.
	Pass int64 `json:"pass"`

	Low int64 `json:"low"`

	// Floor caps a miner that FAILS the fact check.
	Floor int64 `json:"floor"`

	Span int64 `json:"span"`
}

// Buffer lays a hybrid spec out in the ten-int order imw_score_hybrid reads.
//
// Positional, and the WASM reads exactly ten i64s with no length argument: a
// short buffer is read past its end and a reordered one fuses against the
// wrong thresholds. Neither fails loudly, so the layout is pinned by a test.
func (f Fusion) Buffer(spec Spec) [10]int64 {
	return [10]int64{
		spec.Code, spec.Tolerance, f.Mode,
		f.Pass, f.Low, f.Floor, f.Span,
		0, 0, 0, // w_c, w_j, reserved — unused under a gate fusion
	}
}

// parseFusion decodes the fusion column, returning nil for a deterministic
// intent.
func parseFusion(intentID string, raw []byte) (*Fusion, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var f Fusion
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("comparator: %s fusion is not a JSON object: %w", intentID, err)
	}
	// A zero Pass would let a miner that failed the fact check through the
	// gate, which is the one thing the gate exists to prevent.
	if f.Pass <= 0 {
		return nil, fmt.Errorf("comparator: %s declares fusion with pass=%d; "+
			"a miner failing the fact check would pass the gate", intentID, f.Pass)
	}
	if f.Floor >= f.Pass {
		return nil, fmt.Errorf("comparator: %s declares floor=%d at or above pass=%d; "+
			"failing the fact check would score no worse than passing it", intentID, f.Floor, f.Pass)
	}
	return &f, nil
}

// UseSpecs installs database-loaded configs as the live registry, replacing
// the compiled-in intentConfigs map.
//
// Called at startup, after LoadSpecs, and again at every epoch start (the
// intent catalog reload), atomically: asks read the registry concurrently.
// Separate from LoadSpecs so that
// reading the table and committing to it are distinct: a node can load, check
// what it got, and decline — an empty or partial table should leave the
// compiled-in map in place rather than silently disabling every intent.
//
// Replacing rather than merging is deliberate. A merge would leave an intent
// that was REMOVED from the table still being scored from the binary, which is
// the opposite of what removing it meant.
func UseSpecs(specs map[string]IntentConfig) {
	if len(specs) == 0 {
		// Refused rather than applied: an empty table would otherwise turn
		// every wired intent off at once, which looks like every intent
		// failing rather than like a configuration mistake.
		return
	}
	liveConfigs.Store(&specs)
}

// keyParams returns truth_params.key_params as strings by question key, for
// filling {<param>} placeholders in a response format. Non-string values are
// rendered with %v; anything that is not an object is ignored.
func keyParams(params map[string]any) map[string]map[string]string {
	raw, ok := params["key_params"].(map[string]any)
	if !ok || len(raw) == 0 {
		return nil
	}
	out := make(map[string]map[string]string, len(raw))
	for key, v := range raw {
		m, ok := v.(map[string]any)
		if !ok {
			continue
		}
		kp := make(map[string]string, len(m))
		for name, val := range m {
			if s, ok := val.(string); ok {
				kp[name] = s
			} else {
				kp[name] = fmt.Sprintf("%v", val)
			}
		}
		out[key] = kp
	}
	return out
}
