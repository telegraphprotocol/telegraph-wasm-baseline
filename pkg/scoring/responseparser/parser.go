package responseparser

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/engine/llm"
)

// Errors. Every one of them means "do not score this miner on this question";
// none of them is a score of 0. ErrUnavailable is the only one where the model
// never answered, and callers may fall back to the older parse paths for it.
var (
	ErrUnavailable  = errors.New("parser model unavailable")
	ErrBadReply     = errors.New("parser model reply is not {\"value\": ...}")
	ErrNotFound     = errors.New("value not found in the response")
	ErrNotVerbatim  = errors.New("copied value does not appear in the response")
	ErrConvert      = errors.New("copied value cannot be converted")
	ErrOutOfRange   = errors.New("value outside the format's range")
	ErrUnknownUnit  = errors.New("unknown unit")
	ErrNotScoreable = errors.New("reply is a list or object; only one value can be scored")
	ErrNotAllowed   = errors.New("value is not one of the field's allowed values")
	ErrWrongShape   = errors.New("reply shape does not match the field type")
)

// Env var names. Deliberately separate from OPENAI_API_URL / LLM_MODEL: the
// parser model is its own small server (vLLM or llama.cpp), not the routing
// LLM, and must never be silently swapped for it.
const (
	EnvEnabled      = "PARSER_LLM_ENABLED"
	EnvURL          = "PARSER_LLM_URL"
	EnvModel        = "PARSER_LLM_MODEL"
	EnvKey          = "PARSER_LLM_KEY"
	EnvTimeout      = "PARSER_LLM_TIMEOUT_SECONDS"
	EnvShadow       = "PARSER_LLM_SHADOW"
	EnvMaxBodyChars = "PARSER_LLM_MAX_BODY_CHARS"

	// User-request path (POST /engine/v1/ask, WS ask). Only meaningful when
	// PARSER_LLM_ENABLED=true; on by default then.
	EnvUserEnabled = "PARSER_LLM_USER_ENABLED"
	EnvUserTimeout = "PARSER_LLM_USER_TIMEOUT_SECONDS"
)

// Defaults. The model was chosen by testing (86% on 395 cases, 91% on a
// 110-intent final set, 100% valid JSON) against NuExtract-1.5, a fine-tuned
// ModernBERT span extractor and smaller Qwens; serve it with
//
//	vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768
const (
	DefaultURL          = "http://127.0.0.1:8000/v1" // vLLM's default port
	DefaultModel        = "Qwen/Qwen3-4B-Instruct-2507"
	DefaultTimeout      = 60 * time.Second
	DefaultMaxBodyChars = 60000 // ~15-20k tokens; the server allows 32768
	MaxTokens           = 96    // same as the experiments script (--max-new-tokens 96)
	// DefaultUserTimeout bounds shaping one user answer (every field's model
	// call): the user is waiting on it, and a slow parser must not hold an
	// answer that is already in hand. Fields not read in time are
	// "unavailable"; with none read the reply carries the raw answer.
	DefaultUserTimeout = 10 * time.Second
	cacheTTL           = 2 * time.Hour
	cacheErrTTL        = 5 * time.Minute
	cacheMaxEntries    = 20000
)

// Config configures the parser model client.
type Config struct {
	URL          string
	Model        string
	Key          string
	Timeout      time.Duration
	Shadow       bool
	MaxBodyChars int

	// UserDisabled turns off only the user-request path (ParseAll with User); the
	// zero value leaves it on, so enabling the parser enables both.
	UserDisabled bool
	// UserTimeout bounds one user answer's ParseAll (all its fields); 0 =
	// DefaultUserTimeout.
	UserTimeout time.Duration
}

// ConfigFromEnv reads the PARSER_LLM_* variables. enabled is false unless
// PARSER_LLM_ENABLED is a true value; with it false the model path is off and
// scoring behaves exactly as before.
func ConfigFromEnv() (cfg Config, enabled bool, err error) {
	if v := strings.TrimSpace(os.Getenv(EnvEnabled)); v != "" {
		enabled, err = strconv.ParseBool(v)
		if err != nil {
			return Config{}, false, fmt.Errorf("%s=%q is not a boolean", EnvEnabled, v)
		}
	}
	cfg = Config{
		URL:          envOr(EnvURL, DefaultURL),
		Model:        envOr(EnvModel, DefaultModel),
		Key:          os.Getenv(EnvKey),
		Timeout:      DefaultTimeout,
		MaxBodyChars: DefaultMaxBodyChars,
	}
	if v := strings.TrimSpace(os.Getenv(EnvTimeout)); v != "" {
		n, perr := strconv.Atoi(v)
		if perr != nil || n <= 0 {
			return Config{}, false, fmt.Errorf("%s=%q is not a positive number of seconds", EnvTimeout, v)
		}
		cfg.Timeout = time.Duration(n) * time.Second
	}
	if v := strings.TrimSpace(os.Getenv(EnvMaxBodyChars)); v != "" {
		n, perr := strconv.Atoi(v)
		if perr != nil || n <= 0 {
			return Config{}, false, fmt.Errorf("%s=%q is not a positive number", EnvMaxBodyChars, v)
		}
		cfg.MaxBodyChars = n
	}
	if v := strings.TrimSpace(os.Getenv(EnvShadow)); v != "" {
		cfg.Shadow, err = strconv.ParseBool(v)
		if err != nil {
			return Config{}, false, fmt.Errorf("%s=%q is not a boolean", EnvShadow, v)
		}
	}
	cfg.UserTimeout = DefaultUserTimeout
	if v := strings.TrimSpace(os.Getenv(EnvUserTimeout)); v != "" {
		n, perr := strconv.Atoi(v)
		if perr != nil || n <= 0 {
			return Config{}, false, fmt.Errorf("%s=%q is not a positive number of seconds", EnvUserTimeout, v)
		}
		cfg.UserTimeout = time.Duration(n) * time.Second
	}
	if v := strings.TrimSpace(os.Getenv(EnvUserEnabled)); v != "" {
		on, perr := strconv.ParseBool(v)
		if perr != nil {
			return Config{}, false, fmt.Errorf("%s=%q is not a boolean", EnvUserEnabled, v)
		}
		cfg.UserDisabled = !on
	}
	return cfg, enabled, nil
}

func envOr(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

// Completer is the one call the parser makes: llm.Client.Complete.
type Completer interface {
	Complete(ctx context.Context, system, user string, opts *llm.Options) (string, error)
}

// Parser calls the parser model and turns its reply into a scored value.
type Parser struct {
	llm   Completer
	cfg   Config
	cache *replyCache
}

// New builds a parser on its own OpenAI-compatible client.
func New(cfg Config) *Parser {
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	if cfg.MaxBodyChars <= 0 {
		cfg.MaxBodyChars = DefaultMaxBodyChars
	}
	return &Parser{llm: llm.New(cfg.URL, cfg.Key, cfg.Model, cfg.Timeout), cfg: cfg,
		cache: newReplyCache(cacheMaxEntries, cacheTTL)}
}

// Shadow reports whether the parser should only be logged next to the older
// parse paths rather than decide the scored value.
func (p *Parser) Shadow() bool { return p != nil && p.cfg.Shadow }

// UserEnabled reports whether the user-request path (the routed ask reply) is on.
func (p *Parser) UserEnabled() bool { return p != nil && !p.cfg.UserDisabled }

// UserTimeout is the bound on shaping one user answer (all its fields).
func (p *Parser) UserTimeout() time.Duration {
	if p == nil || p.cfg.UserTimeout <= 0 {
		return DefaultUserTimeout
	}
	return p.cfg.UserTimeout
}

// Request is one value to read from one miner response.
type Request struct {
	IntentID string
	Key      string
	Body     string
	Format   *Format
	// KeyParams fills {<param>} placeholders in the format (the key's
	// truth_params.key_params entry). May be nil.
	KeyParams map[string]string
	// Unit is the unit the miner declares it quotes in; "" = the field's own.
	Unit string
	// Slug is for logs only; it does not change the parse.
	Slug string
}

// Result is a parsed value with its audit trail.
type Result struct {
	Value  float64
	Copied string // the text the model copied, verbatim from the response
	Reply  string // the model's raw reply, for auditing disagreements
	Field  string
	Cached bool
}

// Parse reads the scored value out of one response.
//
// One model call per (intent, key, response): the result is cached by
// sha256 of the body, so the audit pass that re-parses scored responses gets
// the SAME reply rather than a second, possibly different, call.
func (p *Parser) Parse(ctx context.Context, req Request) (Result, error) {
	if req.Format == nil {
		return Result{}, fmt.Errorf("intent %s has no response format", req.IntentID)
	}
	if !req.Format.HasScored() {
		// A judgment format has no comparator field. Nothing to score is a
		// configuration gap, not an answer: unavailable, so the caller keeps
		// its older paths.
		return Result{}, fmt.Errorf("%w: intent %s response format has no scored field", ErrUnavailable, req.IntentID)
	}
	params := req.Format.ParamsFor(req.Key, req.KeyParams)
	fl := req.Format.Scored().Fill(req.Key, params)
	unit := FillText(req.Unit, req.Key, params)
	res := Result{Field: fl.Name}

	// A placeholder the key could not fill ("{year}" for a key with no such
	// key_param) would send the model a description that does not say which
	// value is wanted. The format cannot be used for this key, which is a
	// configuration gap rather than an answer, so it reads as unavailable and
	// the caller falls back to the older paths.
	if m := placeholderRE.FindString(fl.Description + " " + fl.Unit); m != "" {
		err := fmt.Errorf("%w: placeholder %s is not filled for key %q (add it to truth_params.key_params)",
			ErrUnavailable, m, req.Key)
		p.log(req, res, err)
		return res, err
	}

	pf := fl.PromptField(unit)
	// The prompt shows the miner's own unit, so the unit is part of the key:
	// two miners with an identical body but different units get different prompts.
	k := cacheKey(req.IntentID, req.Key, pf.Name+"|unit="+pf.Unit, req.Body)
	reply, cached, err := p.reply(ctx, k, pf, req.Body)
	res.Reply, res.Cached = reply, cached
	if err != nil {
		p.log(req, res, err)
		return res, err
	}

	copied, ok := ParseReply(reply)
	switch {
	case !ok:
		err = ErrBadReply
	case copied.Kind == CopiedNone:
		err = ErrNotFound
	case copied.Kind != CopiedText:
		err = ErrNotScoreable
	}
	if err != nil {
		p.log(req, res, err)
		return res, err
	}
	res.Copied = copied.Text

	if !Verbatim(copied.Text, req.Body) {
		err = fmt.Errorf("%w: %q", ErrNotVerbatim, copied.Text)
		p.log(req, res, err)
		return res, err
	}

	// The miner's declared unit may use the same placeholders as the format
	// ("USD per {key}"), so it is filled the same way before matching.
	res.Value, err = Convert(fl, copied.Text, unit)
	p.log(req, res, err)
	return res, err
}

// reply returns the model's reply for the prompt (fl, body), from the cache
// under k or from a call. k must identify the prompt: every input that changes
// UserMessage(fl, body) must be part of it.
func (p *Parser) reply(ctx context.Context, k string, fl Field, body string) (string, bool, error) {
	if e, ok := p.cache.get(k, time.Now()); ok {
		return e.reply, true, e.err
	}
	if len(body) > p.cfg.MaxBodyChars {
		err := fmt.Errorf("%w: response is %d chars, over %s=%d",
			ErrUnavailable, len(body), EnvMaxBodyChars, p.cfg.MaxBodyChars)
		p.cache.put(k, cacheEntry{err: err}, time.Now())
		return "", false, err
	}
	content, err := p.llm.Complete(ctx, SystemPrompt, UserMessage(fl, body),
		&llm.Options{MaxTokens: MaxTokens, Temperature: 0})
	if err != nil {
		err = fmt.Errorf("%w: %v", ErrUnavailable, err)
		// A context cancelled by the caller says nothing about the model:
		// do not let it decide the next attempt.
		if ctx.Err() == nil {
			p.cache.put(k, cacheEntry{err: err}, time.Now())
		}
		return "", false, err
	}
	content = StripThinking(content)
	p.cache.put(k, cacheEntry{reply: content}, time.Now())
	return content, false, nil
}

// log records every parse with the model's raw reply next to the result, so a
// value two validators disagree on can be traced to the reply that caused it.
func (p *Parser) log(req Request, res Result, err error) {
	ev := log.Info()
	if err != nil && !errors.Is(err, ErrNotFound) {
		ev = log.Warn()
	}
	ev = ev.Str("intent", req.IntentID).Str("key", req.Key).Str("miner", req.Slug).
		Str("field", res.Field).Str("model", p.cfg.Model).
		Str("reply", truncate(res.Reply, 400)).Str("copied", truncate(res.Copied, 200)).
		Bool("cached", res.Cached)
	if req.Unit != "" {
		ev = ev.Str("declared_unit", req.Unit)
	}
	if err != nil {
		ev.Err(err).Msg("[response_parser] value not scored")
		return
	}
	ev.Float64("value", res.Value).Msg("[response_parser] parsed")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// placeholderRE matches an unfilled {name} placeholder.
var placeholderRE = regexp.MustCompile(`\{[A-Za-z_][A-Za-z0-9_]*\}`)

// IsUnavailable reports whether err means the model never answered (server
// down, timeout, body too large) as opposed to answering unusably.
func IsUnavailable(err error) bool { return errors.Is(err, ErrUnavailable) }

// ── cache ────────────────────────────────────────────────────────────────────

type cacheEntry struct {
	reply string
	err   error
	at    time.Time
}

// replyCache holds model replies by (intent, key, field, sha256(body)). It is
// bounded and expiring: entries are only needed between the scoring pass and
// the audit pass that follows it, and a response seen again much later should
// be asked again rather than answered from a stale entry forever.
type replyCache struct {
	mu      sync.Mutex
	max     int
	ttl     time.Duration
	entries map[string]cacheEntry
	order   []string
}

func newReplyCache(max int, ttl time.Duration) *replyCache {
	return &replyCache{max: max, ttl: ttl, entries: make(map[string]cacheEntry)}
}

func cacheKey(intentID, key, field, body string) string {
	sum := sha256.Sum256([]byte(body))
	return intentID + "\x00" + key + "\x00" + field + "\x00" + hex.EncodeToString(sum[:])
}

func (c *replyCache) get(k string, now time.Time) (cacheEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[k]
	if !ok {
		return cacheEntry{}, false
	}
	// A failed call is remembered only briefly: long enough that the audit
	// pass straight after scoring sees the same outcome, short enough that a
	// server that comes back is used again.
	ttl := c.ttl
	if e.err != nil {
		ttl = cacheErrTTL
	}
	if now.Sub(e.at) > ttl {
		return cacheEntry{}, false
	}
	return e, true
}

func (c *replyCache) put(k string, e cacheEntry, now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e.at = now
	if _, exists := c.entries[k]; !exists {
		c.order = append(c.order, k)
	}
	c.entries[k] = e
	for len(c.entries) > c.max && len(c.order) > 0 {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.entries, oldest)
	}
}
