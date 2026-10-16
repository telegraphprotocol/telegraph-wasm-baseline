package adapterbuild

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
)

// ObjectInfo is what the overwrite guard needs to know about an existing object.
type ObjectInfo struct {
	Size     int64
	Metadata map[string]string // user metadata; keys compared case-insensitively
	ETag     string
}

// meta returns a user-metadata value case-insensitively (S3 servers canonicalise
// header case differently: "sha256", "Sha256", "X-Amz-Meta-Sha256").
func (o ObjectInfo) meta(key string) string {
	for k, v := range o.Metadata {
		k = strings.TrimPrefix(strings.ToLower(k), "x-amz-meta-")
		if k == strings.ToLower(key) {
			return v
		}
	}
	return ""
}

// ObjectStore is the minimal S3 surface Publish uses — an interface so the
// overwrite guard is unit-tested against a fake.
type ObjectStore interface {
	// Stat returns (info, true, nil) if key exists, (_, false, nil) if it does not.
	Stat(ctx context.Context, key string) (ObjectInfo, bool, error)
	// PutFile uploads a local file to key.
	PutFile(ctx context.Context, key, path, contentType string, meta map[string]string) error
	// URL is the public URL a node fetches key from (what registerWasm records).
	URL(key string) string
	// Describe names the store for logs ("bucket@endpoint").
	Describe() string
}

// Keys returns the per-intent object keys: wasm/<INTENT>/<INTENT>.wasm and its manifest.
func Keys(intent string) (wasmKey, manifestKey string) {
	return "wasm/" + intent + "/" + ArtifactName(intent), "wasm/" + intent + "/" + ManifestName(intent)
}

// PublishResult reports what Publish did.
type PublishResult struct {
	Store          string `json:"store"`
	Key            string `json:"key"`
	ManifestKey    string `json:"manifest_key"`
	URL            string `json:"url"`
	ManifestURL    string `json:"manifest_url"`
	Unchanged      bool   `json:"unchanged"`                 // the key already held these exact bytes
	Overwrote      bool   `json:"overwrote"`                 // replaced DIFFERENT bytes (only with overwrite=true)
	PreviousSHA256 string `json:"previous_sha256,omitempty"` // the replaced/kept object's recorded sha256
	PreviousETag   string `json:"previous_etag,omitempty"`
	PreviousSize   int64  `json:"previous_size,omitempty"`
}

// ErrOverwriteRefused is returned when the per-intent key holds different bytes.
type ErrOverwriteRefused struct {
	Key, ExistingSHA256, ExistingETag, NewSHA256 string
	ExistingSize                                 int64
}

func (e *ErrOverwriteRefused) Error() string {
	old := e.ExistingSHA256
	if old == "" {
		old = "(no sha256 metadata; etag " + e.ExistingETag + ")"
	}
	return fmt.Sprintf("adapterbuild: %s already holds a DIFFERENT object (sha256 %s, %d B); new sha256 %s. "+
		"Per-intent keys are re-fetched by nodes from the on-chain URL on a cache miss, so overwriting changes the "+
		"bytes behind every older registration of this URL (they will then fail their keccak check). "+
		"Refusing; pass --overwrite to replace it deliberately", e.Key, old, e.ExistingSize, e.NewSHA256)
}

// Publish uploads a built artifact and its manifest to the per-intent keys.
//
// Overwrite guard: if the wasm key exists and does not provably hold the same
// bytes (same recorded sha256 and size), Publish refuses unless overwrite is
// true; either way the existing object's hash is logged. An object with no
// sha256 metadata cannot be proven identical and counts as different.
func Publish(ctx context.Context, store ObjectStore, a Artifact, intent string, overwrite bool, logf Logf) (PublishResult, error) {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	wasmKey, manKey := Keys(intent)
	res := PublishResult{
		Store: store.Describe(), Key: wasmKey, ManifestKey: manKey,
		URL: store.URL(wasmKey), ManifestURL: store.URL(manKey),
	}

	existing, exists, err := store.Stat(ctx, wasmKey)
	if err != nil {
		return res, fmt.Errorf("adapterbuild: checking %s: %w", wasmKey, err)
	}
	if exists {
		oldSHA := strings.ToLower(existing.meta("sha256"))
		res.PreviousSHA256, res.PreviousETag, res.PreviousSize = oldSHA, existing.ETag, existing.Size
		logf("existing object at %s: sha256=%q size=%d etag=%s", wasmKey, oldSHA, existing.Size, existing.ETag)
		same := oldSHA != "" && oldSHA == strings.ToLower(a.SHA256) && existing.Size == a.Bytes
		switch {
		case same:
			res.Unchanged = true
			logf("%s already holds these exact bytes — not re-uploading the package", wasmKey)
		case !overwrite:
			return res, &ErrOverwriteRefused{Key: wasmKey, ExistingSHA256: oldSHA, ExistingETag: existing.ETag,
				ExistingSize: existing.Size, NewSHA256: a.SHA256}
		default:
			res.Overwrote = true
			logf("WARNING: --overwrite: replacing %s (old sha256 %q) with sha256 %s", wasmKey, oldSHA, a.SHA256)
		}
	}

	meta := map[string]string{"sha256": a.SHA256, "keccak256": a.Keccak256, "intent": intent}
	if !res.Unchanged {
		logf("uploading %s (%d B) → %s", a.Path, a.Bytes, wasmKey)
		if err := store.PutFile(ctx, wasmKey, a.Path, "application/wasm", meta); err != nil {
			return res, fmt.Errorf("adapterbuild: uploading %s: %w", wasmKey, err)
		}
		got, ok, err := store.Stat(ctx, wasmKey)
		if err != nil || !ok {
			return res, fmt.Errorf("adapterbuild: %s not readable after upload (exists=%v): %v", wasmKey, ok, err)
		}
		if got.Size != a.Bytes || !strings.EqualFold(got.meta("sha256"), a.SHA256) {
			return res, fmt.Errorf("adapterbuild: %s after upload is %d B sha256=%q, uploaded %d B sha256=%s",
				wasmKey, got.Size, got.meta("sha256"), a.Bytes, a.SHA256)
		}
	}
	if err := store.PutFile(ctx, manKey, a.ManifestPath, "application/json", meta); err != nil {
		return res, fmt.Errorf("adapterbuild: uploading %s: %w", manKey, err)
	}
	logf("published %s and %s to %s", wasmKey, manKey, res.Store)
	return res, nil
}

// StoreConfig is the object-store configuration, from env.
type StoreConfig struct {
	Endpoint      string // host:port (scheme stripped)
	Secure        bool
	Bucket        string
	AccessKey     string
	SecretKey     string
	PublicBaseURL string // URL prefix keys are served under; default <scheme>://<endpoint>/<bucket>
	Region        string
}

// StoreConfigFromEnv reads OBJECT_STORE_ENDPOINT, OBJECT_STORE_BUCKET,
// OBJECT_STORE_ACCESS_KEY, OBJECT_STORE_SECRET_KEY, OBJECT_STORE_PUBLIC_BASE_URL,
// OBJECT_STORE_SECURE and OBJECT_STORE_REGION. The endpoint may carry a scheme
// (http:// → insecure, https:// → secure); OBJECT_STORE_SECURE must not
// contradict it.
func StoreConfigFromEnv() (StoreConfig, error) {
	c := StoreConfig{
		Endpoint:      strings.TrimSpace(os.Getenv("OBJECT_STORE_ENDPOINT")),
		Bucket:        strings.TrimSpace(os.Getenv("OBJECT_STORE_BUCKET")),
		AccessKey:     os.Getenv("OBJECT_STORE_ACCESS_KEY"),
		SecretKey:     os.Getenv("OBJECT_STORE_SECRET_KEY"),
		PublicBaseURL: strings.TrimRight(strings.TrimSpace(os.Getenv("OBJECT_STORE_PUBLIC_BASE_URL")), "/"),
		Region:        os.Getenv("OBJECT_STORE_REGION"),
	}
	var missing []string
	for k, v := range map[string]string{
		"OBJECT_STORE_ENDPOINT": c.Endpoint, "OBJECT_STORE_BUCKET": c.Bucket,
		"OBJECT_STORE_ACCESS_KEY": c.AccessKey, "OBJECT_STORE_SECRET_KEY": c.SecretKey,
	} {
		if v == "" {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		return c, fmt.Errorf("adapterbuild: --publish needs %s", strings.Join(sorted(missing), ", "))
	}
	schemeSecure, hasScheme := false, false
	if strings.Contains(c.Endpoint, "://") {
		u, err := url.Parse(c.Endpoint)
		if err != nil || u.Host == "" {
			return c, fmt.Errorf("adapterbuild: OBJECT_STORE_ENDPOINT %q is not a URL or host:port", c.Endpoint)
		}
		switch u.Scheme {
		case "http":
		case "https":
			schemeSecure = true
		default:
			return c, fmt.Errorf("adapterbuild: OBJECT_STORE_ENDPOINT scheme %q (want http or https)", u.Scheme)
		}
		if u.Path != "" && u.Path != "/" {
			return c, fmt.Errorf("adapterbuild: OBJECT_STORE_ENDPOINT must not carry a path (%q); put the bucket in OBJECT_STORE_BUCKET", u.Path)
		}
		hasScheme, c.Endpoint = true, u.Host
	}
	c.Secure = schemeSecure
	if s := strings.TrimSpace(os.Getenv("OBJECT_STORE_SECURE")); s != "" {
		b, err := strconv.ParseBool(s)
		if err != nil {
			return c, fmt.Errorf("adapterbuild: OBJECT_STORE_SECURE=%q is not a boolean", s)
		}
		if hasScheme && b != schemeSecure {
			return c, fmt.Errorf("adapterbuild: OBJECT_STORE_SECURE=%v contradicts the endpoint scheme", b)
		}
		c.Secure = b
	}
	if c.PublicBaseURL == "" {
		scheme := "http"
		if c.Secure {
			scheme = "https"
		}
		c.PublicBaseURL = scheme + "://" + c.Endpoint + "/" + c.Bucket
	}
	return c, nil
}

func sorted(s []string) []string { sort.Strings(s); return s }
