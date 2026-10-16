package adapterbuild

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/telegraphprotocol/telegraph-wasm-baseline/pkg/wasm/wideembed"
)

// HostsRelDir is where the pinned embed hosts and their lock live, relative to
// the repository root. build-hosts.sh writes both; they are committed together.
const HostsRelDir = "wide_runtime/wide_wasm/hosts"

// LockFileName is the lock file inside the hosts directory.
const LockFileName = "hosts.lock.json"

// HostEntry pins one embed host.
type HostEntry struct {
	File     string `json:"file"`
	Features string `json:"features"`
	SHA256   string `json:"sha256"`
	Size     int64  `json:"size"`
}

// Lock is hosts.lock.json: which embed-host bytes are official, and how they
// were built. The packager refuses any host whose sha256 is not recorded here.
type Lock struct {
	Schema          int       `json:"schema"`
	Builder         string    `json:"builder"`
	Toolchain       string    `json:"toolchain"`
	Image           string    `json:"image"`
	ImageDigest     string    `json:"image_digest"`
	RustFlags       string    `json:"rustflags"`
	CargoLockSHA256 string    `json:"cargo_lock_sha256"`
	SrcSHA256       string    `json:"src_sha256"`
	Judgment        HostEntry `json:"judgment"`
	Hybrid          HostEntry `json:"hybrid"`
}

// Entry returns the pinned host for a class.
func (l Lock) Entry(class wideembed.Class) (HostEntry, error) {
	switch class {
	case wideembed.ClassJudgment:
		return l.Judgment, nil
	case wideembed.ClassHybrid:
		return l.Hybrid, nil
	}
	return HostEntry{}, fmt.Errorf("adapterbuild: no pinned host for class %q", class)
}

// Pins reports whether a host sha256 (hex) is any pinned host, and for which class.
func (l Lock) Pins(sha string) (wideembed.Class, bool) {
	sha = strings.ToLower(sha)
	switch sha {
	case strings.ToLower(l.Judgment.SHA256):
		return wideembed.ClassJudgment, true
	case strings.ToLower(l.Hybrid.SHA256):
		return wideembed.ClassHybrid, true
	}
	return "", false
}

// Hosts is a loaded hosts directory: the lock plus where it was read from.
type Hosts struct {
	Dir        string
	Lock       Lock
	LockSHA256 string // sha256 of hosts.lock.json as read (recorded in every manifest)
}

// LoadHosts reads and validates dir/hosts.lock.json.
func LoadHosts(dir string) (*Hosts, error) {
	raw, err := os.ReadFile(filepath.Join(dir, LockFileName))
	if err != nil {
		return nil, fmt.Errorf("adapterbuild: reading host lock: %w", err)
	}
	var l Lock
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&l); err != nil {
		return nil, fmt.Errorf("adapterbuild: %s is not a valid host lock: %w", filepath.Join(dir, LockFileName), err)
	}
	if l.Schema != 1 {
		return nil, fmt.Errorf("adapterbuild: host lock schema %d, this builder understands 1", l.Schema)
	}
	for _, e := range []struct {
		class wideembed.Class
		h     HostEntry
	}{{wideembed.ClassJudgment, l.Judgment}, {wideembed.ClassHybrid, l.Hybrid}} {
		if len(e.h.SHA256) != 64 || e.h.File == "" || e.h.Size <= 0 {
			return nil, fmt.Errorf("adapterbuild: host lock entry for %s is incomplete (file=%q sha256=%q size=%d)",
				e.class, e.h.File, e.h.SHA256, e.h.Size)
		}
		if strings.ContainsAny(e.h.File, `/\`) {
			return nil, fmt.Errorf("adapterbuild: host lock entry for %s names a path, not a file in the hosts dir: %q", e.class, e.h.File)
		}
	}
	sum := sha256.Sum256(raw)
	return &Hosts{Dir: dir, Lock: l, LockSHA256: hex.EncodeToString(sum[:])}, nil
}

// Host returns the pinned host bytes for a class after verifying them against
// the lock. A host whose bytes do not match is refused — the package hash would
// not be reproducible from the committed host.
func (h *Hosts) Host(class wideembed.Class) ([]byte, HostEntry, error) {
	e, err := h.Lock.Entry(class)
	if err != nil {
		return nil, e, err
	}
	p := filepath.Join(h.Dir, e.File)
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, e, fmt.Errorf("adapterbuild: reading pinned %s host: %w", class, err)
	}
	if err := VerifyPinned(b, e); err != nil {
		return nil, e, fmt.Errorf("adapterbuild: %s: %w", p, err)
	}
	return b, e, nil
}

// VerifyPinned checks host bytes against a lock entry.
func VerifyPinned(host []byte, e HostEntry) error {
	sum := sha256.Sum256(host)
	got := hex.EncodeToString(sum[:])
	if !strings.EqualFold(got, e.SHA256) {
		return fmt.Errorf("host sha256 %s is not the pinned %s (hosts.lock.json) — refusing to package into an "+
			"unpinned host; rebuild with build-hosts.sh or restore the committed host", got, e.SHA256)
	}
	if int64(len(host)) != e.Size {
		return fmt.Errorf("host is %d bytes, lock says %d", len(host), e.Size)
	}
	return nil
}

// FindHostsDir resolves the hosts directory: explicit, then $WIDE_HOSTS_DIR,
// then $TELEGRAPH_REPO/<HostsRelDir>, then searching upward from the working
// directory and from the executable's directory for <HostsRelDir>/hosts.lock.json.
func FindHostsDir(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	if d := os.Getenv("WIDE_HOSTS_DIR"); d != "" {
		return d, nil
	}
	if r := os.Getenv("TELEGRAPH_REPO"); r != "" {
		return filepath.Join(r, filepath.FromSlash(HostsRelDir)), nil
	}
	var starts []string
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	for _, s := range starts {
		for d := s; ; d = filepath.Dir(d) {
			cand := filepath.Join(d, filepath.FromSlash(HostsRelDir))
			if _, err := os.Stat(filepath.Join(cand, LockFileName)); err == nil {
				return cand, nil
			}
			if filepath.Dir(d) == d {
				break
			}
		}
	}
	return "", errors.New("adapterbuild: cannot find the pinned hosts directory (" + HostsRelDir +
		"); pass --hosts-dir or set WIDE_HOSTS_DIR / TELEGRAPH_REPO")
}
