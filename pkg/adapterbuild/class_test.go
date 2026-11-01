package adapterbuild

import (
	"strings"
	"testing"
)

func TestResolveClass(t *testing.T) {
	ok := []struct{ intent, explicit, want string }{
		{"VULNERABILITY_TRIAGE", "", "hybrid"},
		{"VULNERABILITY_TRIAGE", "hybrid", "hybrid"},
		{"EMAIL_SECURITY", "", "hybrid"},
		{"CODE_REVIEW", "", "judgment"},
		{"CODE_REVIEW", "judgment", "judgment"},
		{"CODE_REVIEW_GENERAL", "", "judgment"},
		{"EMAIL_SECURITY_GENERAL", "", "hybrid"},
		{"GENERIC", "", "judgment"},
		{"GENERIC", "judgment", "judgment"},
	}
	for _, c := range ok {
		got, err := ResolveClass(c.intent, c.explicit)
		if err != nil || got != c.want {
			t.Errorf("ResolveClass(%s,%q) = %q,%v want %s", c.intent, c.explicit, got, err, c.want)
		}
	}
	bad := []struct{ intent, explicit, wantErr string }{
		{"VULNERABILITY_TRIAGE", "judgment", "disagrees"},
		{"CODE_REVIEW", "hybrid", "disagrees"},
		{"GENERIC", "hybrid", "disagrees"},
		{"FX_NOW", "", "deterministic"},
		{"ROUTE_ETA", "hybrid", "deterministic"},
		{"NOT_AN_INTENT", "", "not a judgment or hybrid intent"},
		{"", "", "intent is required"},
	}
	for _, c := range bad {
		if _, err := ResolveClass(c.intent, c.explicit); err == nil || !strings.Contains(err.Error(), c.wantErr) {
			t.Errorf("ResolveClass(%s,%q) err=%v, want containing %q", c.intent, c.explicit, err, c.wantErr)
		}
	}
}
