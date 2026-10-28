package catalog

import "testing"

func TestCompiledClassOfGeneralAlias(t *testing.T) {
	for _, tc := range []struct {
		intent string
		want   Class
	}{
		{"CODE_REVIEW_GENERAL", ClassJudgment},
		{"EMAIL_SECURITY_GENERAL", ClassHybrid},
	} {
		got, ok := CompiledClassOf(tc.intent)
		if !ok || got != tc.want {
			t.Fatalf("CompiledClassOf(%q) = %q, %v; want %q, true", tc.intent, got, ok, tc.want)
		}
	}
}
