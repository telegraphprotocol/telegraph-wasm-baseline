package runtime

import "testing"

func TestAdapterMetaCheckFloatABI(t *testing.T) {
	valid := AdapterMeta{ABIVersion: FloatAdapterABIVersion, Runtime: AdapterRuntimeONNXF32, Class: "judgment"}
	if err := valid.Check("judgment"); err != nil {
		t.Fatalf("float metadata rejected: %v", err)
	}

	invalid := valid
	invalid.ABIVersion = AdapterABIVersion
	if err := invalid.Check("judgment"); err == nil {
		t.Fatal("float metadata with ABI v1 was accepted")
	}
}
