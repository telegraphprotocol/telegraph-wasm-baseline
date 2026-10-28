package main

import "testing"

func TestIsFinite(t *testing.T) {
	for _, bits := range []uint32{0, 0x3f800000, 0xbf800000} {
		if !isFinite(bits) {
			t.Fatalf("%#x should be finite", bits)
		}
	}
	for _, bits := range []uint32{0x7f800000, 0xff800000, 0x7fc00000} {
		if isFinite(bits) {
			t.Fatalf("%#x should not be finite", bits)
		}
	}
}
