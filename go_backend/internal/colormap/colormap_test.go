package colormap

import (
	"math"
	"testing"
)

func TestPalettes(t *testing.T) {
	p, err := Get("")
	if err != nil || p.Name != "ironbow" {
		t.Fatalf("default palette: %v %v", p, err)
	}
	lo, hi := p.At(0), p.At(1)
	if lo.R > 10 || hi.R != 255 || hi.G != 255 {
		t.Fatalf("ironbow endpoints %v %v", lo, hi)
	}
	if p.At(math.NaN()) != lo || p.At(-3) != lo || p.At(9) != hi {
		t.Fatal("clamping failed")
	}
	w, _ := Get("whitehot")
	if c := w.At(0.5); c.R < 126 || c.R > 129 || c.R != c.G || c.G != c.B {
		t.Fatalf("whitehot midpoint %v", c)
	}
	if _, err := Get("sepia"); err == nil {
		t.Fatal("unknown palette accepted")
	}
	if len(Names()) != 4 {
		t.Fatalf("names %v", Names())
	}
}
