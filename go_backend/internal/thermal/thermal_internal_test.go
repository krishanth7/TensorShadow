package thermal

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestSelectKthMatchesSort(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for trial := 0; trial < 500; trial++ {
		n := 1 + rng.Intn(200)
		a := make([]float64, n)
		for i := range a {
			a[i] = math.Round(rng.NormFloat64()*10) / 2 // many duplicates
		}
		sorted := append([]float64(nil), a...)
		sort.Float64s(sorted)
		k := rng.Intn(n)
		if got := selectKth(a, k); got != sorted[k] {
			t.Fatalf("n=%d k=%d: got %v want %v", n, k, got, sorted[k])
		}
	}
}

func TestCorrectorMatchesReferenceFormula(t *testing.T) {
	for _, app := range []float64{-20, 0, 22, 33.3, 37, 120} {
		for _, eps := range []float64{0.5, 0.95, 0.98, 1} {
			ta, tr := app+kelvinOffset, 21.5+kelvinOffset
			want := math.Pow((math.Pow(ta, 4)-(1-eps)*math.Pow(tr, 4))/eps, 0.25) - kelvinOffset
			if got := CorrectEmissivity(app, eps, 21.5); math.Abs(got-want) > 1e-9 {
				t.Fatalf("app=%v eps=%v: got %v want %v", app, eps, got, want)
			}
		}
	}
}
