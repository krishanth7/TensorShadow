package coreg

import (
	"math"
	"math/rand"
	"testing"
)

// truth is a realistic visible (1280×960) → thermal (160×120) mapping: 1/8
// scale, a 1.5° roll between the sensors, an offset and slight keystone.
var truth = func() Homography {
	a := 1.5 * math.Pi / 180
	s := 0.125
	return Homography{s * math.Cos(a), -s * math.Sin(a), 4, s * math.Sin(a), s * math.Cos(a), -3, 2e-6, -1e-6, 1}
}()

func grid(noise float64, seed int64) []Correspondence {
	rng := rand.New(rand.NewSource(seed))
	var pts []Correspondence
	for _, x := range []float64{160, 640, 1120} {
		for _, y := range []float64{120, 480, 840} {
			v := Point{x, y}
			t, _ := truth.Apply(v)
			t.X += rng.NormFloat64() * noise
			t.Y += rng.NormFloat64() * noise
			pts = append(pts, Correspondence{Visible: v, Thermal: t})
		}
	}
	return pts
}

func maxMappingError(h Homography) float64 {
	var worst float64
	for x := 0.0; x <= 1280; x += 64 {
		for y := 0.0; y <= 960; y += 64 {
			a, _ := h.Apply(Point{x, y})
			b, _ := truth.Apply(Point{x, y})
			worst = math.Max(worst, math.Hypot(a.X-b.X, a.Y-b.Y))
		}
	}
	return worst
}

func TestEstimateRecoversExactHomography(t *testing.T) {
	h, err := Estimate(grid(0, 1), ModelHomography)
	if err != nil {
		t.Fatal(err)
	}
	if e := maxMappingError(h); e > 1e-6 {
		t.Fatalf("noise-free estimate off by %v px", e)
	}
}

func TestEstimateWithNoise(t *testing.T) {
	rig, err := NewRig(RigConfig{ID: "rig", Visible: Size{1280, 960}, Thermal: Size{160, 120}, Points: grid(0.25, 2)})
	if err != nil {
		t.Fatal(err)
	}
	if *rig.RMSE > 0.4 || rig.Quality != QualityGood {
		t.Fatalf("rmse %v quality %s", *rig.RMSE, rig.Quality)
	}
	if e := maxMappingError(rig.Homography); e > 0.6 {
		t.Fatalf("max mapping error %v px with 0.25 px calibration noise", e)
	}
}

func TestAffineModel(t *testing.T) {
	aff := Homography{0.125, 0.01, 3, -0.01, 0.125, 2, 0, 0, 1}
	var pts []Correspondence
	for _, v := range []Point{{0, 0}, {1280, 0}, {0, 960}} {
		p, _ := aff.Apply(v)
		pts = append(pts, Correspondence{Visible: v, Thermal: p})
	}
	h, err := Estimate(pts, ModelAffine)
	if err != nil {
		t.Fatal(err)
	}
	for i := range h {
		if math.Abs(h[i]-aff[i]) > 1e-9 {
			t.Fatalf("affine estimate %v, want %v", h, aff)
		}
	}
	if _, err := Estimate(pts, ModelHomography); err == nil {
		t.Fatal("homography from 3 points accepted")
	}
}

func TestDegenerateAndInvalid(t *testing.T) {
	collinear := []Correspondence{
		{Point{0, 0}, Point{0, 0}}, {Point{1, 1}, Point{1, 1}}, {Point{2, 2}, Point{2, 2}}, {Point{3, 3}, Point{3, 3}},
	}
	if _, err := Estimate(collinear, ModelHomography); err == nil {
		t.Fatal("collinear points accepted")
	}
	bad := []RigConfig{
		{ID: "bad id", Visible: Size{1, 1}, Thermal: Size{1, 1}, Points: grid(0, 1)},
		{ID: "a", Visible: Size{0, 1}, Thermal: Size{1, 1}, Points: grid(0, 1)},
		{ID: "a", Visible: Size{1, 1}, Thermal: Size{1, 1}},
		{ID: "a", Visible: Size{1, 1}, Thermal: Size{1, 1}, Points: grid(0, 1), Homography: &truth},
		{ID: "a", Visible: Size{1, 1}, Thermal: Size{1, 1}, Homography: &Homography{}},
	}
	for i, c := range bad {
		if _, err := NewRig(c); err == nil {
			t.Errorf("case %d accepted", i)
		}
	}
}

func TestInverseAndMapBox(t *testing.T) {
	inv, err := truth.Inverse()
	if err != nil {
		t.Fatal(err)
	}
	p, _ := truth.Apply(Point{700, 300})
	q, _ := inv.Apply(p)
	if math.Hypot(q.X-700, q.Y-300) > 1e-9 {
		t.Fatalf("inverse round trip %v", q)
	}

	zero := 0.0
	rig, _ := NewRig(RigConfig{ID: "r", Visible: Size{1280, 960}, Thermal: Size{160, 120},
		Homography: &Homography{0.125, 0, 0, 0, 0.125, 0, 0, 0, 1}, Margin: &zero})
	if x, y, w, h, ok := rig.MapBox(Box{X: 160, Y: 80, W: 240, H: 320}); !ok || x != 20 || y != 10 || w != 30 || h != 40 {
		t.Fatalf("MapBox = %d,%d,%d,%d,%v", x, y, w, h, ok)
	}
	if _, _, _, _, ok := rig.MapBox(Box{X: 5000, Y: 5000, W: 10, H: 10}); ok {
		t.Fatal("off-frame box mapped")
	}
	rig.Margin = 0.1
	if x, y, w, h, _ := rig.MapBox(Box{X: 160, Y: 80, W: 240, H: 320}); x != 17 || y != 6 || w != 36 || h != 48 {
		t.Fatalf("margin MapBox = %d,%d,%d,%d", x, y, w, h)
	}
}

func TestRegistry(t *testing.T) {
	reg, err := NewRegistry([]RigConfig{{ID: "gate-1", Visible: Size{1280, 960}, Thermal: Size{160, 120}, Points: grid(0, 1)}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reg.Create(RigConfig{ID: "gate-1", Visible: Size{1, 1}, Thermal: Size{1, 1}, Homography: &truth}); err != ErrExists {
		t.Fatalf("duplicate: %v", err)
	}
	if len(reg.List()) != 1 {
		t.Fatal("list")
	}
	if reg.Delete("gate-1") != nil || reg.Delete("gate-1") != ErrNotFound {
		t.Fatal("delete")
	}
}
