package thermal_test

import (
	"math"
	"testing"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/sim"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
)

func ptr(v float64) *float64 { return &v }

func TestEmissivityRoundTrip(t *testing.T) {
	for _, obj := range []float64{30, 34.5, 36.8, 39.2} {
		for _, eps := range []float64{0.95, 0.98, 1} {
			app := thermal.ApparentTemperature(obj, eps, 20)
			if got := thermal.CorrectEmissivity(app, eps, 20); math.Abs(got-obj) > 1e-9 {
				t.Fatalf("obj=%v eps=%v: round trip gave %v", obj, eps, got)
			}
			if eps < 1 && app >= obj {
				t.Fatalf("apparent %v should read below object %v when reflecting a cooler room", app, obj)
			}
		}
	}
}

func TestDecodeFormats(t *testing.T) {
	cfg := thermal.DefaultConfig()
	one := ptr(1)
	cases := []struct {
		name string
		req  thermal.FrameRequest
		want float64
	}{
		{"celsius", thermal.FrameRequest{Format: thermal.FormatCelsius, Data: []float64{36.6}}, 36.6},
		{"kelvin", thermal.FrameRequest{Format: thermal.FormatKelvin, Data: []float64{309.75}}, 36.6},
		{"centikelvin", thermal.FrameRequest{Format: thermal.FormatCentiKelvin, Data: []float64{30975}}, 36.6},
		{"raw16", thermal.FrameRequest{Format: thermal.FormatRaw16, Data: []float64{8000},
			Calibration: thermal.Calibration{Gain: 0.01, Offset: -43.4}}, 36.6},
	}
	for _, c := range cases {
		c.req.Width, c.req.Height = 1, 1
		c.req.Calibration.Emissivity = one
		f, err := thermal.Decode(&c.req, cfg)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if math.Abs(f.Temps[0]-c.want) > 1e-9 {
			t.Fatalf("%s: got %v want %v", c.name, f.Temps[0], c.want)
		}
	}
}

func TestDecodeRejectsBadInput(t *testing.T) {
	cfg := thermal.DefaultConfig()
	bad := []thermal.FrameRequest{
		{Width: 2, Height: 2, Data: []float64{1, 2, 3}},
		{Width: 0, Height: 1, Data: nil},
		{Width: 1, Height: 1, Data: []float64{math.NaN()}},
		{Width: 1, Height: 1, Format: "fahrenheit", Data: []float64{98}},
		{Width: 1, Height: 1, Format: thermal.FormatRaw16, Data: []float64{8000}},
		{Width: 1, Height: 1, Data: []float64{5000}},
		{Width: 1, Height: 1, Data: []float64{30}, Calibration: thermal.Calibration{Emissivity: ptr(0)}},
		{Width: 2000, Height: 2000, Data: make([]float64, 4e6)},
	}
	for i, req := range bad {
		if _, err := thermal.Decode(&req, cfg); err == nil {
			t.Errorf("case %d: expected error", i)
		}
	}
}

func analyseDemo(t *testing.T) thermal.Result {
	t.Helper()
	scene := sim.DemoThermalScene()
	req := thermal.FrameRequest{Width: scene.Width, Height: scene.Height, Format: thermal.FormatCelsius,
		Data: scene.Render(), AmbientTempC: ptr(scene.AmbientC)}
	cfg := thermal.DefaultConfig()
	f, err := thermal.Decode(&req, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return thermal.Analyze(f, nil, cfg)
}

func TestAnalyzeDemoScene(t *testing.T) {
	res := analyseDemo(t)
	if len(res.Subjects) != 4 {
		t.Fatalf("found %d subjects, want 4", len(res.Subjects))
	}
	scene := sim.DemoThermalScene()
	wantStatus := []string{thermal.StatusNormal, thermal.StatusElevated, thermal.StatusFever}
	for i, want := range wantStatus {
		s := res.Subjects[i]
		truth := scene.Subjects[i]
		if math.Abs(s.Canthus.TempC-truth.CanthusC) > 0.15 {
			t.Errorf("subject %d: canthus %.2f°C, truth %.2f°C", i, s.Canthus.TempC, truth.CanthusC)
		}
		if s.Status != want {
			t.Errorf("subject %d: status %s, want %s (core %.2f)", i, s.Status, want, s.CoreEstimate)
		}
		if !s.Liveness.Live {
			t.Errorf("subject %d: real face flagged as spoof: %+v", i, s.Liveness.Checks)
		}
		// Canthus must be in the upper half of the face, near an eye.
		if s.Canthus.Y > int(truth.CY) {
			t.Errorf("subject %d: canthus y=%d below face centre %v", i, s.Canthus.Y, truth.CY)
		}
	}
	spoof := res.Subjects[3]
	if spoof.Liveness.Live {
		t.Fatalf("heated spoof passed liveness: %+v", spoof.Liveness.Checks)
	}
	if res.Summary["spoof_suspected"] != 1 || res.Summary["live"] != 3 {
		t.Fatalf("unexpected summary %v", res.Summary)
	}
	if res.Frame.AmbientSrc != "provided" || *res.Frame.AmbientC != 22 {
		t.Fatalf("ambient not propagated: %+v", res.Frame)
	}
}

func TestExplicitROIs(t *testing.T) {
	scene := sim.DemoThermalScene()
	req := thermal.FrameRequest{Width: scene.Width, Height: scene.Height, Data: scene.Render()}
	cfg := thermal.DefaultConfig()
	f, _ := thermal.Decode(&req, cfg)
	res := thermal.Analyze(f, []thermal.Rect{{X: 85, Y: 18, W: 32, H: 40}, {X: -10, Y: -10, W: 5, H: 5}}, cfg)
	if len(res.Subjects) != 1 || res.Subjects[0].Status != thermal.StatusFever {
		t.Fatalf("ROI analysis wrong: %+v", res.Subjects)
	}
	if res.Frame.AmbientSrc != "estimated_background_median" {
		t.Fatalf("expected estimated ambient, got %s", res.Frame.AmbientSrc)
	}
}

func TestRender(t *testing.T) {
	scene := sim.DemoThermalScene()
	req := thermal.FrameRequest{Width: scene.Width, Height: scene.Height, Data: scene.Render()}
	cfg := thermal.DefaultConfig()
	f, _ := thermal.Decode(&req, cfg)
	img, err := thermal.Render(f, thermal.RenderOptions{Scale: 3, Subjects: thermal.Analyze(f, nil, cfg).Subjects})
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != 480 || b.Dy() != 360 {
		t.Fatalf("bounds %v", b)
	}
	if _, err := thermal.Render(f, thermal.RenderOptions{Palette: "sepia"}); err == nil {
		t.Fatal("unknown palette accepted")
	}
}

func BenchmarkAnalyze160x120(b *testing.B) { benchAnalyze(b, 1) }
func BenchmarkAnalyze640x480(b *testing.B) { benchAnalyze(b, 4) }

func benchAnalyze(b *testing.B, k int) {
	scene := sim.DemoThermalScene()
	scene.Width, scene.Height = scene.Width*k, scene.Height*k
	for i := range scene.Subjects {
		s := &scene.Subjects[i]
		s.CX, s.CY, s.RX, s.RY = s.CX*float64(k), s.CY*float64(k), s.RX*float64(k), s.RY*float64(k)
	}
	data := scene.Render()
	cfg := thermal.DefaultConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := thermal.FrameRequest{Width: scene.Width, Height: scene.Height, Data: data}
		f, err := thermal.Decode(&req, cfg)
		if err != nil {
			b.Fatal(err)
		}
		if res := thermal.Analyze(f, nil, cfg); len(res.Subjects) != 4 {
			b.Fatalf("subjects = %d", len(res.Subjects))
		}
	}
}
