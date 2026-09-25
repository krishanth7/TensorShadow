// Package sim generates deterministic synthetic inputs with known ground
// truth: radiometric thermal scenes and multi-pedestrian crowd sequences.
// It powers the unit tests, the tsdemo walkthrough and the tsload generator.
package sim

import (
	"math"
	"math/rand"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
)

// ThermalSubject is one synthetic face in a thermal scene.
type ThermalSubject struct {
	CX, CY   float64 // face centre (px)
	RX, RY   float64 // face ellipse radii (px)
	SkinC    float64 // mean facial skin temperature (°C)
	CanthusC float64 // inner-canthus temperature (°C)
	// Spoof renders a heated presentation-attack instrument (e.g. a warmed
	// mask or screen): a flat temperature field with no periorbital hotspot.
	Spoof bool
}

// ThermalScene describes a synthetic radiometric frame.
type ThermalScene struct {
	Width, Height int
	AmbientC      float64
	Emissivity    float64 // emissivity of skin used to produce apparent temps
	NoiseC        float64 // sensor NETD-like Gaussian noise σ (°C)
	Seed          int64
	Subjects      []ThermalSubject
}

// Render returns row-major apparent temperatures (°C) as an ε=1 radiometric
// camera would report them.
func (s ThermalScene) Render() []float64 {
	rng := rand.New(rand.NewSource(s.Seed))
	eps := s.Emissivity
	if eps == 0 {
		eps = 0.98
	}
	out := make([]float64, s.Width*s.Height)
	for y := 0; y < s.Height; y++ {
		for x := 0; x < s.Width; x++ {
			// Background: ambient with a gentle vertical gradient (warm ceiling air).
			t := s.AmbientC + 0.6*(1-float64(y)/float64(s.Height))
			emissive := false
			for _, sub := range s.Subjects {
				if v, ok := sub.temperatureAt(float64(x), float64(y)); ok {
					t, emissive = v, true
				}
			}
			t += rng.NormFloat64() * s.NoiseC
			if emissive {
				t = thermal.ApparentTemperature(t, eps, s.AmbientC)
			}
			out[y*s.Width+x] = t
		}
	}
	return out
}

func (s ThermalSubject) temperatureAt(x, y float64) (float64, bool) {
	// Torso/clothing below the face: warm but below skin temperature.
	bx, by := (x-s.CX)/(s.RX*2.2), (y-(s.CY+s.RY*2.6))/(s.RY*1.5)
	torso := bx*bx+by*by <= 1

	dx, dy := (x-s.CX)/s.RX, (y-s.CY)/s.RY
	d := dx*dx + dy*dy
	if d > 1 {
		if torso {
			return 28.0 - 0.8*math.Abs(bx), true
		}
		return 0, false
	}
	if s.Spoof {
		return s.SkinC, true
	}

	// Skin cools towards the face edge; the nose tip is markedly cooler.
	t := s.SkinC + 0.6 - 1.2*d
	t -= 1.8 * gauss(x-s.CX, y-(s.CY+0.18*s.RY), 0.14*s.RX)

	// Two inner-canthus hotspots with a small plateau so that 3×3 sampling
	// recovers the configured canthus temperature.
	sigma := math.Max(1.2, 0.07*s.RX)
	for _, side := range []float64{-1, 1} {
		cx, cy := s.CX+side*0.16*s.RX, s.CY-0.15*s.RY
		r := math.Hypot(x-cx, y-cy)
		g := 1.0
		if r > 1.5 {
			g = math.Exp(-(r - 1.5) * (r - 1.5) / (2 * sigma * sigma))
		}
		t = t*(1-g) + s.CanthusC*g
	}
	return t, true
}

func gauss(dx, dy, sigma float64) float64 {
	return math.Exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
}

// DemoThermalScene is the reference scene used in the README: three people
// (normal, elevated, fever) and one heated spoof.
func DemoThermalScene() ThermalScene {
	return ThermalScene{
		Width: 160, Height: 120, AmbientC: 22, Emissivity: 0.98, NoiseC: 0.05, Seed: 7,
		Subjects: []ThermalSubject{
			{CX: 25, CY: 38, RX: 14, RY: 18, SkinC: 34.2, CanthusC: 36.6},
			{CX: 63, CY: 42, RX: 15, RY: 19, SkinC: 34.8, CanthusC: 37.4},
			{CX: 101, CY: 38, RX: 14, RY: 18, SkinC: 35.6, CanthusC: 38.5},
			{CX: 139, CY: 42, RX: 14, RY: 18, SkinC: 33.5, Spoof: true},
		},
	}
}
