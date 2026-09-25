package sim

import (
	"math/rand"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
)

// CrowdScene describes pedestrians walking horizontally across a camera view.
type CrowdScene struct {
	Width, Height float64
	Pedestrians   int
	Frames        int
	Seed          int64
	BoxW, BoxH    float64
	JitterPx      float64 // detector localisation noise σ (px)
	MissRate      float64 // probability a visible pedestrian is not detected
	MinSpeed      float64 // px/frame
	MaxSpeed      float64 // px/frame
}

// CrowdTruth is the ground truth for a generated sequence.
type CrowdTruth struct {
	// Crossings of the vertical line x = Width/2.
	LeftToRight int
	RightToLeft int
	// Pedestrians visible for at least one frame.
	Visible int
	// Peak number of simultaneously visible pedestrians.
	PeakVisible int
}

// CrowdSequence holds per-frame detections plus ground truth.
type CrowdSequence struct {
	Frames [][]tracking.Detection
	Truth  CrowdTruth
}

type pedestrian struct {
	spawn  int
	x, y   float64
	vx, vy float64
}

// DefaultCrowdScene is the reference 720p scene used in the README.
func DefaultCrowdScene() CrowdScene {
	return CrowdScene{
		Width: 1280, Height: 720, Pedestrians: 24, Frames: 300, Seed: 42,
		BoxW: 56, BoxH: 140, JitterPx: 1.5, MissRate: 0.03, MinSpeed: 5, MaxSpeed: 9,
	}
}

// Generate produces the detection sequence. Pedestrians are assigned to
// well-separated horizontal lanes so ground truth is unambiguous.
func (c CrowdScene) Generate() CrowdSequence {
	rng := rand.New(rand.NewSource(c.Seed))
	lanes := int((c.Height - c.BoxH) / (c.BoxH * 0.45))
	if lanes < 1 {
		lanes = 1
	}
	peds := make([]pedestrian, c.Pedestrians)
	for i := range peds {
		speed := c.MinSpeed + rng.Float64()*(c.MaxSpeed-c.MinSpeed)
		lane := i % lanes
		p := pedestrian{
			spawn: rng.Intn(max(1, c.Frames*6/10)),
			y:     c.BoxH/2 + float64(lane)*(c.BoxH*0.45) + rng.Float64()*6,
			vy:    (rng.Float64() - 0.5) * 0.4,
		}
		if i%2 == 0 {
			p.x, p.vx = c.BoxW/2, speed
		} else {
			p.x, p.vx = c.Width-c.BoxW/2, -speed
		}
		peds[i] = p
	}

	seq := CrowdSequence{Frames: make([][]tracking.Detection, c.Frames)}
	seen := make([]bool, len(peds))
	mid := c.Width / 2
	for f := 0; f < c.Frames; f++ {
		var dets []tracking.Detection
		visible := 0
		for i := range peds {
			p := &peds[i]
			if f < p.spawn {
				continue
			}
			if f > p.spawn {
				prevX := p.x
				p.x += p.vx
				p.y += p.vy
				if prevX < mid && p.x >= mid {
					seq.Truth.LeftToRight++
				} else if prevX >= mid && p.x < mid {
					seq.Truth.RightToLeft++
				}
			}
			if p.x < c.BoxW/2 || p.x > c.Width-c.BoxW/2 {
				continue
			}
			visible++
			seen[i] = true
			if rng.Float64() < c.MissRate {
				continue
			}
			w := c.BoxW * (1 + rng.NormFloat64()*0.02)
			h := c.BoxH * (1 + rng.NormFloat64()*0.02)
			cx := p.x + rng.NormFloat64()*c.JitterPx
			cy := p.y + rng.NormFloat64()*c.JitterPx
			dets = append(dets, tracking.Detection{
				BBox:  tracking.BBox{X: cx - w/2, Y: cy - h/2, W: w, H: h},
				Score: 0.6 + rng.Float64()*0.39,
				Label: "person",
			})
		}
		seq.Truth.PeakVisible = max(seq.Truth.PeakVisible, visible)
		seq.Frames[f] = dets
	}
	for _, s := range seen {
		if s {
			seq.Truth.Visible++
		}
	}
	return seq
}
