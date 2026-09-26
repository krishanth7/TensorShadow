package sim

import (
	"math"
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

	// Appearance: when EmbeddingDim > 0 every pedestrian gets a random
	// identity vector and each detection carries identity + Gaussian noise
	// (EmbeddingNoise is the noise-to-signal ratio), like a ReID network.
	EmbeddingDim   int
	EmbeddingNoise float64
	// Occluder hides pedestrians whose centre lies in [X0, X1] (a pillar).
	Occluder *Occluder
}

// Occluder is a vertical band of the image in which nobody is detected.
type Occluder struct{ X0, X1 float64 }

// CrowdTruth is the ground truth for a generated sequence.
type CrowdTruth struct {
	// Crossings of the vertical line x = Width/2.
	LeftToRight int
	RightToLeft int
	// Pedestrians visible for at least one frame.
	Visible int
	// Peak number of simultaneously visible pedestrians.
	PeakVisible int
	// Longest continuous stretch any pedestrian spent hidden by the occluder.
	LongestOcclusion int
	// HiddenCrossings are midline crossings made behind the occluder by
	// pedestrians who are still hidden when the sequence ends: no tracker can
	// observe them, so accuracy is scored on the observable crossings.
	HiddenCrossings int
}

// CrowdSequence holds per-frame detections plus ground truth.
type CrowdSequence struct {
	Frames [][]tracking.Detection
	Truth  CrowdTruth
}

type pedestrian struct {
	spawn    int
	x, y     float64
	vx, vy   float64
	identity []float64
	hidden   int // consecutive frames behind the occluder
}

// DefaultCrowdScene is the reference 720p scene used in the README.
func DefaultCrowdScene() CrowdScene {
	return CrowdScene{
		Width: 1280, Height: 720, Pedestrians: 24, Frames: 300, Seed: 42,
		BoxW: 56, BoxH: 140, JitterPx: 1.5, MissRate: 0.03, MinSpeed: 5, MaxSpeed: 9,
	}
}

// OccludedCrowdScene is DefaultCrowdScene with a 200 px pillar over the
// counting line and 128-d appearance embeddings. Walking behind the pillar
// takes 22–40 frames, longer than tracking.max_age (15), so pure IoU tracking
// loses every identity there.
func OccludedCrowdScene() CrowdScene {
	c := DefaultCrowdScene()
	c.Occluder = &Occluder{X0: 540, X1: 740}
	c.EmbeddingDim, c.EmbeddingNoise = 128, 0.5
	return c
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
		if c.EmbeddingDim > 0 {
			p.identity = randomUnit(rng, c.EmbeddingDim)
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
			if c.Occluder != nil && p.x >= c.Occluder.X0 && p.x <= c.Occluder.X1 {
				p.hidden++
				seq.Truth.LongestOcclusion = max(seq.Truth.LongestOcclusion, p.hidden)
				continue
			}
			p.hidden = 0
			if rng.Float64() < c.MissRate {
				continue
			}
			w := c.BoxW * (1 + rng.NormFloat64()*0.02)
			h := c.BoxH * (1 + rng.NormFloat64()*0.02)
			cx := p.x + rng.NormFloat64()*c.JitterPx
			cy := p.y + rng.NormFloat64()*c.JitterPx
			det := tracking.Detection{
				BBox:  tracking.BBox{X: cx - w/2, Y: cy - h/2, W: w, H: h},
				Score: 0.6 + rng.Float64()*0.39,
				Label: "person",
			}
			if p.identity != nil {
				sigma := c.EmbeddingNoise / math.Sqrt(float64(c.EmbeddingDim))
				det.Embedding = make([]float64, c.EmbeddingDim)
				for k, v := range p.identity {
					det.Embedding[k] = v + rng.NormFloat64()*sigma
				}
			}
			dets = append(dets, det)
		}
		seq.Truth.PeakVisible = max(seq.Truth.PeakVisible, visible)
		seq.Frames[f] = dets
	}
	for _, s := range seen {
		if s {
			seq.Truth.Visible++
		}
	}
	for _, p := range peds {
		if p.hidden > 0 && (p.x-mid)*p.vx > 0 { // crossed the midline, not yet re-emerged
			seq.Truth.HiddenCrossings++
		}
	}
	return seq
}

func randomUnit(rng *rand.Rand, dim int) []float64 {
	v := make([]float64, dim)
	var n float64
	for i := range v {
		v[i] = rng.NormFloat64()
		n += v[i] * v[i]
	}
	n = math.Sqrt(n)
	for i := range v {
		v[i] /= n
	}
	return v
}
