package sim

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/coreg"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
)

// CoregScene pairs a 1280×960 visible camera with the 160×120 thermal camera
// of DemoThermalScene.
type CoregScene struct {
	Truth       coreg.Homography       // visible → thermal
	Calibration []coreg.Correspondence // heated-target corners seen by both cameras
	FaceBoxes   []thermal.VisibleBox   // what the RGB face detector reports
}

// DemoCoregScene models a typical dual-sensor head: 1/8 scale, a 1.2° roll
// between the sensors, a few pixels of boresight offset and slight keystone.
// Calibration corners carry 0.2 px of localisation noise; RGB face boxes
// carry 3 px of detector jitter.
func DemoCoregScene() CoregScene {
	a := 1.2 * math.Pi / 180
	s := 0.125
	h := coreg.Homography{s * math.Cos(a), -s * math.Sin(a), 3.5, s * math.Sin(a), s * math.Cos(a), -2.5, 1.5e-6, -1e-6, 1}
	rng := rand.New(rand.NewSource(11))

	var cal []coreg.Correspondence
	for _, x := range []float64{200, 640, 1080} {
		for _, y := range []float64{150, 480, 810} {
			v := coreg.Point{X: x, Y: y}
			t, _ := h.Apply(v)
			t.X += rng.NormFloat64() * 0.2
			t.Y += rng.NormFloat64() * 0.2
			cal = append(cal, coreg.Correspondence{Visible: v, Thermal: t})
		}
	}

	inv, _ := h.Inverse()
	var boxes []thermal.VisibleBox
	for i, f := range DemoThermalScene().Subjects {
		// The face ellipse's extent in thermal pixels, taken back to visible pixels.
		var minX, minY, maxX, maxY = math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
		for _, c := range []coreg.Point{{X: f.CX - f.RX, Y: f.CY - f.RY}, {X: f.CX + f.RX, Y: f.CY - f.RY},
			{X: f.CX - f.RX, Y: f.CY + f.RY}, {X: f.CX + f.RX, Y: f.CY + f.RY}} {
			p, _ := inv.Apply(c)
			minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
			minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
		}
		j := func() float64 { return rng.NormFloat64() * 3 }
		boxes = append(boxes, thermal.VisibleBox{
			X: math.Round(minX + j()), Y: math.Round(minY + j()),
			W: math.Round(maxX - minX + j()), H: math.Round(maxY - minY + j()),
			ID: fmt.Sprintf("rgb-face-%d", i+1),
		})
	}
	return CoregScene{Truth: h, Calibration: cal, FaceBoxes: boxes}
}
