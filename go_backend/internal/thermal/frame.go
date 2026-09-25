// Package thermal implements infrared (IR) biometric analysis on radiometric
// thermal frames: unit conversion, emissivity correction, multi-subject skin
// segmentation, inner-canthus temperature screening and thermal liveness.
package thermal

import (
	"errors"
	"fmt"
	"math"
)

// Format identifies how the values in FrameRequest.Data are encoded.
type Format string

const (
	// FormatCelsius: apparent temperature in °C (camera already radiometric).
	FormatCelsius Format = "celsius"
	// FormatKelvin: apparent temperature in K.
	FormatKelvin Format = "kelvin"
	// FormatCentiKelvin: hundredths of a kelvin, as emitted by FLIR Lepton
	// radiometric (TLinear) mode and many USB thermal cores.
	FormatCentiKelvin Format = "centikelvin"
	// FormatRaw16: raw sensor counts, converted with T°C = gain*raw + offset.
	FormatRaw16 Format = "raw16"
)

const kelvinOffset = 273.15

// Config holds analysis parameters. All temperatures are in °C.
type Config struct {
	Emissivity               float64 `yaml:"emissivity" json:"emissivity"`
	ReflectedTempC           float64 `yaml:"reflected_temp_c" json:"reflected_temp_c"`
	SkinMinC                 float64 `yaml:"skin_min_c" json:"skin_min_c"`
	SkinMaxC                 float64 `yaml:"skin_max_c" json:"skin_max_c"`
	MinSubjectPixels         int     `yaml:"min_subject_pixels" json:"min_subject_pixels"`
	MaxSubjects              int     `yaml:"max_subjects" json:"max_subjects"`
	CoreOffsetC              float64 `yaml:"core_offset_c" json:"core_offset_c"`
	ElevatedThresholdC       float64 `yaml:"elevated_threshold_c" json:"elevated_threshold_c"`
	FeverThresholdC          float64 `yaml:"fever_threshold_c" json:"fever_threshold_c"`
	IndeterminateBelowC      float64 `yaml:"indeterminate_below_c" json:"indeterminate_below_c"`
	LivenessMinSpanC         float64 `yaml:"liveness_min_span_c" json:"liveness_min_span_c"`
	LivenessMinAmbientDeltaC float64 `yaml:"liveness_min_ambient_delta_c" json:"liveness_min_ambient_delta_c"`
	MaxPixels                int     `yaml:"max_pixels" json:"max_pixels"`
}

// DefaultConfig returns parameters suitable for indoor screening of human skin.
func DefaultConfig() Config {
	return Config{
		Emissivity:               0.98,
		ReflectedTempC:           22,
		SkinMinC:                 30,
		SkinMaxC:                 42,
		MinSubjectPixels:         120,
		MaxSubjects:              64,
		CoreOffsetC:              0.3,
		ElevatedThresholdC:       37.5,
		FeverThresholdC:          38.0,
		IndeterminateBelowC:      34.0,
		LivenessMinSpanC:         1.5,
		LivenessMinAmbientDeltaC: 4.0,
		MaxPixels:                1 << 20,
	}
}

// Validate reports configuration errors.
func (c Config) Validate() error {
	switch {
	case c.Emissivity <= 0 || c.Emissivity > 1:
		return fmt.Errorf("thermal.emissivity must be in (0,1], got %v", c.Emissivity)
	case c.SkinMinC >= c.SkinMaxC:
		return errors.New("thermal.skin_min_c must be below skin_max_c")
	case c.ElevatedThresholdC > c.FeverThresholdC:
		return errors.New("thermal.elevated_threshold_c must not exceed fever_threshold_c")
	case c.MinSubjectPixels < 1:
		return errors.New("thermal.min_subject_pixels must be >= 1")
	case c.MaxSubjects < 1:
		return errors.New("thermal.max_subjects must be >= 1")
	case c.MaxPixels < 1:
		return errors.New("thermal.max_pixels must be >= 1")
	}
	return nil
}

// Calibration overrides per-frame radiometric parameters. Zero values mean
// "use the server default" (for Emissivity/ReflectedTempC pass a pointer).
type Calibration struct {
	Emissivity     *float64 `json:"emissivity,omitempty"`
	ReflectedTempC *float64 `json:"reflected_temp_c,omitempty"`
	Gain           float64  `json:"gain,omitempty"`   // raw16 only
	Offset         float64  `json:"offset,omitempty"` // raw16 only
}

// Rect is an integer pixel rectangle.
type Rect struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

// FrameRequest is the wire format of a radiometric frame.
type FrameRequest struct {
	Width        int         `json:"width"`
	Height       int         `json:"height"`
	Format       Format      `json:"format"`
	Data         []float64   `json:"data"`
	Calibration  Calibration `json:"calibration"`
	AmbientTempC *float64    `json:"ambient_temp_c,omitempty"`
	// ROIs optionally restricts analysis to caller-supplied face boxes
	// (e.g. from a co-registered visible-light face detector). When empty,
	// subjects are segmented automatically.
	ROIs []Rect `json:"rois,omitempty"`
	// Rig names a visible/IR co-registration rig; VisibleROIs are face boxes
	// from an RGB detector, in visible-camera pixels, mapped into this frame
	// through the rig's homography. Mutually exclusive with ROIs.
	Rig         string       `json:"rig,omitempty"`
	VisibleROIs []VisibleBox `json:"visible_rois,omitempty"`
}

// VisibleBox is a detection box in visible-camera pixels. ID lets callers
// join thermal results back to their RGB track (e.g. a crowd track ID).
type VisibleBox struct {
	X  float64 `json:"x"`
	Y  float64 `json:"y"`
	W  float64 `json:"w"`
	H  float64 `json:"h"`
	ID string  `json:"id,omitempty"`
}

// Frame is a decoded, emissivity-corrected temperature image in °C.
type Frame struct {
	Width, Height int
	Temps         []float64
	Emissivity    float64
	ReflectedC    float64
	AmbientC      *float64
}

// At returns the temperature at (x, y). The caller must stay in bounds.
func (f *Frame) At(x, y int) float64 { return f.Temps[y*f.Width+x] }

// Decode validates req, converts it to °C and applies emissivity correction.
func Decode(req *FrameRequest, cfg Config) (*Frame, error) {
	if req.Width <= 0 || req.Height <= 0 {
		return nil, errors.New("width and height must be positive")
	}
	if req.Width > 8192 || req.Height > 8192 || req.Width*req.Height > cfg.MaxPixels {
		return nil, fmt.Errorf("frame %dx%d exceeds the %d pixel limit", req.Width, req.Height, cfg.MaxPixels)
	}
	if len(req.Data) != req.Width*req.Height {
		return nil, fmt.Errorf("data has %d values, want width*height = %d", len(req.Data), req.Width*req.Height)
	}

	eps := cfg.Emissivity
	if req.Calibration.Emissivity != nil {
		eps = *req.Calibration.Emissivity
	}
	if eps <= 0 || eps > 1 {
		return nil, fmt.Errorf("emissivity must be in (0,1], got %v", eps)
	}
	refl := cfg.ReflectedTempC
	switch {
	case req.Calibration.ReflectedTempC != nil:
		refl = *req.Calibration.ReflectedTempC
	case req.AmbientTempC != nil:
		// Indoors the reflected apparent temperature ≈ ambient air temperature.
		refl = *req.AmbientTempC
	}

	var toC func(float64) float64
	switch req.Format {
	case FormatCelsius, "":
		toC = func(v float64) float64 { return v }
	case FormatKelvin:
		toC = func(v float64) float64 { return v - kelvinOffset }
	case FormatCentiKelvin:
		toC = func(v float64) float64 { return v/100 - kelvinOffset }
	case FormatRaw16:
		if req.Calibration.Gain == 0 {
			return nil, errors.New("raw16 format requires calibration.gain and calibration.offset")
		}
		g, o := req.Calibration.Gain, req.Calibration.Offset
		toC = func(v float64) float64 { return g*v + o }
	default:
		return nil, fmt.Errorf("unsupported format %q", req.Format)
	}

	corr := newCorrector(eps, refl)
	temps := make([]float64, len(req.Data))
	for i, v := range req.Data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("data[%d] is not finite", i)
		}
		t := toC(v)
		if t < -100 || t > 700 {
			return nil, fmt.Errorf("data[%d] decodes to %.2f°C, outside the physical range of a thermal sensor", i, t)
		}
		temps[i] = corr.apply(t)
	}
	return &Frame{
		Width: req.Width, Height: req.Height, Temps: temps,
		Emissivity: eps, ReflectedC: refl, AmbientC: req.AmbientTempC,
	}, nil
}

// CorrectEmissivity converts an apparent (blackbody-equivalent) temperature to
// the object temperature using the Stefan–Boltzmann radiometric model:
//
//	W_app = ε·W_obj + (1−ε)·W_refl,  W ∝ T⁴
func CorrectEmissivity(apparentC, emissivity, reflectedC float64) float64 {
	return newCorrector(emissivity, reflectedC).apply(apparentC)
}

// corrector hoists the per-frame constants of CorrectEmissivity out of the
// per-pixel loop and avoids math.Pow (x⁴ by squaring, ⁴√x as √√x).
type corrector struct {
	identity   bool
	reflTerm   float64 // (1−ε)·T_refl⁴
	invEpsilon float64
}

func newCorrector(emissivity, reflectedC float64) corrector {
	if emissivity >= 1 {
		return corrector{identity: true}
	}
	tr := reflectedC + kelvinOffset
	tr2 := tr * tr
	return corrector{reflTerm: (1 - emissivity) * tr2 * tr2, invEpsilon: 1 / emissivity}
}

func (c corrector) apply(apparentC float64) float64 {
	if c.identity {
		return apparentC
	}
	ta := apparentC + kelvinOffset
	ta2 := ta * ta
	obj4 := (ta2*ta2 - c.reflTerm) * c.invEpsilon
	if obj4 <= 0 {
		return apparentC
	}
	return math.Sqrt(math.Sqrt(obj4)) - kelvinOffset
}

// ApparentTemperature is the inverse of CorrectEmissivity: what a camera
// calibrated for ε=1 reports when viewing an object at objectC.
func ApparentTemperature(objectC, emissivity, reflectedC float64) float64 {
	to := objectC + kelvinOffset
	tr := reflectedC + kelvinOffset
	return math.Pow(emissivity*math.Pow(to, 4)+(1-emissivity)*math.Pow(tr, 4), 0.25) - kelvinOffset
}
