// Package coreg co-registers a visible-light (RGB) camera with a thermal (IR)
// camera so that faces found by an RGB face detector drive the thermal
// regions of interest.
//
// A rig is calibrated from point correspondences — typically the corners of
// a heated calibration target seen by both cameras — and stored as a 3×3
// homography that maps visible pixel coordinates to thermal pixel
// coordinates. The homography is exact for points on one plane; for subjects
// nearer or farther than the calibration plane the residual parallax is
// absorbed by a configurable ROI margin (thermal segmentation inside the ROI
// only keeps skin-temperature pixels, so a slightly larger ROI is safe).
package coreg

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"sync"
)

// Point is a pixel coordinate.
type Point struct {
	X float64 `json:"x" yaml:"x"`
	Y float64 `json:"y" yaml:"y"`
}

// Correspondence pairs the same physical point in both images.
type Correspondence struct {
	Visible Point `json:"visible" yaml:"visible"`
	Thermal Point `json:"thermal" yaml:"thermal"`
}

// Size is an image size in pixels.
type Size struct {
	Width  int `json:"width" yaml:"width"`
	Height int `json:"height" yaml:"height"`
}

// Box is an axis-aligned rectangle (float pixels).
type Box struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	W float64 `json:"w"`
	H float64 `json:"h"`
}

// Homography is a row-major 3×3 projective transform.
type Homography [9]float64

// Apply maps p. ok is false when p maps to the line at infinity.
func (h Homography) Apply(p Point) (Point, bool) {
	w := h[6]*p.X + h[7]*p.Y + h[8]
	if math.Abs(w) < 1e-12 {
		return Point{}, false
	}
	return Point{(h[0]*p.X + h[1]*p.Y + h[2]) / w, (h[3]*p.X + h[4]*p.Y + h[5]) / w}, true
}

func mul(a, b Homography) Homography {
	var c Homography
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				c[i*3+j] += a[i*3+k] * b[k*3+j]
			}
		}
	}
	return c
}

// Inverse returns h⁻¹.
func (h Homography) Inverse() (Homography, error) {
	a, b, c, d, e, f, g, hh, i := h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]
	det := a*(e*i-f*hh) - b*(d*i-f*g) + c*(d*hh-e*g)
	if math.Abs(det) < 1e-15 {
		return Homography{}, errors.New("homography is singular")
	}
	inv := Homography{
		e*i - f*hh, c*hh - b*i, b*f - c*e,
		f*g - d*i, a*i - c*g, c*d - a*f,
		d*hh - e*g, b*g - a*hh, a*e - b*d,
	}
	for k := range inv {
		inv[k] /= det
	}
	return inv, nil
}

func (h Homography) normalised() Homography {
	if h[8] != 0 {
		s := h[8]
		for k := range h {
			h[k] /= s
		}
	}
	return h
}

// Model selects the transform family.
type Model string

const (
	// ModelHomography is an 8-DoF projective transform (≥ 4 points).
	ModelHomography Model = "homography"
	// ModelAffine is a 6-DoF transform (≥ 3 points); more robust when the
	// cameras are near-parallel or the calibration points are few or noisy.
	ModelAffine Model = "affine"
)

// Estimate fits a transform mapping visible → thermal using normalised DLT
// (Hartley conditioning) and linear least squares.
func Estimate(points []Correspondence, model Model) (Homography, error) {
	switch model {
	case ModelHomography, "":
		if len(points) < 4 {
			return Homography{}, errors.New("a homography needs at least 4 correspondences")
		}
	case ModelAffine:
		if len(points) < 3 {
			return Homography{}, errors.New("an affine model needs at least 3 correspondences")
		}
	default:
		return Homography{}, fmt.Errorf("unknown model %q (homography or affine)", model)
	}
	src := make([]Point, len(points))
	dst := make([]Point, len(points))
	for i, c := range points {
		src[i], dst[i] = c.Visible, c.Thermal
		for _, v := range []float64{c.Visible.X, c.Visible.Y, c.Thermal.X, c.Thermal.Y} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return Homography{}, fmt.Errorf("correspondence %d is not finite", i)
			}
		}
	}
	ts, ns, err := conditioner(src)
	if err != nil {
		return Homography{}, err
	}
	td, nd, err := conditioner(dst)
	if err != nil {
		return Homography{}, err
	}

	// Rows of A·h = b with h33 = 1 (and h31 = h32 = 0 for affine).
	nParams := 8
	if model == ModelAffine {
		nParams = 6
	}
	ata := make([][]float64, nParams)
	for i := range ata {
		ata[i] = make([]float64, nParams+1) // augmented with Aᵀb
	}
	addRow := func(row []float64, rhs float64) {
		for i := 0; i < nParams; i++ {
			for j := 0; j < nParams; j++ {
				ata[i][j] += row[i] * row[j]
			}
			ata[i][nParams] += row[i] * rhs
		}
	}
	for i := range ns {
		x, y, u, v := ns[i].X, ns[i].Y, nd[i].X, nd[i].Y
		if model == ModelAffine {
			addRow([]float64{x, y, 1, 0, 0, 0}, u)
			addRow([]float64{0, 0, 0, x, y, 1}, v)
		} else {
			addRow([]float64{x, y, 1, 0, 0, 0, -u * x, -u * y}, u)
			addRow([]float64{0, 0, 0, x, y, 1, -v * x, -v * y}, v)
		}
	}
	sol, err := solve(ata)
	if err != nil {
		return Homography{}, errors.New("correspondences are degenerate (collinear or repeated points)")
	}
	var hn Homography
	copy(hn[:6], sol[:6])
	if model == ModelAffine {
		hn[6], hn[7] = 0, 0
	} else {
		hn[6], hn[7] = sol[6], sol[7]
	}
	hn[8] = 1

	tdInv, err := td.Inverse()
	if err != nil {
		return Homography{}, err
	}
	return mul(tdInv, mul(hn, ts)).normalised(), nil
}

// conditioner returns the similarity transform that moves the centroid to
// the origin and scales the mean distance to √2, plus the transformed points.
func conditioner(pts []Point) (Homography, []Point, error) {
	var cx, cy float64
	for _, p := range pts {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(len(pts))
	cy /= float64(len(pts))
	var md float64
	for _, p := range pts {
		md += math.Hypot(p.X-cx, p.Y-cy)
	}
	md /= float64(len(pts))
	if md < 1e-9 {
		return Homography{}, nil, errors.New("correspondences are degenerate (all points coincide)")
	}
	s := math.Sqrt2 / md
	t := Homography{s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1}
	out := make([]Point, len(pts))
	for i, p := range pts {
		out[i] = Point{s * (p.X - cx), s * (p.Y - cy)}
	}
	return t, out, nil
}

// solve performs Gaussian elimination with partial pivoting on an augmented
// n×(n+1) system.
func solve(m [][]float64) ([]float64, error) {
	n := len(m)
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(m[r][col]) > math.Abs(m[piv][col]) {
				piv = r
			}
		}
		if math.Abs(m[piv][col]) < 1e-10 {
			return nil, errors.New("singular system")
		}
		m[col], m[piv] = m[piv], m[col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col] / m[col][col]
			for c := col; c <= n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = m[i][n] / m[i][i]
	}
	return out, nil
}

// Residuals reports the reprojection error of h on the correspondences, in
// thermal pixels.
func Residuals(h Homography, points []Correspondence) (rmse, maxErr float64) {
	for _, c := range points {
		p, ok := h.Apply(c.Visible)
		if !ok {
			return math.Inf(1), math.Inf(1)
		}
		e := math.Hypot(p.X-c.Thermal.X, p.Y-c.Thermal.Y)
		rmse += e * e
		maxErr = math.Max(maxErr, e)
	}
	return math.Sqrt(rmse / float64(len(points))), maxErr
}

// RigConfig defines a camera pair, from correspondences or a known matrix.
type RigConfig struct {
	ID         string           `json:"id" yaml:"id"`
	Name       string           `json:"name,omitempty" yaml:"name"`
	Visible    Size             `json:"visible" yaml:"visible"`
	Thermal    Size             `json:"thermal" yaml:"thermal"`
	Model      Model            `json:"model,omitempty" yaml:"model"`
	Points     []Correspondence `json:"points,omitempty" yaml:"points"`
	Homography *Homography      `json:"homography,omitempty" yaml:"homography"`
	// Margin enlarges mapped ROIs by this fraction of their size on each side
	// to absorb parallax for subjects off the calibration plane (default 0.1).
	Margin *float64 `json:"margin,omitempty" yaml:"margin"`
}

// Calibration quality grades by RMSE in thermal pixels.
const (
	QualityGood = "good" // < 1 px
	QualityFair = "fair" // < 2.5 px
	QualityPoor = "poor"
)

// Rig is a calibrated visible→thermal camera pair.
type Rig struct {
	ID         string     `json:"id"`
	Name       string     `json:"name,omitempty"`
	Visible    Size       `json:"visible"`
	Thermal    Size       `json:"thermal"`
	Model      Model      `json:"model"`
	Homography Homography `json:"homography"`
	Margin     float64    `json:"margin"`
	Points     int        `json:"points"`
	RMSE       *float64   `json:"rmse_px,omitempty"`
	MaxError   *float64   `json:"max_error_px,omitempty"`
	Quality    string     `json:"quality"`
}

var idPattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,64}$`)

// NewRig validates cfg and calibrates the rig.
func NewRig(cfg RigConfig) (*Rig, error) {
	if !idPattern.MatchString(cfg.ID) {
		return nil, errors.New("id must match [A-Za-z0-9_.-]{1,64}")
	}
	for _, s := range []Size{cfg.Visible, cfg.Thermal} {
		if s.Width < 1 || s.Height < 1 || s.Width > 16384 || s.Height > 16384 {
			return nil, errors.New("visible and thermal sizes must be positive")
		}
	}
	margin := 0.1
	if cfg.Margin != nil {
		margin = *cfg.Margin
	}
	if margin < 0 || margin > 1 {
		return nil, errors.New("margin must be in [0,1]")
	}
	model := cfg.Model
	if model == "" {
		model = ModelHomography
	}
	rig := &Rig{ID: cfg.ID, Name: cfg.Name, Visible: cfg.Visible, Thermal: cfg.Thermal,
		Model: model, Margin: margin, Points: len(cfg.Points)}

	switch {
	case cfg.Homography != nil && len(cfg.Points) > 0:
		return nil, errors.New("give either points or homography, not both")
	case cfg.Homography != nil:
		h := cfg.Homography.normalised()
		if _, err := h.Inverse(); err != nil {
			return nil, err
		}
		rig.Homography, rig.Quality = h, "provided"
		return rig, nil
	}

	h, err := Estimate(cfg.Points, model)
	if err != nil {
		return nil, err
	}
	rmse, maxErr := Residuals(h, cfg.Points)
	rmse, maxErr = math.Round(rmse*1000)/1000, math.Round(maxErr*1000)/1000
	rig.Homography, rig.RMSE, rig.MaxError = h, &rmse, &maxErr
	switch {
	case rmse < 1:
		rig.Quality = QualityGood
	case rmse < 2.5:
		rig.Quality = QualityFair
	default:
		rig.Quality = QualityPoor
	}
	return rig, nil
}

// MapBox projects a visible box into the thermal image: its four corners are
// mapped, the bounding box is taken, enlarged by the rig margin and clipped
// to the thermal frame. ok is false when nothing of the box lands in view.
func (r *Rig) MapBox(b Box) (x, y, w, h int, ok bool) {
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, c := range []Point{{b.X, b.Y}, {b.X + b.W, b.Y}, {b.X, b.Y + b.H}, {b.X + b.W, b.Y + b.H}} {
		p, valid := r.Homography.Apply(c)
		if !valid {
			return 0, 0, 0, 0, false
		}
		minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
		minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
	}
	mx, my := r.Margin*(maxX-minX), r.Margin*(maxY-minY)
	x0 := int(math.Floor(math.Max(0, minX-mx)))
	y0 := int(math.Floor(math.Max(0, minY-my)))
	x1 := int(math.Ceil(math.Min(float64(r.Thermal.Width), maxX+mx)))
	y1 := int(math.Ceil(math.Min(float64(r.Thermal.Height), maxY+my)))
	if x1 <= x0 || y1 <= y0 {
		return 0, 0, 0, 0, false
	}
	return x0, y0, x1 - x0, y1 - y0, true
}

// Errors returned by the registry.
var (
	ErrNotFound = errors.New("rig not found")
	ErrExists   = errors.New("rig already exists")
)

// Registry stores calibrated rigs.
type Registry struct {
	mu   sync.RWMutex
	rigs map[string]*Rig
}

// NewRegistry builds a registry, calibrating any preconfigured rigs.
func NewRegistry(initial []RigConfig) (*Registry, error) {
	reg := &Registry{rigs: map[string]*Rig{}}
	for _, c := range initial {
		if _, err := reg.Create(c); err != nil {
			return nil, fmt.Errorf("coregistration rig %q: %w", c.ID, err)
		}
	}
	return reg, nil
}

// Create calibrates and stores a rig.
func (r *Registry) Create(cfg RigConfig) (*Rig, error) {
	rig, err := NewRig(cfg)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.rigs[rig.ID]; ok {
		return nil, ErrExists
	}
	if len(r.rigs) >= 1024 {
		return nil, errors.New("rig limit reached")
	}
	r.rigs[rig.ID] = rig
	return rig, nil
}

// Get returns a rig by ID.
func (r *Registry) Get(id string) (*Rig, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	rig, ok := r.rigs[id]
	if !ok {
		return nil, ErrNotFound
	}
	return rig, nil
}

// Delete removes a rig.
func (r *Registry) Delete(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.rigs[id]; !ok {
		return ErrNotFound
	}
	delete(r.rigs, id)
	return nil
}

// List returns all rigs sorted by ID.
func (r *Registry) List() []*Rig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Rig, 0, len(r.rigs))
	for _, rig := range r.rigs {
		out = append(out, rig)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
