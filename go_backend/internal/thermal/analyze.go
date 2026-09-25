package thermal

import (
	"math"
	"sort"
	"strconv"
)

// Screening status values.
const (
	StatusNormal        = "normal"
	StatusElevated      = "elevated"
	StatusFever         = "fever"
	StatusIndeterminate = "indeterminate"
)

// Point is an integer pixel coordinate.
type Point struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// SkinStats summarises the skin pixels of one subject (°C).
type SkinStats struct {
	Pixels int     `json:"pixels"`
	MinC   float64 `json:"min_c"`
	MaxC   float64 `json:"max_c"`
	MeanC  float64 `json:"mean_c"`
	StdC   float64 `json:"std_c"`
	P50C   float64 `json:"p50_c"`
	P90C   float64 `json:"p90_c"`
}

// Hotspot is the inner-canthus measurement site.
type Hotspot struct {
	Point
	TempC float64 `json:"temp_c"`
}

// LivenessCheck is one presentation-attack detection test.
type LivenessCheck struct {
	Name   string  `json:"name"`
	Passed bool    `json:"passed"`
	Value  float64 `json:"value"`
	Limit  string  `json:"limit"`
}

// Liveness is the thermal presentation-attack verdict.
type Liveness struct {
	Live   bool            `json:"live"`
	Score  float64         `json:"score"`
	Checks []LivenessCheck `json:"checks"`
}

// Subject is the analysis of one thermal face.
type Subject struct {
	Index        int       `json:"index"`
	ROI          Rect      `json:"roi"`
	Skin         SkinStats `json:"skin"`
	Canthus      Hotspot   `json:"canthus"`
	CoreEstimate float64   `json:"core_estimate_c"`
	Status       string    `json:"status"`
	Liveness     Liveness  `json:"liveness"`
}

// FrameSummary describes the whole frame.
type FrameSummary struct {
	Width      int      `json:"width"`
	Height     int      `json:"height"`
	MinC       float64  `json:"min_c"`
	MaxC       float64  `json:"max_c"`
	MeanC      float64  `json:"mean_c"`
	AmbientC   *float64 `json:"ambient_c"`
	AmbientSrc string   `json:"ambient_source"`
	Emissivity float64  `json:"emissivity"`
	ReflectedC float64  `json:"reflected_c"`
}

// Result is the full analysis output.
type Result struct {
	Frame    FrameSummary   `json:"frame"`
	Subjects []Subject      `json:"subjects"`
	Summary  map[string]int `json:"summary"`
}

// Analyze segments and screens every subject in f. If rois is non-empty the
// given boxes are used instead of automatic segmentation.
func Analyze(f *Frame, rois []Rect, cfg Config) Result {
	res := Result{Frame: summarise(f, cfg), Subjects: []Subject{}, Summary: map[string]int{}}

	var regions []region
	if len(rois) > 0 {
		regions = roiRegions(f, rois, cfg)
	} else {
		regions = segment(f, cfg)
	}

	for i, r := range regions {
		if i >= cfg.MaxSubjects {
			break
		}
		s := analyseRegion(f, r, cfg, res.Frame.AmbientC)
		s.Index = i
		res.Subjects = append(res.Subjects, s)
		res.Summary[s.Status]++
		if s.Liveness.Live {
			res.Summary["live"]++
		} else {
			res.Summary["spoof_suspected"]++
		}
	}
	res.Summary["subjects"] = len(res.Subjects)
	return res
}

func summarise(f *Frame, cfg Config) FrameSummary {
	s := FrameSummary{Width: f.Width, Height: f.Height, Emissivity: f.Emissivity, ReflectedC: f.ReflectedC}
	s.MinC, s.MaxC = math.Inf(1), math.Inf(-1)
	var sum float64
	background := make([]float64, 0, len(f.Temps)/2)
	for _, t := range f.Temps {
		sum += t
		s.MinC = math.Min(s.MinC, t)
		s.MaxC = math.Max(s.MaxC, t)
		if t < cfg.SkinMinC {
			background = append(background, t)
		}
	}
	s.MeanC = round2(sum / float64(len(f.Temps)))
	s.MinC, s.MaxC = round2(s.MinC), round2(s.MaxC)

	switch {
	case f.AmbientC != nil:
		a := *f.AmbientC
		s.AmbientC, s.AmbientSrc = &a, "provided"
	case len(background) > 0:
		a := round2(selectKth(background, len(background)/2))
		s.AmbientC, s.AmbientSrc = &a, "estimated_background_median"
	default:
		s.AmbientSrc = "unknown"
	}
	return s
}

// region is a set of pixel indices belonging to one subject plus its bbox.
// Segmented regions share one label image (labels[i] == id); caller-supplied
// ROIs may overlap, so each carries its own membership mask.
type region struct {
	box    Rect
	labels []int32
	id     int32
	member []bool
	pixels []int
}

func (r *region) contains(i int) bool {
	if r.labels != nil {
		return r.labels[i] == r.id
	}
	return r.member[i]
}

// segment finds 4-connected components of skin-temperature pixels.
func segment(f *Frame, cfg Config) []region {
	n := len(f.Temps)
	label := make([]int32, n)
	var regions []region
	stack := make([]int, 0, 1024)
	var next int32

	for start := 0; start < n; start++ {
		if label[start] != 0 || !isSkin(f.Temps[start], cfg) {
			continue
		}
		next++
		label[start] = next
		stack = append(stack[:0], start)
		var pixels []int
		minX, minY, maxX, maxY := f.Width, f.Height, -1, -1
		for len(stack) > 0 {
			i := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			pixels = append(pixels, i)
			x, y := i%f.Width, i/f.Width
			minX, maxX = min(minX, x), max(maxX, x)
			minY, maxY = min(minY, y), max(maxY, y)
			for _, j := range neighbours4(x, y, f.Width, f.Height) {
				if j >= 0 && label[j] == 0 && isSkin(f.Temps[j], cfg) {
					label[j] = next
					stack = append(stack, j)
				}
			}
		}
		if len(pixels) < cfg.MinSubjectPixels {
			continue
		}
		regions = append(regions, region{
			box:    Rect{X: minX, Y: minY, W: maxX - minX + 1, H: maxY - minY + 1},
			labels: label, id: next, pixels: pixels,
		})
	}
	// Largest subjects first, then left-to-right for a stable order.
	sort.SliceStable(regions, func(a, b int) bool {
		if len(regions[a].pixels) != len(regions[b].pixels) {
			return len(regions[a].pixels) > len(regions[b].pixels)
		}
		return regions[a].box.X < regions[b].box.X
	})
	if len(regions) > cfg.MaxSubjects {
		regions = regions[:cfg.MaxSubjects]
	}
	sort.SliceStable(regions, func(a, b int) bool { return regions[a].box.X < regions[b].box.X })
	return regions
}

func roiRegions(f *Frame, rois []Rect, cfg Config) []region {
	var out []region
	for _, r := range rois {
		x0, y0 := max(r.X, 0), max(r.Y, 0)
		x1, y1 := min(r.X+r.W, f.Width), min(r.Y+r.H, f.Height)
		if x1 <= x0 || y1 <= y0 {
			continue
		}
		member := make([]bool, len(f.Temps))
		var pixels []int
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				i := y*f.Width + x
				if isSkin(f.Temps[i], cfg) {
					member[i] = true
					pixels = append(pixels, i)
				}
			}
		}
		out = append(out, region{box: Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}, member: member, pixels: pixels})
	}
	return out
}

func analyseRegion(f *Frame, r region, cfg Config, ambient *float64) Subject {
	s := Subject{ROI: r.box, Status: StatusIndeterminate}
	if len(r.pixels) == 0 {
		s.Liveness = Liveness{Checks: []LivenessCheck{{Name: "skin_present", Value: 0, Limit: ">0 skin pixels"}}}
		return s
	}

	vals := make([]float64, len(r.pixels))
	var sum, sq float64
	for k, i := range r.pixels {
		t := f.Temps[i]
		vals[k] = t
		sum += t
	}
	mean := sum / float64(len(vals))
	for _, t := range vals {
		sq += (t - mean) * (t - mean)
	}
	sort.Float64s(vals)
	s.Skin = SkinStats{
		Pixels: len(vals),
		MinC:   round2(vals[0]),
		MaxC:   round2(vals[len(vals)-1]),
		MeanC:  round2(mean),
		StdC:   round2(math.Sqrt(sq / float64(len(vals)))),
		P50C:   round2(quantile(vals, 0.50)),
		P90C:   round2(quantile(vals, 0.90)),
	}

	s.Canthus = findCanthus(f, r)
	s.CoreEstimate = round2(s.Canthus.TempC + cfg.CoreOffsetC)
	switch {
	case s.Canthus.TempC < cfg.IndeterminateBelowC:
		s.Status = StatusIndeterminate
	case s.CoreEstimate >= cfg.FeverThresholdC:
		s.Status = StatusFever
	case s.CoreEstimate >= cfg.ElevatedThresholdC:
		s.Status = StatusElevated
	default:
		s.Status = StatusNormal
	}

	s.Liveness = liveness(vals, mean, s.Canthus.TempC, ambient, cfg)
	return s
}

// findCanthus locates the inner-canthus hotspot: the warmest 3×3-averaged
// skin location inside the periorbital band of the face (15–60 % of the ROI
// height). The inner canthus is perfused by the internal carotid artery and is
// the site recommended by ISO/TR 13154 and IEC 80601-2-59 for screening.
func findCanthus(f *Frame, r region) Hotspot {
	y0 := r.box.Y + int(0.15*float64(r.box.H))
	y1 := r.box.Y + int(math.Ceil(0.60*float64(r.box.H)))
	best := Hotspot{TempC: math.Inf(-1)}
	for y := y0; y < y1 && y < f.Height; y++ {
		for x := r.box.X; x < r.box.X+r.box.W && x < f.Width; x++ {
			if !r.contains(y*f.Width + x) {
				continue
			}
			var sum float64
			var cnt int
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := x+dx, y+dy
					if nx < 0 || ny < 0 || nx >= f.Width || ny >= f.Height {
						continue
					}
					if j := ny*f.Width + nx; r.contains(j) {
						sum += f.Temps[j]
						cnt++
					}
				}
			}
			if cnt < 5 {
				continue
			}
			if m := sum / float64(cnt); m > best.TempC {
				best = Hotspot{Point: Point{x, y}, TempC: m}
			}
		}
	}
	if math.IsInf(best.TempC, -1) {
		// Degenerate region: fall back to the hottest member pixel.
		for _, i := range r.pixels {
			if t := f.Temps[i]; t > best.TempC {
				best = Hotspot{Point: Point{i % f.Width, i / f.Width}, TempC: t}
			}
		}
	}
	best.TempC = round2(best.TempC)
	return best
}

// liveness applies thermal presentation-attack detection. Printed photos,
// screens and masks either sit near ambient temperature or, when heated, show
// an unnaturally flat temperature field without the periorbital hotspot that
// arterial perfusion produces on a real face.
func liveness(sorted []float64, mean, canthus float64, ambient *float64, cfg Config) Liveness {
	span := quantile(sorted, 0.98) - quantile(sorted, 0.02)
	median := quantile(sorted, 0.50)
	checks := []LivenessCheck{
		{Name: "physiological_skin_range", Value: round2(mean), Limit: "31.0–38.5 °C",
			Passed: mean >= 31 && mean <= 38.5},
		{Name: "facial_thermal_gradient", Value: round2(span), Limit: ">= " + fmt2(cfg.LivenessMinSpanC) + " °C (p98−p2)",
			Passed: span >= cfg.LivenessMinSpanC},
		{Name: "periorbital_hotspot", Value: round2(canthus - median), Limit: ">= 0.30 °C above facial median",
			Passed: canthus-median >= 0.30},
	}
	if ambient != nil {
		d := mean - *ambient
		checks = append(checks, LivenessCheck{Name: "ambient_contrast", Value: round2(d),
			Limit: ">= " + fmt2(cfg.LivenessMinAmbientDeltaC) + " °C above ambient", Passed: d >= cfg.LivenessMinAmbientDeltaC})
	}
	passed := 0
	for _, c := range checks {
		if c.Passed {
			passed++
		}
	}
	return Liveness{
		Live:   passed == len(checks),
		Score:  round2(float64(passed) / float64(len(checks))),
		Checks: checks,
	}
}

func isSkin(t float64, cfg Config) bool { return t >= cfg.SkinMinC && t <= cfg.SkinMaxC }

func neighbours4(x, y, w, h int) [4]int {
	n := [4]int{-1, -1, -1, -1}
	if x > 0 {
		n[0] = y*w + x - 1
	}
	if x < w-1 {
		n[1] = y*w + x + 1
	}
	if y > 0 {
		n[2] = (y-1)*w + x
	}
	if y < h-1 {
		n[3] = (y+1)*w + x
	}
	return n
}

// quantile assumes sorted input and uses linear interpolation.
func quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	pos := q * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	return sorted[lo] + (pos-float64(lo))*(sorted[hi]-sorted[lo])
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }

func fmt2(v float64) string { return strconv.FormatFloat(round2(v), 'f', -1, 64) }

// selectKth returns the k-th smallest element (0-based) in expected O(n),
// reordering a in place (Hoare's selection with median-of-three pivots).
func selectKth(a []float64, k int) float64 {
	lo, hi := 0, len(a)-1
	for lo < hi {
		mid := lo + (hi-lo)/2
		if a[mid] < a[lo] {
			a[mid], a[lo] = a[lo], a[mid]
		}
		if a[hi] < a[lo] {
			a[hi], a[lo] = a[lo], a[hi]
		}
		if a[hi] < a[mid] {
			a[hi], a[mid] = a[mid], a[hi]
		}
		pivot := a[mid]
		i, j := lo, hi
		for i <= j {
			for a[i] < pivot {
				i++
			}
			for a[j] > pivot {
				j--
			}
			if i <= j {
				a[i], a[j] = a[j], a[i]
				i++
				j--
			}
		}
		switch {
		case k <= j:
			hi = j
		case k >= i:
			lo = i
		default:
			return a[k]
		}
	}
	return a[k]
}
