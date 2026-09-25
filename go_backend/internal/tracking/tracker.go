package tracking

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

// MaxEmbeddingDim bounds appearance-embedding length.
const MaxEmbeddingDim = 2048

// Config holds tracker and session-manager parameters.
type Config struct {
	IoUThreshold     float64       `yaml:"iou_threshold" json:"iou_threshold"`
	MaxAge           int           `yaml:"max_age" json:"max_age"`
	MinHits          int           `yaml:"min_hits" json:"min_hits"`
	MinScore         float64       `yaml:"min_score" json:"min_score"`
	MaxDetections    int           `yaml:"max_detections" json:"max_detections"`
	MaxSessions      int           `yaml:"max_sessions" json:"max_sessions"`
	SessionTTL       time.Duration `yaml:"session_ttl" json:"session_ttl"`
	ProcessNoise     float64       `yaml:"process_noise" json:"process_noise"`
	MeasurementNoise float64       `yaml:"measurement_noise" json:"measurement_noise"`

	// Appearance re-identification (DeepSORT-style). Active for detections
	// that carry an embedding; otherwise the tracker is pure SORT.
	ReIDMaxAge          int     `yaml:"reid_max_age" json:"reid_max_age"`                 // frames a track with appearance survives unseen
	AppearanceThreshold float64 `yaml:"appearance_threshold" json:"appearance_threshold"` // max cosine distance for a match
	FeatureBudget       int     `yaml:"feature_budget" json:"feature_budget"`             // gallery size per track
	GateChi2            float64 `yaml:"gate_chi2" json:"gate_chi2"`                       // Mahalanobis motion gate (2 dof)
}

// DefaultConfig returns SORT/DeepSORT defaults tuned for 10–30 FPS video.
func DefaultConfig() Config {
	return Config{
		IoUThreshold:        0.25,
		MaxAge:              15,
		MinHits:             3,
		MinScore:            0.3,
		MaxDetections:       512,
		MaxSessions:         256,
		SessionTTL:          10 * time.Minute,
		ProcessNoise:        1.0,
		MeasurementNoise:    4.0,
		ReIDMaxAge:          90,
		AppearanceThreshold: 0.25,
		FeatureBudget:       16,
		GateChi2:            9.21, // 99 % quantile of χ² with 2 degrees of freedom
	}
}

// Validate reports configuration errors.
func (c Config) Validate() error {
	switch {
	case c.IoUThreshold <= 0 || c.IoUThreshold >= 1:
		return fmt.Errorf("tracking.iou_threshold must be in (0,1), got %v", c.IoUThreshold)
	case c.MaxAge < 0:
		return errors.New("tracking.max_age must be >= 0")
	case c.MinHits < 1:
		return errors.New("tracking.min_hits must be >= 1")
	case c.MaxDetections < 1:
		return errors.New("tracking.max_detections must be >= 1")
	case c.MaxSessions < 1:
		return errors.New("tracking.max_sessions must be >= 1")
	case c.ProcessNoise <= 0 || c.MeasurementNoise <= 0:
		return errors.New("tracking noise parameters must be positive")
	case c.ReIDMaxAge < c.MaxAge:
		return errors.New("tracking.reid_max_age must be >= max_age")
	case c.AppearanceThreshold <= 0 || c.AppearanceThreshold >= 2:
		return errors.New("tracking.appearance_threshold must be in (0,2)")
	case c.FeatureBudget < 1 || c.FeatureBudget > 256:
		return errors.New("tracking.feature_budget must be in [1,256]")
	case c.GateChi2 <= 0:
		return errors.New("tracking.gate_chi2 must be positive")
	}
	return nil
}

// Detection is one detector output for a frame.
type Detection struct {
	BBox  BBox    `json:"bbox"`
	Score float64 `json:"score"`
	Label string  `json:"label,omitempty"`
	// Embedding is an optional appearance descriptor (e.g. an OSNet / FastReID
	// person embedding or an ArcFace face embedding). It is L2-normalised on
	// ingest; all detections of a session must use the same dimension.
	Embedding []float64 `json:"embedding,omitempty"`
}

// TrackState is the lifecycle state of a track.
type TrackState string

const (
	// StateTentative tracks have not yet accumulated MinHits consecutive hits.
	StateTentative TrackState = "tentative"
	// StateConfirmed tracks were matched on the current frame.
	StateConfirmed TrackState = "confirmed"
	// StateLost tracks are confirmed but currently coasting on prediction.
	StateLost TrackState = "lost"
)

type track struct {
	id        uint64 // public ID, assigned on confirmation (0 while tentative)
	kf        [4]kf1 // cx, cy, w, h
	hits      int
	hitStreak int
	age       int
	missed    int
	confirmed bool
	label     string
	score     float64
	firstSeen time.Time
	lastSeen  time.Time
	prev      Point // filtered centre at the last measured frame
	hasPrev   bool

	gallery      [][]float64 // ring buffer of normalised embeddings
	galleryNext  int
	reidentified int // times this identity was recovered by appearance after a long gap
}

func (t *track) box() BBox {
	w, h := math.Max(t.kf[2].x, 1), math.Max(t.kf[3].x, 1)
	return BBox{X: t.kf[0].x - w/2, Y: t.kf[1].x - h/2, W: w, H: h}
}

func (t *track) center() Point { return Point{t.kf[0].x, t.kf[1].x} }

func (t *track) state() TrackState {
	switch {
	case !t.confirmed:
		return StateTentative
	case t.missed > 0:
		return StateLost
	default:
		return StateConfirmed
	}
}

func (t *track) addFeature(f []float64, budget int) {
	if f == nil {
		return
	}
	if len(t.gallery) < budget {
		t.gallery = append(t.gallery, f)
		return
	}
	t.gallery[t.galleryNext] = f
	t.galleryNext = (t.galleryNext + 1) % budget
}

// appearanceDistance is the smallest cosine distance between f and the
// track's gallery (DeepSORT's nearest-neighbour metric).
func (t *track) appearanceDistance(f []float64) float64 {
	best := math.Inf(1)
	for _, g := range t.gallery {
		var dot float64
		for i := range g {
			dot += g[i] * f[i]
		}
		best = math.Min(best, 1-dot)
	}
	return best
}

// mahalanobis2 is the squared Mahalanobis distance between the predicted
// track centre and p. The variance includes the Kalman position uncertainty,
// measurement noise and a detector-localisation term proportional to box
// height, so the gate widens naturally while a track coasts through an
// occlusion.
func (t *track) mahalanobis2(p Point, measVar float64) float64 {
	h := math.Max(t.kf[3].x, 1)
	loc := 0.1 * h * 0.1 * h
	dx, dy := p.X-t.kf[0].x, p.Y-t.kf[1].x
	return dx*dx/(t.kf[0].p00+measVar+loc) + dy*dy/(t.kf[1].p00+measVar+loc)
}

// Tracker associates detections across frames. It is not safe for concurrent
// use; Session serialises access.
type Tracker struct {
	cfg    Config
	tracks []*track
	nextID uint64
	frame  int
}

// NewTracker returns a tracker with the given configuration.
func NewTracker(cfg Config) *Tracker { return &Tracker{cfg: cfg} }

// UniqueSubjects is the number of confirmed identities ever issued.
func (tr *Tracker) UniqueSubjects() uint64 { return tr.nextID }

type stepResult struct {
	removed    []*track // confirmed tracks retired this frame
	recoveries int      // identities recovered by appearance after > MaxAge frames unseen
}

func (tr *Tracker) maxAgeOf(t *track) int {
	if len(t.gallery) > 0 {
		return tr.cfg.ReIDMaxAge
	}
	return tr.cfg.MaxAge
}

// step runs one predict → associate → update cycle. dets must already carry
// normalised embeddings (or none).
func (tr *Tracker) step(dets []Detection, ts time.Time, appearanceThreshold float64) stepResult {
	tr.frame++
	q, r := tr.cfg.ProcessNoise, tr.cfg.MeasurementNoise
	var res stepResult

	for _, t := range tr.tracks {
		for i := range t.kf {
			t.kf[i].predict(1, q)
		}
		t.age++
	}

	matches := make(map[int]int, len(tr.tracks)) // track index → detection index
	detUsed := make([]bool, len(dets))

	// Stage A — appearance matching cascade: confirmed tracks with a gallery
	// are matched by cosine distance under a Mahalanobis motion gate, most
	// recently seen first so fresh tracks win ties over long-lost ones.
	levels := map[int][]int{}
	for ti, t := range tr.tracks {
		if t.confirmed && len(t.gallery) > 0 {
			levels[t.missed] = append(levels[t.missed], ti)
		}
	}
	ages := make([]int, 0, len(levels))
	for a := range levels {
		ages = append(ages, a)
	}
	sort.Ints(ages)
	for _, age := range ages {
		cands := levels[age]
		var edges []edge
		for li, ti := range cands {
			t := tr.tracks[ti]
			for di, d := range dets {
				if detUsed[di] || d.Embedding == nil || len(d.Embedding) != len(t.gallery[0]) {
					continue
				}
				if t.mahalanobis2(d.BBox.Center(), r) > tr.cfg.GateChi2 {
					continue
				}
				if dist := t.appearanceDistance(d.Embedding); dist <= appearanceThreshold {
					edges = append(edges, edge{a: li, b: di, cost: dist})
				}
			}
		}
		for li, di := range solveSparse(len(cands), len(dets), edges) {
			matches[cands[li]] = di
			detUsed[di] = true
		}
	}

	// Stage B — IoU association (SORT) for everything still unmatched and
	// seen recently enough for its predicted box to be meaningful.
	var restT, restD []int
	for ti, t := range tr.tracks {
		if _, ok := matches[ti]; !ok && t.missed < tr.cfg.MaxAge+1 {
			restT = append(restT, ti)
		}
	}
	for di := range dets {
		if !detUsed[di] {
			restD = append(restD, di)
		}
	}
	var edges []edge
	for a, ti := range restT {
		b := tr.tracks[ti].box()
		for bi, di := range restD {
			if iou := IoU(b, dets[di].BBox); iou >= tr.cfg.IoUThreshold {
				edges = append(edges, edge{a: a, b: bi, cost: 1 - iou})
			}
		}
	}
	for a, bi := range solveSparse(len(restT), len(restD), edges) {
		matches[restT[a]] = restD[bi]
		detUsed[restD[bi]] = true
	}

	// Update matched tracks.
	for ti, di := range matches {
		t, d := tr.tracks[ti], dets[di]
		if t.missed > tr.cfg.MaxAge {
			// Pure SORT would have deleted this track: appearance saved the ID.
			t.reidentified++
			res.recoveries++
		}
		c := d.BBox.Center()
		// Size measurements are noisier than centre measurements.
		t.kf[0].update(c.X, r)
		t.kf[1].update(c.Y, r)
		t.kf[2].update(d.BBox.W, 4*r)
		t.kf[3].update(d.BBox.H, 4*r)
		t.hits++
		t.hitStreak++
		t.missed = 0
		t.score, t.label, t.lastSeen = d.Score, d.Label, ts
		t.addFeature(d.Embedding, tr.cfg.FeatureBudget)
		if !t.confirmed && t.hitStreak >= tr.cfg.MinHits {
			t.confirmed = true
			tr.nextID++
			t.id = tr.nextID
		}
	}
	for ti, t := range tr.tracks {
		if _, ok := matches[ti]; !ok {
			t.missed++
			t.hitStreak = 0
		}
	}

	// Unmatched detections start tentative tracks.
	for di, d := range dets {
		if detUsed[di] {
			continue
		}
		c := d.BBox.Center()
		t := &track{
			kf:        [4]kf1{newKF1(c.X, r), newKF1(c.Y, r), newKF1(d.BBox.W, 4*r), newKF1(d.BBox.H, 4*r)},
			hits:      1,
			hitStreak: 1,
			label:     d.Label,
			score:     d.Score,
			firstSeen: ts,
			lastSeen:  ts,
		}
		t.addFeature(d.Embedding, tr.cfg.FeatureBudget)
		if tr.cfg.MinHits <= 1 {
			t.confirmed = true
			tr.nextID++
			t.id = tr.nextID
		}
		tr.tracks = append(tr.tracks, t)
	}

	// Retire dead tracks: confirmed ones after their max age (longer when an
	// appearance gallery allows re-identification), tentative ones on their
	// first miss (they never earned an identity).
	kept := tr.tracks[:0]
	for _, t := range tr.tracks {
		if t.missed > tr.maxAgeOf(t) || (!t.confirmed && t.missed > 0) {
			if t.confirmed {
				res.removed = append(res.removed, t)
			}
			continue
		}
		kept = append(kept, t)
	}
	for i := len(kept); i < len(tr.tracks); i++ {
		tr.tracks[i] = nil
	}
	tr.tracks = kept
	return res
}

// infeasible is the assignment cost of a pair without an edge. It is large
// enough that the solver first maximises the number of feasible matches.
const infeasible = 1e3

type edge struct {
	a, b int
	cost float64
}

// solveSparse finds a minimum-cost matching using only the given feasible
// edges between na left and nb right nodes. The bipartite graph is split into
// connected components (union-find) and each component is solved optimally
// with the Hungarian algorithm, so crowds become many tiny problems instead
// of one O(n³) problem. It returns left index → right index.
func solveSparse(na, nb int, edges []edge) map[int]int {
	out := map[int]int{}
	if len(edges) == 0 {
		return out
	}
	parent := make([]int, na+nb)
	for i := range parent {
		parent[i] = i
	}
	find := func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	for _, e := range edges {
		parent[find(e.a)] = find(na + e.b)
	}

	type component struct {
		left, right []int
		cost        [][]float64
	}
	comps := map[int]*component{}
	for _, e := range edges {
		if r := find(e.a); comps[r] == nil {
			comps[r] = &component{}
		}
	}
	local := make([]int, na+nb)
	for n := 0; n < na+nb; n++ {
		c := comps[find(n)]
		if c == nil {
			continue
		}
		if n < na {
			local[n] = len(c.left)
			c.left = append(c.left, n)
		} else {
			local[n] = len(c.right)
			c.right = append(c.right, n-na)
		}
	}
	for _, c := range comps {
		c.cost = make([][]float64, len(c.left))
		for i := range c.cost {
			row := make([]float64, len(c.right))
			for k := range row {
				row[k] = infeasible
			}
			c.cost[i] = row
		}
	}
	for _, e := range edges {
		c := comps[find(e.a)]
		if cur := &c.cost[local[e.a]][local[na+e.b]]; e.cost < *cur {
			*cur = e.cost
		}
	}
	for _, c := range comps {
		for i, k := range hungarian(c.cost) {
			if k >= 0 && c.cost[i][k] < infeasible {
				out[c.left[i]] = c.right[k]
			}
		}
	}
	return out
}
