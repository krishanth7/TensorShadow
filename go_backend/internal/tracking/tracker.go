package tracking

import (
	"errors"
	"fmt"
	"math"
	"time"
)

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
}

// DefaultConfig returns SORT-like defaults tuned for 10–30 FPS video.
func DefaultConfig() Config {
	return Config{
		IoUThreshold:     0.25,
		MaxAge:           15,
		MinHits:          3,
		MinScore:         0.3,
		MaxDetections:    512,
		MaxSessions:      256,
		SessionTTL:       10 * time.Minute,
		ProcessNoise:     1.0,
		MeasurementNoise: 4.0,
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
	}
	return nil
}

// Detection is one detector output for a frame.
type Detection struct {
	BBox  BBox    `json:"bbox"`
	Score float64 `json:"score"`
	Label string  `json:"label,omitempty"`
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

// step runs one predict/associate/update cycle and returns the tracks removed
// in this step (for dwell-time accounting).
func (tr *Tracker) step(dets []Detection, ts time.Time) (removed []*track) {
	tr.frame++
	q, r := tr.cfg.ProcessNoise, tr.cfg.MeasurementNoise

	for _, t := range tr.tracks {
		for i := range t.kf {
			t.kf[i].predict(1, q)
		}
		t.age++
	}

	matches, unmatchedDets := tr.associate(dets)

	for ti, di := range matches {
		t, d := tr.tracks[ti], dets[di]
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

	for _, di := range unmatchedDets {
		d := dets[di]
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
		if tr.cfg.MinHits <= 1 {
			t.confirmed = true
			tr.nextID++
			t.id = tr.nextID
		}
		tr.tracks = append(tr.tracks, t)
	}

	// Drop dead tracks: confirmed ones after MaxAge missed frames, tentative
	// ones immediately on their first miss (they never earned an identity).
	kept := tr.tracks[:0]
	for _, t := range tr.tracks {
		if t.missed > tr.cfg.MaxAge || (!t.confirmed && t.missed > 0) {
			if t.confirmed {
				removed = append(removed, t)
			}
			continue
		}
		kept = append(kept, t)
	}
	for i := len(kept); i < len(tr.tracks); i++ {
		tr.tracks[i] = nil
	}
	tr.tracks = kept
	return removed
}

// infeasible is the assignment cost of a pair below the IoU gate. It is large
// enough that the solver first maximises the number of feasible matches.
const infeasible = 1e3

// associate matches existing tracks to detections. Pairs below the IoU
// threshold are gated out, the remaining bipartite graph is split into
// connected components, and each component is solved optimally with the
// Hungarian algorithm on (1 − IoU). In crowds almost all pairs are gated, so
// this turns one O(n³) problem into many tiny ones.
func (tr *Tracker) associate(dets []Detection) (map[int]int, []int) {
	nt, nd := len(tr.tracks), len(dets)
	matches := make(map[int]int, nt)
	if nt == 0 || nd == 0 {
		un := make([]int, nd)
		for i := range un {
			un[i] = i
		}
		return matches, un
	}

	boxes := make([]BBox, nt)
	for i, t := range tr.tracks {
		boxes[i] = t.box()
	}

	// Union-find over nodes [0,nt) = tracks and [nt,nt+nd) = detections.
	parent := make([]int, nt+nd)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	type edge struct {
		t, d int
		iou  float64
	}
	var edges []edge
	for i := 0; i < nt; i++ {
		for j := 0; j < nd; j++ {
			if iou := IoU(boxes[i], dets[j].BBox); iou >= tr.cfg.IoUThreshold {
				edges = append(edges, edge{i, j, iou})
				parent[find(i)] = find(nt + j)
			}
		}
	}

	// Group gated pairs by component; tracks and detections without any
	// feasible pair stay unmatched without entering the solver.
	type component struct {
		tracks, dets []int
		cost         [][]float64
	}
	comps := map[int]*component{}
	for _, e := range edges {
		if r := find(e.t); comps[r] == nil {
			comps[r] = &component{}
		}
	}
	local := make([]int, nt+nd) // node index within its component
	for n := 0; n < nt+nd; n++ {
		c := comps[find(n)]
		if c == nil {
			continue
		}
		if n < nt {
			local[n] = len(c.tracks)
			c.tracks = append(c.tracks, n)
		} else {
			local[n] = len(c.dets)
			c.dets = append(c.dets, n-nt)
		}
	}
	for _, c := range comps {
		c.cost = make([][]float64, len(c.tracks))
		for r := range c.cost {
			row := make([]float64, len(c.dets))
			for k := range row {
				row[k] = infeasible
			}
			c.cost[r] = row
		}
	}
	for _, e := range edges {
		comps[find(e.t)].cost[local[e.t]][local[nt+e.d]] = 1 - e.iou
	}

	matchedDet := make([]bool, nd)
	for _, c := range comps {
		for r, k := range hungarian(c.cost) {
			if k >= 0 && c.cost[r][k] < infeasible {
				matches[c.tracks[r]] = c.dets[k]
				matchedDet[c.dets[k]] = true
			}
		}
	}
	var unmatched []int
	for j, ok := range matchedDet {
		if !ok {
			unmatched = append(unmatched, j)
		}
	}
	return matches, unmatched
}
