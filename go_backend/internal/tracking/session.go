package tracking

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"regexp"
	"sync"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/colormap"
)

// presenceGrace is how many consecutive missed frames a confirmed track may
// coast while still counted as present (smooths over detector dropouts).
const presenceGrace = 2

var idPattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,64}$`)

// Tripwire is a virtual counting line from A to B.
//
// Direction convention: stand at A looking towards B on the displayed image;
// crossing from your right-hand side to your left-hand side counts as "in",
// the opposite as "out". For a vertical line drawn top-to-bottom this makes
// left→right motion on screen "in".
type Tripwire struct {
	ID   string `json:"id"`
	Name string `json:"name,omitempty"`
	A    Point  `json:"a"`
	B    Point  `json:"b"`
}

// Zone is a polygonal region of interest with an optional capacity.
type Zone struct {
	ID       string  `json:"id"`
	Name     string  `json:"name,omitempty"`
	Polygon  []Point `json:"polygon"`
	Capacity int     `json:"capacity,omitempty"`
}

// SessionConfig configures one camera stream.
type SessionConfig struct {
	ID          string     `json:"id"`
	Name        string     `json:"name,omitempty"`
	FrameWidth  float64    `json:"frame_width"`
	FrameHeight float64    `json:"frame_height"`
	AreaM2      float64    `json:"area_m2,omitempty"` // ground area covered by the view
	Lines       []Tripwire `json:"lines,omitempty"`
	Zones       []Zone     `json:"zones,omitempty"`
	HeatmapCols int        `json:"heatmap_cols,omitempty"`
	HeatmapRows int        `json:"heatmap_rows,omitempty"`
}

func (c *SessionConfig) normalise() error {
	if c.ID != "" && !idPattern.MatchString(c.ID) {
		return errors.New("id must match [A-Za-z0-9_.-]{1,64}")
	}
	if !(c.FrameWidth > 0 && c.FrameHeight > 0) || c.FrameWidth > 1e5 || c.FrameHeight > 1e5 {
		return errors.New("frame_width and frame_height must be positive")
	}
	if c.AreaM2 < 0 || math.IsNaN(c.AreaM2) {
		return errors.New("area_m2 must be >= 0")
	}
	if c.HeatmapCols == 0 {
		c.HeatmapCols = 32
	}
	if c.HeatmapRows == 0 {
		c.HeatmapRows = int(math.Max(1, math.Round(float64(c.HeatmapCols)*c.FrameHeight/c.FrameWidth)))
	}
	if c.HeatmapCols < 1 || c.HeatmapRows < 1 || c.HeatmapCols > 256 || c.HeatmapRows > 256 {
		return errors.New("heatmap_cols and heatmap_rows must be in [1,256]")
	}
	if len(c.Lines) > 32 || len(c.Zones) > 32 {
		return errors.New("at most 32 lines and 32 zones per session")
	}
	seen := map[string]bool{}
	for i, l := range c.Lines {
		if l.ID == "" {
			c.Lines[i].ID = fmt.Sprintf("line-%d", i+1)
		}
		if seen["l:"+c.Lines[i].ID] {
			return fmt.Errorf("duplicate line id %q", c.Lines[i].ID)
		}
		seen["l:"+c.Lines[i].ID] = true
		if l.A == l.B {
			return fmt.Errorf("line %q has identical endpoints", c.Lines[i].ID)
		}
	}
	for i, z := range c.Zones {
		if z.ID == "" {
			c.Zones[i].ID = fmt.Sprintf("zone-%d", i+1)
		}
		if seen["z:"+c.Zones[i].ID] {
			return fmt.Errorf("duplicate zone id %q", c.Zones[i].ID)
		}
		seen["z:"+c.Zones[i].ID] = true
		if err := validatePolygon(z.Polygon); err != nil {
			return fmt.Errorf("zone %q: %w", c.Zones[i].ID, err)
		}
		if z.Capacity < 0 {
			return fmt.Errorf("zone %q: capacity must be >= 0", c.Zones[i].ID)
		}
	}
	return nil
}

// FrameInput is one frame of detections for a session.
type FrameInput struct {
	Timestamp  *time.Time  `json:"timestamp,omitempty"`
	Detections []Detection `json:"detections"`
}

// TrackOut is the public view of a track.
type TrackOut struct {
	ID           uint64     `json:"id"`
	State        TrackState `json:"state"`
	BBox         BBox       `json:"bbox"`
	Center       Point      `json:"center"`
	Velocity     Point      `json:"velocity"` // px/frame
	AgeFrames    int        `json:"age_frames"`
	Hits         int        `json:"hits"`
	DwellSeconds float64    `json:"dwell_seconds"`
	Score        float64    `json:"score"`
	Label        string     `json:"label,omitempty"`
}

// LineCount is the running tally for one tripwire.
type LineCount struct {
	ID   string `json:"id"`
	Name string `json:"name,omitempty"`
	In   uint64 `json:"in"`
	Out  uint64 `json:"out"`
	Net  int64  `json:"net"`
}

// ZoneCount is the live occupancy of one zone.
type ZoneCount struct {
	ID           string  `json:"id"`
	Name         string  `json:"name,omitempty"`
	Occupancy    int     `json:"occupancy"`
	Peak         int     `json:"peak"`
	Entries      uint64  `json:"entries"`
	Capacity     int     `json:"capacity,omitempty"`
	Utilisation  float64 `json:"utilisation,omitempty"`
	OverCapacity bool    `json:"over_capacity"`
}

// Density describes how crowded the scene is.
type Density struct {
	PerM2         *float64 `json:"per_m2,omitempty"`
	CoverageRatio float64  `json:"coverage_ratio"`
	Level         string   `json:"level"`
	Basis         string   `json:"basis"`
}

// Flow is the aggregate motion of present subjects.
type Flow struct {
	MeanVX     float64 `json:"mean_vx"`
	MeanVY     float64 `json:"mean_vy"`
	Speed      float64 `json:"speed_px_per_frame"`
	HeadingDeg float64 `json:"heading_deg"`
	Direction  string  `json:"direction"`
	Coherence  float64 `json:"coherence"` // 1 = everyone moving the same way
}

// Dwell summarises time spent in view.
type Dwell struct {
	ActiveAvgSeconds    float64 `json:"active_avg_seconds"`
	ActiveMaxSeconds    float64 `json:"active_max_seconds"`
	CompletedAvgSeconds float64 `json:"completed_avg_seconds"`
	CompletedTracks     uint64  `json:"completed_tracks"`
}

// Analytics is a crowd-analysis snapshot.
type Analytics struct {
	ActiveSubjects  int         `json:"active_subjects"`
	UniqueSubjects  uint64      `json:"unique_subjects"`
	PeakSubjects    int         `json:"peak_subjects"`
	PeakAt          *time.Time  `json:"peak_at,omitempty"`
	FramesProcessed uint64      `json:"frames_processed"`
	Density         Density     `json:"density"`
	Flow            Flow        `json:"flow"`
	Dwell           Dwell       `json:"dwell"`
	Lines           []LineCount `json:"lines"`
	Zones           []ZoneCount `json:"zones"`
}

// FrameResult is returned for every processed frame.
type FrameResult struct {
	SessionID string     `json:"session_id"`
	Frame     uint64     `json:"frame"`
	Timestamp time.Time  `json:"timestamp"`
	Tracks    []TrackOut `json:"tracks"`
	Analytics Analytics  `json:"analytics"`
}

// SessionInfo is the public description of a session.
type SessionInfo struct {
	Config         SessionConfig `json:"config"`
	CreatedAt      time.Time     `json:"created_at"`
	LastActiveAt   time.Time     `json:"last_active_at"`
	Frames         uint64        `json:"frames_processed"`
	ActiveSubjects int           `json:"active_subjects"`
	Subscribers    int           `json:"subscribers"`
}

type zoneState struct {
	inside  map[uint64]bool
	entries uint64
	peak    int
}

// Session tracks subjects for one camera stream. Sessions are independent, so
// different streams are processed fully in parallel.
type Session struct {
	mu        sync.Mutex
	cfg       SessionConfig
	trackCfg  Config
	tracker   *Tracker
	created   time.Time
	lastUsed  time.Time
	frames    uint64
	peak      int
	peakAt    time.Time
	lineIn    []uint64
	lineOut   []uint64
	zones     []zoneState
	heat      []float64
	dwellSum  float64
	dwellN    uint64
	last      Analytics
	lastCount int

	subMu sync.Mutex
	subs  map[chan []byte]struct{}
}

func newSession(cfg SessionConfig, tc Config, now time.Time) *Session {
	s := &Session{
		cfg: cfg, trackCfg: tc, tracker: NewTracker(tc),
		created: now, lastUsed: now,
		lineIn: make([]uint64, len(cfg.Lines)), lineOut: make([]uint64, len(cfg.Lines)),
		zones: make([]zoneState, len(cfg.Zones)),
		heat:  make([]float64, cfg.HeatmapCols*cfg.HeatmapRows),
		subs:  map[chan []byte]struct{}{},
	}
	for i := range s.zones {
		s.zones[i].inside = map[uint64]bool{}
	}
	s.last = s.analyticsLocked(nil, now)
	return s
}

// ID returns the session identifier.
func (s *Session) ID() string { return s.cfg.ID }

// Process ingests one frame of detections and returns tracks plus analytics.
func (s *Session) Process(in FrameInput, now time.Time) (FrameResult, error) {
	if len(in.Detections) > s.trackCfg.MaxDetections {
		return FrameResult{}, fmt.Errorf("%d detections exceeds the per-frame limit of %d", len(in.Detections), s.trackCfg.MaxDetections)
	}
	dets := make([]Detection, 0, len(in.Detections))
	for i, d := range in.Detections {
		if !d.BBox.Valid() {
			return FrameResult{}, fmt.Errorf("detections[%d]: bbox must have finite values and positive w/h", i)
		}
		if d.Score == 0 {
			d.Score = 1 // detectors that do not report confidence
		}
		if d.Score < s.trackCfg.MinScore {
			continue
		}
		dets = append(dets, d)
	}
	ts := now
	if in.Timestamp != nil && !in.Timestamp.IsZero() {
		ts = *in.Timestamp
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastUsed = now
	s.frames++

	for _, t := range s.tracker.step(dets, ts) {
		s.dwellSum += t.lastSeen.Sub(t.firstSeen).Seconds()
		s.dwellN++
		for zi := range s.zones {
			delete(s.zones[zi].inside, t.id)
		}
	}

	var present []*track
	for _, t := range s.tracker.tracks {
		if t.confirmed && t.missed <= presenceGrace {
			present = append(present, t)
		}
		if t.missed > 0 {
			// Coasting: keep prev at the last measured position so a crossing
			// that happens during a detector dropout is caught on re-acquisition.
			continue
		}
		c := t.center()
		if t.confirmed && t.hasPrev {
			s.countCrossings(t.prev, c)
		}
		t.prev, t.hasPrev = c, true
	}

	s.updateZones(present)
	s.accumulateHeat(present)
	if len(present) > s.peak {
		s.peak, s.peakAt = len(present), ts
	}

	res := FrameResult{SessionID: s.cfg.ID, Frame: s.frames, Timestamp: ts, Tracks: make([]TrackOut, 0, len(present))}
	for _, t := range present {
		res.Tracks = append(res.Tracks, trackOut(t, ts))
	}
	res.Analytics = s.analyticsLocked(present, ts)
	s.last = res.Analytics
	s.lastCount = len(present)

	s.publish(res)
	return res, nil
}

func trackOut(t *track, ts time.Time) TrackOut {
	return TrackOut{
		ID: t.id, State: t.state(), BBox: roundBox(t.box()), Center: roundPoint(t.center()),
		Velocity:  roundPoint(Point{t.kf[0].v, t.kf[1].v}),
		AgeFrames: t.age + 1, Hits: t.hits,
		DwellSeconds: round3(ts.Sub(t.firstSeen).Seconds()),
		Score:        round3(t.score), Label: t.label,
	}
}

func (s *Session) countCrossings(from, to Point) {
	for i, l := range s.cfg.Lines {
		if !segmentsIntersect(from, to, l.A, l.B) {
			continue
		}
		// segmentsIntersect partitions the plane into cross >= 0 and cross < 0;
		// moving from the first half-plane to the second is "in".
		if cross(l.A, l.B, from) >= 0 {
			s.lineIn[i]++
		} else {
			s.lineOut[i]++
		}
	}
}

func (s *Session) updateZones(present []*track) {
	for zi, z := range s.cfg.Zones {
		st := &s.zones[zi]
		now := make(map[uint64]bool, len(st.inside))
		for _, t := range present {
			if pointInPolygon(t.center(), z.Polygon) {
				now[t.id] = true
				if !st.inside[t.id] {
					st.entries++
				}
			}
		}
		st.inside = now
		st.peak = max(st.peak, len(now))
	}
}

func (s *Session) accumulateHeat(present []*track) {
	cols, rows := s.cfg.HeatmapCols, s.cfg.HeatmapRows
	for _, t := range present {
		c := t.center()
		cx := int(c.X / s.cfg.FrameWidth * float64(cols))
		cy := int(c.Y / s.cfg.FrameHeight * float64(rows))
		if cx >= 0 && cy >= 0 && cx < cols && cy < rows {
			s.heat[cy*cols+cx]++
		}
	}
}

func (s *Session) analyticsLocked(present []*track, ts time.Time) Analytics {
	a := Analytics{
		ActiveSubjects:  len(present),
		UniqueSubjects:  s.tracker.UniqueSubjects(),
		PeakSubjects:    s.peak,
		FramesProcessed: s.frames,
		Lines:           make([]LineCount, len(s.cfg.Lines)),
		Zones:           make([]ZoneCount, len(s.cfg.Zones)),
	}
	if s.peak > 0 {
		p := s.peakAt
		a.PeakAt = &p
	}

	// Density: physical if the covered ground area is known, otherwise the
	// fraction of the image covered by subject boxes.
	var covered, vx, vy, speedSum float64
	var dwellSum, dwellMax float64
	for _, t := range present {
		b := t.box()
		covered += b.Area()
		vx += t.kf[0].v
		vy += t.kf[1].v
		speedSum += math.Hypot(t.kf[0].v, t.kf[1].v)
		d := ts.Sub(t.firstSeen).Seconds()
		dwellSum += d
		dwellMax = math.Max(dwellMax, d)
	}
	a.Density.CoverageRatio = round3(math.Min(1, covered/(s.cfg.FrameWidth*s.cfg.FrameHeight)))
	if s.cfg.AreaM2 > 0 {
		d := round3(float64(len(present)) / s.cfg.AreaM2)
		a.Density.PerM2, a.Density.Basis = &d, "persons_per_m2"
		a.Density.Level = densityLevel(d, []float64{0.5, 1.5, 3.0})
	} else {
		a.Density.Basis = "image_coverage"
		a.Density.Level = densityLevel(a.Density.CoverageRatio, []float64{0.10, 0.25, 0.45})
	}

	a.Flow.Direction = "none"
	if n := float64(len(present)); n > 0 {
		a.Flow.MeanVX, a.Flow.MeanVY = round3(vx/n), round3(vy/n)
		speed := math.Hypot(vx/n, vy/n)
		a.Flow.Speed = round3(speed)
		a.Flow.HeadingDeg = round3(math.Mod(math.Atan2(vy, vx)*180/math.Pi+360, 360))
		if speedSum > 0 {
			a.Flow.Coherence = round3(speed * n / speedSum)
		}
		a.Flow.Direction = compass(vx/n, vy/n)
		a.Dwell.ActiveAvgSeconds = round3(dwellSum / n)
		a.Dwell.ActiveMaxSeconds = round3(dwellMax)
	}
	a.Dwell.CompletedTracks = s.dwellN
	if s.dwellN > 0 {
		a.Dwell.CompletedAvgSeconds = round3(s.dwellSum / float64(s.dwellN))
	}

	for i, l := range s.cfg.Lines {
		a.Lines[i] = LineCount{ID: l.ID, Name: l.Name, In: s.lineIn[i], Out: s.lineOut[i],
			Net: int64(s.lineIn[i]) - int64(s.lineOut[i])}
	}
	for i, z := range s.cfg.Zones {
		st := s.zones[i]
		zc := ZoneCount{ID: z.ID, Name: z.Name, Occupancy: len(st.inside), Peak: st.peak, Entries: st.entries, Capacity: z.Capacity}
		if z.Capacity > 0 {
			zc.Utilisation = round3(float64(zc.Occupancy) / float64(z.Capacity))
			zc.OverCapacity = zc.Occupancy > z.Capacity
		}
		a.Zones[i] = zc
	}
	return a
}

func densityLevel(v float64, thresholds []float64) string {
	switch {
	case v < thresholds[0]:
		return "low"
	case v < thresholds[1]:
		return "moderate"
	case v < thresholds[2]:
		return "high"
	default:
		return "critical"
	}
}

// compass names the dominant screen direction (y grows downwards).
func compass(vx, vy float64) string {
	if math.Hypot(vx, vy) < 0.5 {
		return "stationary"
	}
	dirs := []string{"east", "south-east", "south", "south-west", "west", "north-west", "north", "north-east"}
	a := math.Mod(math.Atan2(vy, vx)*180/math.Pi+360+22.5, 360)
	return dirs[int(a/45)%8]
}

// Analytics returns the most recent analytics snapshot.
func (s *Session) Analytics() Analytics {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

// Info returns a description of the session.
func (s *Session) Info() SessionInfo {
	s.mu.Lock()
	info := SessionInfo{Config: s.cfg, CreatedAt: s.created, LastActiveAt: s.lastUsed, Frames: s.frames, ActiveSubjects: s.lastCount}
	s.mu.Unlock()
	s.subMu.Lock()
	info.Subscribers = len(s.subs)
	s.subMu.Unlock()
	return info
}

// Heatmap is the normalised spatial occupancy histogram.
type Heatmap struct {
	Cols  int       `json:"cols"`
	Rows  int       `json:"rows"`
	Max   float64   `json:"max_subject_frames"`
	Cells []float64 `json:"cells"` // row-major, normalised to [0,1]
}

// Heatmap returns the occupancy heatmap accumulated so far.
func (s *Session) Heatmap() Heatmap {
	s.mu.Lock()
	defer s.mu.Unlock()
	h := Heatmap{Cols: s.cfg.HeatmapCols, Rows: s.cfg.HeatmapRows, Cells: make([]float64, len(s.heat))}
	for _, v := range s.heat {
		h.Max = math.Max(h.Max, v)
	}
	for i, v := range s.heat {
		if h.Max > 0 {
			h.Cells[i] = round3(v / h.Max)
		}
	}
	return h
}

// HeatmapImage renders the heatmap with a palette, bilinearly upsampled to
// cell pixels per cell.
func (s *Session) HeatmapImage(palette string, cell int) (*image.RGBA, error) {
	pal, err := colormap.Get(palette)
	if err != nil {
		return nil, err
	}
	if cell < 1 {
		cell = 1
	} else if cell > 64 {
		cell = 64
	}
	h := s.Heatmap()
	img := image.NewRGBA(image.Rect(0, 0, h.Cols*cell, h.Rows*cell))
	at := func(x, y int) float64 {
		x = min(max(x, 0), h.Cols-1)
		y = min(max(y, 0), h.Rows-1)
		return h.Cells[y*h.Cols+x]
	}
	for py := 0; py < h.Rows*cell; py++ {
		fy := (float64(py)+0.5)/float64(cell) - 0.5
		y0 := int(math.Floor(fy))
		ty := fy - float64(y0)
		for px := 0; px < h.Cols*cell; px++ {
			fx := (float64(px)+0.5)/float64(cell) - 0.5
			x0 := int(math.Floor(fx))
			tx := fx - float64(x0)
			v := (at(x0, y0)*(1-tx)+at(x0+1, y0)*tx)*(1-ty) + (at(x0, y0+1)*(1-tx)+at(x0+1, y0+1)*tx)*ty
			img.SetRGBA(px, py, pal.At(math.Sqrt(v))) // sqrt: perceptual boost for sparse cells
		}
	}
	return img, nil
}

// Subscribe registers a live listener that receives every FrameResult as
// JSON. Slow listeners drop frames rather than stalling the tracker.
func (s *Session) Subscribe() (<-chan []byte, func()) {
	ch := make(chan []byte, 16)
	s.subMu.Lock()
	s.subs[ch] = struct{}{}
	s.subMu.Unlock()
	var once sync.Once
	return ch, func() {
		once.Do(func() {
			s.subMu.Lock()
			if _, ok := s.subs[ch]; ok {
				delete(s.subs, ch)
				close(ch)
			}
			s.subMu.Unlock()
		})
	}
}

func (s *Session) publish(res FrameResult) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	if len(s.subs) == 0 {
		return
	}
	b, err := json.Marshal(res)
	if err != nil {
		return
	}
	for ch := range s.subs {
		select {
		case ch <- b:
		default: // listener is behind; drop this frame for it
		}
	}
}

func (s *Session) closeSubscribers() {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	for ch := range s.subs {
		delete(s.subs, ch)
		close(ch)
	}
}

func (s *Session) idleSince() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastUsed
}

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }

func roundPoint(p Point) Point { return Point{round3(p.X), round3(p.Y)} }

func roundBox(b BBox) BBox { return BBox{round3(b.X), round3(b.Y), round3(b.W), round3(b.H)} }
