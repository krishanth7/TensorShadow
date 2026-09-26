package tracking

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestIoU(t *testing.T) {
	a := BBox{0, 0, 10, 10}
	cases := []struct {
		b    BBox
		want float64
	}{
		{BBox{0, 0, 10, 10}, 1},
		{BBox{5, 0, 10, 10}, 50.0 / 150},
		{BBox{10, 0, 10, 10}, 0},
		{BBox{2, 2, 5, 5}, 25.0 / 100},
	}
	for _, c := range cases {
		if got := IoU(a, c.b); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("IoU(%v,%v) = %v, want %v", a, c.b, got, c.want)
		}
	}
}

func bruteForce(cost [][]float64) float64 {
	n, m := len(cost), len(cost[0])
	best := math.Inf(1)
	used := make([]bool, m)
	var rec func(i int, acc float64, assigned int)
	rec = func(i int, acc float64, assigned int) {
		if i == n {
			if assigned == min(n, m) {
				best = math.Min(best, acc)
			}
			return
		}
		if n > m { // row may stay unassigned
			rec(i+1, acc, assigned)
		}
		for j := 0; j < m; j++ {
			if !used[j] {
				used[j] = true
				rec(i+1, acc+cost[i][j], assigned+1)
				used[j] = false
			}
		}
	}
	rec(0, 0, 0)
	return best
}

func TestHungarianMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 300; trial++ {
		n, m := 1+rng.Intn(6), 1+rng.Intn(6)
		cost := make([][]float64, n)
		for i := range cost {
			cost[i] = make([]float64, m)
			for j := range cost[i] {
				cost[i][j] = rng.Float64()
			}
		}
		assign := hungarian(cost)
		var total float64
		seen := map[int]bool{}
		count := 0
		for i, j := range assign {
			if j < 0 {
				continue
			}
			if seen[j] {
				t.Fatalf("column %d assigned twice", j)
			}
			seen[j] = true
			total += cost[i][j]
			count++
		}
		if count != min(n, m) {
			t.Fatalf("%dx%d: %d assignments", n, m, count)
		}
		if want := bruteForce(cost); math.Abs(total-want) > 1e-9 {
			t.Fatalf("%dx%d: hungarian %v, optimum %v", n, m, total, want)
		}
	}
}

func TestSegmentsAndPolygons(t *testing.T) {
	if !segmentsIntersect(Point{0, 5}, Point{10, 5}, Point{5, 0}, Point{5, 10}) {
		t.Fatal("crossing segments not detected")
	}
	if segmentsIntersect(Point{0, 5}, Point{4, 5}, Point{5, 0}, Point{5, 10}) {
		t.Fatal("non-crossing segments detected")
	}
	sq := []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	if !pointInPolygon(Point{5, 5}, sq) || pointInPolygon(Point{15, 5}, sq) {
		t.Fatal("pointInPolygon wrong")
	}
	if validatePolygon([]Point{{0, 0}, {1, 1}, {2, 2}}) == nil {
		t.Fatal("degenerate polygon accepted")
	}
}

func TestKalmanConvergesToVelocity(t *testing.T) {
	k := newKF1(0, 4)
	rng := rand.New(rand.NewSource(3))
	for i := 1; i <= 100; i++ {
		k.predict(1, 0.01)
		k.update(float64(i)*7+rng.NormFloat64()*2, 4)
	}
	if math.Abs(k.v-7) > 0.1 {
		t.Fatalf("velocity estimate %v, want ≈7", k.v)
	}
}

func frame(boxes ...BBox) FrameInput {
	in := FrameInput{}
	for _, b := range boxes {
		in.Detections = append(in.Detections, Detection{BBox: b, Score: 0.9})
	}
	return in
}

func TestTrackLifecycle(t *testing.T) {
	cfg := DefaultConfig()
	m := NewManager(cfg)
	s, err := m.Create(SessionConfig{ID: "cam-1", FrameWidth: 640, FrameHeight: 480})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(0, 0)
	var res FrameResult
	for i := 0; i < cfg.MinHits; i++ {
		res, _ = s.Process(frame(BBox{100 + float64(i)*5, 100, 50, 50}), now.Add(time.Duration(i)*100*time.Millisecond))
		if i < cfg.MinHits-1 && len(res.Tracks) != 0 {
			t.Fatalf("frame %d: tentative track leaked: %+v", i, res.Tracks)
		}
	}
	if len(res.Tracks) != 1 || res.Tracks[0].ID != 1 || res.Tracks[0].State != StateConfirmed {
		t.Fatalf("track not confirmed after MinHits: %+v", res.Tracks)
	}
	if res.Tracks[0].DwellSeconds != 0.2 {
		t.Fatalf("dwell = %v, want 0.2", res.Tracks[0].DwellSeconds)
	}

	// Coasting: missing detections keep the identity for up to MaxAge frames.
	for i := 0; i < cfg.MaxAge; i++ {
		res, _ = s.Process(frame(), now)
	}
	res, _ = s.Process(frame(BBox{100 + float64(cfg.MinHits+cfg.MaxAge)*5, 100, 50, 50}), now)
	if len(res.Tracks) != 1 || res.Tracks[0].ID != 1 {
		t.Fatalf("identity lost after coasting: %+v", res.Tracks)
	}
	// Beyond MaxAge the track is dropped and a new identity would be issued.
	for i := 0; i <= cfg.MaxAge+1; i++ {
		res, _ = s.Process(frame(), now)
	}
	if res.Analytics.ActiveSubjects != 0 || res.Analytics.Dwell.CompletedTracks != 1 {
		t.Fatalf("track not retired: %+v", res.Analytics)
	}
}

func TestProcessValidation(t *testing.T) {
	m := NewManager(DefaultConfig())
	s, _ := m.Create(SessionConfig{FrameWidth: 100, FrameHeight: 100})
	if _, err := s.Process(frame(BBox{0, 0, -1, 5}), time.Now()); err == nil {
		t.Fatal("negative width accepted")
	}
	if _, err := s.Process(frame(BBox{math.NaN(), 0, 1, 5}), time.Now()); err == nil {
		t.Fatal("NaN accepted")
	}
	low := FrameInput{Detections: []Detection{{BBox: BBox{0, 0, 10, 10}, Score: 0.1}}}
	for i := 0; i < 5; i++ {
		res, _ := s.Process(low, time.Now())
		if len(res.Tracks) != 0 {
			t.Fatal("low-score detection was tracked")
		}
	}
}

func TestSessionConfigValidation(t *testing.T) {
	m := NewManager(DefaultConfig())
	bad := []SessionConfig{
		{FrameWidth: 0, FrameHeight: 10},
		{ID: "bad id!", FrameWidth: 10, FrameHeight: 10},
		{FrameWidth: 10, FrameHeight: 10, Lines: []Tripwire{{A: Point{1, 1}, B: Point{1, 1}}}},
		{FrameWidth: 10, FrameHeight: 10, Zones: []Zone{{Polygon: []Point{{0, 0}, {1, 0}}}}},
		{FrameWidth: 10, FrameHeight: 10, Lines: []Tripwire{{ID: "x", A: Point{0, 0}, B: Point{1, 1}}, {ID: "x", A: Point{0, 0}, B: Point{2, 1}}}},
	}
	for i, sc := range bad {
		if _, err := m.Create(sc); err == nil {
			t.Errorf("case %d accepted", i)
		}
	}
	if _, err := m.Create(SessionConfig{ID: "a", FrameWidth: 10, FrameHeight: 10}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Create(SessionConfig{ID: "a", FrameWidth: 10, FrameHeight: 10}); err != ErrExists {
		t.Fatalf("duplicate id: %v", err)
	}
}

func TestManagerLimitsAndEviction(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxSessions = 2
	cfg.SessionTTL = time.Minute
	m := NewManager(cfg)
	clock := time.Unix(1000, 0)
	m.now = func() time.Time { return clock }

	a, _ := m.Create(SessionConfig{ID: "a", FrameWidth: 10, FrameHeight: 10})
	m.Create(SessionConfig{ID: "b", FrameWidth: 10, FrameHeight: 10})
	if _, err := m.Create(SessionConfig{ID: "c", FrameWidth: 10, FrameHeight: 10}); err != ErrLimit {
		t.Fatalf("limit not enforced: %v", err)
	}
	ch, _ := a.Subscribe()

	clock = clock.Add(45 * time.Second)
	m.Process("b", frame())
	clock = clock.Add(30 * time.Second)
	if n := m.EvictIdle(); n != 1 {
		t.Fatalf("evicted %d, want 1", n)
	}
	if _, err := m.Get("a"); err != ErrNotFound {
		t.Fatal("idle session a survived")
	}
	if _, open := <-ch; open {
		t.Fatal("subscriber channel not closed on eviction")
	}
	if _, err := m.Get("b"); err != nil {
		t.Fatal("active session b evicted")
	}
}

func TestZonesAndDensity(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinHits = 1
	m := NewManager(cfg)
	s, _ := m.Create(SessionConfig{
		FrameWidth: 1000, FrameHeight: 1000, AreaM2: 4,
		Zones: []Zone{{ID: "door", Polygon: []Point{{0, 0}, {500, 0}, {500, 1000}, {0, 1000}}, Capacity: 1}},
	})
	res, _ := s.Process(frame(BBox{100, 100, 50, 50}, BBox{200, 400, 50, 50}, BBox{800, 400, 50, 50}), time.Now())
	z := res.Analytics.Zones[0]
	if z.Occupancy != 2 || z.Entries != 2 || !z.OverCapacity || z.Utilisation != 2 {
		t.Fatalf("zone stats wrong: %+v", z)
	}
	if d := res.Analytics.Density; d.PerM2 == nil || *d.PerM2 != 0.75 || d.Level != "moderate" {
		t.Fatalf("density wrong: %+v", d)
	}
}

func TestHeatmapAndSubscribe(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinHits = 1
	m := NewManager(cfg)
	s, _ := m.Create(SessionConfig{FrameWidth: 100, FrameHeight: 100, HeatmapCols: 4, HeatmapRows: 4})
	ch, cancel := s.Subscribe()
	defer cancel()
	s.Process(frame(BBox{0, 0, 10, 10}), time.Now())
	select {
	case msg := <-ch:
		if len(msg) == 0 {
			t.Fatal("empty message")
		}
	default:
		t.Fatal("subscriber did not receive the frame")
	}
	h := s.Heatmap()
	if h.Cells[0] != 1 || h.Max != 1 {
		t.Fatalf("heatmap %+v", h)
	}
	if _, err := s.HeatmapImage("ironbow", 8); err != nil {
		t.Fatal(err)
	}
}

func TestEmbeddingValidation(t *testing.T) {
	m := NewManager(DefaultConfig())
	s, _ := m.Create(SessionConfig{FrameWidth: 100, FrameHeight: 100})
	withEmb := func(e []float64) FrameInput {
		return FrameInput{Detections: []Detection{{BBox: BBox{0, 0, 10, 10}, Score: 0.9, Embedding: e}}}
	}
	if _, err := s.Process(withEmb([]float64{0, 0, 0}), time.Now()); err == nil {
		t.Fatal("zero embedding accepted")
	}
	if _, err := s.Process(withEmb([]float64{1, math.Inf(1)}), time.Now()); err == nil {
		t.Fatal("non-finite embedding accepted")
	}
	res, err := s.Process(withEmb([]float64{3, 4}), time.Now())
	if err != nil || !res.Analytics.ReID.Enabled || res.Analytics.ReID.EmbeddingDim != 2 {
		t.Fatalf("valid embedding: %v %+v", err, res.Analytics.ReID)
	}
	if _, err := s.Process(withEmb([]float64{1, 2, 3}), time.Now()); err == nil {
		t.Fatal("dimension change accepted")
	}
	if _, err := m.Create(SessionConfig{FrameWidth: 1, FrameHeight: 1, AppearanceThreshold: 3}); err == nil {
		t.Fatal("bad appearance_threshold accepted")
	}
}

// TestAppearanceResolvesSwap: two people vanish together and reappear with
// swapped positions. IoU/motion alone would swap their IDs; appearance keeps
// each identity with the right person.
func TestAppearanceResolvesSwap(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinHits = 1
	m := NewManager(cfg)
	s, _ := m.Create(SessionConfig{FrameWidth: 1000, FrameHeight: 1000})
	a, b := []float64{1, 0, 0, 0}, []float64{0, 1, 0, 0}
	det := func(x float64, e []float64) Detection {
		return Detection{BBox: BBox{x, 400, 80, 200}, Score: 0.9, Embedding: e}
	}
	now := time.Now()
	var res FrameResult
	for i := 0; i < 5; i++ { // A stands at x=300, B at x=420, both still
		res, _ = s.Process(FrameInput{Detections: []Detection{det(300, a), det(420, b)}}, now)
	}
	idAt := func(r FrameResult, x float64) uint64 {
		for _, tr := range r.Tracks {
			if math.Abs(tr.BBox.X-x) < 20 {
				return tr.ID
			}
		}
		return 0
	}
	idA, idB := idAt(res, 300), idAt(res, 420)
	for i := 0; i < 30; i++ { // both hidden for longer than max_age
		s.Process(FrameInput{}, now)
	}
	res, _ = s.Process(FrameInput{Detections: []Detection{det(300, b), det(420, a)}}, now)
	if idAt(res, 420) != idA || idAt(res, 300) != idB {
		t.Fatalf("identities not preserved across swap: A=%d B=%d, got at300=%d at420=%d", idA, idB, idAt(res, 300), idAt(res, 420))
	}
	if res.Analytics.ReID.Recoveries != 2 {
		t.Fatalf("recoveries = %d, want 2", res.Analytics.ReID.Recoveries)
	}
}
