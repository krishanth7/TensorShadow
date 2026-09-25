package tracking_test

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/sim"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
)

func runScene(t testing.TB, m *tracking.Manager, id string, seq sim.CrowdSequence) tracking.FrameResult {
	t.Helper()
	scene := sim.DefaultCrowdScene()
	_, err := m.Create(tracking.SessionConfig{
		ID: id, FrameWidth: scene.Width, FrameHeight: scene.Height, AreaM2: 60,
		Lines: []tracking.Tripwire{{ID: "mid", A: tracking.Point{X: scene.Width / 2, Y: 0}, B: tracking.Point{X: scene.Width / 2, Y: scene.Height}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	start := time.Unix(1_700_000_000, 0)
	var res tracking.FrameResult
	for i, dets := range seq.Frames {
		ts := start.Add(time.Duration(i) * 100 * time.Millisecond) // 10 FPS
		res, err = m.Process(id, tracking.FrameInput{Timestamp: &ts, Detections: dets})
		if err != nil {
			t.Fatal(err)
		}
	}
	return res
}

// TestCrowdSceneMatchesGroundTruth drives the tracker with a synthetic crowd
// (24 pedestrians, bidirectional flow, 3 % missed detections, 1.5 px jitter)
// and checks identity and counting accuracy against ground truth.
func TestCrowdSceneMatchesGroundTruth(t *testing.T) {
	seq := sim.DefaultCrowdScene().Generate()
	res := runScene(t, tracking.NewManager(tracking.DefaultConfig()), "scene", seq)
	a := res.Analytics

	line := a.Lines[0]
	if int(line.In) != seq.Truth.LeftToRight || int(line.Out) != seq.Truth.RightToLeft {
		t.Errorf("tripwire in/out = %d/%d, truth L→R/R→L = %d/%d",
			line.In, line.Out, seq.Truth.LeftToRight, seq.Truth.RightToLeft)
	}
	if int(a.UniqueSubjects) != seq.Truth.Visible {
		t.Errorf("unique identities = %d, truth = %d (ID switches or fragmentation)", a.UniqueSubjects, seq.Truth.Visible)
	}
	if a.PeakSubjects > seq.Truth.PeakVisible || a.PeakSubjects < seq.Truth.PeakVisible-1 {
		t.Errorf("peak = %d, truth = %d", a.PeakSubjects, seq.Truth.PeakVisible)
	}
	t.Logf("truth: %+v | tracker: unique=%d peak=%d in=%d out=%d dwell=%.2fs",
		seq.Truth, a.UniqueSubjects, a.PeakSubjects, line.In, line.Out, a.Dwell.CompletedAvgSeconds)
}

// TestConcurrentSessions processes many independent streams in parallel
// (run with -race) and checks each produces the same deterministic result.
func TestConcurrentSessions(t *testing.T) {
	seq := sim.DefaultCrowdScene().Generate()
	m := tracking.NewManager(tracking.DefaultConfig())
	const streams = 16
	results := make([]tracking.FrameResult, streams)
	var wg sync.WaitGroup
	for i := 0; i < streams; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = runScene(t, m, "cam-"+string(rune('a'+i)), seq)
		}(i)
	}
	wg.Wait()
	for i := 1; i < streams; i++ {
		if results[i].Analytics.UniqueSubjects != results[0].Analytics.UniqueSubjects ||
			results[i].Analytics.Lines[0] != results[0].Analytics.Lines[0] {
			t.Fatalf("stream %d diverged: %+v vs %+v", i, results[i].Analytics.Lines, results[0].Analytics.Lines)
		}
	}
	if n, _ := m.Totals(); n != streams {
		t.Fatalf("sessions = %d", n)
	}
}

func BenchmarkTrackerFrame(b *testing.B) {
	seq := sim.DefaultCrowdScene().Generate()
	scene := sim.DefaultCrowdScene()
	m := tracking.NewManager(tracking.DefaultConfig())
	sess, _ := m.Create(tracking.SessionConfig{FrameWidth: scene.Width, FrameHeight: scene.Height,
		Lines: []tracking.Tripwire{{A: tracking.Point{X: 640, Y: 0}, B: tracking.Point{X: 640, Y: 720}}}})
	now := time.Now()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess.Process(tracking.FrameInput{Detections: seq.Frames[i%len(seq.Frames)]}, now)
	}
}

func BenchmarkTrackerDense4K(b *testing.B) {
	scene := sim.DefaultCrowdScene()
	scene.Pedestrians, scene.Frames, scene.Height = 400, 400, 2160
	scene.Width = 3840
	seq := scene.Generate()
	m := tracking.NewManager(tracking.DefaultConfig())
	sess, _ := m.Create(tracking.SessionConfig{FrameWidth: scene.Width, FrameHeight: scene.Height})
	// Warm up to steady state and find the busiest frame.
	busiest := 0
	for i, f := range seq.Frames {
		if len(f) > len(seq.Frames[busiest]) {
			busiest = i
		}
	}
	now := time.Now()
	for i := 0; i < busiest; i++ { // warm up tracks to steady state
		sess.Process(tracking.FrameInput{Detections: seq.Frames[i]}, now)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess.Process(tracking.FrameInput{Detections: seq.Frames[busiest-2+i%3]}, now)
	}
	b.ReportMetric(float64(len(seq.Frames[busiest])), "subjects/frame")
}

// TestAccuracyAcrossSeeds is the accuracy regression gate quoted in the
// README: 50 random scenes, 1,200 pedestrians and 1,200 line crossings.
func TestAccuracyAcrossSeeds(t *testing.T) {
	if testing.Short() {
		t.Skip("accuracy sweep skipped in -short mode")
	}
	var crossErr, crossTotal, idErr, idTotal int
	for seed := int64(1); seed <= 50; seed++ {
		scene := sim.DefaultCrowdScene()
		scene.Seed = seed
		seq := scene.Generate()
		a := runScene(t, tracking.NewManager(tracking.DefaultConfig()), fmt.Sprintf("seed-%d", seed), seq).Analytics
		crossErr += absInt(int(a.Lines[0].In)-seq.Truth.LeftToRight) + absInt(int(a.Lines[0].Out)-seq.Truth.RightToLeft)
		crossTotal += seq.Truth.LeftToRight + seq.Truth.RightToLeft
		idErr += absInt(int(a.UniqueSubjects) - seq.Truth.Visible)
		idTotal += seq.Truth.Visible
	}
	t.Logf("tripwire: %d/%d crossings counted correctly; identities: %d/%d correct",
		crossTotal-crossErr, crossTotal, idTotal-idErr, idTotal)
	if crossErr > 0 {
		t.Errorf("tripwire count error %d over %d crossings", crossErr, crossTotal)
	}
	if float64(idErr) > 0.01*float64(idTotal) {
		t.Errorf("identity error %d over %d subjects exceeds 1%%", idErr, idTotal)
	}
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
