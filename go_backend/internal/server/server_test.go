package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/config"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/sim"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/workerpool"
)

func newTestServer(t *testing.T, mutate func(*config.Config)) *httptest.Server {
	t.Helper()
	cfg := config.Default()
	cfg.Concurrency.RateLimitRPS = 0
	if mutate != nil {
		mutate(&cfg)
	}
	pool := workerpool.New(cfg.Concurrency.Workers, cfg.Concurrency.QueueSize)
	crowd := tracking.NewManager(cfg.Tracking)
	s := New(cfg, Deps{Pool: pool, Crowd: crowd,
		Web: fstest.MapFS{"index.html": {Data: []byte("<html>dashboard</html>")}}})
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() {
		crowd.CloseAll()
		ts.Close()
		pool.Shutdown(context.Background())
	})
	return ts
}

func do(t *testing.T, method, url string, body any) (*http.Response, []byte) {
	t.Helper()
	var rdr io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rdr = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, url, rdr)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp, out
}

func TestHealthStatsMetricsDashboard(t *testing.T) {
	ts := newTestServer(t, nil)
	resp, body := do(t, "GET", ts.URL+"/api/v1/health", nil)
	if resp.StatusCode != 200 || !strings.Contains(string(body), `"status":"up"`) {
		t.Fatalf("health: %d %s", resp.StatusCode, body)
	}
	if resp.Header.Get("X-Request-ID") == "" {
		t.Fatal("missing X-Request-ID")
	}
	resp, body = do(t, "GET", ts.URL+"/api/v1/stats", nil)
	var stats map[string]any
	json.Unmarshal(body, &stats)
	if resp.StatusCode != 200 || stats["pool"] == nil || stats["crowd"] == nil {
		t.Fatalf("stats: %d %s", resp.StatusCode, body)
	}
	_, body = do(t, "GET", ts.URL+"/metrics", nil)
	for _, want := range []string{
		`tensorshadow_http_requests_total{route="GET /api/v1/health",method="GET",code="200"} 1`,
		"tensorshadow_pool_workers", "tensorshadow_http_request_duration_seconds_bucket",
	} {
		if !strings.Contains(string(body), want) {
			t.Fatalf("metrics missing %q:\n%s", want, body)
		}
	}
	_, body = do(t, "GET", ts.URL+"/", nil)
	if string(body) != "<html>dashboard</html>" {
		t.Fatalf("dashboard: %s", body)
	}
	resp, _ = do(t, "POST", ts.URL+"/api/v1/health", nil)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("wrong method status %d", resp.StatusCode)
	}
}

func TestPredictKeepsV1Contract(t *testing.T) {
	ts := newTestServer(t, nil)
	resp, body := do(t, "POST", ts.URL+"/api/v1/predict", map[string]any{"data": []float64{0, 0}})
	var out struct {
		Status string  `json:"status"`
		Result float64 `json:"result"`
	}
	json.Unmarshal(body, &out)
	if resp.StatusCode != 200 || out.Status != "success" || out.Result != 0.5 {
		t.Fatalf("predict: %d %s", resp.StatusCode, body)
	}
	resp, _ = do(t, "POST", ts.URL+"/api/v1/predict", map[string]any{"data": []float64{}})
	if resp.StatusCode != 400 {
		t.Fatalf("empty data accepted: %d", resp.StatusCode)
	}
}

func demoFrame() thermal.FrameRequest {
	scene := sim.DemoThermalScene()
	amb := scene.AmbientC
	return thermal.FrameRequest{Width: scene.Width, Height: scene.Height, Format: thermal.FormatCelsius,
		Data: scene.Render(), AmbientTempC: &amb}
}

func TestThermalEndpoints(t *testing.T) {
	ts := newTestServer(t, nil)
	resp, body := do(t, "POST", ts.URL+"/api/v1/thermal/analyze", demoFrame())
	if resp.StatusCode != 200 {
		t.Fatalf("analyze: %d %s", resp.StatusCode, body)
	}
	var res thermalResponse
	json.Unmarshal(body, &res)
	if len(res.Subjects) != 4 || res.Summary["fever"] != 1 || res.Summary["spoof_suspected"] != 1 || res.RequestID == "" {
		t.Fatalf("analyze result: %s", body)
	}

	resp, body = do(t, "POST", ts.URL+"/api/v1/thermal/render?palette=rainbow&scale=2&annotate=true", demoFrame())
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "image/png" {
		t.Fatalf("render: %d %s", resp.StatusCode, body)
	}
	img, err := png.Decode(bytes.NewReader(body))
	if err != nil || img.Bounds().Dx() != 320 {
		t.Fatalf("render png: %v %v", err, img.Bounds())
	}

	for _, bad := range []struct {
		url  string
		body any
	}{
		{"/api/v1/thermal/analyze", map[string]any{"width": 2, "height": 2, "data": []float64{1}}},
		{"/api/v1/thermal/analyze", map[string]any{"width": 1, "height": 1, "data": []float64{30}, "unknown": 1}},
		{"/api/v1/thermal/render?palette=sepia", demoFrame()},
		{"/api/v1/thermal/render?scale=99", demoFrame()},
	} {
		resp, body := do(t, "POST", ts.URL+bad.url, bad.body)
		if resp.StatusCode != 400 || !strings.Contains(string(body), `"error"`) {
			t.Errorf("%s: %d %s", bad.url, resp.StatusCode, body)
		}
	}
}

func TestBodyLimit(t *testing.T) {
	ts := newTestServer(t, func(c *config.Config) { c.Server.MaxBodyBytes = 4096 })
	resp, body := do(t, "POST", ts.URL+"/api/v1/thermal/analyze", demoFrame())
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d %s", resp.StatusCode, body)
	}
}

func TestCrowdEndToEnd(t *testing.T) {
	ts := newTestServer(t, nil)
	scene := sim.DefaultCrowdScene()
	seq := scene.Generate()

	resp, body := do(t, "POST", ts.URL+"/api/v1/crowd/sessions", tracking.SessionConfig{
		ID: "lobby", FrameWidth: scene.Width, FrameHeight: scene.Height, AreaM2: 60,
		Lines: []tracking.Tripwire{{ID: "gate", A: tracking.Point{X: 640, Y: 0}, B: tracking.Point{X: 640, Y: 720}}},
		Zones: []tracking.Zone{{ID: "left", Polygon: []tracking.Point{{X: 0, Y: 0}, {X: 640, Y: 0}, {X: 640, Y: 720}, {X: 0, Y: 720}}, Capacity: 10}},
	})
	if resp.StatusCode != 201 || resp.Header.Get("Location") != "/api/v1/crowd/sessions/lobby" {
		t.Fatalf("create: %d %s", resp.StatusCode, body)
	}
	resp, _ = do(t, "POST", ts.URL+"/api/v1/crowd/sessions", tracking.SessionConfig{ID: "lobby", FrameWidth: 1, FrameHeight: 1})
	if resp.StatusCode != 409 {
		t.Fatalf("duplicate create: %d", resp.StatusCode)
	}

	// Subscribe to the live stream before sending frames.
	streamReq, _ := http.NewRequest("GET", ts.URL+"/api/v1/crowd/sessions/lobby/stream", nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	streamResp, err := http.DefaultClient.Do(streamReq.WithContext(ctx))
	if err != nil || streamResp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("stream: %v", err)
	}
	defer streamResp.Body.Close()
	gotEvent := make(chan string, 1)
	go func() {
		sc := bufio.NewScanner(streamResp.Body)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			if strings.HasPrefix(sc.Text(), "data: {") {
				gotEvent <- sc.Text()
				return
			}
		}
	}()
	time.Sleep(50 * time.Millisecond) // let the subscription register

	var last tracking.FrameResult
	for _, dets := range seq.Frames {
		resp, body := do(t, "POST", ts.URL+"/api/v1/crowd/sessions/lobby/frames", tracking.FrameInput{Detections: dets})
		if resp.StatusCode != 200 {
			t.Fatalf("frame: %d %s", resp.StatusCode, body)
		}
		json.Unmarshal(body, &last)
	}
	select {
	case ev := <-gotEvent:
		if !strings.Contains(ev, `"session_id":"lobby"`) {
			t.Fatalf("unexpected event %s", ev)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no SSE event received")
	}

	a := last.Analytics
	if int(a.Lines[0].In) != seq.Truth.LeftToRight || int(a.Lines[0].Out) != seq.Truth.RightToLeft {
		t.Fatalf("line counts %+v vs truth %+v", a.Lines[0], seq.Truth)
	}
	if a.Zones[0].Entries == 0 || a.FramesProcessed != uint64(scene.Frames) {
		t.Fatalf("analytics %+v", a)
	}

	_, body = do(t, "GET", ts.URL+"/api/v1/crowd/sessions/lobby/analytics", nil)
	if !strings.Contains(string(body), `"unique_subjects":24`) {
		t.Fatalf("analytics endpoint: %s", body)
	}
	_, body = do(t, "GET", ts.URL+"/api/v1/crowd/sessions/lobby/heatmap", nil)
	if !strings.Contains(string(body), `"cols":32`) {
		t.Fatalf("heatmap: %s", body)
	}
	resp, body = do(t, "GET", ts.URL+"/api/v1/crowd/sessions/lobby/heatmap?format=png&cell=4", nil)
	if _, err := png.Decode(bytes.NewReader(body)); err != nil || resp.StatusCode != 200 {
		t.Fatalf("heatmap png: %d %v", resp.StatusCode, err)
	}
	_, body = do(t, "GET", ts.URL+"/api/v1/crowd/sessions", nil)
	if !strings.Contains(string(body), `"count":1`) {
		t.Fatalf("list: %s", body)
	}

	resp, _ = do(t, "DELETE", ts.URL+"/api/v1/crowd/sessions/lobby", nil)
	if resp.StatusCode != 204 {
		t.Fatalf("delete: %d", resp.StatusCode)
	}
	resp, body = do(t, "POST", ts.URL+"/api/v1/crowd/sessions/lobby/frames", tracking.FrameInput{})
	if resp.StatusCode != 404 || !strings.Contains(string(body), "session_not_found") {
		t.Fatalf("frame after delete: %d %s", resp.StatusCode, body)
	}
}

func TestRateLimit(t *testing.T) {
	ts := newTestServer(t, func(c *config.Config) {
		c.Concurrency.RateLimitRPS = 1
		c.Concurrency.RateLimitBurst = 3
	})
	codes := map[int]int{}
	for i := 0; i < 6; i++ {
		resp, _ := do(t, "GET", ts.URL+"/api/v1/stats", nil)
		codes[resp.StatusCode]++
		if resp.StatusCode == 429 && resp.Header.Get("Retry-After") == "" {
			t.Fatal("429 without Retry-After")
		}
	}
	if codes[200] != 3 || codes[429] != 3 {
		t.Fatalf("codes %v", codes)
	}
	// Health checks are never rate limited.
	if resp, _ := do(t, "GET", ts.URL+"/api/v1/health", nil); resp.StatusCode != 200 {
		t.Fatalf("health limited: %d", resp.StatusCode)
	}
}

func TestBackpressureReturns503(t *testing.T) {
	cfg := config.Default()
	cfg.Concurrency.RateLimitRPS = 0
	pool := workerpool.New(1, 0) // one worker, no queue
	defer pool.Shutdown(context.Background())
	ts := httptest.NewServer(New(cfg, Deps{Pool: pool, Crowd: tracking.NewManager(cfg.Tracking)}).Handler())
	defer ts.Close()

	// Occupy the only worker. release is closed exactly once, and before the
	// deferred pool shutdown even if an assertion below fails.
	release, busy := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	go func() {
		// With no queue, a hand-off only succeeds once the worker goroutine is
		// parked on its receive, so retry until the blocking task is accepted.
		for {
			_, err := pool.Submit(context.Background(), func(context.Context) (any, error) {
				close(busy)
				<-release
				return nil, nil
			})
			if !errors.Is(err, workerpool.ErrQueueFull) {
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()
	<-busy

	resp, body := do(t, "POST", ts.URL+"/api/v1/predict", map[string]any{"data": []float64{1}})
	if resp.StatusCode != 503 || resp.Header.Get("Retry-After") != "1" || !strings.Contains(string(body), "overloaded") {
		t.Fatalf("saturated pool: %d %s", resp.StatusCode, body)
	}
	unblock()
	// Once the worker frees up, requests succeed again.
	deadline := time.Now().Add(2 * time.Second)
	for {
		resp, _ = do(t, "POST", ts.URL+"/api/v1/predict", map[string]any{"data": []float64{1}})
		if resp.StatusCode == 200 || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("pool did not recover: %d", resp.StatusCode)
	}
}

func TestCORSPreflight(t *testing.T) {
	ts := newTestServer(t, func(c *config.Config) { c.Server.CORSOrigins = []string{"https://ops.example"} })
	req, _ := http.NewRequest("OPTIONS", ts.URL+"/api/v1/crowd/sessions", nil)
	req.Header.Set("Origin", "https://ops.example")
	req.Header.Set("Access-Control-Request-Method", "POST")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 204 || resp.Header.Get("Access-Control-Allow-Origin") != "https://ops.example" {
		t.Fatalf("preflight: %d %v", resp.StatusCode, resp.Header)
	}
}

func TestThermalBinaryIngest(t *testing.T) {
	ts := newTestServer(t, nil)
	scene := sim.DemoThermalScene()
	temps := scene.Render()

	// FLIR Lepton-style radiometric buffer: uint16 centikelvin, little endian.
	buf := make([]byte, 2*len(temps))
	for i, c := range temps {
		binary.LittleEndian.PutUint16(buf[2*i:], uint16(math.Round((c+273.15)*100)))
	}
	url := fmt.Sprintf("%s/api/v1/thermal/analyze?width=%d&height=%d&ambient_c=22", ts.URL, scene.Width, scene.Height)
	resp, err := http.Post(url, "application/octet-stream", bytes.NewReader(buf))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var res thermalResponse
	json.Unmarshal(body, &res)
	if resp.StatusCode != 200 || len(res.Subjects) != 4 || res.Summary["fever"] != 1 || res.Summary["spoof_suspected"] != 1 {
		t.Fatalf("binary analyze: %d %s", resp.StatusCode, body)
	}

	for _, bad := range []struct {
		query string
		body  []byte
	}{
		{"width=160&height=120", buf[:100]},
		{"width=160&height=120", append(append([]byte{}, buf...), 0, 0)},
		{"width=160&height=120&dtype=int8", buf},
		{"width=x&height=120", buf},
		{"width=160&height=120&roi=1,2,3", buf},
	} {
		resp, err := http.Post(ts.URL+"/api/v1/thermal/analyze?"+bad.query, "application/octet-stream", bytes.NewReader(bad.body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 400 {
			t.Errorf("%s (%d bytes): status %d, want 400", bad.query, len(bad.body), resp.StatusCode)
		}
	}
}
