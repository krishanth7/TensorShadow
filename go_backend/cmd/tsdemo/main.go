// Command tsdemo exercises a running TensorShadow backend end to end: IR
// thermal screening of a synthetic radiometric frame and multi-subject crowd
// tracking of a synthetic 720p pedestrian sequence. It prints a report and
// writes the rendered thermal image and crowd heatmap as PNG files.
//
//	go run ./go_backend -addr :8080 &
//	go run ./go_backend/cmd/tsdemo -addr http://localhost:8080 -out docs/images
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/sim"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
)

var client = &http.Client{Timeout: 30 * time.Second}

func main() {
	addr := flag.String("addr", "http://localhost:8080", "backend base URL")
	out := flag.String("out", "docs/images", "directory for generated PNG files")
	flag.Parse()
	if err := run(strings.TrimRight(*addr, "/"), *out); err != nil {
		fmt.Fprintln(os.Stderr, "tsdemo:", err)
		os.Exit(1)
	}
}

func run(base, out string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	var health map[string]any
	if err := call("GET", base+"/api/v1/health", nil, &health); err != nil {
		return fmt.Errorf("backend not reachable at %s: %w", base, err)
	}
	fmt.Printf("TensorShadow backend %v at %s (status: %v)\n\n", health["version"], base, health["status"])

	if err := modelDemo(base); err != nil {
		return err
	}
	if err := thermalDemo(base, out); err != nil {
		return err
	}
	if err := coregDemo(base, out); err != nil {
		return err
	}
	if err := crowdDemo(base, out); err != nil {
		return err
	}
	return reidDemo(base)
}

func modelDemo(base string) error {
	fmt.Println("━━ Model serving (/api/v1/predict) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	var info struct {
		Backend string `json:"backend"`
		URL     string `json:"url"`
		Model   string `json:"model"`
		Status  struct {
			Ready    bool           `json:"ready"`
			State    string         `json:"state"`
			Versions []string       `json:"versions"`
			Metadata map[string]any `json:"metadata"`
		} `json:"status"`
	}
	if err := call("GET", base+"/api/v1/model", nil, &info); err != nil {
		return err
	}
	fmt.Printf("backend %s  model %s  url %s  ready=%v state=%s versions=%v\n",
		info.Backend, info.Model, orDash(info.URL), info.Status.Ready, info.Status.State, info.Status.Versions)

	batch := [][]float64{
		{0.4, -0.2, 0.1, 0.3, -0.5, 0.2, 0.0, 0.6, -0.1, 0.3},
		{-1.2, 0.5, 0.3, -0.4, 0.1, -0.9, -0.2, -0.3, 0.2, -0.8},
	}
	var res struct {
		Predictions []struct {
			Outputs []float64 `json:"outputs"`
			Result  float64   `json:"result"`
			Class   int       `json:"class"`
		} `json:"predictions"`
		Backend   string  `json:"backend"`
		Model     string  `json:"model"`
		Version   string  `json:"version"`
		LatencyMs float64 `json:"latency_ms"`
		Degraded  bool    `json:"degraded"`
	}
	if err := call("POST", base+"/api/v1/predict", map[string]any{"instances": batch}, &res); err != nil {
		return err
	}
	for i, p := range res.Predictions {
		fmt.Printf("instance %d → class %d  p=%.6f  outputs=%v\n", i, p.Class, p.Result, p.Outputs)
	}
	fmt.Printf("served by %s (model %s, version %s) in %.2f ms, degraded=%v\n\n",
		res.Backend, res.Model, orDash(res.Version), res.LatencyMs, res.Degraded)
	return nil
}

func coregDemo(base, out string) error {
	fmt.Println("━━ Visible/IR co-registration → thermal ROIs from the RGB detector ━━━━━━━━━━")
	scene := sim.DemoCoregScene()
	id := fmt.Sprintf("demo-rig-%d", time.Now().UnixNano())
	var rig struct {
		RMSE     float64 `json:"rmse_px"`
		MaxError float64 `json:"max_error_px"`
		Quality  string  `json:"quality"`
		Points   int     `json:"points"`
	}
	if err := call("POST", base+"/api/v1/coreg/rigs", map[string]any{
		"id": id, "name": "Demo dual-sensor head",
		"visible": map[string]int{"width": 1280, "height": 960}, "thermal": map[string]int{"width": 160, "height": 120},
		"points": scene.Calibration,
	}, &rig); err != nil {
		return err
	}
	defer call("DELETE", base+"/api/v1/coreg/rigs/"+id, nil, nil)
	fmt.Printf("rig calibrated from %d heated-target points: RMSE %.3f px, max %.3f px (%s)\n\n",
		rig.Points, rig.RMSE, rig.MaxError, rig.Quality)

	th := sim.DemoThermalScene()
	amb := th.AmbientC
	req := thermal.FrameRequest{Width: th.Width, Height: th.Height, Data: th.Render(), AmbientTempC: &amb,
		Rig: id, VisibleROIs: scene.FaceBoxes}
	var res struct {
		thermal.Result
		Coreg struct {
			Mapped int `json:"mapped"`
		} `json:"coregistration"`
	}
	if err := call("POST", base+"/api/v1/thermal/analyze", req, &res); err != nil {
		return err
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "RGB detection\tVisible box (x,y,w,h)\tThermal ROI\tCanthus\tStatus\tLiveness")
	for _, s := range res.Subjects {
		v := s.VisibleROI
		live := "LIVE"
		if !s.Liveness.Live {
			live = "SPOOF"
		}
		fmt.Fprintf(tw, "%s\t%.0f,%.0f,%.0f,%.0f\t%d,%d,%d,%d\t%.2f °C\t%s\t%s\n", v.ID, v.X, v.Y, v.W, v.H,
			s.ROI.X, s.ROI.Y, s.ROI.W, s.ROI.H, s.Canthus.TempC, strings.ToUpper(s.Status), live)
	}
	tw.Flush()

	png, err := raw("POST", base+"/api/v1/thermal/render?palette=ironbow&scale=4&annotate=true", req)
	if err != nil {
		return err
	}
	p := filepath.Join(out, "thermal_coreg.png")
	if err := os.WriteFile(p, png, 0o644); err != nil {
		return err
	}
	fmt.Printf("\n%d RGB boxes mapped into the thermal frame; annotated render → %s\n\n", res.Coreg.Mapped, p)
	return nil
}

func reidDemo(base string) error {
	fmt.Println("━━ Appearance re-identification through long occlusion ━━━━━━━━━━━━━━━━━━━━━")
	scene := sim.OccludedCrowdScene()
	seq := scene.Generate()
	fmt.Printf("720p scene, %d pedestrians, a %g px pillar over the counting line hides everyone for up to %d frames (max_age = 15)\n\n",
		scene.Pedestrians, scene.Occluder.X1-scene.Occluder.X0, seq.Truth.LongestOcclusion)

	type outcome struct {
		unique, crossings, recoveries int
	}
	runOnce := func(withEmbeddings bool) (outcome, error) {
		id := fmt.Sprintf("reid-%v-%d", withEmbeddings, time.Now().UnixNano())
		cfg := tracking.SessionConfig{ID: id, FrameWidth: scene.Width, FrameHeight: scene.Height,
			Lines: []tracking.Tripwire{{ID: "mid", A: tracking.Point{X: scene.Width / 2, Y: 0}, B: tracking.Point{X: scene.Width / 2, Y: scene.Height}}}}
		if err := call("POST", base+"/api/v1/crowd/sessions", cfg, nil); err != nil {
			return outcome{}, err
		}
		defer call("DELETE", base+"/api/v1/crowd/sessions/"+id, nil, nil)
		var last tracking.FrameResult
		for _, f := range seq.Frames {
			dets := f
			if !withEmbeddings {
				dets = make([]tracking.Detection, len(f))
				for i, d := range f {
					d.Embedding = nil
					dets[i] = d
				}
			}
			if err := call("POST", base+"/api/v1/crowd/sessions/"+id+"/frames", tracking.FrameInput{Detections: dets}, &last); err != nil {
				return outcome{}, err
			}
		}
		a := last.Analytics
		return outcome{int(a.UniqueSubjects), int(a.Lines[0].In + a.Lines[0].Out), int(a.ReID.Recoveries)}, nil
	}
	iou, err := runOnce(false)
	if err != nil {
		return err
	}
	reid, err := runOnce(true)
	if err != nil {
		return err
	}
	observable := seq.Truth.LeftToRight + seq.Truth.RightToLeft - seq.Truth.HiddenCrossings
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Metric\tIoU only (SORT)\tReID (DeepSORT-style)\tGround truth")
	fmt.Fprintf(tw, "Unique identities\t%d\t%d\t%d\n", iou.unique, reid.unique, seq.Truth.Visible)
	fmt.Fprintf(tw, "Line crossings counted\t%d\t%d\t%d\n", iou.crossings, reid.crossings, observable)
	fmt.Fprintf(tw, "Identities recovered after occlusion\t%d\t%d\t–\n", iou.recoveries, reid.recoveries)
	tw.Flush()
	fmt.Println()
	return nil
}

func orDash(s string) string {
	if s == "" {
		return "–"
	}
	return s
}

func thermalDemo(base, out string) error {
	fmt.Println("━━ Infrared thermal screening ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	scene := sim.DemoThermalScene()
	amb := scene.AmbientC
	req := thermal.FrameRequest{Width: scene.Width, Height: scene.Height, Format: thermal.FormatCelsius,
		Data: scene.Render(), AmbientTempC: &amb}

	var res struct {
		thermal.Result
		ProcessingMs float64 `json:"processing_ms"`
	}
	if err := call("POST", base+"/api/v1/thermal/analyze", req, &res); err != nil {
		return err
	}
	fmt.Printf("Frame %dx%d  range %.2f–%.2f °C  ambient %.1f °C (%s)  ε=%.2f  processed in %.2f ms\n\n",
		res.Frame.Width, res.Frame.Height, res.Frame.MinC, res.Frame.MaxC, *res.Frame.AmbientC,
		res.Frame.AmbientSrc, res.Frame.Emissivity, res.ProcessingMs)

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Subject\tROI (x,y,w,h)\tSkin mean\tCanthus\tTruth\tCore est.\tStatus\tLiveness")
	for i, s := range res.Subjects {
		truth := "spoof"
		if !scene.Subjects[i].Spoof {
			truth = fmt.Sprintf("%.2f °C", scene.Subjects[i].CanthusC)
		}
		live := fmt.Sprintf("LIVE (%.2f)", s.Liveness.Score)
		if !s.Liveness.Live {
			var failed []string
			for _, c := range s.Liveness.Checks {
				if !c.Passed {
					failed = append(failed, c.Name)
				}
			}
			live = fmt.Sprintf("SPOOF (%.2f): %s", s.Liveness.Score, strings.Join(failed, ", "))
		}
		fmt.Fprintf(tw, "#%d\t%d,%d,%d,%d\t%.2f °C\t%.2f °C\t%s\t%.2f °C\t%s\t%s\n", s.Index,
			s.ROI.X, s.ROI.Y, s.ROI.W, s.ROI.H, s.Skin.MeanC, s.Canthus.TempC, truth, s.CoreEstimate,
			strings.ToUpper(s.Status), live)
	}
	tw.Flush()
	fmt.Printf("\nSummary: %v\n", res.Summary)

	png, err := raw("POST", base+"/api/v1/thermal/render?palette=ironbow&scale=4&annotate=true", req)
	if err != nil {
		return err
	}
	p := filepath.Join(out, "thermal_ironbow.png")
	if err := os.WriteFile(p, png, 0o644); err != nil {
		return err
	}
	fmt.Printf("Rendered annotated thermal image → %s (%d bytes)\n\n", p, len(png))
	return nil
}

func crowdDemo(base, out string) error {
	fmt.Println("━━ Multi-subject tracking & crowd analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	scene := sim.DefaultCrowdScene()
	seq := scene.Generate()
	id := fmt.Sprintf("demo-%d", time.Now().Unix())
	cfg := tracking.SessionConfig{
		ID: id, Name: "Concourse camera 1", FrameWidth: scene.Width, FrameHeight: scene.Height, AreaM2: 60,
		Lines: []tracking.Tripwire{{ID: "gate", Name: "Centre gate",
			A: tracking.Point{X: scene.Width / 2, Y: 0}, B: tracking.Point{X: scene.Width / 2, Y: scene.Height}}},
		Zones: []tracking.Zone{{ID: "west", Name: "West concourse", Capacity: 10, Polygon: []tracking.Point{
			{X: 0, Y: 0}, {X: scene.Width / 2, Y: 0}, {X: scene.Width / 2, Y: scene.Height}, {X: 0, Y: scene.Height}}}},
	}
	if err := call("POST", base+"/api/v1/crowd/sessions", cfg, nil); err != nil {
		return err
	}
	defer call("DELETE", base+"/api/v1/crowd/sessions/"+id, nil, nil)

	start := time.Unix(1_767_225_600, 0).UTC() // fixed clock: 10 FPS sequence
	var last tracking.FrameResult
	var total time.Duration
	for i, dets := range seq.Frames {
		ts := start.Add(time.Duration(i) * 100 * time.Millisecond)
		t0 := time.Now()
		if err := call("POST", base+"/api/v1/crowd/sessions/"+id+"/frames",
			tracking.FrameInput{Timestamp: &ts, Detections: dets}, &last); err != nil {
			return err
		}
		total += time.Since(t0)
		if (i+1)%60 == 0 {
			a := last.Analytics
			fmt.Printf("frame %3d  active %2d  unique %2d  gate in/out %2d/%-2d  west zone %2d  density %.2f/m² (%s)  flow %s\n",
				i+1, a.ActiveSubjects, a.UniqueSubjects, a.Lines[0].In, a.Lines[0].Out, a.Zones[0].Occupancy,
				*a.Density.PerM2, a.Density.Level, a.Flow.Direction)
		}
	}
	a := last.Analytics
	fmt.Printf("\n%d frames posted, mean round-trip %.2f ms/frame\n\n", len(seq.Frames),
		float64(total.Microseconds())/1000/float64(len(seq.Frames)))

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Metric\tTracker\tGround truth")
	fmt.Fprintf(tw, "Unique subjects\t%d\t%d\n", a.UniqueSubjects, seq.Truth.Visible)
	fmt.Fprintf(tw, "Peak simultaneous\t%d\t%d\n", a.PeakSubjects, seq.Truth.PeakVisible)
	fmt.Fprintf(tw, "Gate crossings L→R (in)\t%d\t%d\n", a.Lines[0].In, seq.Truth.LeftToRight)
	fmt.Fprintf(tw, "Gate crossings R→L (out)\t%d\t%d\n", a.Lines[0].Out, seq.Truth.RightToLeft)
	fmt.Fprintf(tw, "West zone entries / peak\t%d / %d\t–\n", a.Zones[0].Entries, a.Zones[0].Peak)
	fmt.Fprintf(tw, "Avg dwell (completed)\t%.2f s\t–\n", a.Dwell.CompletedAvgSeconds)
	tw.Flush()

	png, err := raw("GET", base+"/api/v1/crowd/sessions/"+id+"/heatmap?format=png&palette=ironbow&cell=20", nil)
	if err != nil {
		return err
	}
	p := filepath.Join(out, "crowd_heatmap.png")
	if err := os.WriteFile(p, png, 0o644); err != nil {
		return err
	}
	fmt.Printf("\nOccupancy heatmap → %s (%d bytes)\n", p, len(png))
	return nil
}

func call(method, url string, in, out any) error {
	b, err := raw(method, url, in)
	if err != nil || out == nil {
		return err
	}
	return json.Unmarshal(b, out)
}

// raw sends one request. It honours the API's back-pressure signals: on
// 429 (rate limited) or 503 (overloaded) it waits for Retry-After and retries,
// so the walkthrough works against a server running the default rate limit.
func raw(method, url string, in any) ([]byte, error) {
	var payload []byte
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return nil, err
		}
		payload = b
	}
	const maxAttempts = 30
	for attempt := 1; ; attempt++ {
		var body io.Reader
		if payload != nil {
			body = bytes.NewReader(payload)
		}
		req, err := http.NewRequest(method, url, body)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		b, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		retryable := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable
		if retryable && attempt < maxAttempts {
			wait := time.Second
			if secs, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && secs >= 0 {
				wait = time.Duration(secs) * time.Second
			}
			time.Sleep(max(wait, 50*time.Millisecond))
			continue
		}
		if resp.StatusCode >= 300 {
			return nil, fmt.Errorf("%s %s: %s: %s", method, url, resp.Status, bytes.TrimSpace(b))
		}
		return b, nil
	}
}
