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

	if err := thermalDemo(base, out); err != nil {
		return err
	}
	return crowdDemo(base, out)
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

func raw(method, url string, in any) ([]byte, error) {
	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return nil, err
		}
		body = bytes.NewReader(b)
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
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%s %s: %s: %s", method, url, resp.Status, bytes.TrimSpace(b))
	}
	return b, nil
}
