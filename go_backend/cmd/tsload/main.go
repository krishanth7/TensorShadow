// Command tsload is a closed-loop HTTP load generator for the TensorShadow
// backend. Each virtual client owns its own crowd session (a camera stream)
// and/or posts thermal frames, then latency percentiles and throughput are
// reported.
//
//	go run ./go_backend/cmd/tsload -addr http://localhost:8080 -c 64 -d 15s -scenario mixed
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/sim"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
)

type sample struct {
	kind   string
	lat    time.Duration
	status int
}

func main() {
	addr := flag.String("addr", "http://localhost:8080", "backend base URL")
	conc := flag.Int("c", 32, "concurrent clients (one camera stream each)")
	dur := flag.Duration("d", 10*time.Second, "test duration")
	scenario := flag.String("scenario", "crowd", "crowd | thermal | mixed")
	encoding := flag.String("thermal-encoding", "binary", "thermal frame encoding: binary (uint16 centikelvin) | json")
	flag.Parse()
	base := strings.TrimRight(*addr, "/")

	transport := &http.Transport{MaxIdleConns: *conc * 2, MaxIdleConnsPerHost: *conc * 2, IdleConnTimeout: time.Minute}
	client := &http.Client{Transport: transport, Timeout: 30 * time.Second}

	seq := sim.DefaultCrowdScene().Generate()
	frames := make([][]byte, len(seq.Frames))
	for i, d := range seq.Frames {
		frames[i], _ = json.Marshal(tracking.FrameInput{Detections: d})
	}
	scene := sim.DemoThermalScene()
	amb := scene.AmbientC
	temps := scene.Render()
	thermalURL := base + "/api/v1/thermal/analyze"
	thermalType := "application/json"
	var thermalBody []byte
	if *encoding == "binary" {
		thermalBody = make([]byte, 2*len(temps))
		for i, c := range temps {
			binary.LittleEndian.PutUint16(thermalBody[2*i:], uint16(math.Round((c+273.15)*100)))
		}
		thermalURL += fmt.Sprintf("?width=%d&height=%d&ambient_c=%g", scene.Width, scene.Height, amb)
		thermalType = "application/octet-stream"
	} else {
		thermalBody, _ = json.Marshal(thermal.FrameRequest{Width: scene.Width, Height: scene.Height,
			Data: temps, AmbientTempC: &amb})
	}

	fmt.Printf("tsload: %d clients, %s, scenario=%s, target=%s\n", *conc, *dur, *scenario, base)
	fmt.Printf("payloads: crowd frame ≈ %d B (avg %.1f detections), thermal frame %d B (%dx%d, %s)\n",
		avgLen(frames), avgDets(seq.Frames), len(thermalBody), scene.Width, scene.Height, *encoding)

	var (
		mu      sync.Mutex
		samples []sample
		errs    atomic.Int64
		wg      sync.WaitGroup
	)
	deadline := time.Now().Add(*dur)
	started := time.Now()
	for c := 0; c < *conc; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			sid := fmt.Sprintf("load-%d-%d", time.Now().UnixNano(), c)
			useCrowd := *scenario == "crowd" || (*scenario == "mixed" && c%4 != 0)
			if useCrowd {
				body, _ := json.Marshal(tracking.SessionConfig{ID: sid, FrameWidth: 1280, FrameHeight: 720,
					Lines: []tracking.Tripwire{{A: tracking.Point{X: 640, Y: 0}, B: tracking.Point{X: 640, Y: 720}}}})
				if st, _, err := post(client, base+"/api/v1/crowd/sessions", "application/json", body); err != nil || st != 201 {
					errs.Add(1)
					return
				}
				defer func() {
					req, _ := http.NewRequest("DELETE", base+"/api/v1/crowd/sessions/"+sid, nil)
					if resp, err := client.Do(req); err == nil {
						resp.Body.Close()
					}
				}()
			}
			local := make([]sample, 0, 4096)
			for i := 0; time.Now().Before(deadline); i++ {
				url, kind, ctype, body := base+"/api/v1/crowd/sessions/"+sid+"/frames", "crowd", "application/json", frames[i%len(frames)]
				if !useCrowd {
					url, kind, ctype, body = thermalURL, "thermal", thermalType, thermalBody
				}
				st, lat, err := post(client, url, ctype, body)
				if err != nil {
					errs.Add(1)
					continue
				}
				local = append(local, sample{kind, lat, st})
			}
			mu.Lock()
			samples = append(samples, local...)
			mu.Unlock()
		}(c)
	}
	wg.Wait()
	elapsed := time.Since(started)
	report(samples, elapsed, errs.Load())
}

func post(c *http.Client, url, ctype string, body []byte) (int, time.Duration, error) {
	t0 := time.Now()
	resp, err := c.Post(url, ctype, bytes.NewReader(body))
	if err != nil {
		return 0, 0, err
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return resp.StatusCode, time.Since(t0), nil
}

func report(samples []sample, elapsed time.Duration, transportErrs int64) {
	byKind := map[string][]sample{}
	for _, s := range samples {
		byKind[s.kind] = append(byKind[s.kind], s)
	}
	kinds := make([]string, 0, len(byKind))
	for k := range byKind {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)

	fmt.Printf("\n%-8s %9s %10s %8s %8s %8s %8s %8s  %s\n", "kind", "requests", "req/s", "p50", "p90", "p99", "max", "mean", "status codes")
	total := 0
	for _, k := range kinds {
		ss := byKind[k]
		total += len(ss)
		lats := make([]time.Duration, len(ss))
		codes := map[int]int{}
		var sum time.Duration
		for i, s := range ss {
			lats[i] = s.lat
			sum += s.lat
			codes[s.status]++
		}
		sort.Slice(lats, func(i, j int) bool { return lats[i] < lats[j] })
		pct := func(q float64) string { return ms(lats[int(q*float64(len(lats)-1))]) }
		fmt.Printf("%-8s %9d %10.0f %8s %8s %8s %8s %8s  %v\n", k, len(ss), float64(len(ss))/elapsed.Seconds(),
			pct(0.50), pct(0.90), pct(0.99), ms(lats[len(lats)-1]), ms(sum/time.Duration(len(ss))), codes)
	}
	fmt.Printf("\ntotal: %d requests in %s → %.0f req/s, transport errors: %d\n",
		total, elapsed.Round(time.Millisecond), float64(total)/elapsed.Seconds(), transportErrs)
	if total == 0 {
		os.Exit(1)
	}
}

func ms(d time.Duration) string { return fmt.Sprintf("%.2fms", float64(d.Microseconds())/1000) }

func avgLen(b [][]byte) int {
	n := 0
	for _, x := range b {
		n += len(x)
	}
	return n / len(b)
}

func avgDets(f [][]tracking.Detection) float64 {
	n := 0
	for _, x := range f {
		n += len(x)
	}
	return float64(n) / float64(len(f))
}
