# TensorShadow Platinum

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go 1.22+](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00bfa5?style=flat&logo=google)](https://mediapipe.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?style=flat&logo=three.js)](https://threejs.org/)
[![OpenAPI 3](https://img.shields.io/badge/OpenAPI-3.0-6BA539?style=flat&logo=openapiinitiative)](api/openapi.yaml)

**TensorShadow** is an enterprise-grade biometric intelligence platform. It combines deep neural networks and real-time computer vision for facial analysis, ocular tracking and dental metrics. A **high-concurrency Go backend** adds **infrared (IR) thermal screening** and **multi-subject crowd analytics**.

---

## 💎 Key Features

- **Tricolour Biometric Segmentation:**
  - 🟠 **Orange Matrix:** 468-point high-density facial topography.
  - 🔵 **Blue Ocular Points:** High-precision iris and pupil tracking.
  - 🟢 **Green Maxillary Scan:** Real-time dental and oral geometry analysis.
- **Neural Demographic Estimation:** Deep-learning-based age and gender estimation with live confidence scoring.
- **Subject-Specific ML Training:** A manual calibration mode that locks neural weights to a specific face for higher accuracy.
- **3D Neural Projection:** Live WebGL/Three.js volumetric output showing real-time Z-depth and head orientation.
- **Enterprise UI:** A professional dashboard with millisecond latency monitoring and biometric data export.

### 🆕 New in v2.0
- ⚡ **Go High-Concurrency Backend:**
  - A bounded worker pool with back-pressure (503 + `Retry-After`), per-IP rate limiting and per-job timeouts.
  - Prometheus metrics and graceful shutdown.
  - **~11,800 tracking frames/s** on 4 vCPUs.
- 🌡️ **Infrared Thermal Biometrics:**
  - Radiometric decoding, including raw FLIR Lepton `uint16` buffers.
  - Emissivity correction.
  - Multi-face segmentation.
  - **Inner-canthus fever screening.**
  - **Thermal liveness / anti-spoofing.**
  - False-colour rendering.
- 👥 **Multi-Subject Tracking & Crowd Analysis:**
  - A Kalman + Hungarian tracker with stable IDs.
  - Tripwire in/out counting.
  - Zone occupancy and capacity alarms.
  - Density levels, crowd flow, dwell time and occupancy heatmaps.
  - A live SSE stream.
  - The dashboard now tracks **up to 6 faces** at once.

---

## 🏗️ Architecture

```text
 Browser dashboard ──(face boxes)──┐        ┌──────────── Go backend ────────────┐
 IR cameras (raw uint16 / JSON) ───┼──────► │ middleware: request-ID · logging · │
 CCTV person detectors ────────────┘        │ CORS · rate limit · body limit     │
                                            │            ▼                        │
                                            │ bounded worker pool (back-pressure)│
                                            │     ▼                     ▼        │
                                            │ internal/thermal   internal/tracking│
                                            │ ε-correction       Kalman + gated   │
                                            │ segmentation       Hungarian, lines,│
                                            │ canthus · liveness zones, heatmap   │
                                            │            ▼                        │
                                            │ JSON · PNG · SSE · /metrics         │
                                            └─────────────────────────────────────┘
```

- **Neural Engine:** MediaPipe FaceMesh (multi-face) + TensorFlow.js.
- **Visualiser:** Three.js 3D rendering + Canvas 2D overlay with live track IDs.
- **Backend:** Go `net/http` service with a worker pool, zero-dependency Prometheus metrics and an embedded dashboard. The Flask server remains as a front-end prototype.
- **ML Pipe:** Scikit-Learn MLP fallback for localised "on-the-fly" training.

The full design is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and every endpoint is specified in [`api/openapi.yaml`](api/openapi.yaml).

---

## 📁 Project Structure

```text
TensorShadow/
├── api/openapi.yaml          # OpenAPI 3 spec (kept in sync with the router by a test)
├── clients/python/           # Dependency-free Python client
├── configs/config.yaml       # Server, concurrency, thermal and tracking settings
├── data/                     # Training datasets and stats
├── docs/                     # Architecture notes and generated output images
├── go_backend/
│   ├── main.go               # Server entrypoint (graceful shutdown)
│   ├── cmd/tsdemo/           # End-to-end feature walkthrough
│   ├── cmd/tsload/           # Closed-loop HTTP load generator
│   ├── internal/
│   │   ├── workerpool/       # Bounded pool with back-pressure
│   │   ├── thermal/          # IR decoding, screening, liveness, rendering
│   │   ├── tracking/         # Multi-subject tracker + crowd analytics
│   │   ├── server/           # HTTP API, middleware, SSE
│   │   ├── metrics/          # Prometheus exposition
│   │   ├── config/  colormap/  sim/
│   ├── web/index.html        # Dashboard (embedded in the Go binary)
│   └── sim_server.py         # Flask prototype serving the same dashboard
├── r_analysis/               # Statistical reporting scripts
├── tensorflow_model/         # ML training and model architecture
├── Dockerfile  Makefile  requirements.txt
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Go 1.22+ (backend)
- Python 3.9+ and pip (ML tooling, Flask prototype)

### 2. Run the Go backend and dashboard
```bash
cd TensorShadow
go run ./go_backend            # or: make build && ./bin/go_backend
```
Open **http://localhost:8080** in your browser. The dashboard connects to the crowd-tracking API automatically and labels every face with a persistent ID.

Useful flags and settings: `-addr :9090`, `-log-level debug`, `-config path.yaml`. The environment variables `TS_PORT`, `TS_WORKERS`, `TS_QUEUE_SIZE` and `TS_RATE_LIMIT_RPS` override the config file.

### 3. See every feature in 5 seconds
```bash
make demo        # runs go_backend/cmd/tsdemo against the running server
```

### 4. Python tooling (unchanged)
```bash
pip install -r requirements.txt
python go_backend/sim_server.py   # front-end-only prototype (no Go APIs)
```

### 5. Docker
```bash
docker build -t tensorshadow . && docker run -p 8080:8080 tensorshadow
```

---

## 🧠 Usage Examples

### ML Calibration
Use the **Calibration Panel** on the left to train the model to your specific face for better age and gender accuracy.

### 3D Volumetric Scan
The right panel shows a live rotating point cloud of your facial structure. Turn your head to see depth sensing in action.

### Infrared thermal screening
Upload a radiometric frame. The most efficient format is a raw FLIR Lepton-style buffer of little-endian `uint16` centikelvin values:
```bash
curl -X POST -H 'Content-Type: application/octet-stream' --data-binary @frame.bin \
  'http://localhost:8080/api/v1/thermal/analyze?width=160&height=120&ambient_c=22'

# False-colour PNG with status boxes and canthus crosshairs
curl -X POST -H 'Content-Type: application/octet-stream' --data-binary @frame.bin -o thermal.png \
  'http://localhost:8080/api/v1/thermal/render?width=160&height=120&palette=ironbow&scale=4&annotate=true'
```
JSON input also works (`{"width":160,"height":120,"format":"celsius","data":[…]}`), as do the `kelvin` and `raw16` (with `gain`/`offset`) formats.

### Multi-subject crowd tracking
Open one session per camera, then post each frame's detections (from MediaPipe, YOLO or any detector):
```bash
curl -X POST localhost:8080/api/v1/crowd/sessions -H 'Content-Type: application/json' -d '{
  "id": "lobby-cam-1", "frame_width": 1280, "frame_height": 720, "area_m2": 60,
  "lines": [{"id": "gate", "a": {"x": 640, "y": 0}, "b": {"x": 640, "y": 720}}],
  "zones": [{"id": "queue", "capacity": 10,
             "polygon": [{"x":0,"y":0},{"x":640,"y":0},{"x":640,"y":720},{"x":0,"y":720}]}]}'

curl -X POST localhost:8080/api/v1/crowd/sessions/lobby-cam-1/frames -H 'Content-Type: application/json' \
  -d '{"detections": [{"bbox": {"x": 600, "y": 200, "w": 60, "h": 150}, "score": 0.94, "label": "person"}]}'

curl -N localhost:8080/api/v1/crowd/sessions/lobby-cam-1/stream        # live SSE feed
curl -o heat.png 'localhost:8080/api/v1/crowd/sessions/lobby-cam-1/heatmap?format=png'
```

### Python client
```python
from tensorshadow_client import TensorShadowClient   # clients/python/
ts = TensorShadowClient("http://localhost:8080")
result = ts.thermal_analyze_raw(buffer, width=160, height=120, ambient_c=22.0)
ts.create_session("lobby-cam-1", 1280, 720, area_m2=60)
frame = ts.post_frame("lobby-cam-1", [{"bbox": {"x": 600, "y": 200, "w": 60, "h": 150}}])
```

### API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness probe |
| `GET` | `/api/v1/stats` | Runtime, pool, latency and crowd statistics |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/api/v1/predict` | v1-compatible model scoring |
| `POST` | `/api/v1/thermal/analyze` | Segment, screen and liveness-check all faces |
| `POST` | `/api/v1/thermal/render` | False-colour PNG (ironbow, rainbow, whitehot, blackhot) |
| `POST/GET` | `/api/v1/crowd/sessions` | Create or list tracking sessions |
| `GET/DELETE` | `/api/v1/crowd/sessions/{id}` | Describe or close a session |
| `POST` | `/api/v1/crowd/sessions/{id}/frames` | Ingest detections → tracks + analytics |
| `GET` | `/api/v1/crowd/sessions/{id}/analytics` | Latest analytics snapshot |
| `GET` | `/api/v1/crowd/sessions/{id}/heatmap` | Occupancy heatmap (JSON or `?format=png`) |
| `GET` | `/api/v1/crowd/sessions/{id}/stream` | Live Server-Sent Events |

---

## 📊 Outputs

Every output below was produced by the code in this repository. The scenes are synthetic and generated with known ground truth (`go_backend/internal/sim`), so accuracy is measured rather than estimated. Timings were taken on a **4 vCPU Intel Xeon @ 2.1 GHz** cloud container with Go 1.24, and the load generator ran **on the same machine**, so the throughput figures are conservative.

### 1. Infrared thermal screening (`make demo`)

The reference frame is a 160×120 radiometric image of three real faces at different body temperatures plus one **heated spoof** (a warmed mask/screen at a skin-like 33.5 °C):

```text
━━ Infrared thermal screening ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame 160x120  range 21.86–38.65 °C  ambient 22.0 °C (provided)  ε=0.98  processed in 1.51 ms

Subject  ROI (x,y,w,h)  Skin mean  Canthus   Truth     Core est.  Status         Liveness
#0       11,20,29,37    34.29 °C   36.62 °C  36.60 °C  36.92 °C   NORMAL         LIVE (1.00)
#1       48,23,31,39    34.88 °C   37.40 °C  37.40 °C  37.70 °C   ELEVATED       LIVE (1.00)
#2       87,20,29,37    35.72 °C   38.52 °C  38.50 °C  38.82 °C   FEVER          LIVE (1.00)
#3       125,24,29,37   33.50 °C   33.56 °C  spoof     33.86 °C   INDETERMINATE  SPOOF (0.50): facial_thermal_gradient, periorbital_hotspot
```

The measured canthus temperature is within **0.02 °C** of ground truth for every real face. The spoof passes the temperature-range checks but is caught because it has no facial thermal gradient and no periorbital hotspot.

<p align="center">
  <img src="docs/images/thermal_ironbow.png" width="640" alt="Annotated ironbow thermal image: green normal, amber elevated, red fever, blue spoof; white crosshairs mark the measured inner canthus">
  <br><em><code>POST /api/v1/thermal/render?palette=ironbow&amp;annotate=true</code>. Box colours: 🟩 normal · 🟨 elevated · 🟥 fever · 🟦 spoof suspected. The crosshair marks the measured inner canthus.</em>
</p>

<details>
<summary>Sample <code>/thermal/analyze</code> response (fever subject, from a raw <code>uint16</code> centikelvin upload)</summary>

```json
{
  "roi": { "x": 87, "y": 20, "w": 29, "h": 37 },
  "skin": { "pixels": 785, "min_c": 34.36, "max_c": 38.65, "mean_c": 35.72, "std_c": 0.75, "p50_c": 35.55, "p90_c": 36.29 },
  "canthus": { "x": 99, "y": 35, "temp_c": 38.52 },
  "core_estimate_c": 38.82,
  "status": "fever",
  "liveness": {
    "live": true, "score": 1,
    "checks": [
      { "name": "physiological_skin_range", "passed": true, "value": 35.72, "limit": "31.0–38.5 °C" },
      { "name": "facial_thermal_gradient",  "passed": true, "value": 3.46,  "limit": ">= 1.5 °C (p98−p2)" },
      { "name": "periorbital_hotspot",      "passed": true, "value": 2.97,  "limit": ">= 0.30 °C above facial median" },
      { "name": "ambient_contrast",         "passed": true, "value": 13.72, "limit": ">= 4 °C above ambient" }
    ]
  }
}
```
</details>

### 2. Multi-subject tracking & crowd analysis (`make demo`)

The reference scene is 720p at 10 FPS with 24 pedestrians walking in both directions. Detections include 1.5 px localisation jitter and 3 % missed detections. A tripwire sits at the centre gate and a capacity-10 zone covers the west concourse.

```text
━━ Multi-subject tracking & crowd analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
frame  60  active  6  unique  6  gate in/out  0/0   west zone  5  density 0.10/m² (low)  flow east
frame 120  active 14  unique 14  gate in/out  2/0   west zone  5  density 0.23/m² (low)  flow stationary
frame 180  active 22  unique 23  gate in/out  6/3   west zone  8  density 0.37/m² (low)  flow west
frame 240  active 17  unique 24  gate in/out  9/9   west zone 10  density 0.28/m² (low)  flow west
frame 300  active  9  unique 24  gate in/out 12/12  west zone  5  density 0.15/m² (low)  flow west

300 frames posted, mean round-trip 0.60 ms/frame

Metric                    Tracker  Ground truth
Unique subjects           24       24
Peak simultaneous         23       23
Gate crossings L→R (in)   12       12
Gate crossings R→L (out)  12       12
West zone entries / peak  24 / 13  –
Avg dwell (completed)     16.27 s  –
```

"Flow stationary" at frame 120 is correct: two equal crowds walking in opposite directions cancel out, and the API reports this through the `coherence` field.

<p align="center">
  <img src="docs/images/crowd_heatmap.png" width="640" alt="Crowd occupancy heatmap showing horizontal pedestrian lanes">
  <br><em><code>GET /api/v1/crowd/sessions/{id}/heatmap?format=png</code>: occupancy accumulated over 300 frames. The walking lanes are clearly visible.</em>
</p>

**Accuracy across 50 randomised scenes** (`TestAccuracyAcrossSeeds`, enforced in CI):

| Metric | Result |
|---|---|
| Tripwire crossings counted correctly | **1,200 / 1,200 (100 %)** |
| Identities correct (no ID switch or fragmentation) | **1,199 / 1,200 (99.9 %)** |

<details>
<summary>Sample <code>/frames</code> response (4th frame of two people; one has just crossed the gate)</summary>

```json
{
  "session_id": "lobby-cam-1", "frame": 4, "timestamp": "2026-09-25T10:00:03Z",
  "tracks": [
    { "id": 1, "state": "confirmed", "bbox": { "x": 623.98, "y": 200, "w": 60, "h": 150 },
      "center": { "x": 653.98, "y": 275 }, "velocity": { "x": 7.993, "y": 0 },
      "age_frames": 4, "hits": 4, "dwell_seconds": 3, "score": 0.94, "label": "person" },
    { "id": 2, "state": "confirmed", "bbox": { "x": 200, "y": 300, "w": 58, "h": 148 },
      "center": { "x": 229, "y": 374 }, "velocity": { "x": 0, "y": 0 },
      "age_frames": 4, "hits": 4, "dwell_seconds": 3, "score": 0.88, "label": "person" }
  ],
  "analytics": {
    "active_subjects": 2, "unique_subjects": 2, "peak_subjects": 2, "frames_processed": 4,
    "density": { "per_m2": 0.033, "coverage_ratio": 0.019, "level": "low", "basis": "persons_per_m2" },
    "flow": { "mean_vx": 3.997, "mean_vy": 0, "speed_px_per_frame": 3.997, "heading_deg": 0, "direction": "east", "coherence": 1 },
    "dwell": { "active_avg_seconds": 3, "active_max_seconds": 3, "completed_avg_seconds": 0, "completed_tracks": 0 },
    "lines": [ { "id": "gate", "in": 1, "out": 0, "net": 1 } ],
    "zones": [ { "id": "queue", "occupancy": 1, "peak": 1, "entries": 1, "capacity": 10, "utilisation": 0.1, "over_capacity": false } ]
  }
}
```
</details>

### 3. Dashboard multi-subject mode

<p align="center">
  <img src="docs/images/dashboard_crowd.png" width="800" alt="Dashboard showing three tracked subjects with IDs and dwell times and the Crowd Analysis panel reporting backend ONLINE">
  <br><em>Headless-browser capture of the dashboard served by the Go backend. The sandbox had no camera and no CDN access, so synthetic face landmarks were fed through the dashboard's own <code>crowdSubmit()</code> path. The IDs and dwell times come from the Go tracker, and the page reports disabled 3D and vision features instead of crashing.</em>
</p>

### 4. High-concurrency load tests (`go_backend/cmd/tsload`)

These are closed-loop tests of 15 s each, with the rate limiter disabled (`TS_RATE_LIMIT_RPS=0`) and default config (4 workers, queue 1024):

| Scenario | Clients | Throughput | p50 | p90 | p99 | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Crowd tracking frames (12.5 detections avg) | 64 | **11,806 req/s** | 3.51 ms | 13.11 ms | 24.56 ms | 0 |
| Thermal 160×120, raw `uint16` (38 KB) | 32 | **1,927 req/s** | 15.92 ms | 22.42 ms | 30.63 ms | 0 |
| Thermal 160×120, JSON (356 KB) | 32 | 460 req/s | 50.08 ms | 153.65 ms | 286.88 ms | 0 |
| Mixed: 75 % crowd / 25 % thermal | 64 | 5,258 req/s | 10.72 / 13.56 ms | 18.81 / 22.56 ms | 28.72 / 33.59 ms | 0 |

After roughly 293,000 requests the server reported `errors_total: 0`, `rejected: 0` and 11 goroutines at rest (no leaks), and every load-test session had been cleaned up. Binary thermal ingest is **4.2× faster** than JSON.

Back-pressure and rate limiting were checked as well. With the default per-IP limit of 200 req/s, the same crowd test was answered with fast `429` + `Retry-After` responses (p50 1.0 ms). A saturated pool returns `503 overloaded` + `Retry-After: 1` (`TestBackpressureReturns503`).

### 5. Micro-benchmarks (`make bench`)

```text
BenchmarkAnalyze160x120-4        1221     890028 ns/op    651713 B/op     98 allocs/op
BenchmarkAnalyze640x480-4          56   18682458 ns/op  13295445 B/op    138 allocs/op
BenchmarkTrackerFrame-4         65402      16166 ns/op      8507 B/op    167 allocs/op
BenchmarkTrackerDense4K-4         394    2616764 ns/op    396.0 subjects/frame
BenchmarkSubmit-4             1211054        957 ns/op       144 B/op      2 allocs/op
```

| Workload | Before optimisation | After |
|---|---:|---:|
| Thermal analysis, 160×120 | 4.32 ms | **0.89 ms** (4.9×): no `math.Pow` on the hot path, O(n) median |
| Thermal analysis, 640×480 | 79.1 ms | **18.7 ms** (4.2×) |
| Tracker, dense 4K scene (396 subjects/frame) | 22.4 ms | **2.6 ms** (8.6×): gated, component-wise Hungarian |

### 6. Test suite (`make race`, `make cover`)

```text
ok   go_backend/internal/colormap     coverage: 93.3% of statements
ok   go_backend/internal/config       coverage: 77.1% of statements
ok   go_backend/internal/metrics      coverage: 94.2% of statements
ok   go_backend/internal/server       coverage: 81.2% of statements
ok   go_backend/internal/thermal      coverage: 91.7% of statements
ok   go_backend/internal/tracking     coverage: 91.2% of statements
ok   go_backend/internal/workerpool   coverage: 88.9% of statements
```

All packages pass under the race detector, on both Go 1.22 and Go 1.24. The server suite also passed 100 consecutive stress runs. CI (`.github/workflows/ci.yml`) runs gofmt, vet and the race tests, then boots the binary and runs `tsdemo` and the Python client against it.

---

## 🗺️ Roadmap
- [x] Implement Go-based high-concurrency backend.
- [x] Add Infrared (IR) support for thermal biometric scanning.
- [x] Integrate Multi-Subject tracking (Crowd Analysis).
- [ ] Wire `/api/v1/predict` to TensorFlow Serving / ONNX Runtime.
- [ ] Appearance embeddings (DeepSORT-style re-identification) for long occlusions.
- [ ] Visible/IR co-registration to drive thermal ROIs from the RGB face detector.

> ⚠️ Thermal screening output is decision support, not a medical diagnosis. Production deployments need a blackbody reference source and site calibration of `thermal.core_offset_c`.

## 🤝 Contributing
Contributions are welcome! Please follow the corporate development guidelines in [CONTRIBUTING.md](CONTRIBUTING.md). Run `make lint race` before opening a pull request.

## 📄 License
This project is licensed under the MIT License.
