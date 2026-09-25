# TensorShadow Architecture

TensorShadow has a browser vision front end (MediaPipe + Three.js), a Go
backend for high-concurrency processing, model servers (TensorFlow Serving and
ONNX Runtime), and Python/R tooling for model training and offline analysis.

```text
 ┌──────────────── Browser dashboard (go_backend/web/index.html) ────────────────┐
 │  MediaPipe FaceMesh (≤6 faces) → face boxes ──► POST /crowd/sessions/{id}/frames │
 │  Three.js 3-D projection · live track IDs + dwell overlay ◄── tracks/analytics   │
 └──────────────────────────────────────┬──────────────────────────────────────────┘
          IR cameras (FLIR Lepton, …) ──┤  raw uint16 / JSON frames
          CCTV person detectors ────────┤  detections per frame
                                        ▼
 ┌──────────────────────────── Go backend (go_backend/) ───────────────────────────┐
 │ net/http ─► recover ─► request-ID ─► access log ─► CORS ─► per-route:            │
 │             metrics · token-bucket rate limit (429) · body limit (413)           │
 │                                        │                                         │
 │            I/O-bound: decode JSON / binary on the request goroutine              │
 │                                        ▼                                         │
 │            ┌──────── bounded worker pool (N = CPU cores) ────────┐               │
 │            │ queue full → 503 + Retry-After   · job timeout → 504 │               │
 │            └───────┬─────────────────────────────────┬───────────┘               │
 │                    ▼                                 ▼                           │
 │  internal/thermal (+ internal/coreg)   internal/tracking                         │
 │  rig: RGB boxes → homography → ROIs    Manager (RWMutex, TTL eviction)           │
 │  decode → ε-correction → segment       └─ Session per camera (own mutex)         │
 │  → canthus → screen → liveness            Kalman predict → appearance cascade    │
 │  → false-colour PNG                        (ReID) → gated IoU Hungarian          │
 │                                            → lifecycle → tripwires, zones,       │
 │                                            density, flow, dwell, heatmap → SSE   │
 │                                                                                  │
 │  internal/inference: /predict ─► TF Serving | ONNX Runtime (OIP v2) | baseline   │
 │   timeouts · retries · circuit breaker · bulkhead · degraded fallback            │
 │  /metrics (Prometheus) · /api/v1/stats · graceful shutdown (SIGTERM drains)      │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

## 1. High-concurrency backend (`go_backend/`)

| Concern | Design |
|---|---|
| Routing | Go 1.22 `net/http` method + path patterns. The only dependency is `gopkg.in/yaml.v3`. |
| CPU isolation | Heavy work is submitted to `internal/workerpool`: a fixed set of workers (default = CPU cores) fed by a bounded queue (default 1024). Request goroutines only do I/O and decoding. |
| Back-pressure | `Submit` never blocks on a full queue. It returns `ErrQueueFull` at once, and the handler answers `503 overloaded` with `Retry-After: 1`. Latency stays bounded under overload. |
| Timeouts | Each job runs under `concurrency.job_timeout` (504 on expiry). Jobs whose caller has already gone are skipped when dequeued (`expired` counter). |
| Fairness | A token-bucket rate limiter per client IP (`429 rate_limited` + `Retry-After`). Health and metrics are exempt. |
| Parallelism across streams | Each crowd session has its own mutex. The manager takes a read lock only for lookups, so camera streams never contend with each other. |
| Streaming | `GET …/stream` uses Server-Sent Events. Each subscriber has a 16-frame buffer and a slow consumer drops frames rather than stalling the tracker. The write deadline is lifted per stream through `http.ResponseController`. |
| Observability | `X-Request-ID` on every response; structured `slog` logs (JSON in production); Prometheus `/metrics` covering request counters, latency histograms per route, pool and crowd gauges; JSON `/api/v1/stats`. |
| Resilience | Panics are recovered both in HTTP handlers and inside pool tasks. Idle sessions are evicted by a janitor. On SIGTERM the server stops accepting connections, closes SSE streams, drains in-flight requests and then drains the pool. |
| Hardening | Strict JSON decoding (unknown fields rejected), a body size limit, validated query parameters and a pixel budget per frame. The container image runs as distroless non-root. |

## 2. Infrared thermal biometrics (`internal/thermal`)

1. **Decode**: accepts `celsius`, `kelvin`, `centikelvin` (FLIR Lepton TLinear) or `raw16` with a linear `gain`/`offset` calibration. Input can be JSON or a little-endian `uint16`/`float32` buffer. The binary form is about 9× smaller and about 4× faster end to end.
2. **Emissivity correction**: Stefan–Boltzmann radiometric model
   `T_obj = ((T_app⁴ − (1−ε)·T_refl⁴) / ε)^¼`, with skin ε = 0.98. The reflected temperature defaults to ambient. Per-frame constants are hoisted and no `math.Pow` runs on the hot path.
3. **Segmentation**: 4-connected components of skin-temperature pixels (30–42 °C by default), with a minimum area filter. Callers can supply face ROIs instead, for example from a co-registered visible-light detector.
4. **Screening**: the maximum 3×3-averaged temperature in the periorbital band (15–60 % of face height) gives the **inner canthus**. This is the measurement site recommended by ISO/TR 13154 and IEC 80601-2-59. A configurable offset converts it to a core estimate, which is then classified as `normal` / `elevated` (≥ 37.5 °C) / `fever` (≥ 38.0 °C). A cold or occluded canthus is reported as `indeterminate`.
5. **Liveness (presentation-attack detection)**: photos, screens and masks either sit near ambient or, when heated, show a flat field. There are four explainable checks: physiological skin range, facial thermal gradient (p98 − p2), a periorbital hotspot above the facial median, and contrast against ambient.
6. **Rendering**: ironbow, rainbow, white-hot and black-hot palettes, with optional overlays of status-coloured boxes and canthus crosshairs.

> Screening output is decision support, not a medical diagnosis. Deployments need a blackbody reference and site calibration of `core_offset_c`.

## 3. Multi-subject tracking & crowd analysis (`internal/tracking`)

**Tracker (SORT family).**
- Each track carries four decoupled constant-velocity Kalman filters (cx, cy, w, h), so updates are allocation-free and O(1).
- Association is **gated**: pairs below the IoU threshold are infeasible. The bipartite graph splits into connected components (union-find), and each component is solved optimally with the Hungarian algorithm on `1 − IoU`. In crowds this turns one O(n³) problem into many tiny ones: 396 subjects per frame drop from 22 ms to 2.5 ms.
- Lifecycle: *tentative* until `min_hits` consecutive matches, when a public ID is issued. A track then goes *confirmed* ⇄ *lost* (coasting on prediction) and is removed after `max_age` missed frames.

**Crowd analytics per session.**
- Active, unique and peak subjects.
- Density in persons/m² when `area_m2` is known, otherwise the fraction of the image covered by boxes. Four levels: low / moderate / high / critical.
- Flow: mean velocity, heading, compass direction and coherence (1 = everyone moving the same way).
- Dwell time for active and completed tracks.
- **Tripwires** with directional in/out counts. The previous position only advances on measured frames, so a crossing that happens during a detector dropout is still counted.
- **Zones**: point-in-polygon occupancy, peak, entries, and capacity alarms.
- A spatial occupancy **heatmap**, returned as JSON or a bilinear PNG.

## 4. Browser dashboard (`go_backend/web/index.html`)

The Go binary embeds the dashboard, and the Flask prototype serves the same file.
- MediaPipe FaceMesh now detects up to 6 faces. Their boxes are posted to a per-tab tracking session and drawn back with stable IDs and dwell times.
- Each CDN dependency is optional. If Three.js or MediaPipe cannot load (offline or air-gapped sites), only that feature turns off and the backend link keeps working.

## 5. Verification

- Unit and end-to-end HTTP tests run under `-race` in CI (`.github/workflows/ci.yml`).
- Accuracy gates: 50 randomised synthetic scenes, checked against ground truth for tripwire counts and identities; a further 50 occluded scenes compare re-identification against pure IoU tracking.
- A live parity test (`TestLiveModelServers`, a CI job) runs the Go client against real TF Serving and ONNX Runtime serving the same weights.
- `api/openapi.yaml` is kept in sync with the router by `TestOpenAPICoversAllRoutes`.
- `tsdemo` (feature walkthrough) and `tsload` (closed-loop load generator) run against a live server.

## 6. Model serving (`internal/inference`, `tensorflow_model/`)

```text
 POST /api/v1/predict ─► validate (dims, batch ≤ max_batch, finite)
        │
        ▼  bulkhead: ≤ max_concurrent calls in flight (else 503)
  circuit breaker ── open? ──────────────────────────────┐
        │ closed / half-open probe                        │
        ▼                                                 ▼
  tfserving:   POST /v1/models/{m}[/versions/{v}]:predict    baseline logistic
  onnxruntime: POST /v2/models/{m}[/versions/{v}]/infer     (fallback: true →
               (Open Inference Protocol v2; input name        "degraded": true)
                discovered from GET /v2/models/{m})
        │  timeout per call · retries on 5xx/429/network · 4xx = client error
        ▼
  {result, class, outputs} per instance + backend, model, version, latency
```

- `tensorflow_model/export_models.py` trains the Keras DNN (Dense 128 → 64 → softmax) and exports **the same weights** twice. One copy is a SavedModel for TF Serving (`serving/tensorshadow/1`, signature `serving_default`, input `features`, output `probabilities`), built with Keras `ExportArchive` so that TF Serving can restore the Keras 3 variables. The other is an ONNX graph for ONNX Runtime (`onnx/tensorshadow/1/model.onnx`, opset 17). The exporter fails if the reloaded SavedModel and ONNX Runtime disagree by more than 1e-5.
- `tensorflow_model/onnx_server.py` is a stdlib + `onnxruntime` implementation of the Open Inference Protocol v2 over a Triton-style model repository. The Go client works unchanged against it, against Triton's onnxruntime backend, and against KServe.
- `GET /api/v1/model` reports backend readiness (TF Serving `model_version_status`, or v2 `ready` + metadata) and counters. Prometheus gauges cover requests, failures, fallbacks and breaker state.

## 7. Appearance re-identification (DeepSORT-style)

SORT loses a track after `max_age` missed frames, so anyone hidden longer than that returns with a new ID, and tripwire crossings made while hidden are lost. When detections carry an `embedding`, the tracker adds appearance:

1. **Gallery:** every track stores its last `feature_budget` L2-normalised embeddings. The appearance distance is the minimum cosine distance to the gallery (DeepSORT's nearest-neighbour metric).
2. **Matching cascade (stage A):** confirmed tracks with a gallery are matched first, most recently seen first. A pair is feasible only if its cosine distance is ≤ `appearance_threshold` **and** it passes a **Mahalanobis motion gate** (χ², 2 dof, 99 %) on the Kalman-predicted centre. The gate variance grows while a track coasts, so it widens automatically during an occlusion but still rejects implausible jumps. The same sparse component-wise Hungarian solver as the IoU stage is used.
3. **IoU stage (B):** tentative tracks, tracks without appearance, and anything left over are associated by IoU exactly as before, so detections without embeddings behave like pure SORT.
4. **Longer memory:** tracks with appearance survive `reid_max_age` frames (default 90) instead of `max_age` (15). A match after more than `max_age` frames is counted as a *recovery* (`analytics.reid.recoveries`, `track.reidentified`). Crossings are evaluated from the last *measured* position, so a line crossed behind an occluder is still counted when the person reappears.

Embeddings can come from any ReID or face-recognition network (OSNet/FastReID for people, ArcFace for faces). The dashboard sends a 128-bin Hellinger-normalised colour histogram of the face and hair region: a classic appearance cue that is cheap to compute in the browser.

## 8. Visible/IR co-registration (`internal/coreg`)

- **Calibration:** point correspondences (≥ 4 for a homography, ≥ 3 for affine), typically the corners of a heated calibration target seen by both cameras. The estimator is **normalised DLT** (Hartley conditioning) with linear least squares, and reports the reprojection RMSE and maximum error in thermal pixels, graded good (< 1 px), fair (< 2.5 px) or poor. A known matrix can also be supplied. Rigs are created through the API or preloaded from `coregistration.rigs` in the config.
- **Runtime:** `/thermal/analyze` and `/thermal/render` accept `rig` plus `visible_rois`, the RGB detector's boxes in visible pixels (as `vbox=x,y,w,h[,id]` query parameters for binary uploads). The four corners of each box are projected, their bounding box is taken, enlarged by `margin` (default 10 %) to absorb parallax for subjects off the calibration plane, and clipped to the thermal frame. Only skin-temperature pixels inside the ROI are measured, so the margin does not bias the reading.
- **Traceability:** each thermal subject carries `roi_index` and echoes its `visible_roi` (including the caller's `id`, e.g. a crowd track ID), so fever-screening results join back to RGB identities. Boxes that miss the thermal view are listed in `coregistration.outside_frame`.

## Legacy components

- **TensorFlow** (`tensorflow_model/`): the training scripts (`model.py`, `train.py`, `inference.py`) plus the serving exporter above.
- **R** (`r_analysis/`): `ggplot2` reporting on exported analytics.
