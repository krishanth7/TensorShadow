# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-09-25

This release completes the remaining roadmap items.

### Added
- **`/api/v1/predict` wired to TensorFlow Serving and ONNX Runtime.**
  - `internal/inference` supports the TF Serving REST API and the Open Inference Protocol v2 (ONNX Runtime via Triton, KServe or the bundled server), alongside the in-process baseline.
  - Batch `instances` requests, per-call timeouts, retries, a circuit breaker, a concurrency bulkhead and an optional degraded fallback.
  - New `GET /api/v1/model` endpoint and Prometheus gauges.
  - `tensorflow_model/export_models.py` exports one trained model as a SavedModel and as ONNX, with a parity check. `tensorflow_model/onnx_server.py` is an Open Inference Protocol v2 server built on ONNX Runtime.
  - A CI job tests parity against real TF Serving and ONNX Runtime.
- **DeepSORT-style appearance re-identification.**
  - Detections can carry an `embedding`.
  - Tracks keep a feature gallery.
  - A matching cascade gated by Mahalanobis distance runs before IoU association.
  - Tracks with appearance survive `reid_max_age` frames.
  - Recoveries are reported in analytics and per track.
  - The dashboard sends colour-histogram embeddings.
- **Visible/IR co-registration.**
  - Rigs are calibrated by normalised DLT (homography or affine) with a reprojection-error grade.
  - `/api/v1/coreg/rigs` endpoints; rigs can also be preloaded from config.
  - `rig` + `visible_rois` (or `vbox`) on the thermal endpoints drive thermal ROIs from RGB face boxes; results link back through `visible_roi`.

### Fixed
- `build_model` passed `optimiser=` to Keras `compile` (the keyword is `optimizer`), which raised `TypeError` whenever TensorFlow was installed.

### Changed
- `/api/v1/predict` no longer runs on the CPU worker pool, because model-server I/O is bounded by the inference bulkhead. The v1 `{"data": [...]}` form and its `result` field are unchanged.

## [2.0.0] - 2026-09-25

This release delivers all three roadmap items.

### Added
- **Go high-concurrency backend.**
  - The Gin/Viper stub (which did not compile) is replaced with a `net/http` server.
  - A bounded worker pool gives back-pressure (503 + `Retry-After`), per-job timeouts and panic isolation.
  - Per-IP token-bucket rate limiting.
  - Request IDs, structured `slog` logging, Prometheus `/metrics` and JSON `/api/v1/stats`.
  - Strict request validation and body limits.
  - Graceful SIGTERM shutdown that also closes SSE streams.
  - Configuration from YAML with defaults and environment overrides.
- **Infrared (IR) thermal biometric scanning** (`/api/v1/thermal/*`).
  - Radiometric decoding: °C, K, centikelvin (FLIR Lepton) and raw16 with gain/offset. Input can be JSON or raw `uint16`/`float32` buffers.
  - Stefan–Boltzmann emissivity correction.
  - Multi-subject skin segmentation.
  - Inner-canthus fever screening.
  - Four-check thermal liveness (presentation-attack detection).
  - False-colour PNG rendering with annotations.
- **Multi-subject tracking & crowd analysis** (`/api/v1/crowd/*`).
  - Kalman-filter tracker with gated, component-wise Hungarian association.
  - Per-camera sessions with tripwire in/out counting, zone occupancy and capacity, density levels, flow, dwell time and an occupancy heatmap (JSON/PNG).
  - Live Server-Sent Events stream.
- Dashboard: multi-face detection (up to 6) with live track IDs from the Go backend, and graceful degradation when CDN libraries are unavailable.
- `tsdemo` walkthrough and `tsload` load generator (`go_backend/cmd/`), a Python client (`clients/python/`), Makefile, Dockerfile and GitHub Actions CI.
- Complete OpenAPI 3 specification, with a test that keeps it in sync with the router.

### Changed
- The dashboard HTML moved to `go_backend/web/index.html`. The Go binary embeds it and the Flask prototype serves the same file.
- `/api/v1/predict` keeps its v1 contract and now returns a deterministic baseline score (`model: baseline-logistic`) instead of a constant.
- `/api/v1/stats` reports live runtime, pool and latency statistics instead of mock data.
- `smoke_test.py` checks the new layout and exits non-zero when anything is missing.

## [1.0.0] - 2026-02-21

- Initial release of the TensorShadow framework: TensorFlow training and
  inference scripts, Go API template, R analytics and documentation.
