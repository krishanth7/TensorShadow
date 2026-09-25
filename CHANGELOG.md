# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
