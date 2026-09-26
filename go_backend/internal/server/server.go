// Package server exposes the TensorShadow HTTP API: health and metrics,
// infrared thermal screening, and multi-subject crowd tracking.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"net/http"
	"runtime"
	"strconv"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/config"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/coreg"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/inference"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/metrics"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/workerpool"
)

// Version is the API version reported by /health and /stats.
const Version = "2.1.0"

// Server wires configuration, the worker pool and domain services to HTTP.
type Server struct {
	cfg     config.Config
	pool    *workerpool.Pool
	crowd   *tracking.Manager
	infer   *inference.Engine
	coreg   *coreg.Registry
	metrics *metrics.Registry
	limiter *rateLimiter
	log     *slog.Logger
	web     fs.FS
	started time.Time
	handler http.Handler
	routes  []string // registered "METHOD /path" patterns
}

// Deps are the collaborators a Server needs.
type Deps struct {
	Pool    *workerpool.Pool
	Crowd   *tracking.Manager
	Infer   *inference.Engine // optional; defaults to the in-process baseline
	Coreg   *coreg.Registry   // optional; defaults to an empty rig registry
	Metrics *metrics.Registry
	Logger  *slog.Logger
	Web     fs.FS // optional; serves index.html at "/"
}

// New builds the server and its routing table.
func New(cfg config.Config, d Deps) *Server {
	if d.Logger == nil {
		d.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if d.Metrics == nil {
		d.Metrics = metrics.New()
	}
	if d.Infer == nil {
		d.Infer, _ = inference.New(inference.DefaultConfig(), nil)
	}
	if d.Coreg == nil {
		d.Coreg, _ = coreg.NewRegistry(nil)
	}
	s := &Server{
		cfg: cfg, pool: d.Pool, crowd: d.Crowd, infer: d.Infer, coreg: d.Coreg, metrics: d.Metrics, log: d.Logger, web: d.Web,
		limiter: newRateLimiter(cfg.Concurrency.RateLimitRPS, cfg.Concurrency.RateLimitBurst),
		started: time.Now(),
	}
	s.registerGauges()

	mux := http.NewServeMux()
	s.route(mux, "GET /api/v1/health", false, s.handleHealth)
	s.route(mux, "GET /api/v1/stats", true, s.handleStats)
	s.route(mux, "POST /api/v1/predict", true, s.handlePredict)
	s.route(mux, "GET /api/v1/model", true, s.handleModelInfo)

	s.route(mux, "POST /api/v1/thermal/analyze", true, s.handleThermalAnalyze)
	s.route(mux, "POST /api/v1/thermal/render", true, s.handleThermalRender)
	s.route(mux, "GET /api/v1/thermal/palettes", true, s.handlePalettes)

	s.route(mux, "POST /api/v1/coreg/rigs", true, s.handleCreateRig)
	s.route(mux, "GET /api/v1/coreg/rigs", true, s.handleListRigs)
	s.route(mux, "GET /api/v1/coreg/rigs/{id}", true, s.handleGetRig)
	s.route(mux, "DELETE /api/v1/coreg/rigs/{id}", true, s.handleDeleteRig)
	s.route(mux, "POST /api/v1/coreg/rigs/{id}/map", true, s.handleMapBoxes)

	s.route(mux, "POST /api/v1/crowd/sessions", true, s.handleCreateSession)
	s.route(mux, "GET /api/v1/crowd/sessions", true, s.handleListSessions)
	s.route(mux, "GET /api/v1/crowd/sessions/{id}", true, s.handleGetSession)
	s.route(mux, "DELETE /api/v1/crowd/sessions/{id}", true, s.handleDeleteSession)
	s.route(mux, "POST /api/v1/crowd/sessions/{id}/frames", true, s.handleFrame)
	s.route(mux, "GET /api/v1/crowd/sessions/{id}/analytics", true, s.handleAnalytics)
	s.route(mux, "GET /api/v1/crowd/sessions/{id}/heatmap", true, s.handleHeatmap)
	s.route(mux, "GET /api/v1/crowd/sessions/{id}/stream", false, s.handleStream)

	s.route(mux, "GET /metrics", false, s.handleMetrics)
	s.route(mux, "GET /{$}", false, s.handleDashboard)

	s.handler = s.withRecovery(withRequestID(s.withAccessLog(s.withCORS(mux))))
	return s
}

// Handler returns the root HTTP handler.
func (s *Server) Handler() http.Handler { return s.handler }

// route registers h with per-route metrics and optional rate limiting.
func (s *Server) route(mux *http.ServeMux, pattern string, limited bool, h http.HandlerFunc) {
	s.routes = append(s.routes, pattern)
	mux.Handle(pattern, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w}
		defer func() { s.metrics.ObserveRequest(pattern, r.Method, rec.code(), time.Since(start)) }()

		if limited && s.limiter != nil {
			if ok, wait := s.limiter.allow(clientIP(r)); !ok {
				rec.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(wait.Seconds()))))
				writeError(rec, r, http.StatusTooManyRequests, "rate_limited", "request rate limit exceeded")
				return
			}
		}
		if r.Body != nil {
			r.Body = http.MaxBytesReader(rec, r.Body, s.cfg.Server.MaxBodyBytes)
		}
		h(rec, r)
	}))
}

func (s *Server) registerGauges() {
	for _, g := range []metrics.Gauge{
		{Name: "tensorshadow_pool_workers", Help: "Worker goroutines.", Fn: func() float64 { return float64(s.pool.Stats().Workers) }},
		{Name: "tensorshadow_pool_queue_depth", Help: "Jobs waiting in the queue.", Fn: func() float64 { return float64(s.pool.Stats().QueueDepth) }},
		{Name: "tensorshadow_pool_in_flight", Help: "Jobs executing.", Fn: func() float64 { return float64(s.pool.Stats().InFlight) }},
		{Name: "tensorshadow_pool_completed_total", Help: "Jobs completed.", Fn: func() float64 { return float64(s.pool.Stats().Completed) }},
		{Name: "tensorshadow_pool_rejected_total", Help: "Jobs rejected because the queue was full.", Fn: func() float64 { return float64(s.pool.Stats().Rejected) }},
		{Name: "tensorshadow_crowd_sessions", Help: "Active crowd tracking sessions.", Fn: func() float64 { n, _ := s.crowd.Totals(); return float64(n) }},
		{Name: "tensorshadow_crowd_active_subjects", Help: "Subjects currently tracked across all sessions.", Fn: func() float64 { _, n := s.crowd.Totals(); return float64(n) }},
		{Name: "tensorshadow_inference_requests_total", Help: "Prediction requests.", Fn: func() float64 { return float64(s.infer.Stats().Requests) }},
		{Name: "tensorshadow_inference_backend_failures_total", Help: "Model-server calls that failed after retries.", Fn: func() float64 { return float64(s.infer.Stats().Failures) }},
		{Name: "tensorshadow_inference_fallbacks_total", Help: "Predictions served by the baseline fallback.", Fn: func() float64 { return float64(s.infer.Stats().Fallbacks) }},
		{Name: "tensorshadow_inference_breaker_open", Help: "1 while the model-server circuit breaker is open.", Fn: func() float64 {
			if s.infer.Stats().BreakerOpen {
				return 1
			}
			return 0
		}},
		{Name: "tensorshadow_goroutines", Help: "Goroutines.", Fn: func() float64 { return float64(runtime.NumGoroutine()) }},
	} {
		s.metrics.RegisterGauge(g)
	}
}

// apiError is a client-facing error with an HTTP status.
type apiError struct {
	status  int
	code    string
	message string
}

func (e *apiError) Error() string { return e.message }

func badRequest(format string, a ...any) error {
	return &apiError{http.StatusBadRequest, "invalid_request", fmt.Sprintf(format, a...)}
}

type errorBody struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	RequestID string `json:"request_id,omitempty"`
}

// writeAPIError writes an *apiError (or a generic 400 for other errors).
func (s *Server) writeAPIError(w http.ResponseWriter, r *http.Request, err error) {
	var ae *apiError
	if errors.As(err, &ae) {
		writeError(w, r, ae.status, ae.code, ae.message)
		return
	}
	writeError(w, r, http.StatusBadRequest, "invalid_request", err.Error())
}

func writeError(w http.ResponseWriter, r *http.Request, status int, code, msg string) {
	var b errorBody
	b.Error.Code, b.Error.Message, b.RequestID = code, msg, requestID(r.Context())
	writeJSON(w, status, b)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// decodeJSON strictly decodes the request body into v.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
				fmt.Sprintf("request body exceeds %d bytes", tooBig.Limit))
			return false
		}
		writeError(w, r, http.StatusBadRequest, "invalid_json", err.Error())
		return false
	}
	if dec.More() {
		writeError(w, r, http.StatusBadRequest, "invalid_json", "request body must contain a single JSON value")
		return false
	}
	return true
}

// exec runs CPU-bound work on the worker pool and maps pool errors to HTTP.
// It returns ok=false after writing an error response.
func (s *Server) exec(w http.ResponseWriter, r *http.Request, task workerpool.Task) (any, bool) {
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Concurrency.JobTimeout)
	defer cancel()
	v, err := s.pool.Submit(ctx, task)
	if err == nil {
		return v, true
	}
	var ae *apiError
	switch {
	case errors.As(err, &ae):
		writeError(w, r, ae.status, ae.code, ae.message)
	case errors.Is(err, workerpool.ErrQueueFull):
		w.Header().Set("Retry-After", "1")
		writeError(w, r, http.StatusServiceUnavailable, "overloaded", "server is at capacity, retry shortly")
	case errors.Is(err, workerpool.ErrClosed):
		writeError(w, r, http.StatusServiceUnavailable, "shutting_down", "server is shutting down")
	case errors.Is(err, context.DeadlineExceeded):
		writeError(w, r, http.StatusGatewayTimeout, "timeout", "processing exceeded the job timeout")
	case errors.Is(err, context.Canceled):
		// Client went away; nothing useful to send.
	default:
		s.log.Error("task failed", "error", err, "request_id", requestID(r.Context()))
		writeError(w, r, http.StatusInternalServerError, "internal", "internal server error")
	}
	return nil, false
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":         "up",
		"service":        "TensorShadow-backend",
		"version":        Version,
		"uptime_seconds": math.Round(time.Since(s.started).Seconds()*10) / 10,
	})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	sessions, subjects := s.crowd.Totals()
	writeJSON(w, http.StatusOK, map[string]any{
		"service":   "TensorShadow-backend",
		"version":   Version,
		"env":       s.cfg.Server.Env,
		"http":      s.metrics.Summary(),
		"pool":      s.pool.Stats(),
		"crowd":     map[string]int{"sessions": sessions, "active_subjects": subjects},
		"inference": map[string]any{"backend": s.infer.Config().Backend, "stats": s.infer.Stats()},
		"runtime": map[string]any{
			"go_version":    runtime.Version(),
			"num_cpu":       runtime.NumCPU(),
			"goroutines":    runtime.NumGoroutine(),
			"heap_alloc_mb": math.Round(float64(ms.HeapAlloc)/(1<<20)*100) / 100,
			"gc_cycles":     ms.NumGC,
		},
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_ = s.metrics.WritePrometheus(w)
}

func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	if s.web == nil {
		writeJSON(w, http.StatusOK, map[string]string{"service": "TensorShadow-backend", "docs": "/api/v1/health"})
		return
	}
	b, err := fs.ReadFile(s.web, "index.html")
	if err != nil {
		writeError(w, r, http.StatusNotFound, "not_found", "dashboard not bundled")
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(b)
}
