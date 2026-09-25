// Package inference routes /api/v1/predict to a model server.
//
// Two production protocols are supported:
//
//   - "tfserving":   TensorFlow Serving REST API (POST /v1/models/{m}:predict)
//   - "onnxruntime": Open Inference Protocol v2 (POST /v2/models/{m}/infer), as
//     served for ONNX Runtime models by Triton's onnxruntime backend, KServe,
//     or the bundled tensorflow_model/onnx_server.py.
//
// An in-process "baseline" logistic scorer needs no server. Remote backends
// get per-call timeouts, bounded retries, a circuit breaker and an optional
// fallback to the baseline, so a model-server outage degrades the endpoint
// instead of failing it.
package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Backend names.
const (
	BackendBaseline    = "baseline"
	BackendTFServing   = "tfserving"
	BackendONNXRuntime = "onnxruntime"
)

// Config selects and tunes the model backend.
type Config struct {
	Backend          string        `yaml:"backend" json:"backend"`
	URL              string        `yaml:"url" json:"url"`
	Model            string        `yaml:"model" json:"model"`
	Version          string        `yaml:"version" json:"version,omitempty"`
	InputName        string        `yaml:"input_name" json:"input_name,omitempty"`   // v2; discovered from metadata when empty
	OutputName       string        `yaml:"output_name" json:"output_name,omitempty"` // optional output selection
	InputDim         int           `yaml:"input_dim" json:"input_dim"`               // 0 = do not validate
	MaxBatch         int           `yaml:"max_batch" json:"max_batch"`
	Timeout          time.Duration `yaml:"timeout" json:"timeout"`
	Retries          int           `yaml:"retries" json:"retries"`
	MaxConcurrent    int           `yaml:"max_concurrent" json:"max_concurrent"`
	Fallback         bool          `yaml:"fallback" json:"fallback"`
	BreakerThreshold int           `yaml:"breaker_threshold" json:"breaker_threshold"`
	BreakerCooldown  time.Duration `yaml:"breaker_cooldown" json:"breaker_cooldown"`
}

// DefaultConfig uses the in-process baseline.
func DefaultConfig() Config {
	return Config{
		Backend:          BackendBaseline,
		Model:            "tensorshadow",
		InputDim:         0,
		MaxBatch:         256,
		Timeout:          2 * time.Second,
		Retries:          1,
		MaxConcurrent:    64,
		Fallback:         true,
		BreakerThreshold: 5,
		BreakerCooldown:  10 * time.Second,
	}
}

// Validate reports configuration errors.
func (c Config) Validate() error {
	switch c.Backend {
	case BackendBaseline:
	case BackendTFServing, BackendONNXRuntime:
		if !strings.HasPrefix(c.URL, "http://") && !strings.HasPrefix(c.URL, "https://") {
			return fmt.Errorf("inference.url must be an http(s) URL for backend %q", c.Backend)
		}
		if c.Model == "" {
			return errors.New("inference.model is required for remote backends")
		}
	default:
		return fmt.Errorf("inference.backend must be baseline, tfserving or onnxruntime, got %q", c.Backend)
	}
	switch {
	case c.Timeout <= 0:
		return errors.New("inference.timeout must be positive")
	case c.Retries < 0 || c.Retries > 5:
		return errors.New("inference.retries must be in [0,5]")
	case c.MaxBatch < 1:
		return errors.New("inference.max_batch must be >= 1")
	case c.MaxConcurrent < 1:
		return errors.New("inference.max_concurrent must be >= 1")
	case c.BreakerThreshold < 1 || c.BreakerCooldown <= 0:
		return errors.New("inference breaker settings must be positive")
	case c.InputDim < 0:
		return errors.New("inference.input_dim must be >= 0")
	}
	return nil
}

// predictor is one protocol implementation.
type predictor interface {
	predict(ctx context.Context, instances [][]float64) ([][]float64, string, error) // outputs, model version
	status(ctx context.Context) Status
}

// Status describes backend readiness.
type Status struct {
	Ready    bool           `json:"ready"`
	State    string         `json:"state"`
	Detail   string         `json:"detail,omitempty"`
	Versions []string       `json:"versions,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// InputError is a client error (bad shape or values); it is never retried
// and never triggers the fallback.
type InputError struct{ msg string }

func (e *InputError) Error() string { return e.msg }

func inputErr(format string, a ...any) error { return &InputError{fmt.Sprintf(format, a...)} }

// remoteError is an error returned by the model server itself.
type remoteError struct {
	status int
	msg    string
}

func (e *remoteError) Error() string {
	return fmt.Sprintf("model server returned %d: %s", e.status, e.msg)
}

// retryable reports whether a failed call may succeed on retry.
func retryable(err error) bool {
	var re *remoteError
	if errors.As(err, &re) {
		return re.status >= 500 || re.status == http.StatusTooManyRequests
	}
	return !errors.Is(err, context.Canceled)
}

// ErrUnavailable is returned when the backend failed and fallback is off.
var ErrUnavailable = errors.New("model backend unavailable")

// ErrBusy is returned when MaxConcurrent calls are already in flight.
var ErrBusy = errors.New("model backend at concurrency limit")

// Result is one scored instance.
type Result struct {
	Outputs []float64 `json:"outputs"`
	Score   float64   `json:"result"` // highest output value (probability of the top class)
	Class   int       `json:"class"`  // index of the highest output
}

// Response is the outcome of a Predict call.
type Response struct {
	Results        []Result `json:"-"`
	Backend        string   `json:"backend"`
	Model          string   `json:"model"`
	Version        string   `json:"version,omitempty"`
	Degraded       bool     `json:"degraded,omitempty"`
	DegradedReason string   `json:"degraded_reason,omitempty"`
	LatencyMs      float64  `json:"latency_ms"`
}

// Stats are cumulative counters.
type Stats struct {
	Requests    int64   `json:"requests"`
	Instances   int64   `json:"instances"`
	Failures    int64   `json:"backend_failures"`
	Retries     int64   `json:"retries"`
	Fallbacks   int64   `json:"fallbacks"`
	Rejected    int64   `json:"rejected_busy"`
	BreakerOpen bool    `json:"breaker_open"`
	AvgMs       float64 `json:"avg_backend_latency_ms"`
}

// Info is returned by GET /api/v1/model.
type Info struct {
	Backend  string `json:"backend"`
	URL      string `json:"url,omitempty"`
	Model    string `json:"model"`
	Version  string `json:"version,omitempty"`
	InputDim int    `json:"input_dim,omitempty"`
	Fallback bool   `json:"fallback"`
	Status   Status `json:"status"`
	Stats    Stats  `json:"stats"`
}

// Engine is the concurrency-safe inference front end.
type Engine struct {
	cfg      Config
	remote   predictor // nil for the baseline backend
	baseline baseline
	sem      chan struct{}

	mu          sync.Mutex
	consecutive int
	openUntil   time.Time
	now         func() time.Time

	requests, instances, failures, retries, fallbacks, rejected atomic.Int64
	backendNanos, backendCalls                                  atomic.Int64
}

// New builds an engine for cfg. client may be nil.
func New(cfg Config, client *http.Client) (*Engine, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if client == nil {
		client = &http.Client{Transport: &http.Transport{
			MaxIdleConns: cfg.MaxConcurrent, MaxIdleConnsPerHost: cfg.MaxConcurrent, IdleConnTimeout: 90 * time.Second,
		}}
	}
	e := &Engine{cfg: cfg, sem: make(chan struct{}, cfg.MaxConcurrent), now: time.Now}
	base := strings.TrimRight(cfg.URL, "/")
	switch cfg.Backend {
	case BackendTFServing:
		e.remote = &tfServing{client: client, base: base, cfg: cfg}
	case BackendONNXRuntime:
		e.remote = &openInference{client: client, base: base, cfg: cfg}
	}
	return e, nil
}

// Config returns the engine configuration.
func (e *Engine) Config() Config { return e.cfg }

// Predict scores a batch of feature vectors.
func (e *Engine) Predict(ctx context.Context, instances [][]float64) (Response, error) {
	if err := e.validate(instances); err != nil {
		return Response{}, err
	}
	e.requests.Add(1)
	e.instances.Add(int64(len(instances)))
	start := time.Now()

	if e.remote == nil {
		outs, _, _ := e.baseline.predict(ctx, instances)
		return e.respond(outs, BackendBaseline, "baseline-logistic", "", start, ""), nil
	}

	var reason string
	if e.breakerOpen() {
		reason = "circuit breaker open after repeated backend failures"
	} else {
		outs, version, err := e.callRemote(ctx, instances)
		if err == nil {
			return e.respond(outs, e.cfg.Backend, e.cfg.Model, version, start, ""), nil
		}
		var ie *InputError
		if errors.As(err, &ie) || errors.Is(err, ErrBusy) || errors.Is(err, context.Canceled) {
			return Response{}, err
		}
		reason = err.Error()
	}

	if !e.cfg.Fallback {
		return Response{}, fmt.Errorf("%w: %s", ErrUnavailable, reason)
	}
	e.fallbacks.Add(1)
	outs, _, _ := e.baseline.predict(ctx, instances)
	return e.respond(outs, BackendBaseline, "baseline-logistic", "", start, reason), nil
}

func (e *Engine) validate(instances [][]float64) error {
	if len(instances) == 0 {
		return inputErr("at least one instance is required")
	}
	if len(instances) > e.cfg.MaxBatch {
		return inputErr("batch of %d exceeds inference.max_batch (%d)", len(instances), e.cfg.MaxBatch)
	}
	dim := len(instances[0])
	for i, inst := range instances {
		if len(inst) == 0 || len(inst) > 4096 {
			return inputErr("instance %d must contain 1–4096 numbers", i)
		}
		if len(inst) != dim {
			return inputErr("instance %d has %d features, instance 0 has %d", i, len(inst), dim)
		}
		if e.cfg.InputDim > 0 && len(inst) != e.cfg.InputDim {
			return inputErr("model expects %d features, got %d", e.cfg.InputDim, len(inst))
		}
		for j, v := range inst {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return inputErr("instance %d feature %d is not finite", i, j)
			}
		}
	}
	return nil
}

func (e *Engine) callRemote(ctx context.Context, instances [][]float64) ([][]float64, string, error) {
	select {
	case e.sem <- struct{}{}:
		defer func() { <-e.sem }()
	default:
		e.rejected.Add(1)
		return nil, "", ErrBusy
	}

	var lastErr error
	for attempt := 0; attempt <= e.cfg.Retries; attempt++ {
		if attempt > 0 {
			e.retries.Add(1)
			select {
			case <-time.After(time.Duration(attempt) * 25 * time.Millisecond):
			case <-ctx.Done():
				return nil, "", ctx.Err()
			}
		}
		callCtx, cancel := context.WithTimeout(ctx, e.cfg.Timeout)
		t0 := time.Now()
		outs, version, err := e.remote.predict(callCtx, instances)
		cancel()
		if err == nil {
			if len(outs) != len(instances) {
				err = fmt.Errorf("model server returned %d predictions for %d instances", len(outs), len(instances))
			} else {
				e.backendNanos.Add(int64(time.Since(t0)))
				e.backendCalls.Add(1)
				e.recordSuccess()
				return outs, version, nil
			}
		}
		var ie *InputError
		if errors.As(err, &ie) {
			return nil, "", err
		}
		var re *remoteError
		if errors.As(err, &re) && re.status >= 400 && re.status < 500 && re.status != http.StatusTooManyRequests {
			// The model rejected the input (e.g. wrong shape): a client error.
			return nil, "", inputErr("model rejected the input: %s", re.msg)
		}
		lastErr = err
		if ctx.Err() != nil || !retryable(err) {
			break
		}
	}
	e.failures.Add(1)
	e.recordFailure()
	return nil, "", lastErr
}

func (e *Engine) breakerOpen() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.now().Before(e.openUntil)
}

func (e *Engine) recordSuccess() {
	e.mu.Lock()
	e.consecutive = 0
	e.mu.Unlock()
}

func (e *Engine) recordFailure() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.consecutive++
	if e.consecutive >= e.cfg.BreakerThreshold {
		// Half-open after the cooldown: the next request probes the backend.
		e.openUntil = e.now().Add(e.cfg.BreakerCooldown)
		e.consecutive = e.cfg.BreakerThreshold - 1
	}
}

func (e *Engine) respond(outs [][]float64, backend, model, version string, start time.Time, degraded string) Response {
	r := Response{Backend: backend, Model: model, Version: version,
		LatencyMs: math.Round(float64(time.Since(start).Microseconds())) / 1000}
	if degraded != "" {
		r.Degraded, r.DegradedReason = true, degraded
	}
	r.Results = make([]Result, len(outs))
	for i, o := range outs {
		res := Result{Outputs: make([]float64, len(o)), Score: math.Inf(-1)}
		for j, v := range o {
			res.Outputs[j] = math.Round(v*1e6) / 1e6
			if v > res.Score {
				res.Score, res.Class = v, j
			}
		}
		res.Score = math.Round(res.Score*1e6) / 1e6
		r.Results[i] = res
	}
	return r
}

// Info reports configuration, readiness and counters.
func (e *Engine) Info(ctx context.Context) Info {
	info := Info{Backend: e.cfg.Backend, Model: e.cfg.Model, Version: e.cfg.Version,
		InputDim: e.cfg.InputDim, Fallback: e.cfg.Fallback, Stats: e.Stats()}
	if e.remote == nil {
		info.Model = "baseline-logistic"
		info.Status = Status{Ready: true, State: "AVAILABLE", Detail: "in-process baseline logistic scorer"}
		return info
	}
	info.URL = e.cfg.URL
	sctx, cancel := context.WithTimeout(ctx, e.cfg.Timeout)
	defer cancel()
	info.Status = e.remote.status(sctx)
	return info
}

// Stats returns cumulative counters.
func (e *Engine) Stats() Stats {
	s := Stats{
		Requests: e.requests.Load(), Instances: e.instances.Load(), Failures: e.failures.Load(),
		Retries: e.retries.Load(), Fallbacks: e.fallbacks.Load(), Rejected: e.rejected.Load(),
		BreakerOpen: e.breakerOpen(),
	}
	if n := e.backendCalls.Load(); n > 0 {
		s.AvgMs = math.Round(float64(e.backendNanos.Load())/float64(n)/1e3) / 1e3
	}
	return s
}

// baseline is the dependency-free logistic scorer kept from API v1:
// p = σ(mean(x)), returned as a single output.
type baseline struct{}

func (baseline) predict(_ context.Context, instances [][]float64) ([][]float64, string, error) {
	out := make([][]float64, len(instances))
	for i, inst := range instances {
		var sum float64
		for _, v := range inst {
			sum += v
		}
		out[i] = []float64{1 / (1 + math.Exp(-sum/float64(len(inst))))}
	}
	return out, "", nil
}

func (baseline) status(context.Context) Status { return Status{Ready: true, State: "AVAILABLE"} }
