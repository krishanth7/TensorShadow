// Package metrics records HTTP request metrics and exposes them in the
// Prometheus text exposition format without external dependencies.
package metrics

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"sync"
	"time"
)

// Buckets are latency histogram upper bounds in seconds.
var Buckets = []float64{0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5}

type reqKey struct {
	route, method string
	code          int
}

type histogram struct {
	counts []uint64 // per bucket (non-cumulative), plus +Inf at the end
	sum    float64
	n      uint64
}

// Gauge is a lazily evaluated metric.
type Gauge struct {
	Name, Help string
	Fn         func() float64
}

// Registry is a concurrency-safe metrics registry.
type Registry struct {
	mu       sync.Mutex
	requests map[reqKey]uint64
	latency  map[string]*histogram
	all      *histogram
	gauges   []Gauge
	started  time.Time
}

// New returns an empty registry.
func New() *Registry {
	return &Registry{
		requests: map[reqKey]uint64{},
		latency:  map[string]*histogram{},
		all:      newHistogram(),
		started:  time.Now(),
	}
}

func newHistogram() *histogram { return &histogram{counts: make([]uint64, len(Buckets)+1)} }

func (h *histogram) observe(sec float64) {
	i := sort.SearchFloat64s(Buckets, sec)
	h.counts[i]++
	h.sum += sec
	h.n++
}

// quantile estimates the q-quantile by linear interpolation inside buckets.
func (h *histogram) quantile(q float64) float64 {
	if h.n == 0 {
		return 0
	}
	rank := q * float64(h.n)
	var cum uint64
	for i, c := range h.counts {
		if float64(cum+c) >= rank {
			lo := 0.0
			if i > 0 {
				lo = Buckets[i-1]
			}
			hi := lo * 2
			if i < len(Buckets) {
				hi = Buckets[i]
			}
			if c == 0 {
				return hi
			}
			return lo + (hi-lo)*(rank-float64(cum))/float64(c)
		}
		cum += c
	}
	return Buckets[len(Buckets)-1]
}

// ObserveRequest records one completed HTTP request.
func (r *Registry) ObserveRequest(route, method string, code int, d time.Duration) {
	sec := d.Seconds()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests[reqKey{route, method, code}]++
	h, ok := r.latency[route]
	if !ok {
		h = newHistogram()
		r.latency[route] = h
	}
	h.observe(sec)
	r.all.observe(sec)
}

// RegisterGauge adds a gauge evaluated at scrape time.
func (r *Registry) RegisterGauge(g Gauge) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.gauges = append(r.gauges, g)
}

// Summary is a compact overview for JSON endpoints.
type Summary struct {
	RequestsTotal uint64  `json:"requests_total"`
	ErrorsTotal   uint64  `json:"errors_total"`
	P50Ms         float64 `json:"p50_ms"`
	P95Ms         float64 `json:"p95_ms"`
	P99Ms         float64 `json:"p99_ms"`
	MeanMs        float64 `json:"mean_ms"`
	UptimeSeconds float64 `json:"uptime_seconds"`
}

// Summary returns aggregate request statistics.
func (r *Registry) Summary() Summary {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := Summary{UptimeSeconds: math.Round(time.Since(r.started).Seconds()*10) / 10}
	for k, v := range r.requests {
		s.RequestsTotal += v
		if k.code >= 500 {
			s.ErrorsTotal += v
		}
	}
	ms := func(sec float64) float64 { return math.Round(sec*1e6) / 1e3 }
	s.P50Ms, s.P95Ms, s.P99Ms = ms(r.all.quantile(0.5)), ms(r.all.quantile(0.95)), ms(r.all.quantile(0.99))
	if r.all.n > 0 {
		s.MeanMs = ms(r.all.sum / float64(r.all.n))
	}
	return s
}

// WritePrometheus writes all metrics in text exposition format 0.0.4.
func (r *Registry) WritePrometheus(w io.Writer) error {
	r.mu.Lock()
	keys := make([]reqKey, 0, len(r.requests))
	for k := range r.requests {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.route != b.route {
			return a.route < b.route
		}
		if a.method != b.method {
			return a.method < b.method
		}
		return a.code < b.code
	})
	var b []byte
	b = append(b, "# HELP tensorshadow_http_requests_total HTTP requests by route, method and status.\n# TYPE tensorshadow_http_requests_total counter\n"...)
	for _, k := range keys {
		b = fmt.Appendf(b, "tensorshadow_http_requests_total{route=%q,method=%q,code=\"%d\"} %d\n", k.route, k.method, k.code, r.requests[k])
	}

	routes := make([]string, 0, len(r.latency))
	for k := range r.latency {
		routes = append(routes, k)
	}
	sort.Strings(routes)
	b = append(b, "# HELP tensorshadow_http_request_duration_seconds HTTP request latency.\n# TYPE tensorshadow_http_request_duration_seconds histogram\n"...)
	for _, route := range routes {
		h := r.latency[route]
		var cum uint64
		for i, ub := range Buckets {
			cum += h.counts[i]
			b = fmt.Appendf(b, "tensorshadow_http_request_duration_seconds_bucket{route=%q,le=%q} %d\n", route, strconv.FormatFloat(ub, 'g', -1, 64), cum)
		}
		cum += h.counts[len(Buckets)]
		b = fmt.Appendf(b, "tensorshadow_http_request_duration_seconds_bucket{route=%q,le=\"+Inf\"} %d\n", route, cum)
		b = fmt.Appendf(b, "tensorshadow_http_request_duration_seconds_sum{route=%q} %g\n", route, h.sum)
		b = fmt.Appendf(b, "tensorshadow_http_request_duration_seconds_count{route=%q} %d\n", route, h.n)
	}
	gauges := append([]Gauge(nil), r.gauges...)
	r.mu.Unlock()

	b = fmt.Appendf(b, "# HELP tensorshadow_uptime_seconds Process uptime.\n# TYPE tensorshadow_uptime_seconds gauge\ntensorshadow_uptime_seconds %g\n", time.Since(r.started).Seconds())
	for _, g := range gauges {
		b = fmt.Appendf(b, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", g.Name, g.Help, g.Name, g.Name, g.Fn())
	}
	_, err := w.Write(b)
	return err
}
