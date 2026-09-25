package metrics

import (
	"strings"
	"testing"
	"time"
)

func TestSummaryAndExposition(t *testing.T) {
	r := New()
	for i := 0; i < 90; i++ {
		r.ObserveRequest("GET /a", "GET", 200, 2*time.Millisecond)
	}
	for i := 0; i < 10; i++ {
		r.ObserveRequest("GET /a", "GET", 503, 200*time.Millisecond)
	}
	r.RegisterGauge(Gauge{Name: "x_gauge", Help: "test", Fn: func() float64 { return 7 }})

	s := r.Summary()
	if s.RequestsTotal != 100 || s.ErrorsTotal != 10 {
		t.Fatalf("summary counts %+v", s)
	}
	if s.P50Ms < 1 || s.P50Ms > 2.5 || s.P99Ms < 100 || s.P99Ms > 250 {
		t.Fatalf("quantiles %+v", s)
	}

	var sb strings.Builder
	if err := r.WritePrometheus(&sb); err != nil {
		t.Fatal(err)
	}
	out := sb.String()
	for _, want := range []string{
		`tensorshadow_http_requests_total{route="GET /a",method="GET",code="200"} 90`,
		`tensorshadow_http_request_duration_seconds_bucket{route="GET /a",le="0.0025"} 90`,
		`tensorshadow_http_request_duration_seconds_bucket{route="GET /a",le="+Inf"} 100`,
		`tensorshadow_http_request_duration_seconds_count{route="GET /a"} 100`,
		"x_gauge 7",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}
