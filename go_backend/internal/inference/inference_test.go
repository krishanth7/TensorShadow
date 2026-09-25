package inference

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fakeTFServing implements the TF Serving REST predict + status endpoints.
func fakeTFServing(t *testing.T, fail *atomic.Int32) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "GET" && r.URL.Path == "/v1/models/ts":
			w.Write([]byte(`{"model_version_status":[{"version":"3","state":"AVAILABLE","status":{"error_code":"OK"}}]}`))
		case r.Method == "POST" && r.URL.Path == "/v1/models/ts:predict":
			if fail != nil && fail.Load() > 0 {
				fail.Add(-1)
				w.WriteHeader(503)
				w.Write([]byte(`{"error":"model warming up"}`))
				return
			}
			var req struct {
				Instances [][]float64 `json:"instances"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			if len(req.Instances[0]) != 3 {
				w.WriteHeader(400)
				w.Write([]byte(`{"error":"Incompatible shapes: expected [?,3]"}`))
				return
			}
			var preds [][]float64
			for _, in := range req.Instances {
				p := 1 / (1 + math.Exp(-in[0]))
				preds = append(preds, []float64{1 - p, p})
			}
			json.NewEncoder(w).Encode(map[string]any{"predictions": preds})
		default:
			w.WriteHeader(404)
		}
	}))
}

// fakeOpenInference implements the v2 metadata, ready and infer endpoints.
func fakeOpenInference(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/models/ts":
			w.Write([]byte(`{"name":"ts","versions":["1"],"platform":"onnxruntime_onnx","inputs":[{"name":"features","datatype":"FP32","shape":[-1,3]}],"outputs":[{"name":"probabilities","datatype":"FP32","shape":[-1,2]}]}`))
		case "/v2/models/ts/ready":
			w.Write([]byte(`{"ready":true}`))
		case "/v2/models/ts/infer":
			var req struct {
				Inputs []v2Tensor `json:"inputs"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			in := req.Inputs[0]
			if in.Name != "features" || len(in.Shape) != 2 || in.Shape[1] != 3 {
				w.WriteHeader(400)
				w.Write([]byte(`{"error":"unknown input"}`))
				return
			}
			n, d := in.Shape[0], in.Shape[1]
			out := v2Tensor{Name: "probabilities", Shape: []int{n, 2}, Datatype: "FP32"}
			for i := 0; i < n; i++ {
				p := 1 / (1 + math.Exp(-in.Data[i*d]))
				out.Data = append(out.Data, 1-p, p)
			}
			json.NewEncoder(w).Encode(map[string]any{"model_name": "ts", "model_version": "1", "outputs": []v2Tensor{out}})
		default:
			w.WriteHeader(404)
		}
	}))
}

func remoteCfg(backend, url string) Config {
	c := DefaultConfig()
	c.Backend, c.URL, c.Model = backend, url, "ts"
	return c
}

func TestBaselineKeepsV1Semantics(t *testing.T) {
	e, _ := New(DefaultConfig(), nil)
	r, err := e.Predict(context.Background(), [][]float64{{0, 0}, {10, 10}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Results[0].Score != 0.5 || r.Results[1].Score < 0.9999 || r.Backend != BackendBaseline {
		t.Fatalf("baseline %+v", r)
	}
}

func TestProtocols(t *testing.T) {
	tfs := fakeTFServing(t, nil)
	defer tfs.Close()
	v2 := fakeOpenInference(t)
	defer v2.Close()

	for _, c := range []struct{ backend, url string }{{BackendTFServing, tfs.URL}, {BackendONNXRuntime, v2.URL}} {
		e, err := New(remoteCfg(c.backend, c.url), nil)
		if err != nil {
			t.Fatal(err)
		}
		r, err := e.Predict(context.Background(), [][]float64{{2, 0, 0}, {-2, 0, 0}})
		if err != nil {
			t.Fatalf("%s: %v", c.backend, err)
		}
		if r.Degraded || r.Backend != c.backend || len(r.Results) != 2 {
			t.Fatalf("%s: %+v", c.backend, r)
		}
		if r.Results[0].Class != 1 || r.Results[1].Class != 0 || math.Abs(r.Results[0].Score-0.880797) > 1e-6 {
			t.Fatalf("%s: results %+v", c.backend, r.Results)
		}
		info := e.Info(context.Background())
		if !info.Status.Ready || info.Status.State != "AVAILABLE" {
			t.Fatalf("%s: status %+v", c.backend, info.Status)
		}
		// A model-side shape error is a client error, not an outage.
		_, err = e.Predict(context.Background(), [][]float64{{1, 2}})
		var ie *InputError
		if !errors.As(err, &ie) {
			t.Fatalf("%s: shape error = %v, want InputError", c.backend, err)
		}
		if e.Stats().Fallbacks != 0 {
			t.Fatalf("%s: client error triggered fallback", c.backend)
		}
	}
}

func TestRetryRecoversTransientFailure(t *testing.T) {
	var fail atomic.Int32
	fail.Store(1)
	srv := fakeTFServing(t, &fail)
	defer srv.Close()
	e, _ := New(remoteCfg(BackendTFServing, srv.URL), nil)
	r, err := e.Predict(context.Background(), [][]float64{{1, 1, 1}})
	if err != nil || r.Degraded || e.Stats().Retries != 1 {
		t.Fatalf("r=%+v err=%v stats=%+v", r, err, e.Stats())
	}
}

func TestFallbackAndCircuitBreaker(t *testing.T) {
	var fail atomic.Int32
	fail.Store(1000)
	srv := fakeTFServing(t, &fail)
	defer srv.Close()
	cfg := remoteCfg(BackendTFServing, srv.URL)
	cfg.Retries, cfg.BreakerThreshold, cfg.BreakerCooldown = 0, 3, time.Minute
	e, _ := New(cfg, nil)
	clock := time.Unix(0, 0)
	e.now = func() time.Time { return clock }

	for i := 0; i < 3; i++ {
		r, err := e.Predict(context.Background(), [][]float64{{0, 0, 0}})
		if err != nil || !r.Degraded || r.Backend != BackendBaseline || !strings.Contains(r.DegradedReason, "503") {
			t.Fatalf("call %d: %+v %v", i, r, err)
		}
	}
	if !e.Stats().BreakerOpen {
		t.Fatal("breaker should be open after 3 failures")
	}
	before := fail.Load()
	r, _ := e.Predict(context.Background(), [][]float64{{0, 0, 0}})
	if fail.Load() != before || !strings.Contains(r.DegradedReason, "circuit breaker") {
		t.Fatal("open breaker must short-circuit without calling the backend")
	}

	// After the cooldown a single probe goes through; recovery closes the breaker.
	fail.Store(0)
	clock = clock.Add(2 * time.Minute)
	if r, _ := e.Predict(context.Background(), [][]float64{{0, 0, 0}}); r.Degraded {
		t.Fatalf("probe after cooldown should succeed: %+v", r)
	}
	if e.Stats().BreakerOpen {
		t.Fatal("breaker should close after a successful probe")
	}
}

func TestNoFallbackReturnsUnavailable(t *testing.T) {
	cfg := remoteCfg(BackendONNXRuntime, "http://127.0.0.1:1") // nothing listens on port 1
	cfg.Fallback, cfg.Retries, cfg.Timeout = false, 0, 500*time.Millisecond
	e, _ := New(cfg, nil)
	if _, err := e.Predict(context.Background(), [][]float64{{1}}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v, want ErrUnavailable", err)
	}
	if st := e.Info(context.Background()).Status; st.Ready {
		t.Fatalf("unreachable backend reported ready: %+v", st)
	}
}

func TestValidation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.InputDim, cfg.MaxBatch = 3, 2
	e, _ := New(cfg, nil)
	for _, in := range [][][]float64{nil, {{1, 2}}, {{1, 2, 3}, {1, 2}}, {{1, 2, math.NaN()}}, {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}} {
		var ie *InputError
		if _, err := e.Predict(context.Background(), in); !errors.As(err, &ie) {
			t.Errorf("input %v: err = %v", in, err)
		}
	}
	bad := []Config{{Backend: "pytorch"}, {Backend: BackendTFServing, URL: "localhost:8501"}}
	for _, c := range bad {
		d := DefaultConfig()
		d.Backend, d.URL = c.Backend, c.URL
		if d.Validate() == nil {
			t.Errorf("config %+v accepted", c)
		}
	}
}

// TestLiveModelServers runs against real servers when their URLs are set:
//
//	TS_TFSERVING_URL=http://localhost:8501 TS_ONNX_URL=http://localhost:8001 go test ./go_backend/internal/inference/
//
// Both serve the same exported weights (tensorflow_model/export_models.py),
// so their outputs must agree.
func TestLiveModelServers(t *testing.T) {
	tfsURL, onnxURL := os.Getenv("TS_TFSERVING_URL"), os.Getenv("TS_ONNX_URL")
	if tfsURL == "" || onnxURL == "" {
		t.Skip("set TS_TFSERVING_URL and TS_ONNX_URL to run against real model servers")
	}
	batch := [][]float64{
		{0.4, -0.2, 0.1, 0.3, -0.5, 0.2, 0.0, 0.6, -0.1, 0.3},
		{-1.2, 0.5, 0.3, -0.4, 0.1, -0.9, -0.2, -0.3, 0.2, -0.8},
	}
	var outputs [2][]Result
	for i, c := range []struct{ backend, url string }{{BackendTFServing, tfsURL}, {BackendONNXRuntime, onnxURL}} {
		cfg := DefaultConfig()
		cfg.Backend, cfg.URL, cfg.Model, cfg.Fallback, cfg.InputDim = c.backend, c.url, "tensorshadow", false, 10
		e, err := New(cfg, nil)
		if err != nil {
			t.Fatal(err)
		}
		r, err := e.Predict(context.Background(), batch)
		if err != nil {
			t.Fatalf("%s: %v", c.backend, err)
		}
		outputs[i] = r.Results
		t.Logf("%-11s version=%q results=%+v status=%+v", c.backend, r.Version, r.Results, e.Info(context.Background()).Status.State)
	}
	for i := range batch {
		for j := range outputs[0][i].Outputs {
			if d := math.Abs(outputs[0][i].Outputs[j] - outputs[1][i].Outputs[j]); d > 1e-5 {
				t.Fatalf("instance %d output %d: TF Serving %v vs ONNX Runtime %v", i, j, outputs[0][i].Outputs, outputs[1][i].Outputs)
			}
		}
	}
}
