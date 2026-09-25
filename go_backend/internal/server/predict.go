package server

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/inference"
)

// predictRequest accepts the v1 single-vector form ("data") or a batch
// ("instances"), mirroring the TensorFlow Serving request shape.
type predictRequest struct {
	Data      []float64   `json:"data,omitempty"`
	Instances [][]float64 `json:"instances,omitempty"`
}

type predictResponse struct {
	Status string `json:"status"`
	// Single-instance ("data") requests: the v1 fields.
	*inference.Result
	// Batch ("instances") requests.
	Predictions []inference.Result `json:"predictions,omitempty"`
	inference.Response
	Timestamp time.Time `json:"timestamp"`
	RequestID string    `json:"request_id"`
}

// handlePredict scores feature vectors on the configured model backend
// (TensorFlow Serving, ONNX Runtime via Open Inference Protocol v2, or the
// in-process baseline). Model-server I/O does not occupy the CPU worker pool;
// the engine bounds its own concurrency instead.
func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req predictRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	batch := req.Instances != nil
	if batch == (req.Data != nil) {
		writeError(w, r, http.StatusBadRequest, "invalid_request", `provide exactly one of "data" (one vector) or "instances" (a batch)`)
		return
	}
	instances := req.Instances
	if !batch {
		instances = [][]float64{req.Data}
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Concurrency.JobTimeout)
	defer cancel()
	res, err := s.infer.Predict(ctx, instances)
	if err != nil {
		var ie *inference.InputError
		switch {
		case errors.As(err, &ie):
			writeError(w, r, http.StatusBadRequest, "invalid_request", ie.Error())
		case errors.Is(err, inference.ErrBusy):
			w.Header().Set("Retry-After", "1")
			writeError(w, r, http.StatusServiceUnavailable, "overloaded", "model backend is at its concurrency limit")
		case errors.Is(err, inference.ErrUnavailable):
			w.Header().Set("Retry-After", "5")
			writeError(w, r, http.StatusServiceUnavailable, "model_unavailable", err.Error())
		case errors.Is(err, context.DeadlineExceeded):
			writeError(w, r, http.StatusGatewayTimeout, "timeout", "model backend did not answer in time")
		case errors.Is(err, context.Canceled):
		default:
			s.log.Error("predict failed", "error", err, "request_id", requestID(r.Context()))
			writeError(w, r, http.StatusInternalServerError, "internal", "internal server error")
		}
		return
	}

	out := predictResponse{Status: "success", Response: res, Timestamp: time.Now().UTC(), RequestID: requestID(r.Context())}
	if batch {
		out.Predictions = res.Results
	} else {
		out.Result = &res.Results[0]
	}
	writeJSON(w, http.StatusOK, out)
}

// handleModelInfo reports the model backend, its readiness and counters.
func (s *Server) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.infer.Info(r.Context()))
}
