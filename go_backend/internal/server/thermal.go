package server

import (
	"bytes"
	"context"
	"image/png"
	"net/http"
	"strconv"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/colormap"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
)

type thermalResponse struct {
	thermal.Result
	ProcessingMs float64 `json:"processing_ms"`
	RequestID    string  `json:"request_id"`
}

func (s *Server) handleThermalAnalyze(w http.ResponseWriter, r *http.Request) {
	req, ok := s.readFrame(w, r)
	if !ok {
		return
	}
	cfg := s.cfg.Thermal
	v, ok := s.exec(w, r, func(context.Context) (any, error) {
		start := time.Now()
		f, err := thermal.Decode(&req, cfg)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		res := thermal.Analyze(f, req.ROIs, cfg)
		return thermalResponse{Result: res, ProcessingMs: float64(time.Since(start).Microseconds()) / 1000}, nil
	})
	if !ok {
		return
	}
	resp := v.(thermalResponse)
	resp.RequestID = requestID(r.Context())
	writeJSON(w, http.StatusOK, resp)
}

// handleThermalRender returns a false-colour PNG of the frame. Query
// parameters: palette, scale (1–16), min_c, max_c, annotate (bool). The frame
// may be JSON or a binary sensor buffer (see readFrame).
func (s *Server) handleThermalRender(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	opt := thermal.RenderOptions{Palette: q.Get("palette"), Scale: 4}
	if _, err := colormap.Get(opt.Palette); err != nil {
		writeError(w, r, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if v := q.Get("scale"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > 16 {
			writeError(w, r, http.StatusBadRequest, "invalid_request", "scale must be an integer in [1,16]")
			return
		}
		opt.Scale = n
	}
	for name, dst := range map[string]**float64{"min_c": &opt.MinC, "max_c": &opt.MaxC} {
		if v := q.Get(name); v != "" {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				writeError(w, r, http.StatusBadRequest, "invalid_request", name+" must be a number")
				return
			}
			*dst = &f
		}
	}
	annotate, _ := strconv.ParseBool(q.Get("annotate"))

	req, ok := s.readFrame(w, r)
	if !ok {
		return
	}
	cfg := s.cfg.Thermal
	v, ok := s.exec(w, r, func(context.Context) (any, error) {
		f, err := thermal.Decode(&req, cfg)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		if annotate {
			opt.Subjects = thermal.Analyze(f, req.ROIs, cfg).Subjects
		}
		img, err := thermal.Render(f, opt)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		var buf bytes.Buffer
		enc := png.Encoder{CompressionLevel: png.BestSpeed}
		if err := enc.Encode(&buf, img); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	})
	if !ok {
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(v.([]byte))
}

func (s *Server) handlePalettes(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"palettes": colormap.Names(), "default": "ironbow"})
}
