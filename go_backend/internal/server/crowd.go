package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image/png"
	"net/http"
	"strconv"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/colormap"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
)

func (s *Server) sessionError(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case errors.Is(err, tracking.ErrNotFound):
		writeError(w, r, http.StatusNotFound, "session_not_found", "no session with id "+strconv.Quote(r.PathValue("id")))
	case errors.Is(err, tracking.ErrExists):
		writeError(w, r, http.StatusConflict, "session_exists", err.Error())
	case errors.Is(err, tracking.ErrLimit):
		writeError(w, r, http.StatusTooManyRequests, "session_limit", err.Error())
	default:
		writeError(w, r, http.StatusBadRequest, "invalid_request", err.Error())
	}
}

func (s *Server) handleCreateSession(w http.ResponseWriter, r *http.Request) {
	var sc tracking.SessionConfig
	if !decodeJSON(w, r, &sc) {
		return
	}
	sess, err := s.crowd.Create(sc)
	if err != nil {
		s.sessionError(w, r, err)
		return
	}
	w.Header().Set("Location", "/api/v1/crowd/sessions/"+sess.ID())
	writeJSON(w, http.StatusCreated, sess.Info())
}

func (s *Server) handleListSessions(w http.ResponseWriter, r *http.Request) {
	list := s.crowd.List()
	writeJSON(w, http.StatusOK, map[string]any{"sessions": list, "count": len(list)})
}

func (s *Server) handleGetSession(w http.ResponseWriter, r *http.Request) {
	sess, err := s.crowd.Get(r.PathValue("id"))
	if err != nil {
		s.sessionError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, sess.Info())
}

func (s *Server) handleDeleteSession(w http.ResponseWriter, r *http.Request) {
	if err := s.crowd.Delete(r.PathValue("id")); err != nil {
		s.sessionError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleFrame ingests one frame of detections. Frames for the same session
// are serialised by the session; clients should send them in order.
func (s *Server) handleFrame(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	sess, err := s.crowd.Get(id)
	if err != nil {
		s.sessionError(w, r, err)
		return
	}
	var in tracking.FrameInput
	if !decodeJSON(w, r, &in) {
		return
	}
	v, ok := s.exec(w, r, func(context.Context) (any, error) {
		res, err := sess.Process(in, time.Now())
		if err != nil {
			return nil, badRequest("%v", err)
		}
		return res, nil
	})
	if !ok {
		return
	}
	writeJSON(w, http.StatusOK, v)
}

func (s *Server) handleAnalytics(w http.ResponseWriter, r *http.Request) {
	sess, err := s.crowd.Get(r.PathValue("id"))
	if err != nil {
		s.sessionError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"session_id": sess.ID(), "analytics": sess.Analytics()})
}

// handleHeatmap returns the occupancy heatmap as JSON, or as a PNG with
// ?format=png (optional palette and cell size in pixels).
func (s *Server) handleHeatmap(w http.ResponseWriter, r *http.Request) {
	sess, err := s.crowd.Get(r.PathValue("id"))
	if err != nil {
		s.sessionError(w, r, err)
		return
	}
	q := r.URL.Query()
	if q.Get("format") != "png" {
		writeJSON(w, http.StatusOK, map[string]any{"session_id": sess.ID(), "heatmap": sess.Heatmap()})
		return
	}
	palette := q.Get("palette")
	if _, err := colormap.Get(palette); err != nil {
		writeError(w, r, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	cell := 16
	if v := q.Get("cell"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > 64 {
			writeError(w, r, http.StatusBadRequest, "invalid_request", "cell must be an integer in [1,64]")
			return
		}
		cell = n
	}
	v, ok := s.exec(w, r, func(context.Context) (any, error) {
		img, err := sess.HeatmapImage(palette, cell)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
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

// handleStream pushes every processed frame to the client as Server-Sent
// Events (event: frame), with a heartbeat comment every 15 s.
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	sess, err := s.crowd.Get(r.PathValue("id"))
	if err != nil {
		s.sessionError(w, r, err)
		return
	}
	rc := http.NewResponseController(w)
	_ = rc.SetWriteDeadline(time.Time{}) // long-lived: lift the server write timeout

	ch, cancel := sess.Subscribe()
	defer cancel()

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "retry: 2000\n: connected to session %s\n\n", sess.ID())
	if err := rc.Flush(); err != nil {
		return
	}

	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			fmt.Fprint(w, ": ping\n\n")
		case msg, ok := <-ch:
			if !ok {
				fmt.Fprint(w, "event: close\ndata: {}\n\n")
				rc.Flush()
				return
			}
			w.Write([]byte("event: frame\ndata: "))
			w.Write(msg)
			w.Write([]byte("\n\n"))
		}
		if err := rc.Flush(); err != nil {
			return
		}
	}
}
