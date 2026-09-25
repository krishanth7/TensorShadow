package server

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/coreg"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
)

// coregInfo reports how visible boxes were mapped into a thermal frame.
type coregInfo struct {
	Rig          string `json:"rig"`
	Quality      string `json:"calibration_quality"`
	Mapped       int    `json:"mapped"`
	OutsideFrame []int  `json:"outside_frame,omitempty"` // indexes of visible_rois that miss the thermal view
}

// applyRig maps req.VisibleROIs through the named rig into thermal ROIs. It
// returns nil when the request does not use co-registration.
func (s *Server) applyRig(req *thermal.FrameRequest) (*coregInfo, error) {
	if req.Rig == "" {
		if len(req.VisibleROIs) > 0 {
			return nil, badRequest("visible_rois require a co-registration rig")
		}
		return nil, nil
	}
	if len(req.ROIs) > 0 {
		return nil, badRequest("use either rois (thermal pixels) or rig + visible_rois, not both")
	}
	if len(req.VisibleROIs) == 0 || len(req.VisibleROIs) > 64 {
		return nil, badRequest("rig requires 1–64 visible_rois")
	}
	rig, err := s.coreg.Get(req.Rig)
	if err != nil {
		return nil, &apiError{http.StatusNotFound, "rig_not_found", "no co-registration rig " + strconv.Quote(req.Rig)}
	}
	if req.Width != rig.Thermal.Width || req.Height != rig.Thermal.Height {
		return nil, badRequest("frame is %dx%d but rig %q is calibrated for a %dx%d thermal camera",
			req.Width, req.Height, rig.ID, rig.Thermal.Width, rig.Thermal.Height)
	}
	info := &coregInfo{Rig: rig.ID, Quality: rig.Quality}
	req.ROIs = make([]thermal.Rect, len(req.VisibleROIs))
	for i, b := range req.VisibleROIs {
		for _, v := range []float64{b.X, b.Y, b.W, b.H} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, badRequest("visible_rois[%d] is not finite", i)
			}
		}
		if b.W <= 0 || b.H <= 0 {
			return nil, badRequest("visible_rois[%d] must have positive w and h", i)
		}
		x, y, w, h, ok := rig.MapBox(coreg.Box{X: b.X, Y: b.Y, W: b.W, H: b.H})
		if !ok {
			info.OutsideFrame = append(info.OutsideFrame, i) // empty ROI: skipped by analysis
			continue
		}
		req.ROIs[i] = thermal.Rect{X: x, Y: y, W: w, H: h}
		info.Mapped++
	}
	return info, nil
}

// attachVisible links each subject back to the RGB box that produced it.
func attachVisible(res *thermal.Result, req *thermal.FrameRequest) {
	if len(req.VisibleROIs) == 0 {
		return
	}
	for i := range res.Subjects {
		if idx := res.Subjects[i].ROIIndex; idx != nil && *idx < len(req.VisibleROIs) {
			vb := req.VisibleROIs[*idx]
			res.Subjects[i].VisibleROI = &vb
		}
	}
}

func (s *Server) handleCreateRig(w http.ResponseWriter, r *http.Request) {
	var cfg coreg.RigConfig
	if !decodeJSON(w, r, &cfg) {
		return
	}
	rig, err := s.coreg.Create(cfg)
	switch {
	case errors.Is(err, coreg.ErrExists):
		writeError(w, r, http.StatusConflict, "rig_exists", err.Error())
	case err != nil:
		writeError(w, r, http.StatusBadRequest, "invalid_request", err.Error())
	default:
		w.Header().Set("Location", "/api/v1/coreg/rigs/"+rig.ID)
		writeJSON(w, http.StatusCreated, rig)
	}
}

func (s *Server) handleListRigs(w http.ResponseWriter, r *http.Request) {
	rigs := s.coreg.List()
	writeJSON(w, http.StatusOK, map[string]any{"rigs": rigs, "count": len(rigs)})
}

func (s *Server) rigOr404(w http.ResponseWriter, r *http.Request) *coreg.Rig {
	rig, err := s.coreg.Get(r.PathValue("id"))
	if err != nil {
		writeError(w, r, http.StatusNotFound, "rig_not_found", "no co-registration rig "+strconv.Quote(r.PathValue("id")))
		return nil
	}
	return rig
}

func (s *Server) handleGetRig(w http.ResponseWriter, r *http.Request) {
	if rig := s.rigOr404(w, r); rig != nil {
		writeJSON(w, http.StatusOK, rig)
	}
}

func (s *Server) handleDeleteRig(w http.ResponseWriter, r *http.Request) {
	if s.coreg.Delete(r.PathValue("id")) != nil {
		writeError(w, r, http.StatusNotFound, "rig_not_found", "no co-registration rig "+strconv.Quote(r.PathValue("id")))
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleMapBoxes maps visible boxes into thermal pixels without analysing a
// frame — useful for verifying a calibration or overlaying ROIs client-side.
func (s *Server) handleMapBoxes(w http.ResponseWriter, r *http.Request) {
	rig := s.rigOr404(w, r)
	if rig == nil {
		return
	}
	var req struct {
		Boxes []thermal.VisibleBox `json:"boxes"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Boxes) == 0 || len(req.Boxes) > 256 {
		writeError(w, r, http.StatusBadRequest, "invalid_request", "boxes must contain 1–256 entries")
		return
	}
	type mapped struct {
		Visible thermal.VisibleBox `json:"visible"`
		Thermal *thermal.Rect      `json:"thermal"` // null when outside the thermal view
	}
	out := make([]mapped, len(req.Boxes))
	for i, b := range req.Boxes {
		out[i].Visible = b
		if !(b.W > 0 && b.H > 0) {
			writeError(w, r, http.StatusBadRequest, "invalid_request", fmt.Sprintf("boxes[%d] must have positive w and h", i))
			return
		}
		if x, y, bw, bh, ok := rig.MapBox(coreg.Box{X: b.X, Y: b.Y, W: b.W, H: b.H}); ok {
			out[i].Thermal = &thermal.Rect{X: x, Y: y, W: bw, H: bh}
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"rig": rig.ID, "boxes": out})
}
