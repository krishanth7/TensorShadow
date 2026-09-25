package server

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"
	"strconv"
	"strings"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
)

// readFrame decodes a thermal frame from either JSON (application/json) or a
// raw little-endian sensor buffer (application/octet-stream).
//
// Binary query parameters:
//
//	width, height   frame size (required)
//	dtype           uint16 (default) | float32
//	format          centikelvin (default for uint16) | raw16 | celsius (default for float32) | kelvin
//	gain, offset    raw16 calibration: T°C = gain·raw + offset
//	emissivity, reflected_temp_c, ambient_c
//	roi             x,y,w,h — may be repeated
func (s *Server) readFrame(w http.ResponseWriter, r *http.Request) (thermal.FrameRequest, bool) {
	var req thermal.FrameRequest
	ct, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if ct != "application/octet-stream" {
		return req, decodeJSON(w, r, &req)
	}
	if err := s.parseBinaryFrame(r, &req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
				fmt.Sprintf("request body exceeds %d bytes", tooBig.Limit))
		} else {
			writeError(w, r, http.StatusBadRequest, "invalid_request", err.Error())
		}
		return req, false
	}
	return req, true
}

func (s *Server) parseBinaryFrame(r *http.Request, req *thermal.FrameRequest) error {
	q := r.URL.Query()
	var err error
	intParam := func(name string) int {
		if err != nil {
			return 0
		}
		v, e := strconv.Atoi(q.Get(name))
		if e != nil {
			err = fmt.Errorf("query parameter %s must be an integer", name)
		}
		return v
	}
	floatParam := func(name string) *float64 {
		if err != nil || q.Get(name) == "" {
			return nil
		}
		v, e := strconv.ParseFloat(q.Get(name), 64)
		if e != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			err = fmt.Errorf("query parameter %s must be a finite number", name)
			return nil
		}
		return &v
	}

	req.Width, req.Height = intParam("width"), intParam("height")
	req.Calibration.Emissivity = floatParam("emissivity")
	req.Calibration.ReflectedTempC = floatParam("reflected_temp_c")
	req.AmbientTempC = floatParam("ambient_c")
	if g := floatParam("gain"); g != nil {
		req.Calibration.Gain = *g
	}
	if o := floatParam("offset"); o != nil {
		req.Calibration.Offset = *o
	}
	if err != nil {
		return err
	}
	for _, roi := range q["roi"] {
		parts := strings.Split(roi, ",")
		if len(parts) != 4 {
			return fmt.Errorf("roi %q must be x,y,w,h", roi)
		}
		var v [4]int
		for i, p := range parts {
			if v[i], err = strconv.Atoi(strings.TrimSpace(p)); err != nil {
				return fmt.Errorf("roi %q must contain integers", roi)
			}
		}
		req.ROIs = append(req.ROIs, thermal.Rect{X: v[0], Y: v[1], W: v[2], H: v[3]})
	}

	if req.Width <= 0 || req.Height <= 0 || req.Width*req.Height > s.cfg.Thermal.MaxPixels {
		return fmt.Errorf("width×height must be in [1, %d]", s.cfg.Thermal.MaxPixels)
	}
	n := req.Width * req.Height

	dtype := q.Get("dtype")
	switch dtype {
	case "", "uint16":
		req.Format = thermal.Format(q.Get("format"))
		if req.Format == "" {
			req.Format = thermal.FormatCentiKelvin
		}
		buf := make([]byte, 2*n)
		if err := readExact(r.Body, buf); err != nil {
			return err
		}
		req.Data = make([]float64, n)
		for i := range req.Data {
			req.Data[i] = float64(binary.LittleEndian.Uint16(buf[2*i:]))
		}
	case "float32":
		req.Format = thermal.Format(q.Get("format"))
		if req.Format == "" {
			req.Format = thermal.FormatCelsius
		}
		buf := make([]byte, 4*n)
		if err := readExact(r.Body, buf); err != nil {
			return err
		}
		req.Data = make([]float64, n)
		for i := range req.Data {
			req.Data[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:])))
		}
	default:
		return fmt.Errorf("dtype must be uint16 or float32, got %q", dtype)
	}
	return nil
}

// readExact fills buf and requires the body to end exactly there.
func readExact(body io.Reader, buf []byte) error {
	if _, err := io.ReadFull(body, buf); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return err
		}
		return fmt.Errorf("body shorter than width×height×sample size (%d bytes)", len(buf))
	}
	var extra [1]byte
	if n, _ := body.Read(extra[:]); n > 0 {
		return fmt.Errorf("body longer than width×height×sample size (%d bytes)", len(buf))
	}
	return nil
}
