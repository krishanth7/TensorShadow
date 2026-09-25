package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net"
	"net/http"
	"runtime/debug"
	"slices"
	"sync"
	"time"
)

type ctxKey int

const requestIDKey ctxKey = iota

// requestID returns the request ID stored in ctx.
func requestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey).(string)
	return id
}

// statusRecorder captures the response status while remaining transparent to
// http.ResponseController (Flush / SetWriteDeadline) via Unwrap.
type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (r *statusRecorder) WriteHeader(code int) {
	if r.status == 0 {
		r.status = code
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(b)
	r.bytes += n
	return n, err
}

func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func (r *statusRecorder) code() int {
	if r.status == 0 {
		return http.StatusOK
	}
	return r.status
}

// withRecovery converts panics into 500 responses.
func (s *Server) withRecovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				if v == http.ErrAbortHandler {
					panic(v)
				}
				s.log.Error("panic serving request", "panic", v, "path", r.URL.Path,
					"request_id", requestID(r.Context()), "stack", string(debug.Stack()))
				writeError(w, r, http.StatusInternalServerError, "internal", "internal server error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// withRequestID propagates X-Request-ID or generates one.
func withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" || len(id) > 128 {
			var b [8]byte
			rand.Read(b[:])
			id = hex.EncodeToString(b[:])
		}
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), requestIDKey, id)))
	})
}

// withAccessLog logs every request; health and metrics scrapes at debug level.
func (s *Server) withAccessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r)
		level := slog.LevelInfo
		switch {
		case rec.code() >= 500:
			level = slog.LevelError
		case r.URL.Path == "/api/v1/health" || r.URL.Path == "/metrics":
			level = slog.LevelDebug
		}
		s.log.LogAttrs(r.Context(), level, "http",
			slog.String("method", r.Method), slog.String("path", r.URL.Path),
			slog.Int("status", rec.code()), slog.Int("bytes", rec.bytes),
			slog.Duration("duration", time.Since(start)),
			slog.String("remote", clientIP(r)), slog.String("request_id", requestID(r.Context())))
	})
}

// withCORS applies the configured CORS policy and answers preflight requests.
func (s *Server) withCORS(next http.Handler) http.Handler {
	allowAll := slices.Contains(s.cfg.Server.CORSOrigins, "*")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		if origin := r.Header.Get("Origin"); origin != "" {
			switch {
			case allowAll:
				h.Set("Access-Control-Allow-Origin", "*")
			case slices.Contains(s.cfg.Server.CORSOrigins, origin):
				h.Set("Access-Control-Allow-Origin", origin)
				h.Add("Vary", "Origin")
			}
			if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
				h.Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
				h.Set("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
				h.Set("Access-Control-Max-Age", "600")
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// rateLimiter is a per-client token bucket.
type rateLimiter struct {
	rps, burst float64
	mu         sync.Mutex
	buckets    map[string]*bucket
	calls      uint64
	now        func() time.Time
}

type bucket struct {
	tokens float64
	last   time.Time
}

func newRateLimiter(rps float64, burst int) *rateLimiter {
	if rps <= 0 {
		return nil
	}
	return &rateLimiter{rps: rps, burst: float64(burst), buckets: map[string]*bucket{}, now: time.Now}
}

// allow consumes a token for key. When denied it returns the wait until the
// next token is available.
func (l *rateLimiter) allow(key string) (bool, time.Duration) {
	now := l.now()
	l.mu.Lock()
	defer l.mu.Unlock()

	l.calls++
	if l.calls%4096 == 0 {
		for k, b := range l.buckets {
			if now.Sub(b.last) > time.Minute {
				delete(l.buckets, k)
			}
		}
	}

	b, ok := l.buckets[key]
	if !ok {
		b = &bucket{tokens: l.burst, last: now}
		l.buckets[key] = b
	}
	b.tokens = min(l.burst, b.tokens+now.Sub(b.last).Seconds()*l.rps)
	b.last = now
	if b.tokens >= 1 {
		b.tokens--
		return true, 0
	}
	return false, time.Duration((1 - b.tokens) / l.rps * float64(time.Second))
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
