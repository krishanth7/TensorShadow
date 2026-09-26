// Command go_backend runs the TensorShadow high-concurrency API server.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/config"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/coreg"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/inference"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/metrics"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/server"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/workerpool"
	"github.com/TensorShadow/TensorShadow/go_backend/web"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "tensorshadow:", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", "", "path to config.yaml (default: search configs/config.yaml)")
	addr := flag.String("addr", "", "listen address, overrides server.port (e.g. :9090)")
	logLevel := flag.String("log-level", "info", "log level: debug, info, warn, error")
	logFormat := flag.String("log-format", "", "log format: text or json (default: json in production)")
	flag.Parse()

	cfg, used, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	log := newLogger(*logLevel, *logFormat, cfg.Server.Env)
	if used == "" {
		log.Warn("no config file found, using built-in defaults")
	}

	pool := workerpool.New(cfg.Concurrency.Workers, cfg.Concurrency.QueueSize)
	crowd := tracking.NewManager(cfg.Tracking)
	infer, err := inference.New(cfg.Inference, nil)
	if err != nil {
		return err
	}
	rigs, err := coreg.NewRegistry(cfg.Coreg.Rigs)
	if err != nil {
		return err
	}
	srv := server.New(cfg, server.Deps{
		Pool: pool, Crowd: crowd, Infer: infer, Coreg: rigs, Metrics: metrics.New(), Logger: log, Web: web.FS,
	})

	listen := ":" + cfg.Server.Port
	if *addr != "" {
		listen = *addr
	}
	httpSrv := &http.Server{
		Addr:              listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       cfg.Server.ReadTimeout,
		WriteTimeout:      cfg.Server.WriteTimeout,
		IdleTimeout:       cfg.Server.IdleTimeout,
		ErrorLog:          slog.NewLogLogger(log.Handler(), slog.LevelWarn),
	}
	// Close SSE streams first so Shutdown is not held open by them.
	httpSrv.RegisterOnShutdown(crowd.CloseAll)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go crowd.RunJanitor(ctx, time.Minute, func(n int) { log.Info("evicted idle crowd sessions", "count", n) })

	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	ps := pool.Stats()
	log.Info("TensorShadow backend listening",
		"addr", ln.Addr().String(), "version", server.Version, "env", cfg.Server.Env, "config", used,
		"workers", ps.Workers, "queue", ps.QueueCapacity,
		"model_backend", cfg.Inference.Backend, "model_url", cfg.Inference.URL)
	if cfg.Inference.Backend != inference.BackendBaseline {
		st := infer.Info(ctx).Status
		log.Info("model backend status", "ready", st.Ready, "state", st.State, "detail", st.Detail)
	}

	errCh := make(chan error, 1)
	go func() { errCh <- httpSrv.Serve(ln) }()

	select {
	case err := <-errCh:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
		log.Info("shutdown signal received, draining")
	}

	shCtx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()
	if err := httpSrv.Shutdown(shCtx); err != nil {
		log.Error("http shutdown", "error", err)
	}
	if err := pool.Shutdown(shCtx); err != nil {
		log.Error("worker pool shutdown", "error", err)
	}
	log.Info("shutdown complete", "pool", pool.Stats())
	return nil
}

func newLogger(level, format, env string) *slog.Logger {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		lvl = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lvl}
	if format == "" {
		format = "text"
		if strings.EqualFold(env, "production") {
			format = "json"
		}
	}
	if format == "json" {
		return slog.New(slog.NewJSONHandler(os.Stdout, opts))
	}
	return slog.New(slog.NewTextHandler(os.Stdout, opts))
}
