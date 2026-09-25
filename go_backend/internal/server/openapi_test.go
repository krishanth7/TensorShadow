package server

import (
	"context"
	"os"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/config"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/workerpool"
)

// TestOpenAPICoversAllRoutes keeps api/openapi.yaml in sync with the router.
func TestOpenAPICoversAllRoutes(t *testing.T) {
	raw, err := os.ReadFile("../../../api/openapi.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var spec struct {
		Paths map[string]map[string]any `yaml:"paths"`
	}
	if err := yaml.Unmarshal(raw, &spec); err != nil {
		t.Fatal(err)
	}
	cfg := config.Default()
	pool := workerpool.New(1, 1)
	defer pool.Shutdown(context.Background())
	s := New(cfg, Deps{Pool: pool, Crowd: tracking.NewManager(cfg.Tracking)})
	for _, pattern := range s.routes {
		method, path, _ := strings.Cut(pattern, " ")
		if path == "/{$}" { // dashboard
			continue
		}
		ops, ok := spec.Paths[path]
		if !ok {
			t.Errorf("route %s is not documented in api/openapi.yaml", pattern)
			continue
		}
		if _, ok := ops[strings.ToLower(method)]; !ok {
			t.Errorf("method %s of %s is not documented", method, path)
		}
	}
}
