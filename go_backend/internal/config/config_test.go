package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadRepositoryConfig(t *testing.T) {
	cfg, used, err := Load("../../../configs/config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if used == "" {
		t.Fatal("expected config file to be used")
	}
	if cfg.Server.Env != "production" || cfg.Tracking.SessionTTL != 10*time.Minute || cfg.Thermal.Emissivity != 0.98 {
		t.Fatalf("unexpected values: %+v", cfg)
	}
}

func TestPartialFileKeepsDefaults(t *testing.T) {
	p := filepath.Join(t.TempDir(), "c.yaml")
	os.WriteFile(p, []byte("server:\n  port: 9000\nthermal:\n  fever_threshold_c: 38.3\n"), 0o600)
	cfg, _, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Port != "9000" || cfg.Thermal.FeverThresholdC != 38.3 {
		t.Fatalf("overrides not applied: %+v", cfg.Server)
	}
	if cfg.Thermal.ElevatedThresholdC != 37.5 || cfg.Concurrency.QueueSize != 1024 {
		t.Fatal("defaults lost for keys absent from the file")
	}
}

func TestEnvOverrides(t *testing.T) {
	t.Setenv("TS_PORT", "9123")
	t.Setenv("TS_WORKERS", "3")
	if _, _, err := Load(filepath.Join(t.TempDir(), "missing.yaml")); err == nil {
		t.Fatal("explicit missing path should error")
	}
	saved := DefaultPaths
	t.Cleanup(func() { DefaultPaths = saved })
	DefaultPaths = []string{filepath.Join(t.TempDir(), "none.yaml")}
	cfg, used, err := Load("")
	if err != nil || used != "" {
		t.Fatalf("defaults-only load failed: %v %q", err, used)
	}
	if cfg.Server.Port != "9123" || cfg.Concurrency.Workers != 3 {
		t.Fatalf("env overrides not applied: port=%s workers=%d", cfg.Server.Port, cfg.Concurrency.Workers)
	}
}

func TestValidateRejectsBadValues(t *testing.T) {
	cases := map[string]func(*Config){
		"port":       func(c *Config) { c.Server.Port = "http" },
		"emissivity": func(c *Config) { c.Thermal.Emissivity = 1.5 },
		"thresholds": func(c *Config) { c.Thermal.ElevatedThresholdC = 39 },
		"iou":        func(c *Config) { c.Tracking.IoUThreshold = 0 },
		"burst":      func(c *Config) { c.Concurrency.RateLimitBurst = 0 },
	}
	for name, mutate := range cases {
		c := Default()
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("%s: expected validation error", name)
		}
	}
}
