// Package config loads TensorShadow backend configuration from YAML with
// compiled-in defaults and environment-variable overrides.
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/thermal"
	"github.com/TensorShadow/TensorShadow/go_backend/internal/tracking"
)

// Server holds HTTP server settings.
type Server struct {
	Port            string        `yaml:"port"`
	Env             string        `yaml:"env"`
	ReadTimeout     time.Duration `yaml:"read_timeout"`
	WriteTimeout    time.Duration `yaml:"write_timeout"`
	IdleTimeout     time.Duration `yaml:"idle_timeout"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout"`
	MaxBodyBytes    int64         `yaml:"max_body_bytes"`
	CORSOrigins     []string      `yaml:"cors_origins"`
}

// Concurrency holds worker-pool and rate-limit settings.
type Concurrency struct {
	Workers        int           `yaml:"workers"`
	QueueSize      int           `yaml:"queue_size"`
	JobTimeout     time.Duration `yaml:"job_timeout"`
	RateLimitRPS   float64       `yaml:"rate_limit_rps"`
	RateLimitBurst int           `yaml:"rate_limit_burst"`
}

// Config is the complete backend configuration.
type Config struct {
	Server      Server          `yaml:"server"`
	Concurrency Concurrency     `yaml:"concurrency"`
	Thermal     thermal.Config  `yaml:"thermal"`
	Tracking    tracking.Config `yaml:"tracking"`
}

// Default returns the built-in configuration.
func Default() Config {
	return Config{
		Server: Server{
			Port:            "8080",
			Env:             "development",
			ReadTimeout:     15 * time.Second,
			WriteTimeout:    30 * time.Second,
			IdleTimeout:     120 * time.Second,
			ShutdownTimeout: 20 * time.Second,
			MaxBodyBytes:    16 << 20,
			CORSOrigins:     []string{"*"},
		},
		Concurrency: Concurrency{
			Workers:        0,
			QueueSize:      1024,
			JobTimeout:     5 * time.Second,
			RateLimitRPS:   200,
			RateLimitBurst: 400,
		},
		Thermal:  thermal.DefaultConfig(),
		Tracking: tracking.DefaultConfig(),
	}
}

// DefaultPaths are searched, in order, when no explicit path is given.
var DefaultPaths = []string{"configs/config.yaml", "../configs/config.yaml"}

// Load reads configuration from path (or the first existing DefaultPaths
// entry when path is empty), then applies environment overrides. It returns
// the path actually used ("" when running on defaults only).
func Load(path string) (Config, string, error) {
	cfg := Default()
	used := ""
	candidates := DefaultPaths
	if path != "" {
		candidates = []string{path}
	}
	for _, p := range candidates {
		b, err := os.ReadFile(p)
		if errors.Is(err, os.ErrNotExist) && path == "" {
			continue
		}
		if err != nil {
			return cfg, "", fmt.Errorf("read config: %w", err)
		}
		if err := yaml.Unmarshal(b, &cfg); err != nil {
			return cfg, "", fmt.Errorf("parse %s: %w", p, err)
		}
		used = p
		break
	}
	if err := applyEnv(&cfg); err != nil {
		return cfg, used, err
	}
	return cfg, used, cfg.Validate()
}

func applyEnv(c *Config) error {
	if v := os.Getenv("TS_PORT"); v != "" {
		c.Server.Port = v
	}
	if v := os.Getenv("TS_ENV"); v != "" {
		c.Server.Env = v
	}
	for name, dst := range map[string]*int{"TS_WORKERS": &c.Concurrency.Workers, "TS_QUEUE_SIZE": &c.Concurrency.QueueSize} {
		if v := os.Getenv(name); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			*dst = n
		}
	}
	if v := os.Getenv("TS_RATE_LIMIT_RPS"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return fmt.Errorf("TS_RATE_LIMIT_RPS: %w", err)
		}
		c.Concurrency.RateLimitRPS = f
	}
	return nil
}

// Validate reports invalid settings.
func (c Config) Validate() error {
	if p, err := strconv.Atoi(c.Server.Port); err != nil || p < 1 || p > 65535 {
		return fmt.Errorf("server.port %q is not a valid TCP port", c.Server.Port)
	}
	if c.Server.MaxBodyBytes < 1024 {
		return errors.New("server.max_body_bytes must be at least 1024")
	}
	if c.Concurrency.Workers < 0 || c.Concurrency.QueueSize < 0 {
		return errors.New("concurrency.workers and queue_size must be >= 0")
	}
	if c.Concurrency.JobTimeout <= 0 {
		return errors.New("concurrency.job_timeout must be positive")
	}
	if c.Concurrency.RateLimitRPS < 0 || (c.Concurrency.RateLimitRPS > 0 && c.Concurrency.RateLimitBurst < 1) {
		return errors.New("concurrency.rate_limit_burst must be >= 1 when rate limiting is enabled")
	}
	if err := c.Thermal.Validate(); err != nil {
		return err
	}
	return c.Tracking.Validate()
}
