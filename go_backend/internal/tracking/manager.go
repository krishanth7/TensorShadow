package tracking

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sort"
	"sync"
	"time"
)

var (
	// ErrNotFound is returned for unknown session IDs.
	ErrNotFound = errors.New("session not found")
	// ErrExists is returned when creating a session whose ID is taken.
	ErrExists = errors.New("session already exists")
	// ErrLimit is returned when the session limit is reached.
	ErrLimit = errors.New("session limit reached")
)

// Manager owns all tracking sessions. Lookups take a read lock only, so frame
// processing for different sessions never contends on the manager.
type Manager struct {
	cfg      Config
	mu       sync.RWMutex
	sessions map[string]*Session
	now      func() time.Time
}

// NewManager creates an empty manager.
func NewManager(cfg Config) *Manager {
	return &Manager{cfg: cfg, sessions: map[string]*Session{}, now: time.Now}
}

// Config returns the tracker configuration.
func (m *Manager) Config() Config { return m.cfg }

// Create validates sc and registers a new session. An empty ID is generated.
func (m *Manager) Create(sc SessionConfig) (*Session, error) {
	if err := sc.normalise(); err != nil {
		return nil, err
	}
	if sc.ID == "" {
		sc.ID = newID()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.sessions[sc.ID]; ok {
		return nil, ErrExists
	}
	if len(m.sessions) >= m.cfg.MaxSessions {
		return nil, ErrLimit
	}
	s := newSession(sc, m.cfg, m.now())
	m.sessions[sc.ID] = s
	return s, nil
}

// Get returns the session with the given ID.
func (m *Manager) Get(id string) (*Session, error) {
	m.mu.RLock()
	s, ok := m.sessions[id]
	m.mu.RUnlock()
	if !ok {
		return nil, ErrNotFound
	}
	return s, nil
}

// Delete removes a session and disconnects its subscribers.
func (m *Manager) Delete(id string) error {
	m.mu.Lock()
	s, ok := m.sessions[id]
	delete(m.sessions, id)
	m.mu.Unlock()
	if !ok {
		return ErrNotFound
	}
	s.closeSubscribers()
	return nil
}

// List returns all sessions ordered by creation time.
func (m *Manager) List() []SessionInfo {
	m.mu.RLock()
	all := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		all = append(all, s)
	}
	m.mu.RUnlock()
	out := make([]SessionInfo, len(all))
	for i, s := range all {
		out[i] = s.Info()
	}
	sort.Slice(out, func(a, b int) bool { return out[a].CreatedAt.Before(out[b].CreatedAt) })
	return out
}

// Totals returns the session count and total present subjects.
func (m *Manager) Totals() (sessions, subjects int) {
	for _, info := range m.List() {
		sessions++
		subjects += info.ActiveSubjects
	}
	return
}

// Process runs one frame through the given session.
func (m *Manager) Process(id string, in FrameInput) (FrameResult, error) {
	s, err := m.Get(id)
	if err != nil {
		return FrameResult{}, err
	}
	return s.Process(in, m.now())
}

// EvictIdle deletes sessions idle for longer than the configured TTL and
// returns how many were removed.
func (m *Manager) EvictIdle() int {
	if m.cfg.SessionTTL <= 0 {
		return 0
	}
	cutoff := m.now().Add(-m.cfg.SessionTTL)
	var stale []string
	m.mu.RLock()
	for id, s := range m.sessions {
		if s.idleSince().Before(cutoff) {
			stale = append(stale, id)
		}
	}
	m.mu.RUnlock()
	n := 0
	for _, id := range stale {
		if m.Delete(id) == nil {
			n++
		}
	}
	return n
}

// RunJanitor evicts idle sessions every interval until ctx is cancelled.
func (m *Manager) RunJanitor(ctx context.Context, interval time.Duration, onEvict func(int)) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if n := m.EvictIdle(); n > 0 && onEvict != nil {
				onEvict(n)
			}
		}
	}
}

// CloseAll disconnects every subscriber (used during shutdown).
func (m *Manager) CloseAll() {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, s := range m.sessions {
		s.closeSubscribers()
	}
}

func newID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return time.Now().Format("20060102T150405.000000000")
	}
	return hex.EncodeToString(b[:])
}
