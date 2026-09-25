// Package workerpool provides a bounded, back-pressured goroutine pool.
//
// CPU-heavy work (thermal frame analysis, multi-subject tracking) is executed
// on a fixed number of workers fed by a bounded queue. When the queue is full
// Submit fails fast with ErrQueueFull instead of letting latency grow without
// bound, which lets the HTTP layer shed load with a 503 + Retry-After.
package workerpool

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

var (
	// ErrQueueFull is returned when the job queue has no free capacity.
	ErrQueueFull = errors.New("workerpool: queue full")
	// ErrClosed is returned when submitting to a pool that is shutting down.
	ErrClosed = errors.New("workerpool: pool closed")
	// ErrPanic wraps a panic raised inside a task.
	ErrPanic = errors.New("workerpool: task panicked")
)

// Task is a unit of work. It must honour ctx cancellation where practical.
type Task func(ctx context.Context) (any, error)

type result struct {
	val any
	err error
}

type job struct {
	ctx      context.Context
	task     Task
	out      chan result
	enqueued time.Time
}

// Pool is a fixed-size worker pool with a bounded queue.
type Pool struct {
	jobs    chan job
	workers int

	mu     sync.RWMutex
	closed bool
	wg     sync.WaitGroup

	inFlight  atomic.Int64
	completed atomic.Int64
	failed    atomic.Int64
	rejected  atomic.Int64
	expired   atomic.Int64
	panics    atomic.Int64
	waitNanos atomic.Int64
	busyNanos atomic.Int64
}

// Stats is a point-in-time snapshot of pool counters.
type Stats struct {
	Workers        int     `json:"workers"`
	QueueCapacity  int     `json:"queue_capacity"`
	QueueDepth     int     `json:"queue_depth"`
	InFlight       int64   `json:"in_flight"`
	Completed      int64   `json:"completed"`
	Failed         int64   `json:"failed"`
	Rejected       int64   `json:"rejected"`
	Expired        int64   `json:"expired"`
	Panics         int64   `json:"panics"`
	AvgQueueWaitMs float64 `json:"avg_queue_wait_ms"`
	AvgExecMs      float64 `json:"avg_exec_ms"`
}

// New starts a pool. workers <= 0 means runtime.NumCPU(); queueSize < 0 is
// treated as 0 (unbuffered hand-off: a job is accepted only if a worker is idle).
func New(workers, queueSize int) *Pool {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if queueSize < 0 {
		queueSize = 0
	}
	p := &Pool{jobs: make(chan job, queueSize), workers: workers}
	p.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go p.worker()
	}
	return p
}

// Submit enqueues t and blocks until it completes or ctx is done. It never
// blocks on a full queue: ErrQueueFull is returned immediately.
func (p *Pool) Submit(ctx context.Context, t Task) (any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	j := job{ctx: ctx, task: t, out: make(chan result, 1), enqueued: time.Now()}

	p.mu.RLock()
	if p.closed {
		p.mu.RUnlock()
		return nil, ErrClosed
	}
	select {
	case p.jobs <- j:
		p.mu.RUnlock()
	default:
		p.mu.RUnlock()
		p.rejected.Add(1)
		return nil, ErrQueueFull
	}

	select {
	case r := <-j.out:
		return r.val, r.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (p *Pool) worker() {
	defer p.wg.Done()
	for j := range p.jobs {
		p.waitNanos.Add(int64(time.Since(j.enqueued)))
		if err := j.ctx.Err(); err != nil {
			// The caller gave up while the job was queued; skip the work.
			p.expired.Add(1)
			j.out <- result{err: err}
			continue
		}
		p.inFlight.Add(1)
		start := time.Now()
		val, err := p.run(j)
		p.busyNanos.Add(int64(time.Since(start)))
		p.inFlight.Add(-1)
		if err != nil {
			p.failed.Add(1)
		} else {
			p.completed.Add(1)
		}
		j.out <- result{val: val, err: err}
	}
}

func (p *Pool) run(j job) (val any, err error) {
	defer func() {
		if r := recover(); r != nil {
			p.panics.Add(1)
			val, err = nil, fmt.Errorf("%w: %v", ErrPanic, r)
		}
	}()
	return j.task(j.ctx)
}

// Shutdown stops accepting work, drains the queue and waits for workers to
// exit or ctx to expire.
func (p *Pool) Shutdown(ctx context.Context) error {
	p.mu.Lock()
	if !p.closed {
		p.closed = true
		close(p.jobs)
	}
	p.mu.Unlock()

	done := make(chan struct{})
	go func() { p.wg.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Stats returns a snapshot of the pool's counters.
func (p *Pool) Stats() Stats {
	s := Stats{
		Workers:       p.workers,
		QueueCapacity: cap(p.jobs),
		QueueDepth:    len(p.jobs),
		InFlight:      p.inFlight.Load(),
		Completed:     p.completed.Load(),
		Failed:        p.failed.Load(),
		Rejected:      p.rejected.Load(),
		Expired:       p.expired.Load(),
		Panics:        p.panics.Load(),
	}
	if dequeued := s.Completed + s.Failed + s.Expired; dequeued > 0 {
		s.AvgQueueWaitMs = float64(p.waitNanos.Load()) / float64(dequeued) / 1e6
	}
	if ran := s.Completed + s.Failed; ran > 0 {
		s.AvgExecMs = float64(p.busyNanos.Load()) / float64(ran) / 1e6
	}
	return s
}
