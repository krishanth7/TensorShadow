package workerpool

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSubmitReturnsResult(t *testing.T) {
	p := New(2, 4)
	defer p.Shutdown(context.Background())

	v, err := p.Submit(context.Background(), func(context.Context) (any, error) { return 42, nil })
	if err != nil || v.(int) != 42 {
		t.Fatalf("got (%v, %v), want (42, nil)", v, err)
	}
}

func TestQueueFullIsRejectedImmediately(t *testing.T) {
	p := New(1, 1)
	defer p.Shutdown(context.Background())

	release := make(chan struct{})
	started := make(chan struct{})
	block := func(context.Context) (any, error) { <-release; return nil, nil }

	// Occupy the single worker, then fill the single queue slot.
	go p.Submit(context.Background(), func(ctx context.Context) (any, error) { close(started); return block(ctx) })
	<-started
	go p.Submit(context.Background(), block)
	deadline := time.Now().Add(time.Second)
	for p.Stats().QueueDepth != 1 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}

	start := time.Now()
	_, err := p.Submit(context.Background(), block)
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("err = %v, want ErrQueueFull", err)
	}
	if time.Since(start) > 50*time.Millisecond {
		t.Fatal("Submit blocked on a full queue")
	}
	if p.Stats().Rejected != 1 {
		t.Fatalf("rejected = %d, want 1", p.Stats().Rejected)
	}
	close(release)
}

func TestPanicIsRecovered(t *testing.T) {
	p := New(1, 1)
	defer p.Shutdown(context.Background())

	_, err := p.Submit(context.Background(), func(context.Context) (any, error) { panic("boom") })
	if !errors.Is(err, ErrPanic) {
		t.Fatalf("err = %v, want ErrPanic", err)
	}
	// The worker must survive the panic.
	if _, err := p.Submit(context.Background(), func(context.Context) (any, error) { return 1, nil }); err != nil {
		t.Fatalf("pool unusable after panic: %v", err)
	}
	if p.Stats().Panics != 1 {
		t.Fatalf("panics = %d, want 1", p.Stats().Panics)
	}
}

func TestContextCancellation(t *testing.T) {
	p := New(1, 1)
	defer p.Shutdown(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := p.Submit(ctx, func(ctx context.Context) (any, error) { <-ctx.Done(); return nil, ctx.Err() })
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want DeadlineExceeded", err)
	}
}

func TestShutdownDrainsAndRejects(t *testing.T) {
	p := New(4, 64)
	var ran atomic.Int64
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.Submit(context.Background(), func(context.Context) (any, error) {
				time.Sleep(time.Millisecond)
				ran.Add(1)
				return nil, nil
			})
		}()
	}
	wg.Wait()
	if err := p.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if ran.Load() != 32 {
		t.Fatalf("ran = %d, want 32", ran.Load())
	}
	if _, err := p.Submit(context.Background(), func(context.Context) (any, error) { return nil, nil }); !errors.Is(err, ErrClosed) {
		t.Fatalf("err = %v, want ErrClosed", err)
	}
}

func BenchmarkSubmit(b *testing.B) {
	p := New(0, 4096)
	defer p.Shutdown(context.Background())
	task := func(context.Context) (any, error) { return nil, nil }
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			for {
				if _, err := p.Submit(context.Background(), task); !errors.Is(err, ErrQueueFull) {
					break
				}
			}
		}
	})
}
