package tracking

// kf1 is a 1-D constant-velocity Kalman filter with state [position, velocity].
// A box is tracked with four independent filters (cx, cy, w, h); decoupling the
// axes keeps the update allocation-free and O(1) while matching the accuracy
// of a full 8-state filter for the diagonal noise models SORT uses.
type kf1 struct {
	x, v               float64 // state
	p00, p01, p10, p11 float64 // covariance
}

func newKF1(z, measVar float64) kf1 {
	return kf1{x: z, p00: measVar, p11: 100 * measVar}
}

// predict advances the state by dt using white-noise-acceleration process
// noise with spectral density q.
func (k *kf1) predict(dt, q float64) {
	k.x += k.v * dt
	// P = F P Fᵀ + Q with F = [[1 dt] [0 1]].
	p00 := k.p00 + dt*(k.p10+k.p01) + dt*dt*k.p11
	p01 := k.p01 + dt*k.p11
	p10 := k.p10 + dt*k.p11
	p11 := k.p11
	dt2 := dt * dt
	k.p00 = p00 + q*dt2*dt2/4
	k.p01 = p01 + q*dt2*dt/2
	k.p10 = p10 + q*dt2*dt/2
	k.p11 = p11 + q*dt2
}

// update fuses a position measurement z with variance r.
func (k *kf1) update(z, r float64) {
	y := z - k.x
	s := k.p00 + r
	k0, k1 := k.p00/s, k.p10/s
	k.x += k0 * y
	k.v += k1 * y
	p00, p01 := k.p00, k.p01
	k.p00 = (1 - k0) * p00
	k.p01 = (1 - k0) * p01
	k.p10 -= k1 * p00
	k.p11 -= k1 * p01
}
