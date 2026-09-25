package tracking

import "math"

// hungarian solves the rectangular linear assignment problem, minimising the
// total cost. It returns assign[i] = column assigned to row i, or -1.
// Complexity is O(n²·m) with n = min(rows, cols).
func hungarian(cost [][]float64) []int {
	n := len(cost)
	if n == 0 {
		return nil
	}
	m := len(cost[0])
	if m == 0 {
		out := make([]int, n)
		for i := range out {
			out[i] = -1
		}
		return out
	}
	if n > m {
		// Solve the transposed problem so that rows <= cols.
		t := make([][]float64, m)
		for j := range t {
			t[j] = make([]float64, n)
			for i := 0; i < n; i++ {
				t[j][i] = cost[i][j]
			}
		}
		colOf := hungarian(t)
		out := make([]int, n)
		for i := range out {
			out[i] = -1
		}
		for j, i := range colOf {
			if i >= 0 {
				out[i] = j
			}
		}
		return out
	}

	// Jonker-Volgenant style shortest augmenting path with potentials
	// (1-indexed; column 0 is a virtual source).
	u := make([]float64, n+1)
	v := make([]float64, m+1)
	p := make([]int, m+1)
	way := make([]int, m+1)
	minv := make([]float64, m+1)
	used := make([]bool, m+1)

	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		for j := range minv {
			minv[j] = math.Inf(1)
			used[j] = false
		}
		for {
			used[j0] = true
			i0, delta, j1 := p[j0], math.Inf(1), 0
			for j := 1; j <= m; j++ {
				if used[j] {
					continue
				}
				if cur := cost[i0-1][j-1] - u[i0] - v[j]; cur < minv[j] {
					minv[j], way[j] = cur, j0
				}
				if minv[j] < delta {
					delta, j1 = minv[j], j
				}
			}
			for j := 0; j <= m; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for j0 != 0 {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
		}
	}

	out := make([]int, n)
	for i := range out {
		out[i] = -1
	}
	for j := 1; j <= m; j++ {
		if p[j] != 0 {
			out[p[j]-1] = j - 1
		}
	}
	return out
}
