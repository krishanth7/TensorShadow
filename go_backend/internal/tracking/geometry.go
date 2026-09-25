// Package tracking implements real-time multi-subject tracking and crowd
// analytics: a SORT-style tracker (Kalman prediction + Hungarian assignment on
// IoU), plus per-stream analytics such as occupancy, density, flow, dwell
// time, virtual tripwire counting and zone occupancy.
package tracking

import (
	"errors"
	"math"
)

// Point is a 2-D coordinate in pixels.
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// BBox is an axis-aligned box: top-left corner plus width and height.
type BBox struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	W float64 `json:"w"`
	H float64 `json:"h"`
}

// Center returns the box centre.
func (b BBox) Center() Point { return Point{b.X + b.W/2, b.Y + b.H/2} }

// Area returns the box area (0 for degenerate boxes).
func (b BBox) Area() float64 {
	if b.W <= 0 || b.H <= 0 {
		return 0
	}
	return b.W * b.H
}

// Valid reports whether the box has finite, positive dimensions.
func (b BBox) Valid() bool {
	for _, v := range []float64{b.X, b.Y, b.W, b.H} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return b.W > 0 && b.H > 0
}

// IoU returns the intersection-over-union of a and b.
func IoU(a, b BBox) float64 {
	ix := math.Min(a.X+a.W, b.X+b.W) - math.Max(a.X, b.X)
	iy := math.Min(a.Y+a.H, b.Y+b.H) - math.Max(a.Y, b.Y)
	if ix <= 0 || iy <= 0 {
		return 0
	}
	inter := ix * iy
	union := a.Area() + b.Area() - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// cross returns the z-component of (b−a)×(c−a): >0 when c lies to the left of
// the directed line a→b (in image coordinates, y pointing down, "left" is
// counter-clockwise on screen).
func cross(a, b, c Point) float64 {
	return (b.X-a.X)*(c.Y-a.Y) - (b.Y-a.Y)*(c.X-a.X)
}

// segmentsIntersect reports whether segment p1–p2 properly crosses q1–q2
// (touching at an endpoint counts on one side only, which avoids double
// counting when a trajectory vertex lies exactly on a tripwire).
func segmentsIntersect(p1, p2, q1, q2 Point) bool {
	d1 := cross(q1, q2, p1)
	d2 := cross(q1, q2, p2)
	d3 := cross(p1, p2, q1)
	d4 := cross(p1, p2, q2)
	return ((d1 < 0 && d2 >= 0) || (d1 >= 0 && d2 < 0)) &&
		((d3 < 0 && d4 >= 0) || (d3 >= 0 && d4 < 0))
}

// pointInPolygon uses the even–odd ray casting rule.
func pointInPolygon(p Point, poly []Point) bool {
	in := false
	for i, j := 0, len(poly)-1; i < len(poly); j, i = i, i+1 {
		a, b := poly[i], poly[j]
		if (a.Y > p.Y) != (b.Y > p.Y) &&
			p.X < (b.X-a.X)*(p.Y-a.Y)/(b.Y-a.Y)+a.X {
			in = !in
		}
	}
	return in
}

func polygonArea(poly []Point) float64 {
	var s float64
	for i, j := 0, len(poly)-1; i < len(poly); j, i = i, i+1 {
		s += poly[j].X*poly[i].Y - poly[i].X*poly[j].Y
	}
	return math.Abs(s) / 2
}

func validatePolygon(poly []Point) error {
	if len(poly) < 3 {
		return errors.New("polygon needs at least 3 vertices")
	}
	if polygonArea(poly) == 0 {
		return errors.New("polygon has zero area")
	}
	return nil
}
