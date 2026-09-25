// Package colormap maps scalar values to false-colour palettes used for
// thermal imagery and crowd heatmaps.
package colormap

import (
	"fmt"
	"image/color"
	"math"
	"sort"
)

type stop struct {
	pos     float64
	r, g, b float64
}

// Palette maps t in [0,1] to a colour.
type Palette struct {
	Name  string
	stops []stop
	lut   [256]color.RGBA
}

func newPalette(name string, stops []stop) *Palette {
	p := &Palette{Name: name, stops: stops}
	for i := range p.lut {
		p.lut[i] = p.interp(float64(i) / 255)
	}
	return p
}

func (p *Palette) interp(t float64) color.RGBA {
	s := p.stops
	if t <= s[0].pos {
		return color.RGBA{uint8(s[0].r), uint8(s[0].g), uint8(s[0].b), 255}
	}
	for i := 1; i < len(s); i++ {
		if t <= s[i].pos {
			a, b := s[i-1], s[i]
			f := (t - a.pos) / (b.pos - a.pos)
			return color.RGBA{
				uint8(math.Round(a.r + f*(b.r-a.r))),
				uint8(math.Round(a.g + f*(b.g-a.g))),
				uint8(math.Round(a.b + f*(b.b-a.b))),
				255,
			}
		}
	}
	l := s[len(s)-1]
	return color.RGBA{uint8(l.r), uint8(l.g), uint8(l.b), 255}
}

// At returns the colour for t; values outside [0,1] are clamped, NaN maps to 0.
func (p *Palette) At(t float64) color.RGBA {
	if !(t > 0) { // also catches NaN
		t = 0
	} else if t > 1 {
		t = 1
	}
	return p.lut[int(math.Round(t*255))]
}

var palettes = map[string]*Palette{
	"ironbow": newPalette("ironbow", []stop{
		{0.00, 0, 0, 12},
		{0.18, 40, 0, 110},
		{0.38, 150, 0, 150},
		{0.55, 215, 40, 70},
		{0.72, 245, 115, 0},
		{0.88, 255, 205, 25},
		{1.00, 255, 255, 235},
	}),
	"rainbow": newPalette("rainbow", []stop{
		{0.00, 20, 0, 90},
		{0.20, 0, 70, 255},
		{0.40, 0, 210, 210},
		{0.60, 60, 230, 40},
		{0.80, 255, 210, 0},
		{1.00, 255, 20, 0},
	}),
	"whitehot": newPalette("whitehot", []stop{{0, 0, 0, 0}, {1, 255, 255, 255}}),
	"blackhot": newPalette("blackhot", []stop{{0, 255, 255, 255}, {1, 0, 0, 0}}),
}

// Get looks up a palette by name. An empty name returns ironbow.
func Get(name string) (*Palette, error) {
	if name == "" {
		name = "ironbow"
	}
	p, ok := palettes[name]
	if !ok {
		return nil, fmt.Errorf("unknown palette %q (available: %v)", name, Names())
	}
	return p, nil
}

// Names lists the available palettes in sorted order.
func Names() []string {
	out := make([]string, 0, len(palettes))
	for k := range palettes {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
