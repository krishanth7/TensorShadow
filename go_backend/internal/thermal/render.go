package thermal

import (
	"image"
	"image/color"

	"github.com/TensorShadow/TensorShadow/go_backend/internal/colormap"
)

// RenderOptions controls false-colour rendering.
type RenderOptions struct {
	Palette string   // ironbow (default), rainbow, whitehot, blackhot
	MinC    *float64 // colour scale lower bound; default frame minimum
	MaxC    *float64 // colour scale upper bound; default frame maximum
	Scale   int      // nearest-neighbour upscale factor, 1–16
	// Subjects, when non-nil, are drawn as status-coloured boxes with a
	// crosshair on the canthus measurement site.
	Subjects []Subject
}

var statusColour = map[string]color.RGBA{
	StatusNormal:        {34, 197, 94, 255},
	StatusElevated:      {250, 204, 21, 255},
	StatusFever:         {239, 68, 68, 255},
	StatusIndeterminate: {148, 163, 184, 255},
}

var spoofColour = color.RGBA{56, 189, 248, 255}

// Render produces a false-colour image of f.
func Render(f *Frame, opt RenderOptions) (*image.RGBA, error) {
	pal, err := colormap.Get(opt.Palette)
	if err != nil {
		return nil, err
	}
	scale := opt.Scale
	if scale < 1 {
		scale = 1
	} else if scale > 16 {
		scale = 16
	}

	lo, hi := f.Temps[0], f.Temps[0]
	for _, t := range f.Temps {
		lo, hi = min(lo, t), max(hi, t)
	}
	if opt.MinC != nil {
		lo = *opt.MinC
	}
	if opt.MaxC != nil {
		hi = *opt.MaxC
	}
	span := hi - lo
	if span <= 0 {
		span = 1
	}

	img := image.NewRGBA(image.Rect(0, 0, f.Width*scale, f.Height*scale))
	for y := 0; y < f.Height; y++ {
		for x := 0; x < f.Width; x++ {
			c := pal.At((f.At(x, y) - lo) / span)
			for sy := 0; sy < scale; sy++ {
				row := (y*scale + sy) * img.Stride
				for sx := 0; sx < scale; sx++ {
					o := row + (x*scale+sx)*4
					img.Pix[o], img.Pix[o+1], img.Pix[o+2], img.Pix[o+3] = c.R, c.G, c.B, 255
				}
			}
		}
	}

	for _, s := range opt.Subjects {
		c := statusColour[s.Status]
		if !s.Liveness.Live {
			c = spoofColour
		}
		r := s.ROI
		drawRect(img, r.X*scale, r.Y*scale, (r.X+r.W)*scale-1, (r.Y+r.H)*scale-1, max(1, scale/2), c)
		cx, cy := s.Canthus.X*scale+scale/2, s.Canthus.Y*scale+scale/2
		arm := 3 * scale
		white := color.RGBA{255, 255, 255, 255}
		drawRect(img, cx-arm, cy, cx+arm, cy, 1, white)
		drawRect(img, cx, cy-arm, cx, cy+arm, 1, white)
	}
	return img, nil
}

// drawRect strokes the rectangle (x0,y0)-(x1,y1) inclusive with thickness t.
func drawRect(img *image.RGBA, x0, y0, x1, y1, t int, c color.RGBA) {
	b := img.Bounds()
	fill := func(ax, ay, bx, by int) {
		for y := max(ay, b.Min.Y); y <= by && y < b.Max.Y; y++ {
			for x := max(ax, b.Min.X); x <= bx && x < b.Max.X; x++ {
				img.SetRGBA(x, y, c)
			}
		}
	}
	if x0 == x1 || y0 == y1 { // a line
		fill(x0, y0, x1, y1)
		return
	}
	fill(x0, y0, x1, y0+t-1)
	fill(x0, y1-t+1, x1, y1)
	fill(x0, y0, x0+t-1, y1)
	fill(x1-t+1, y0, x1, y1)
}
