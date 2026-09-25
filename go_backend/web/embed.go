// Package web embeds the browser dashboard so the Go binary is self-contained.
// The same index.html is served by the Flask prototype (go_backend/sim_server.py).
package web

import "embed"

// FS contains index.html.
//
//go:embed index.html
var FS embed.FS
