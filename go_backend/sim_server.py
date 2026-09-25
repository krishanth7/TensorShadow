"""Flask prototype server for the TensorShadow dashboard.

The production backend is the Go service in this directory (``go run
./go_backend``), which serves the same dashboard plus the thermal and crowd
tracking APIs. This lightweight server is kept for front-end-only work: the
dashboard detects that the Go API is absent and runs multi-face rendering
locally.
"""

import os

from flask import Flask, Response

app = Flask(__name__)

DASHBOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")


@app.route("/")
def dashboard() -> Response:
    with open(DASHBOARD_PATH, encoding="utf-8") as fh:
        return Response(fh.read(), mimetype="text/html")


if __name__ == "__main__":
    app.run(port=int(os.environ.get("TS_PORT", "8080")), host="0.0.0.0")
