"""Minimal, dependency-free Python client for the TensorShadow Go backend.

Example::

    from tensorshadow_client import TensorShadowClient

    ts = TensorShadowClient("http://localhost:8080")
    print(ts.health())

    # Infrared: a FLIR Lepton-style uint16 centikelvin buffer (row-major).
    result = ts.thermal_analyze_raw(buffer, width=160, height=120, ambient_c=22.0)

    # Crowd: one session per camera, then post detections frame by frame.
    ts.create_session("lobby-cam", frame_width=1280, frame_height=720, area_m2=60,
                      lines=[{"id": "gate", "a": {"x": 640, "y": 0}, "b": {"x": 640, "y": 720}}])
    out = ts.post_frame("lobby-cam", [{"bbox": {"x": 100, "y": 80, "w": 60, "h": 150}, "score": 0.93}])
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence


class TensorShadowError(RuntimeError):
    """Raised for non-2xx API responses; carries the API error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class TensorShadowClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- transport -----------------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[bytes] = None,
                 content_type: str = "application/json", raw: bool = False) -> Any:
        req = urllib.request.Request(self.base_url + path, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as err:
            payload = err.read()
            try:
                e = json.loads(payload)["error"]
                raise TensorShadowError(err.code, e["code"], e["message"]) from None
            except (ValueError, KeyError):
                raise TensorShadowError(err.code, "http_error", payload.decode(errors="replace")) from None
        if raw:
            return data
        return json.loads(data) if data else None

    def _json(self, method: str, path: str, payload: Any = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode()
        return self._request(method, path, body)

    # -- system --------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self._json("GET", "/api/v1/health")

    def stats(self) -> Dict[str, Any]:
        return self._json("GET", "/api/v1/stats")

    # -- model serving -------------------------------------------------------
    def predict(self, features: Sequence[float]) -> Dict[str, Any]:
        """Score one vector (TF Serving / ONNX Runtime / baseline, per server config)."""
        return self._json("POST", "/api/v1/predict", {"data": list(features)})

    def predict_batch(self, instances: Sequence[Sequence[float]]) -> Dict[str, Any]:
        return self._json("POST", "/api/v1/predict", {"instances": [list(x) for x in instances]})

    def model_info(self) -> Dict[str, Any]:
        return self._json("GET", "/api/v1/model")

    # -- visible/IR co-registration -----------------------------------------
    def create_rig(self, rig_id: str, visible_size: Sequence[int], thermal_size: Sequence[int],
                   points: List[Dict[str, Any]], **options: Any) -> Dict[str, Any]:
        """Calibrate a rig from [{"visible": {"x":..,"y":..}, "thermal": {"x":..,"y":..}}, ...]."""
        return self._json("POST", "/api/v1/coreg/rigs", {
            "id": rig_id, "points": points,
            "visible": {"width": visible_size[0], "height": visible_size[1]},
            "thermal": {"width": thermal_size[0], "height": thermal_size[1]}, **options})

    def delete_rig(self, rig_id: str) -> None:
        self._request("DELETE", f"/api/v1/coreg/rigs/{urllib.parse.quote(rig_id)}")

    def thermal_analyze_coregistered(self, buffer: bytes, width: int, height: int, rig: str,
                                     visible_boxes: List[Dict[str, Any]], **params: Any) -> Dict[str, Any]:
        """Analyse a raw uint16 frame using face boxes from the RGB camera.

        ``visible_boxes`` are {"x","y","w","h"[, "id"]} in visible-camera pixels.
        """
        vbox = [",".join(str(b[k]) for k in ("x", "y", "w", "h")) + (f",{b['id']}" if b.get("id") else "")
                for b in visible_boxes]
        query = urllib.parse.urlencode({"width": width, "height": height, "rig": rig, "vbox": vbox, **params}, doseq=True)
        return self._request("POST", "/api/v1/thermal/analyze?" + query, bytes(buffer), "application/octet-stream")

    # -- infrared thermal ----------------------------------------------------
    def thermal_analyze(self, temps_c: Sequence[float], width: int, height: int,
                        ambient_c: Optional[float] = None, **calibration: float) -> Dict[str, Any]:
        """Analyse a frame of apparent temperatures in °C (JSON transport)."""
        payload: Dict[str, Any] = {"width": width, "height": height, "format": "celsius",
                                   "data": list(temps_c), "calibration": calibration}
        if ambient_c is not None:
            payload["ambient_temp_c"] = ambient_c
        return self._json("POST", "/api/v1/thermal/analyze", payload)

    def thermal_analyze_raw(self, buffer: bytes, width: int, height: int, dtype: str = "uint16",
                            fmt: Optional[str] = None, **params: Any) -> Dict[str, Any]:
        """Analyse a raw little-endian sensor buffer (≈9× smaller than JSON)."""
        query = {"width": width, "height": height, "dtype": dtype, **params}
        if fmt:
            query["format"] = fmt
        path = "/api/v1/thermal/analyze?" + urllib.parse.urlencode(query)
        return self._request("POST", path, bytes(buffer), "application/octet-stream")

    def thermal_render_raw(self, buffer: bytes, width: int, height: int, palette: str = "ironbow",
                           scale: int = 4, annotate: bool = True, **params: Any) -> bytes:
        """Render a raw buffer to a false-colour PNG (returns PNG bytes)."""
        query = {"width": width, "height": height, "palette": palette, "scale": scale,
                 "annotate": str(annotate).lower(), **params}
        path = "/api/v1/thermal/render?" + urllib.parse.urlencode(query)
        return self._request("POST", path, bytes(buffer), "application/octet-stream", raw=True)

    # -- crowd analysis ------------------------------------------------------
    def create_session(self, session_id: str, frame_width: float, frame_height: float,
                       **options: Any) -> Dict[str, Any]:
        return self._json("POST", "/api/v1/crowd/sessions",
                          {"id": session_id, "frame_width": frame_width, "frame_height": frame_height, **options})

    def post_frame(self, session_id: str, detections: List[Dict[str, Any]],
                   timestamp: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"detections": detections}
        if timestamp:
            payload["timestamp"] = timestamp
        return self._json("POST", f"/api/v1/crowd/sessions/{urllib.parse.quote(session_id)}/frames", payload)

    def analytics(self, session_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/api/v1/crowd/sessions/{urllib.parse.quote(session_id)}/analytics")

    def heatmap_png(self, session_id: str, palette: str = "ironbow", cell: int = 16) -> bytes:
        q = urllib.parse.urlencode({"format": "png", "palette": palette, "cell": cell})
        return self._request("GET", f"/api/v1/crowd/sessions/{urllib.parse.quote(session_id)}/heatmap?{q}", raw=True)

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"/api/v1/crowd/sessions/{urllib.parse.quote(session_id)}")


if __name__ == "__main__":
    import array
    import sys

    client = TensorShadowClient(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080")
    print("health:", client.health())

    # A 22 °C room with one 20×24 px face: skin cools towards the edges and the
    # inner canthus reads 37.6 °C apparent. The server corrects for skin
    # emissivity (ε = 0.98), so it reports the canthus ≈ 0.3 °C warmer.
    def ck(celsius: float) -> int:
        return round((celsius + 273.15) * 100)

    w, h = 64, 48
    frame = array.array("H", [ck(22.0)] * (w * h))
    for y in range(12, 36):
        for x in range(22, 42):
            frame[y * w + x] = ck(35.0 - 0.15 * abs(x - 31.5) - 0.1 * abs(y - 24))
    for y in range(18, 21):
        for x in range(28, 31):
            frame[y * w + x] = ck(37.6)
    res = client.thermal_analyze_raw(frame.tobytes(), w, h, ambient_c=22)
    s = res["subjects"][0]
    print(f"thermal: canthus {s['canthus']['temp_c']} °C -> {s['status']}, live={s['liveness']['live']}")

    model = client.model_info()
    pred = client.predict([0.4, -0.2, 0.1, 0.3, -0.5, 0.2, 0.0, 0.6, -0.1, 0.3])
    print(f"predict: backend={pred['backend']} class={pred['class']} p={pred['result']} "
          f"(model ready={model['status']['ready']})")

    # Co-registration: the thermal camera sees the visible scene at 1/10 scale.
    pts = [{"visible": {"x": x, "y": y}, "thermal": {"x": x / 10, "y": y / 10}}
           for x in (40, 320, 600) for y in (40, 240, 440)]
    client.create_rig("py-rig", (640, 480), (w, h), pts)
    try:
        face = {"x": 220, "y": 120, "w": 200, "h": 240, "id": "rgb-1"}
        res = client.thermal_analyze_coregistered(frame.tobytes(), w, h, "py-rig", [face], ambient_c=22)
        s = res["subjects"][0]
        print(f"coreg: RGB box {s['visible_roi']['id']} -> thermal ROI {s['roi']} -> {s['status']}")
    finally:
        client.delete_rig("py-rig")

    client.create_session("py-demo", 1280, 720,
                          lines=[{"id": "gate", "a": {"x": 640, "y": 0}, "b": {"x": 640, "y": 720}}])
    try:
        for i in range(12):
            out = client.post_frame("py-demo", [{"bbox": {"x": 560 + 12 * i, "y": 200, "w": 60, "h": 150}}])
        a = out["analytics"]
        print(f"crowd: {a['active_subjects']} tracked, gate in/out {a['lines'][0]['in']}/{a['lines'][0]['out']}")
    finally:
        client.delete_session("py-demo")
