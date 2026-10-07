"""Localhost dashboard: play sample clips with pose overlay, run the full
assessment pipeline (auto-calibrate → reps → score), and accept uploads.

Stdlib only (no Flask/FastAPI). Pose tracks and browser-playable transcodes are
made on first request and cached under data/interim/.
"""

from __future__ import annotations

import cgi
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from ..config import interim_dir, load_pipeline_config, load_rules, PROJECT_ROOT
from ..datasets import dataset_path, registry
from ..geometry import angle_deg
from ..schema import JOINTS, PoseSequence
from ..signal import preprocess

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
STATIC = Path(__file__).parent / "static"
UPLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "uploads"
_pose_lock = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_estimator = None


# ── clip discovery ──────────────────────────────────────────────────────────

def discover() -> dict[str, dict]:
    """All sample clips + uploads on disk, keyed by a URL-safe id."""
    clips = {}
    # Dataset clips
    for ds, d in registry()["datasets"].items():
        root = dataset_path(ds)
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            m = re.match(r"(pushup|situp)_", f.name)
            ex = m.group(1) if m else (d["exercises"][0] if len(d.get("exercises", [])) == 1 else "unknown")
            cid = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{ds}__{f.stem}")
            clips[cid] = {"id": cid, "dataset": ds, "name": f.name, "exercise": ex, "path": str(f),
                          "size_mb": round(f.stat().st_size / 1e6, 2)}
    # Uploaded clips
    if UPLOAD_DIR.exists():
        for f in sorted(UPLOAD_DIR.iterdir()):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            cid = re.sub(r"[^A-Za-z0-9_.-]", "-", f"upload__{f.stem}")
            clips[cid] = {"id": cid, "dataset": "upload", "name": f.name, "exercise": "unknown", "path": str(f),
                          "size_mb": round(f.stat().st_size / 1e6, 2)}
    return clips


def _lock(key: str) -> threading.Lock:
    return _locks.setdefault(key, threading.Lock())


# ── pose estimation (cached) ───────────────────────────────────────────────

def pose_track(clip: dict) -> tuple[PoseSequence, float]:
    global _estimator
    cache = interim_dir() / "pose" / f"{clip['id']}.npz"
    with _lock("pose:" + clip["id"]):
        if cache.exists():
            return PoseSequence.load(cache), 0.0
        cache.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with _pose_lock:
            if _estimator is None:
                from ..pose import MediaPipePoseEstimator
                _estimator = MediaPipePoseEstimator.from_config(load_pipeline_config())
            seq = _estimator.estimate_video(clip["path"], progress=False)
        seq.save(cache)
        return seq, time.time() - t0


def web_video(clip: dict, fps: float) -> Path:
    out = interim_dir() / "web" / f"{clip['id']}.mp4"
    with _lock("video:" + clip["id"]):
        if out.exists():
            return out
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not on PATH")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", clip["path"], "-an",
             "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps:.4f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(tmp)],
            check=True)
        tmp.replace(out)
        return out


# ── pose-only payload (same as before) ─────────────────────────────────────

def _num(a: np.ndarray, nd: int) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def _frames(seq: PoseSequence) -> list:
    w, h = seq.size
    norm = seq.xy / np.array([w, h])
    return [[None if np.isnan(norm[t, j, 0]) else [round(float(norm[t, j, 0]), 4), round(float(norm[t, j, 1]), 4),
                                                   round(float(seq.conf[t, j]), 2)]
             for j in range(len(JOINTS))] for t in range(len(seq))]


def pose_payload(clip: dict) -> dict:
    raw, took = pose_track(clip)
    cfg = load_pipeline_config()
    clean = preprocess(raw, cfg)
    w, h = raw.size
    chain = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    side_score = {s: float(np.mean([raw.joint_conf(f"{s}_{n}").mean() for n in chain])) for s in ("l", "r")}
    side = max(side_score, key=side_score.get)
    P = clean.side_points(side)
    signals = {
        "elbow": angle_deg(P["shoulder"], P["elbow"], P["wrist"]),
        "hip": angle_deg(P["shoulder"], P["hip"], P["knee"]),
        "knee": angle_deg(P["hip"], P["knee"], P["ankle"]),
        "body": angle_deg(P["shoulder"], P["hip"], P["ankle"]),
    }
    detected = float(np.mean(~np.isnan(raw.xy[:, :, 0]).all(axis=1)))
    return {
        "clip": {k: clip[k] for k in ("id", "dataset", "name", "exercise")},
        "fps": raw.fps, "frames": len(raw), "size": [w, h], "side": side, "joints": list(JOINTS),
        "raw": _frames(raw), "smooth": _frames(clean),
        "signals": {k: _num(v, 1) for k, v in signals.items()},
        "stats": {"detected_frac": round(detected, 3),
                  "mean_conf_near": round(float(np.mean([raw.joint_conf(f"{side}_{n}").mean() for n in chain])), 3),
                  "pose_seconds": round(took, 1)},
    }


# ── full analysis payload ──────────────────────────────────────────────────

def analyze_payload(clip: dict, exercise: str, age: int = 25, sex: str = "m") -> dict:
    """Run the complete pipeline: pose → auto-calibrate → reps → score."""
    from ..auto_calibrate import auto_calibrate
    from ..calibration import CalibrationError
    from ..pipeline import analyze_sequence
    from ..scoring import ScoreTable

    cfg = load_pipeline_config()
    rules = load_rules(exercise)
    raw, took = pose_track(clip)

    # Auto-calibrate
    try:
        profile, segments = auto_calibrate(exercise, raw, cfg, rules)
    except CalibrationError as e:
        return {"error": "calibration_failed", "reasons": e.reasons}

    # Analyze
    # Use the end of the calibration window as the timer start
    t_start = segments["start"][1] + 0.5
    result = analyze_sequence(exercise, raw, profile, cfg, rules, t_start=t_start)

    # Score
    score_info = None
    try:
        table = ScoreTable.load()
        pts = table.points(exercise, sex, age, result.summary["valid"])
        score_info = {
            "station_points": pts,
            "max_station_points": 25,
            "age": age, "sex": sex,
            "age_band": f"{(age // 3) * 3 - 1}-{(age // 3) * 3 + 1}" if age >= 22 else "<22",
        }
    except Exception:
        pass  # scoring table not available

    # Build rep log
    rep_log = []
    for r in result.reps:
        checks_detail = []
        for c in r.checks:
            checks_detail.append({
                "name": c.name, "passed": c.passed,
                "value": round(c.value, 2) if np.isfinite(c.value) else None,
                "threshold": round(c.threshold, 2) if np.isfinite(c.threshold) else None,
                "unit": c.unit,
            })
        rep_log.append({
            "rep": r.index, "t_start": round(r.t_start, 2), "t_end": round(r.t_end, 2) if r.t_end else None,
            "valid": r.valid, "in_window": r.in_window,
            "reasons": r.reasons, "checks": checks_detail,
        })

    # Pose data for overlay (reuse the smooth track)
    w, h = raw.size
    clean = result.clean
    side = profile.side

    P = clean.side_points(side)
    signals = {
        "elbow": angle_deg(P["shoulder"], P["elbow"], P["wrist"]),
        "hip": angle_deg(P["shoulder"], P["hip"], P["knee"]),
        "knee": angle_deg(P["hip"], P["knee"], P["ankle"]),
        "body": angle_deg(P["shoulder"], P["hip"], P["ankle"]),
    }

    return {
        "clip": {k: clip[k] for k in ("id", "dataset", "name", "exercise")},
        "exercise": exercise,
        "fps": raw.fps, "frames": len(raw), "size": [w, h],
        "side": side, "joints": list(JOINTS),
        "smooth": _frames(clean), "raw": _frames(raw),
        "signals": {k: _num(v, 1) for k, v in signals.items()},
        "calibration": {
            "segments": {k: [round(v[0], 2), round(v[1], 2)] for k, v in segments.items()},
            "rom": profile.rom,
            "warnings": profile.warnings,
        },
        "reps": rep_log,
        "summary": result.summary,
        "score": score_info,
        "stats": {"pose_seconds": round(took, 1)},
    }


# ── HTTP handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "ippt-dashboard"

    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

    def _file_range(self, path: Path, ctype: str):
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(size - int(m.group(2)), 0)
            end = min(end, size - 1)
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0 and (chunk := f.read(min(1 << 20, left))):
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        qs = parse_qs(urlparse(self.path).query)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/clips":
                return self._json(sorted(({k: v for k, v in c.items() if k != "path"} for c in discover().values()),
                                         key=lambda c: (c["exercise"], c["dataset"], c["name"])))

            m = re.fullmatch(r"/api/clips/([^/]+)/(pose|video|analyze)", path)
            if m:
                clip = discover().get(m.group(1))
                if clip is None:
                    return self._json({"error": "unknown clip"}, 404)
                action = m.group(2)
                if action == "pose":
                    return self._json(pose_payload(clip))
                if action == "video":
                    fps = pose_track(clip)[0].fps
                    return self._file_range(web_video(clip, fps), "video/mp4")
                if action == "analyze":
                    exercise = qs.get("exercise", [clip.get("exercise", "pushup")])[0]
                    if exercise not in ("pushup", "situp"):
                        return self._json({"error": "exercise must be pushup or situp"}, 400)
                    age = int(qs.get("age", ["25"])[0])
                    sex = qs.get("sex", ["m"])[0]
                    return self._json(analyze_payload(clip, exercise, age, sex))

            self._json({"error": "not found"}, 404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/upload":
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype:
                    return self._json({"error": "expected multipart/form-data"}, 400)
                boundary = ctype.split("boundary=")[1].strip()
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                # Parse multipart
                environ = {
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                    "CONTENT_LENGTH": str(content_length),
                }
                from io import BytesIO
                fs = cgi.FieldStorage(fp=BytesIO(body), environ=environ, keep_blank_values=True)
                file_item = fs["file"] if "file" in fs else None
                if file_item is None or not file_item.filename:
                    return self._json({"error": "no file uploaded"}, 400)

                # Save
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(file_item.filename).name)
                dest = UPLOAD_DIR / safe_name
                # Avoid collision
                stem, ext = dest.stem, dest.suffix
                i = 1
                while dest.exists():
                    dest = UPLOAD_DIR / f"{stem}_{i}{ext}"
                    i += 1
                dest.write_bytes(file_item.file.read())
                cid = re.sub(r"[^A-Za-z0-9_.-]", "-", f"upload__{dest.stem}")
                return self._json({"id": cid, "name": dest.name, "size_mb": round(dest.stat().st_size / 1e6, 2)})
            self._json({"error": "not found"}, 404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"IPPT dashboard on {url}  ({len(discover())} clips; Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
