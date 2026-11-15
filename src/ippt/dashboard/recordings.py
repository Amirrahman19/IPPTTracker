"""Existing recordings for the dashboard: list web-playable clips and estimate reps from cached pose tracks.

Clips live in data/interim/web (override with IPPT_RECORDINGS_DIR) and their
pose tracks in data/interim/pose. Neither is in git, so a fresh deploy simply
shows an empty library.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np

from ..config import interim_dir, load_pipeline_config, load_rules
from ..geometry import angle_deg
from ..live import StreamingRepCounter
from ..schema import JOINT_INDEX, SIDE_JOINTS, PoseSequence

_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+\.mp4$")
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


def recordings_dir() -> Path:
    return Path(os.environ.get("IPPT_RECORDINGS_DIR", interim_dir() / "web"))


def _pose_path(name: str) -> Path:
    return interim_dir() / "pose" / (Path(name).stem + ".npz")


def _exercise_of(name: str) -> str:
    # names look like "<dataset>__<clip>.mp4"; the dataset part (e.g. hmdb51_pushup_situp) is no hint
    n = Path(name).stem.split("__", 1)[-1].lower().replace("-", "").replace("_", "")
    return "situp" if "situp" in n else "pushup"


def _title(name: str) -> str:
    stem = Path(name).stem
    return stem.split("__", 1)[-1].replace("_", " ").replace("-", " ")


def media_path(name: str) -> Path | None:
    """Resolve a clip name to a file inside the recordings dir, or None (no path traversal)."""
    if not _NAME_RE.match(name):
        return None
    p = recordings_dir() / name
    return p if p.is_file() else None


def list_recordings() -> list[dict]:
    d = recordings_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.mp4")):
        if not _NAME_RE.match(p.name):
            continue
        out.append({
            "name": p.name,
            "title": _title(p.name),
            "exercise": _exercise_of(p.name),
            "size_kb": round(p.stat().st_size / 1024),
            "analysed": _pose_path(p.name).is_file(),
        })
    return out


def _signal(seq: PoseSequence, side: str, kind: str) -> np.ndarray:
    pts = {n: seq.joint(f"{side}_{n}") for n in SIDE_JOINTS}
    if kind == "elbow_angle":
        return np.array([angle_deg(pts["shoulder"][i], pts["elbow"][i], pts["wrist"][i])
                         for i in range(len(seq))], dtype=float)
    dx = pts["shoulder"][:, 0] - pts["hip"][:, 0]
    dy = pts["shoulder"][:, 1] - pts["hip"][:, 1]
    return np.degrees(np.arctan2(dy, dx))


def analyse(name: str) -> dict | None:
    """Estimate reps for a clip from its cached pose track (default ROM, no per-person calibration)."""
    if media_path(name) is None:
        return None
    with _cache_lock:
        if name in _cache:
            return _cache[name]
    exercise = _exercise_of(name)
    result: dict = {"name": name, "exercise": exercise, "available": False}
    pose = _pose_path(name)
    if pose.is_file():
        try:
            from ..calibration import select_side
            from ..signal import preprocess
            cfg = load_pipeline_config()
            rules = load_rules(exercise)
            seq = preprocess(PoseSequence.load(pose), cfg)
            side, _ = select_side(seq, slice(0, len(seq)), rules["required_joints"])
            seg = rules["segmentation"]
            sig = _signal(seq, side, seg["signal"])
            lo, hi = seg["default_rom"]["lo"], seg["default_rom"]["hi"]
            finite = sig[~np.isnan(sig)]
            if len(finite) > 10:
                # Self-calibrate: use the clip's own range of motion when it is wide enough to be real.
                p_lo, p_hi = float(np.percentile(finite, 5)), float(np.percentile(finite, 95))
                if p_hi - p_lo >= 30:
                    lo, hi = p_lo, p_hi
            rom = hi - lo
            if seg["direction"] == "below":
                counter = StreamingRepCounter(high=hi - seg["exit_frac"] * rom, low=hi - seg["enter_frac"] * rom)
            else:
                counter = StreamingRepCounter(high=lo + seg["enter_frac"] * rom, low=lo + seg["exit_frac"] * rom)
                counter.state = "down"
            reps = []
            for t, v in zip(seq.t, sig):
                if counter.update(float(v), float(t)):
                    reps.append(round(float(t), 2))
            step = max(1, len(seq) // 300)
            history = [[round(float(t), 2), round(float(v), 1)]
                       for t, v in zip(seq.t[::step], sig[::step]) if not np.isnan(v)]
            result.update(available=True, side=side, rep_count=len(reps), rep_times=reps,
                          signal=history, signal_label="Elbow angle" if exercise == "pushup" else "Torso angle",
                          note="Estimate: range of motion taken from the clip itself, not an official calibration.")
        except Exception:
            import traceback
            traceback.print_exc()
    with _cache_lock:
        _cache[name] = result
    return result
