"""Auto-calibration: detect the starting pose from the video without manual timestamps.

The user records themselves: hold the starting position for a few seconds, then
do their reps. This module finds that static window automatically and builds
calibration segments from it.

Push-up starting position: arms locked at the top (plank).
Sit-up starting position: lying flat, shoulder blades on the mat.
"""

from __future__ import annotations

import numpy as np

from .calibration import Profile, calibrate, CalibrationError
from .config import load_pipeline_config, load_rules
from .schema import PoseSequence
from .signal import preprocess


def _joint_motion(seq: PoseSequence, side: str, joints: list[str], window: int) -> np.ndarray:
    """Per-frame measure of how much the near-side joints move over a sliding window.

    Returns (T,) array. Low values mean the person is holding still.
    """
    pts = np.stack([seq.joint(f"{side}_{j}") for j in joints], axis=1)  # (T, J, 2)
    h = float(seq.size[1])
    pts = pts / h  # normalise by image height so the threshold is resolution-independent

    motion = np.full(len(seq), np.inf)
    for i in range(window, len(seq)):
        chunk = pts[i - window:i]  # (window, J, 2)
        med = np.nanmedian(chunk, axis=0, keepdims=True)
        wander = np.nanmax(np.linalg.norm(chunk - med, axis=-1))
        motion[i - 1] = wander
    return motion


def _select_side(seq: PoseSequence, joints: list[str]) -> str:
    """Pick the side with better average confidence over the whole clip."""
    score = {s: float(np.mean([seq.joint_conf(f"{s}_{j}").mean() for j in joints])) for s in ("l", "r")}
    return "l" if score["l"] > score["r"] else "r"


def find_static_window(seq: PoseSequence, exercise: str, cfg: dict,
                       min_duration_s: float = 1.5, max_search_s: float = 15.0,
                       motion_threshold: float = 0.015) -> tuple[float, float]:
    """Find the first period where the person holds still for at least `min_duration_s`.

    Searches only the first `max_search_s` seconds of the clip.
    Returns (t_start, t_end) in seconds.

    Raises CalibrationError if no static window is found.
    """
    rules = load_rules(exercise)
    joints = rules["required_joints"]
    side = _select_side(seq, joints)
    fps = seq.fps or 30.0
    window_frames = max(3, int(0.5 * fps))  # half-second sliding window

    motion = _joint_motion(seq, side, joints, window_frames)

    # Search within the first max_search_s
    max_frame = min(len(seq), int(max_search_s * fps))
    min_frames = max(int(min_duration_s * fps), window_frames + 1)

    # Find the longest run of low-motion frames in the search region
    still = motion[:max_frame] < motion_threshold
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i in range(max_frame):
        if still[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    if best_len < min_frames:
        # Try with a more relaxed threshold
        relaxed = motion_threshold * 2.0
        still = motion[:max_frame] < relaxed
        best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
        for i in range(max_frame):
            if still[i]:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
            else:
                cur_len = 0

    if best_len < min_frames:
        raise CalibrationError([
            f"no static starting pose found in the first {max_search_s:.0f}s "
            f"(hold the starting position still for at least {min_duration_s:.0f}s before doing reps)"
        ])

    t0 = float(seq.t[best_start])
    t1 = float(seq.t[min(best_start + best_len - 1, len(seq) - 1)])
    return t0, t1


def auto_calibrate(exercise: str, raw: PoseSequence,
                   cfg: dict | None = None, rules: dict | None = None) -> tuple[Profile, dict]:
    """Full auto-calibration: detect the static start, find reps after it, calibrate.

    Returns (Profile, segments_dict) where segments_dict has the time windows
    that were auto-detected, for display in the UI.
    """
    cfg = cfg or load_pipeline_config()
    rules = rules or load_rules(exercise)
    clean = preprocess(raw, cfg)

    # 1. Find the static start window
    t_start, t_end = find_static_window(clean, exercise, cfg)

    # 2. Everything after the static window is potential reps
    #    Use up to 30s after calibration as the reps window for ROM estimation
    clip_end = float(clean.t[-1])
    reps_start = t_end + 0.5  # small gap after the static period
    reps_end = min(reps_start + 30.0, clip_end)

    segments: dict[str, tuple[float, float]] = {"start": (t_start, t_end)}
    if reps_end - reps_start > 2.0:
        segments["reps"] = (reps_start, reps_end)

    profile = calibrate(exercise, clean, segments, cfg, rules)
    return profile, segments
