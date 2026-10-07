"""Planar kinematic simulator of side-view push-ups and sit-ups.

Produces PoseSequence objects with known ground truth (including deliberate
form faults) so calibration, features and rep logic can be tested without
footage. Not a substitute for real recordings: it only checks that the code does
what the rules say.
"""

from __future__ import annotations

import numpy as np

from .schema import JOINT_INDEX, JOINTS, PoseSequence

BODY = dict(upper_arm=0.30, forearm=0.26, torso=0.50, thigh=0.45, shin=0.43)
SIZE = (1280, 720)
PX_PER_M = 400.0

PUSHUP_REP = dict(bottom=85.0, top=178.0, dur=1.6, sag=0.0, knee_drop=0.0)
SITUP_REP = dict(top=85.0, low=4.0, dur=2.0, foot_lift=0.0, bridge=0.0)
SITUP_LYING = 4.0


def _bump(n: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, n)
    return s, 0.5 - 0.5 * np.cos(2 * np.pi * s)


def _timeline(exercise: str, test_reps: list[dict], calib_reps: int, fps: int):
    """Per-frame parameter arrays plus the calibration segments and test start time."""
    defaults = PUSHUP_REP if exercise == "pushup" else SITUP_REP
    keys = list(defaults)
    cols = {k: [] for k in keys}

    REST = dict(defaults)
    if exercise == "pushup":
        REST.update(bottom=PUSHUP_REP["top"], sag=0.0, knee_drop=0.0)
    else:
        REST.update(top=SITUP_LYING, low=SITUP_LYING, foot_lift=0.0, bridge=0.0)

    primary = "bottom" if exercise == "pushup" else "top"

    def hold(sec):
        # Stay where the last rep ended, so a rep that never regains lockout / lying stays that way.
        for k in keys:
            val = cols[primary][-1] if k == primary and cols[primary] else REST[k]
            cols[k].extend([val] * int(sec * fps))

    def rep(spec):
        r = {**defaults, **spec}
        n = max(int(r["dur"] * fps), 8)
        s, b = _bump(n)
        if exercise == "pushup":
            prev = cols["bottom"][-1] if cols["bottom"] else REST["bottom"]
            base = prev + (r["top"] - prev) * s
            cols["bottom"].extend(base - (base - r["bottom"]) * b)
            cols["top"].extend([r["top"]] * n)
            cols["dur"].extend([r["dur"]] * n)
            cols["sag"].extend(r["sag"] * b)
            cols["knee_drop"].extend(r["knee_drop"] * b)
        else:
            prev = cols["top"][-1] if cols["top"] else REST["top"]
            base = prev + (r["low"] - prev) * s
            cols["top"].extend(base + (r["top"] - base) * b)
            cols["low"].extend([r["low"]] * n)
            cols["dur"].extend([r["dur"]] * n)
            cols["foot_lift"].extend(r["foot_lift"] * b)
            cols["bridge"].extend(r["bridge"] * b)

    hold(3.0)
    calib_start = (0.5, 2.5)
    c0 = len(cols[keys[0]]) / fps
    for _ in range(calib_reps):
        rep({"dur": 4.0})
        hold(0.5)
    c1 = len(cols[keys[0]]) / fps
    hold(2.0)
    test_start = len(cols[keys[0]]) / fps
    for spec in test_reps:
        rep(spec)
        hold(0.8)
    hold(1.0)
    segments = {"start": calib_start}
    if calib_reps:
        segments["reps"] = (c0, c1)
    return {k: np.asarray(v, float) for k, v in cols.items()}, segments, test_start


def _pushup_points(p: dict, body: dict) -> dict[str, np.ndarray]:
    ua, fa = body["upper_arm"], body["forearm"]
    La = body["shin"] + body["thigh"] + body["torso"]
    ah, wy = 0.08, 0.04
    e = np.radians(p["bottom"])
    n = len(e)
    W = np.stack([np.zeros(n), np.full(n, wy)], -1)
    E = np.stack([np.zeros(n), np.full(n, wy + fa)], -1)
    S = E + ua * np.stack([np.sin(e), -np.cos(e)], -1)
    th = np.arcsin(np.clip((S[:, 1] - ah) / La, -1, 1))
    u = np.stack([np.cos(th), np.sin(th)], -1)
    perp = np.stack([-np.sin(th), np.cos(th)], -1)
    ankle = S - La * u
    hip = ankle + (body["shin"] + body["thigh"]) * u + perp * p["sag"][:, None]
    knee = ankle + body["shin"] * u + perp * (0.4 * p["sag"])[:, None]
    knee[:, 1] -= p["knee_drop"]
    off = lambda x, y: np.stack([np.full(n, x), np.full(n, y)], -1)  # noqa: E731
    return {
        "nose": S + 0.28 * u - 0.05 * perp, "ear": S + 0.18 * u - 0.03 * perp, "shoulder": S, "elbow": E,
        "wrist": W, "pinky": off(0.04, 0.0), "index": off(0.09, 0.0), "hip": hip, "knee": knee, "ankle": ankle,
        "heel": ankle + off(-0.05, 0.02), "toe": np.stack([ankle[:, 0] + 0.20, np.zeros(n)], -1),
    }


def _situp_points(p: dict, body: dict) -> dict[str, np.ndarray]:
    psi = np.radians(p["top"])
    n = len(psi)
    hh = 0.12
    hip = np.stack([np.zeros(n), hh + p["bridge"]], -1)
    d = np.stack([np.cos(psi), np.sin(psi)], -1)
    S = hip + body["torso"] * d
    beta = np.radians(55.0)
    knee0 = np.array([-body["thigh"] * np.cos(beta), hh + body["thigh"] * np.sin(beta)])
    knee = np.tile(knee0, (n, 1)) + np.stack([np.zeros(n), 0.5 * p["bridge"]], -1)
    ay = 0.08
    ankle_x = knee0[0] - np.sqrt(body["shin"] ** 2 - (knee0[1] - ay) ** 2)
    lift = p["foot_lift"]
    ankle = np.stack([np.full(n, ankle_x), ay + lift], -1)
    ear = S + 0.15 * d
    rot = lambda a: np.stack([np.cos(psi + a), np.sin(psi + a)], -1)  # noqa: E731
    wrist = ear + 0.02 * rot(0.5)
    return {
        "nose": S + 0.25 * d + 0.05 * rot(np.pi / 2), "ear": ear, "shoulder": S, "elbow": wrist + 0.12 * rot(np.pi / 2),
        "wrist": wrist, "pinky": wrist + 0.03 * rot(0.0), "index": wrist + 0.05 * rot(0.0), "hip": hip, "knee": knee,
        "ankle": ankle, "heel": ankle + np.stack([np.full(n, 0.04), np.full(n, -0.08)], -1),
        "toe": ankle + np.stack([np.full(n, -0.18), np.full(n, -0.08)], -1),
    }


def make_session(exercise: str, test_reps: list[dict] | None = None, calib_reps: int = 3, fps: int = 30,
                 seed: int = 0, noise_px: float = 1.0, dropout: float = 0.0, roll_deg: float = 0.0,
                 mirror: bool = False, body: dict | None = None):
    """Simulate a calibration block followed by test reps.

    Returns (PoseSequence, calibration_segments, test_start_time_s). `test_reps` is a
    list of dicts overriding PUSHUP_REP / SITUP_REP (e.g. {"bottom": 120} for a shallow push-up).
    """
    body = {**BODY, **(body or {})}
    test_reps = [{}] * 5 if test_reps is None else test_reps
    p, segments, test_start = _timeline(exercise, test_reps, calib_reps, fps)
    n = len(next(iter(p.values())))
    if exercise == "pushup":
        pts = _pushup_points({"bottom": p["bottom"], "sag": p["sag"], "knee_drop": p["knee_drop"]}, body)
    else:
        pts = _situp_points({"top": p["top"], "bridge": p["bridge"], "foot_lift": p["foot_lift"]}, body)

    rng = np.random.default_rng(seed)
    world = np.zeros((n, len(JOINTS), 2))
    for name in JOINTS:
        if name == "nose":
            world[:, JOINT_INDEX[name]] = pts["nose"]
            continue
        side, short = name.split("_", 1)
        world[:, JOINT_INDEX[name]] = pts[short] + (0.01 if side == "r" else 0.0)  # far side sits just behind
    if mirror:
        world[..., 0] *= -1
    a = np.radians(roll_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    world = world @ R.T
    xy = np.stack([100 + PX_PER_M * (world[..., 0] - world[..., 0].min()),
                   SIZE[1] - 80 - PX_PER_M * (world[..., 1] - world[..., 1].min())], -1)
    xy += rng.normal(0, noise_px, xy.shape)
    conf = np.full((n, len(JOINTS)), 0.95)
    if dropout > 0:
        miss = rng.random((n, len(JOINTS))) < dropout
        xy[miss] = np.nan
        conf[miss] = 0.0
    return PoseSequence(np.arange(n) / fps, xy, conf, SIZE, f"sim-{exercise}"), segments, test_start
