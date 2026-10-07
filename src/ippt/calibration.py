"""Per-participant calibration.

Calibration answers "how do I measure this person?", never "what do I demand of
this person?". From a short clip it derives:

  * the near-side limb chain (so it cannot flip mid-set),
  * the ground line and facing direction (camera tilt and mirroring cancel out),
  * segment lengths (so every distance is body-relative),
  * natural posture offsets (so a normal hip contour is not read as sag),
  * the observed range of motion, used only to place rep-segmentation thresholds.

The validity thresholds in the rules file stay fixed. Where a calibrated value
feeds one, it can only tighten it (see `clamp_tighter`).

Calibration segments are time windows (seconds) in the calibration clip:
  start     required  static start posture. Push-up: arms locked at the top.
                      Sit-up: lying flat, shoulder blades on the mat.
  opposite  optional  static opposite posture. Push-up: bottom. Sit-up: full-range top.
  reps      optional  a few slow reference reps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .features import body_offset, compute_features, level_pose
from .geometry import Leveler, dist, fit_line_angle, signed_line_distance
from .schema import PoseSequence

# Candidate physical ground-contact points, best first. Both exercises have two
# body points resting on the floor/mat that give a true ground direction.
GROUND_CANDIDATES = {
    "pushup": [("toe", "index"), ("toe", "wrist"), ("ankle", "wrist")],
    "situp": [("toe", "heel"), ("toe", "ankle")],
}


class CalibrationError(Exception):
    """Calibration rejected. `.reasons` lists every failed gate."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("calibration rejected: " + "; ".join(self.reasons))


def clamp_tighter(calibrated: float, hard: float, lower_bound: bool) -> float:
    """Combine a calibrated limit with a hard rule limit, never loosening the rule.

    lower_bound=True : the measurement must be >= limit, so take the larger value.
    lower_bound=False: the measurement must be <= limit, so take the smaller value.
    """
    return max(calibrated, hard) if lower_bound else min(calibrated, hard)


@dataclass
class Profile:
    exercise: str
    side: str
    leveler: Leveler
    lengths: dict[str, float]
    baseline: dict
    rom: dict
    quality: dict = field(default_factory=dict)
    feasibility: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple, np.ndarray)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return o
        return clean({
            "exercise": self.exercise, "side": self.side, "leveler": self.leveler.to_dict(),
            "lengths": self.lengths, "baseline": self.baseline, "rom": self.rom,
            "quality": self.quality, "feasibility": self.feasibility, "warnings": self.warnings,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(d["exercise"], d["side"], Leveler.from_dict(d["leveler"]), d["lengths"], d["baseline"],
                   d["rom"], d.get("quality", {}), d.get("feasibility", {}), d.get("warnings", []))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


def _median_point(p: np.ndarray) -> np.ndarray:
    return np.nanmedian(p, axis=0)


def _p90(x: np.ndarray) -> float:
    return float(np.nanpercentile(x, 90)) if np.any(~np.isnan(x)) else float("nan")


def select_side(seq: PoseSequence, window: slice, joints: list[str]) -> tuple[str, dict[str, float]]:
    """Pick the limb chain the camera sees best during the start segment."""
    score = {s: float(np.mean([seq.joint_conf(f"{s}_{j}")[window].mean() for j in joints])) for s in ("l", "r")}
    return ("l" if score["l"] > score["r"] else "r"), score


def _ground_leveler(exercise: str, seq: PoseSequence, side: str, window: slice, cc: dict, reasons: list[str]):
    min_sep = cc["min_ground_separation"] * seq.size[1]
    for names in GROUND_CANDIDATES[exercise]:
        pts = [_median_point(seq.joint(f"{side}_{n}")[window]) for n in names]
        if any(np.isnan(p).any() for p in pts):
            continue
        up = np.array([[p[0], -p[1]] for p in pts])
        if np.linalg.norm(up[0] - up[1]) < min_sep:
            continue
        return Leveler(tuple(up.mean(axis=0)), fit_line_angle(up)), names
    reasons.append("no pair of ground-contact points is visible and far enough apart to fit a ground line")
    return None, None


def calibrate(exercise: str, seq: PoseSequence, segments: dict[str, tuple[float, float]],
              cfg: dict, rules: dict) -> Profile:
    """Build a Profile from a preprocessed calibration sequence, or raise CalibrationError."""
    cc = cfg["calibration"]
    if "start" not in segments:
        raise ValueError("calibration segments must include 'start'")
    sl = {k: seq.window(*v) for k, v in segments.items()}
    start = sl["start"]
    reasons: list[str] = []
    warnings: list[str] = []

    n_start = start.stop - start.start
    if n_start < cc["min_start_frames"]:
        reasons.append(f"start segment has {n_start} frames, need {cc['min_start_frames']}")
        raise CalibrationError(reasons)

    required = rules["required_joints"]
    side, side_scores = select_side(seq, start, required)
    mean_conf = side_scores[side]
    if mean_conf < cc["min_mean_conf"]:
        reasons.append(f"near-side keypoint confidence {mean_conf:.2f} < {cc['min_mean_conf']}")

    # Side-view check: in a true side view the two shoulders nearly overlap in x.
    near_sh, far_sh = seq.joint(f"{side}_shoulder"), seq.joint(f"{'r' if side == 'l' else 'l'}_shoulder")
    torso_px = _p90(dist(near_sh, seq.joint(f"{side}_hip"))[start])
    if not np.isfinite(torso_px) or torso_px <= 0:
        reasons.append("torso (shoulder to hip) is not measurable in the start segment")
        raise CalibrationError(reasons)
    frontal = float(np.nanmedian(np.abs(near_sh[start, 0] - far_sh[start, 0]))) / torso_px
    if np.isfinite(frontal) and frontal > cc["max_frontal_ratio"]:
        reasons.append(f"not a side view: shoulder x-gap is {frontal:.2f} torso lengths (limit {cc['max_frontal_ratio']})")

    # Static start posture must actually be static.
    for j in required:
        p = seq.joint(f"{side}_{j}")[start]
        wander = float(np.nanpercentile(np.linalg.norm(p - np.nanmedian(p, axis=0), axis=1), 90)) / torso_px
        if wander > cc["max_start_motion"]:
            reasons.append(f"start posture not static: {j} wanders {wander:.2f} torso lengths (limit {cc['max_start_motion']})")
            break

    lev, ground_names = _ground_leveler(exercise, seq, side, start, cc, reasons)
    if reasons:
        raise CalibrationError(reasons)

    # Facing: make the head side +x in every video, whichever way the participant faces the camera.
    lp = level_pose(seq, lev, side)
    dx = float(np.nanmedian(lp["shoulder"][start, 0] - lp["ankle"][start, 0]))
    lev = Leveler(lev.origin, lev.theta, 1.0 if dx >= 0 else -1.0)
    lp = level_pose(seq, lev, side)
    if float(np.nanmedian(lp["shoulder"][start, 1])) < 0:
        warnings.append("shoulder is below the ground line: camera may be upside down or the ground points are wrong")

    # Segment lengths: 90th percentile over all calibration frames (limbs at full projected length).
    idx = np.zeros(len(seq), dtype=bool)
    for s in sl.values():
        idx[s] = True
    d = lambda a, b: _p90(dist(lp[a], lp[b])[idx])  # noqa: E731
    lengths = {
        "upper_arm": d("shoulder", "elbow"), "forearm": d("elbow", "wrist"), "torso": d("shoulder", "hip"),
        "thigh": d("hip", "knee"), "shin": d("knee", "ankle"), "foot": d("heel", "toe"), "body": d("shoulder", "ankle"),
    }
    needed = {"torso", "body"} | ({"upper_arm", "forearm"} if exercise == "pushup" else {"thigh", "shin"})
    missing = [k for k in needed if not np.isfinite(lengths[k]) or lengths[k] <= 0]
    if missing:
        raise CalibrationError([f"segment length(s) not measurable: {', '.join(sorted(missing))}"])
    for k, v in lengths.items():
        if k not in needed and (not np.isfinite(v) or v <= 0):
            warnings.append(f"{k} length not measurable; using a fallback")
            lengths[k] = 0.5 * lengths["shin"] if k == "foot" and np.isfinite(lengths["shin"]) else lengths["torso"] * 0.5

    # Natural posture baseline from the start segment.
    m = {k: _median_point(lp[k][start]) for k in ("ankle", "heel", "toe", "hip", "knee")}
    baseline = {
        "body_offset": float(np.nanmedian(body_offset({k: lp[k][start] for k in ("hip", "ankle", "shoulder")}, lengths["torso"]))),
        "knee_offset": float(np.nanmedian(signed_line_distance(lp["knee"][start], lp["ankle"][start], lp["shoulder"][start]))
                             / lengths["torso"]),
        "hip_y": float(m["hip"][1]),
        "hip_height": float(m["hip"][1] / lengths["body"]),
        "knee_height": float(m["knee"][1] / lengths["body"]),
        "ankle": m["ankle"].tolist(), "heel": m["heel"].tolist(), "toe": m["toe"].tolist(),
    }
    if exercise == "pushup" and (baseline["hip_height"] <= 0.02 or baseline["knee_height"] <= 0.02):
        raise CalibrationError(["start posture has the hips or knees on the ground; hold a straight plank for calibration"])
    for k in ("hip_height", "knee_height"):  # ratios divide by these; keep them away from zero
        baseline[k] = max(baseline[k], 1e-3)

    prof = Profile(exercise, side, lev, lengths, baseline, {}, warnings=warnings)

    # Range of motion on the primary signal, used only to place segmentation thresholds.
    seg = rules["segmentation"]
    feats = compute_features(lp, prof)
    sig = feats[seg["signal"]]
    start_val = float(np.nanmedian(sig[start]))
    opp_val = float(np.nanmedian(sig[sl["opposite"]])) if "opposite" in sl else None
    reps_lo = reps_hi = None
    if "reps" in sl:
        reps_lo, reps_hi = (float(np.nanpercentile(sig[sl["reps"]], q)) for q in (5, 95))
    vals = [v for v in (start_val, opp_val, reps_lo, reps_hi) if v is not None]
    lo, hi, source = min(vals), max(vals), ("reps" if reps_lo is not None else "static")
    if len(vals) < 2 or hi - lo < 10.0:
        warnings.append("calibration range of motion is missing or too small; using default segmentation range")
        d_rom = seg["default_rom"]
        lo, hi, source = d_rom["lo"], d_rom["hi"], "default"
    prof.rom = {"lo": lo, "hi": hi, "start": start_val, "source": source}

    prof.feasibility = _feasibility(exercise, feats, sl, rules)
    prof.quality = {
        "mean_conf": mean_conf, "side_scores": side_scores, "frontal_ratio": frontal,
        "ground_points": list(ground_names), "n_start_frames": n_start,
    }
    return prof


def _feasibility(exercise: str, feats: dict, sl: dict, rules: dict) -> dict:
    """Informational: could the participant hit the rule limits during calibration?"""
    c = rules["checks"]
    rng = [s for k, s in sl.items() if k in ("opposite", "reps")]
    out: dict = {}
    if exercise == "pushup":
        out["start_lockout_ok"] = bool(np.nanmedian(feats["elbow_angle"][sl["start"]]) >= c["lockout"]["min_elbow_deg"])
        if rng:
            out["reached_depth"] = bool(min(np.nanmin(feats["upper_arm_incline_deg"][s]) for s in rng)
                                        <= c["depth"]["max_upper_arm_incline_deg"])
    else:
        if rng:
            out["reached_top"] = bool(max(np.nanmax(feats["torso_angle_deg"][s]) for s in rng) >= c["top"]["min_torso_deg"])
    return out
