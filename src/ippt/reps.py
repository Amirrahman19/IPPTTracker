"""Rep segmentation and validity checks.

Two stages, deliberately separate:

  1. Segmentation finds attempts (a dip below / rise above a hysteresis band on
     one primary signal). The band is placed from the participant's calibrated
     range of motion so jitter is not counted and weak ROM is still detected.
  2. Evaluation runs fixed rule checks on each attempt. A rep is valid only if
     every check passes, and each failure keeps its measured value and limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .calibration import Profile, clamp_tighter
from .schema import PoseSequence


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    threshold: float
    unit: str = ""

    def text(self) -> str:
        op = "OK" if self.passed else "FAIL"
        return f"{self.name} {op} ({self.value:.2f}{self.unit} vs limit {self.threshold:.2f}{self.unit})"


@dataclass
class Rep:
    index: int
    t_start: float
    t_end: float | None        # None when the clip ended mid-rep
    valid: bool
    reasons: list[str]
    checks: list[Check] = field(default_factory=list)
    in_window: bool = True     # inside the timed station window

    def explain(self) -> str:
        if self.valid:
            return "valid"
        return "no-count: " + "; ".join(c.text() for c in self.checks if not c.passed) if self.reasons else "no-count"


def segmentation_thresholds(prof: Profile, rules: dict) -> dict:
    """Thresholds on the primary signal, plus the same band in "depth" coordinates.

    Depth coordinate d: 0 at the home posture, larger the further into the rep.
    Push-up home is the locked-out top (d = hi - elbow); sit-up home is lying flat (d = torso - lo).
    """
    seg = rules["segmentation"]
    lo, hi = prof.rom["lo"], prof.rom["hi"]
    rom = hi - lo
    if seg["direction"] == "below":
        enter = min(hi - seg["enter_frac"] * rom, seg["enter_max"])
        exit_ = max(hi - seg["exit_frac"] * rom, enter + 3.0)
        enter_d, exit_d = hi - enter, hi - exit_
    else:
        enter = max(lo + seg["enter_frac"] * rom, lo + seg["enter_min_deg"])
        exit_ = max(min(lo + seg["exit_frac"] * rom, enter - 2.0), lo + 3.0)
        enter_d, exit_d = enter - lo, exit_ - lo
    return {"enter": enter, "exit": exit_, "enter_d": enter_d, "exit_d": exit_d,
            "min_prominence": seg["min_excursion_frac"] * rom}


def depth_coordinate(sig: np.ndarray, prof: Profile, rules: dict) -> np.ndarray:
    if rules["segmentation"]["direction"] == "below":
        return prof.rom["hi"] - sig
    return sig - prof.rom["lo"]


def _fill_nan(x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def detect_attempts(d: np.ndarray, enter_d: float, exit_d: float, min_prominence: float) -> list[tuple[int, int, int | None]]:
    """Each deep point (a prominent peak of the depth coordinate) is one attempt: (start, deepest, end).

    start: last time before the deepest point the participant was at home (d <= exit_d).
    end:   first return home after it. If they descend again without getting home, the end is the
           best point reached before the next descent, so the missed lockout/return is still judged
           on that attempt. None if the clip ends before they get home.
    """
    x = _fill_nan(d)
    peaks, _ = find_peaks(x, height=enter_d, prominence=min_prominence)
    out: list[tuple[int, int, int | None]] = []
    prev_end = 0
    n = len(x)
    for k, v in enumerate(peaks):
        nxt = int(peaks[k + 1]) if k + 1 < len(peaks) else n
        back = np.flatnonzero(x[prev_end:v] <= exit_d)
        start = prev_end + int(back[-1]) if len(back) else prev_end + int(np.argmin(x[prev_end:v + 1]))
        fwd = np.flatnonzero(x[v:nxt] <= exit_d)
        if len(fwd):
            end = int(v + fwd[0])
        elif nxt < n:
            end = int(v + np.argmin(x[v:nxt]))
        else:
            end = None
        out.append((start, int(v), end))
        prev_end = end if end is not None else n
    return out


def visible_mask(clean: PoseSequence, side: str, rules: dict) -> np.ndarray:
    """Frames where every required near-side joint is available after gating and short-gap interpolation."""
    return np.all([~np.isnan(clean.joint(f"{side}_{j}")).any(axis=-1) for j in rules["required_joints"]], axis=0)


def _chk(name: str, value: float, thr: float, upper: bool, unit: str = "") -> Check:
    """upper=True: pass when value <= thr. upper=False: pass when value >= thr. NaN fails."""
    ok = bool(value <= thr) if upper else bool(value >= thr)
    return Check(name, ok, float(value), float(thr), unit)


def _nan(fn, x: np.ndarray) -> float:
    return float(fn(x)) if np.any(~np.isnan(x)) else float("nan")


def _pushup_checks(f: dict, w: slice, end: slice, prof: Profile, rules: dict) -> list[Check]:
    c = rules["checks"]
    return [
        _chk("depth", _nan(np.nanmin, f["upper_arm_incline_deg"][w]), c["depth"]["max_upper_arm_incline_deg"], True, "deg"),
        _chk("lockout", _nan(np.nanmax, f["elbow_angle"][end]), c["lockout"]["min_elbow_deg"], False, "deg"),
        _chk("body_sag", -_nan(np.nanmin, f["body_dev"][w]), c["body_line"]["max_sag_norm"], True),
        _chk("body_pike", _nan(np.nanmax, f["body_dev"][w]), c["body_line"]["max_pike_norm"], True),
        _chk("knee_contact", -_nan(np.nanmin, f["knee_dev"][w]), c["ground_contact"]["max_knee_drop_norm"], True),
    ]


def _situp_checks(f: dict, w: slice, end: slice, prof: Profile, rules: dict) -> list[Check]:
    c = rules["checks"]
    ret = c["return"]
    # The calibrated lying angle can only tighten the return limit, never loosen it.
    ret_limit = clamp_tighter(prof.rom["start"] + ret["tolerance_deg"], ret["max_torso_deg"], lower_bound=False)
    out = [
        _chk("top", _nan(np.nanmax, f["torso_angle_deg"][w]), c["top"]["min_torso_deg"], False, "deg"),
        _chk("return", _nan(np.nanmin, f["torso_angle_deg"][end]), ret_limit, True, "deg"),
        _chk("feet_anchor", _nan(np.nanmax, f["foot_drift"][w]), c["feet_anchor"]["max_drift_foot_lengths"], True),
        _chk("hip_bridge", _nan(np.nanmax, f["hip_rise"][w]), c["hip_bridge"]["max_rise_torso"], True),
    ]
    k = c["knee_angle"]
    if k.get("enabled"):
        ka = float(np.nanmedian(f["knee_angle"][w]))
        out.append(_chk("knee_min", ka, k["min_deg"], False, "deg"))
        out.append(_chk("knee_max", ka, k["max_deg"], True, "deg"))
    h = c["hands"]
    if h.get("enabled"):
        out.append(_chk("hands", _nan(np.nanmax, f["wrist_ear_dist"][w]), h["max_wrist_ear_upper_arm"], True))
    return out


_CHECKS = {"pushup": _pushup_checks, "situp": _situp_checks}


def evaluate(exercise: str, feats: dict[str, np.ndarray], t: np.ndarray, visible: np.ndarray,
             prof: Profile, rules: dict, t_start: float | None = None) -> list[Rep]:
    seg = rules["segmentation"]
    th = segmentation_thresholds(prof, rules)
    d = depth_coordinate(feats[seg["signal"]], prof, rules)
    n = len(d)
    fps = (n - 1) / (t[-1] - t[0]) if n > 1 else 30.0
    settle = max(1, int(round(seg["settle_s"] * fps)))
    attempts = detect_attempts(d, th["enter_d"], th["exit_d"], th["min_prominence"])
    min_vis = rules["checks"]["visibility"]["min_visible_fraction"]

    reps: list[Rep] = []
    for k, (i0, deepest, i1) in enumerate(attempts):
        if i1 is None:
            reps.append(Rep(len(reps) + 1, float(t[i0]), None, False, ["incomplete"],
                            [Check("incomplete", False, float("nan"), float("nan"))]))
            continue
        if t[i1] - t[i0] < seg["min_duration_s"]:
            continue  # too brief to be a rep: tracker glitch
        nxt = attempts[k + 1][1] if k + 1 < len(attempts) else n
        w = slice(i0, i1 + 1)
        end = slice(deepest, min(i1 + settle + 1, nxt, n))  # best position reached after the deepest point
        checks = _CHECKS[exercise](feats, w, end, prof, rules)
        checks.insert(0, _chk("visibility", float(np.mean(visible[w])), min_vis, False))
        failed = [c.name for c in checks if not c.passed]
        reps.append(Rep(len(reps) + 1, float(t[i0]), float(t[i1]), not failed, failed, checks))

    if reps:
        t0 = reps[0].t_start if t_start is None else t_start
        limit = t0 + rules["session"]["duration_s"]
        for r in reps:
            r.in_window = r.t_end is not None and t0 <= r.t_start and r.t_end <= limit
    return reps


def summarize(reps: list[Rep]) -> dict:
    timed = [r for r in reps if r.in_window]
    return {
        "valid": sum(r.valid for r in timed),
        "no_count": sum(not r.valid for r in timed),
        "attempts": len(timed),
        "outside_window": len(reps) - len(timed),
    }
