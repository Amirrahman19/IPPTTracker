"""Frame-by-frame live pipeline: pose estimation, streaming calibration, rep counting.

Designed for a phone camera sending JPEG frames over WebSocket.
States: waiting -> calibrating -> counting.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .calibration import CalibrationError, Profile, calibrate, select_side
from .config import load_pipeline_config, load_rules
from .features import compute_features, level_pose
from .geometry import Leveler, angle_deg
from .schema import JOINTS, JOINT_INDEX, PoseSequence, SIDE_JOINTS


class StreamingOneEuro:
    """Per-frame One Euro filter for a single 2D point."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 8.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx = np.zeros(2)
        self._prev_raw: np.ndarray | None = None
        self._prev_t: float | None = None

    def update(self, xy: np.ndarray, t: float) -> np.ndarray:
        if np.isnan(xy).any():
            self._x = None
            self._dx = np.zeros(2)
            self._prev_raw = None
            return xy
        dt = max(t - self._prev_t, 1e-3) if self._prev_t is not None else 1 / 15
        if self._x is not None and self._prev_raw is not None:
            dx = (xy - self._prev_raw) / dt
            ad = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * self.d_cutoff * dt))
            self._dx = ad * dx + (1 - ad) * self._dx
            cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
            a = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))
            self._x = a * xy + (1 - a) * self._x
        else:
            self._x = xy.copy()
            self._dx = np.zeros(2)
        self._prev_raw = xy.copy()
        self._prev_t = t
        return self._x.copy()


class StreamingRepCounter:
    """Hysteresis-based rep counter that works on a single signal value per frame."""

    def __init__(self, high: float, low: float):
        self.high = high
        self.low = low
        self.state = "up"
        self.count = 0
        self.last_rep_t: float | None = None

    def update(self, value: float, t: float) -> bool:
        if np.isnan(value):
            return False
        if self.state == "up" and value < self.low:
            self.state = "down"
        elif self.state == "down" and value > self.high:
            self.state = "up"
            self.count += 1
            self.last_rep_t = t
            return True
        return False


CALIB_HOLD_S = 2.5
CALIB_MIN_FRAMES = 30


class LiveSession:
    """Manages a single live assessment session."""

    def __init__(self):
        self.exercise = "pushup"
        self.state = "waiting"
        self.profile: Profile | None = None
        self.rep_counter: StreamingRepCounter | None = None
        self.rep_count = 0
        self.signal_history: list[tuple[float, float]] = []
        self.rep_events: list[dict] = []
        self._estimator = None
        self._live_lm = None
        self._calib_ts: list[float] = []
        self._calib_xys: list[np.ndarray] = []
        self._calib_confs: list[np.ndarray] = []
        self._calib_size: tuple[int, int] | None = None
        self._filters: dict[str, StreamingOneEuro] = {}
        self._t0: float | None = None
        self._frame_count = 0
        self._still_frames = 0
        self._prev_xy: np.ndarray | None = None
        self._cfg = load_pipeline_config()
        self._rules = load_rules(self.exercise)
        self._side: str | None = None

    def _get_live_lm(self):
        if self._live_lm is None:
            from .pose import MediaPipePoseEstimator
            if self._estimator is None:
                self._estimator = MediaPipePoseEstimator.from_config(self._cfg)
            self._live_lm = self._estimator.open_live()
        return self._live_lm

    def set_exercise(self, exercise: str):
        if exercise not in ("pushup", "situp"):
            return
        self.exercise = exercise
        self._rules = load_rules(exercise)
        self.reset()

    def reset(self):
        self.state = "waiting"
        self.profile = None
        self.rep_counter = None
        self.rep_count = 0
        self.signal_history = []
        self.rep_events = []
        self._calib_ts = []
        self._calib_xys = []
        self._calib_confs = []
        self._calib_size = None
        self._filters = {}
        self._t0 = None
        self._frame_count = 0
        self._still_frames = 0
        self._prev_xy = None
        self._side = None
        if self._live_lm is not None:
            self._live_lm.close()
            self._live_lm = None

    def _estimate_frame(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._get_live_lm().estimate(bgr)

    def _smooth_xy(self, xy: np.ndarray, t: float) -> np.ndarray:
        out = xy.copy()
        e = self._cfg["preprocess"]["one_euro"]
        for j in range(len(JOINTS)):
            key = str(j)
            if key not in self._filters:
                self._filters[key] = StreamingOneEuro(e["min_cutoff"], e["beta"], e["d_cutoff"])
            out[j] = self._filters[key].update(xy[j], t)
        return out

    def _check_stillness(self, xy: np.ndarray, h: int) -> bool:
        if self._prev_xy is None:
            self._prev_xy = xy.copy()
            return False
        ok = ~np.isnan(xy[:, 0]) & ~np.isnan(self._prev_xy[:, 0])
        if ok.sum() < 6:
            self._prev_xy = xy.copy()
            return False
        motion = np.nanmax(np.linalg.norm(xy[ok] - self._prev_xy[ok], axis=-1)) / h
        self._prev_xy = xy.copy()
        return motion < 0.02

    def _try_calibrate(self) -> bool:
        if len(self._calib_ts) < CALIB_MIN_FRAMES:
            return False
        t = np.array(self._calib_ts)
        xy = np.stack(self._calib_xys)
        conf = np.stack(self._calib_confs)
        seq = PoseSequence(t, xy, conf, self._calib_size, "live")

        from .signal import preprocess
        clean = preprocess(seq, self._cfg)

        required = self._rules["required_joints"]
        side, _ = select_side(clean, slice(0, len(clean)), required)
        self._side = side

        seg_rules = self._rules["segmentation"]
        d_rom = seg_rules["default_rom"]
        segments = {"start": (float(t[0]), float(t[-1]))}
        try:
            prof = calibrate(self.exercise, clean, segments, self._cfg, self._rules)
            if prof.rom["source"] == "default":
                prof.rom = {"lo": d_rom["lo"], "hi": d_rom["hi"],
                            "start": prof.rom["start"], "source": "default"}
        except CalibrationError:
            return False

        self.profile = prof
        lo, hi = prof.rom["lo"], prof.rom["hi"]
        rom = hi - lo

        if seg_rules["direction"] == "below":
            enter = hi - seg_rules["enter_frac"] * rom
            exit_ = hi - seg_rules["exit_frac"] * rom
            self.rep_counter = StreamingRepCounter(high=exit_, low=enter)
        else:
            enter = lo + seg_rules["enter_frac"] * rom
            exit_ = lo + seg_rules["exit_frac"] * rom
            self.rep_counter = StreamingRepCounter(high=enter, low=exit_)
            self.rep_counter.state = "down"

        return True

    def _get_signal_value(self, smooth_xy: np.ndarray) -> float:
        if self._side is None or self.profile is None:
            return float("nan")
        side = self._side
        pts = {}
        for n in SIDE_JOINTS:
            idx = JOINT_INDEX[f"{side}_{n}"]
            pts[n] = smooth_xy[idx]
        sig = self._rules["segmentation"]["signal"]
        if sig == "elbow_angle":
            return float(angle_deg(pts["shoulder"], pts["elbow"], pts["wrist"]))
        elif sig == "torso_angle_deg":
            dx = pts["shoulder"][0] - pts["hip"][0]
            dy = pts["shoulder"][1] - pts["hip"][1]
            return float(np.degrees(np.arctan2(dy, dx)))
        return float("nan")

    def process_frame(self, jpeg_bytes: bytes) -> dict:
        bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"state": self.state, "error": "bad_frame"}

        h, w = bgr.shape[:2]
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        self._frame_count += 1

        xy, conf = self._estimate_frame(bgr)

        min_conf = self._cfg["preprocess"]["min_conf"]
        clean_xy = xy.copy()
        clean_xy[conf < min_conf] = np.nan
        smooth_xy = self._smooth_xy(clean_xy, t)

        norm_joints = self._normalize_joints(smooth_xy, w, h)

        if self.state == "waiting":
            is_still = self._check_stillness(xy, h)
            person_visible = np.sum(conf > min_conf) >= 6
            if is_still and person_visible:
                self._still_frames += 1
            else:
                self._still_frames = 0
            if self._still_frames >= 5:
                self.state = "calibrating"
                self._calib_ts = []
                self._calib_xys = []
                self._calib_confs = []
                self._calib_size = (w, h)
                self._still_frames = 0

            return {
                "state": self.state,
                "joints": norm_joints,
                "rep_count": 0,
                "signal": None,
                "signal_history": [],
                "message": "Hold starting position still...",
                "still_progress": min(self._still_frames / 5, 1.0) if self._still_frames > 0 else 0,
            }

        if self.state == "calibrating":
            self._calib_ts.append(t)
            self._calib_xys.append(xy)
            self._calib_confs.append(conf)

            elapsed = t - self._calib_ts[0] if self._calib_ts else 0
            progress = min(elapsed / CALIB_HOLD_S, 1.0)

            if elapsed >= CALIB_HOLD_S:
                if self._try_calibrate():
                    self.state = "counting"
                    return {
                        "state": self.state,
                        "joints": norm_joints,
                        "rep_count": 0,
                        "signal": None,
                        "signal_history": [],
                        "message": "Calibrated! Start your reps.",
                        "side": self._side,
                    }
                else:
                    self.state = "waiting"
                    self._still_frames = 0
                    self._prev_xy = None
                    return {
                        "state": "waiting",
                        "joints": norm_joints,
                        "rep_count": 0,
                        "signal": None,
                        "signal_history": [],
                        "message": "Calibration failed — hold still in starting position.",
                    }

            return {
                "state": self.state,
                "joints": norm_joints,
                "rep_count": 0,
                "signal": None,
                "signal_history": [],
                "message": f"Calibrating... hold still ({progress:.0%})",
                "calib_progress": progress,
            }

        sig_val = self._get_signal_value(smooth_xy)
        rep_completed = False
        if self.rep_counter and not np.isnan(sig_val):
            rep_completed = self.rep_counter.update(sig_val, t)
            if rep_completed:
                self.rep_count = self.rep_counter.count
                self.rep_events.append({"t": round(t, 2), "rep": self.rep_count})

        if not np.isnan(sig_val):
            self.signal_history.append((round(t, 2), round(sig_val, 1)))
            if len(self.signal_history) > 600:
                self.signal_history = self.signal_history[-600:]

        return {
            "state": self.state,
            "joints": norm_joints,
            "rep_count": self.rep_count,
            "signal": round(sig_val, 1) if not np.isnan(sig_val) else None,
            "signal_history": self.signal_history[-120:],
            "rep_completed": rep_completed,
            "counter_state": self.rep_counter.state if self.rep_counter else None,
            "side": self._side,
        }

    def _normalize_joints(self, xy: np.ndarray, w: int, h: int) -> list:
        out = []
        for j in range(len(JOINTS)):
            if np.isnan(xy[j, 0]):
                out.append(None)
            else:
                out.append([round(float(xy[j, 0]) / w, 4), round(float(xy[j, 1]) / h, 4)])
        return out
