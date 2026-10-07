"""Estimator-agnostic skeleton and the per-video pose container.

Every backend is adapted into this one joint list, so the rest of the pipeline
never sees backend-specific indices. Joints a backend cannot provide stay NaN
with zero confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINTS = (
    "nose",
    "l_ear", "r_ear",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_pinky", "r_pinky",
    "l_index", "r_index",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
    "l_heel", "r_heel",
    "l_toe", "r_toe",
)
JOINT_INDEX = {name: i for i, name in enumerate(JOINTS)}

# Side-specific joint names, used as f"{side}_{name}". "index" is the fingertip, "toe" the foot index.
SIDE_JOINTS = ("ear", "shoulder", "elbow", "wrist", "pinky", "index", "hip", "knee", "ankle", "heel", "toe")

# MediaPipe BlazePose landmark indices (33-point model).
MEDIAPIPE_INDEX = {
    "nose": 0,
    "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30,
    "l_toe": 31, "r_toe": 32,
}


@dataclass
class PoseSequence:
    """Pose track for one video. Coordinates are pixels (x right, y down)."""

    t: np.ndarray          # (T,) seconds
    xy: np.ndarray         # (T, J, 2) float, NaN where missing
    conf: np.ndarray       # (T, J) in [0, 1]
    size: tuple[int, int]  # (width, height)
    source: str = ""

    def __len__(self) -> int:
        return len(self.t)

    @property
    def fps(self) -> float:
        return float((len(self.t) - 1) / (self.t[-1] - self.t[0])) if len(self.t) > 1 else 0.0

    def with_xy(self, xy: np.ndarray) -> "PoseSequence":
        return PoseSequence(self.t, xy, self.conf, self.size, self.source)

    def window(self, t0: float, t1: float) -> slice:
        i0 = int(np.searchsorted(self.t, t0, side="left"))
        i1 = int(np.searchsorted(self.t, t1, side="right"))
        return slice(i0, i1)

    def joint(self, name: str) -> np.ndarray:
        return self.xy[:, JOINT_INDEX[name]]

    def joint_conf(self, name: str) -> np.ndarray:
        return self.conf[:, JOINT_INDEX[name]]

    def side_points(self, side: str) -> dict[str, np.ndarray]:
        """Near-side chain as {short name: (T, 2)}."""
        return {n: self.joint(f"{side}_{n}") for n in SIDE_JOINTS}

    def side_conf(self, side: str) -> dict[str, np.ndarray]:
        return {n: self.joint_conf(f"{side}_{n}") for n in SIDE_JOINTS}

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, t=self.t, xy=self.xy, conf=self.conf,
            size=np.asarray(self.size), source=np.asarray(self.source),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        d = np.load(path, allow_pickle=False)
        return cls(d["t"], d["xy"], d["conf"], (int(d["size"][0]), int(d["size"][1])), str(d["source"]))
