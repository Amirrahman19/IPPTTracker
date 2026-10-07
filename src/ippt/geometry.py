"""2D geometry helpers and the ground-levelling transform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b between segments b->a and b->c, in degrees [0, 180]."""
    u, v = a - b, c - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def signed_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perpendicular distance of p from line a->b; positive when p is to the left of a->b."""
    v, w = b - a, p - a
    with np.errstate(invalid="ignore", divide="ignore"):
        return (v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]) / np.linalg.norm(v, axis=-1)


def fit_line_angle(points: np.ndarray) -> float:
    """Direction (radians, in (-pi/2, pi/2]) of the least-squares line through `points` (N, 2)."""
    c = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ang = float(np.arctan2(vt[0, 1], vt[0, 0]))
    if ang > np.pi / 2:
        ang -= np.pi
    elif ang <= -np.pi / 2:
        ang += np.pi
    return ang


@dataclass
class Leveler:
    """Maps image pixels to a ground-aligned frame: ground is y = 0, y is up, the head side is +x.

    Distances are preserved (pixels), so camera tilt cancels while scale is kept.
    """

    origin: tuple[float, float]   # point on the ground line, in y-up image coordinates
    theta: float                  # ground-line angle in y-up image coordinates (radians)
    flip: float = 1.0             # -1 mirrors x so the participant always faces +x

    def apply(self, xy: np.ndarray) -> np.ndarray:
        p = np.stack([xy[..., 0], -xy[..., 1]], axis=-1) - np.asarray(self.origin)
        c, s = np.cos(-self.theta), np.sin(-self.theta)
        x = c * p[..., 0] - s * p[..., 1]
        y = s * p[..., 0] + c * p[..., 1]
        return np.stack([self.flip * x, y], axis=-1)

    def to_dict(self) -> dict:
        return {"origin": list(self.origin), "theta": self.theta, "flip": self.flip}

    @classmethod
    def from_dict(cls, d: dict) -> "Leveler":
        return cls(tuple(d["origin"]), float(d["theta"]), float(d["flip"]))
