"""Confidence gating, short-gap interpolation and One Euro smoothing."""

from __future__ import annotations

import numpy as np

from .schema import PoseSequence


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of at most `max_gap` frames along axis 0. Longer gaps stay NaN."""
    flat = x.reshape(len(x), -1).astype(float).copy()
    for j in range(flat.shape[1]):
        col = flat[:, j]
        idx = np.flatnonzero(~np.isnan(col))
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
            a, b = idx[k], idx[k + 1]
            col[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [col[a], col[b]])
    return flat.reshape(x.shape)


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))


def one_euro(x: np.ndarray, t: np.ndarray, min_cutoff: float = 1.5,
             beta: float = 8.0, d_cutoff: float = 1.0) -> np.ndarray:
    """Offline One Euro filter along axis 0. NaN stays NaN and resets that element's state."""
    out = np.full(x.shape, np.nan)
    state_x = np.full(x.shape[1:], np.nan)
    state_dx = np.zeros(x.shape[1:])
    prev_raw = np.full(x.shape[1:], np.nan)
    with np.errstate(invalid="ignore"):
        for i in range(len(x)):
            xi = x[i]
            ok = ~np.isnan(xi)
            had = ok & ~np.isnan(state_x)
            dt = max(float(t[i] - t[i - 1]), 1e-3) if i else 1 / 30
            dx = np.where(had, (xi - prev_raw) / dt, 0.0)
            ad = _alpha(d_cutoff, dt)
            dx_hat = np.where(had, ad * dx + (1 - ad) * state_dx, 0.0)
            a = _alpha(min_cutoff + beta * np.abs(dx_hat), dt)
            x_hat = np.where(had, a * xi + (1 - a) * state_x, xi)
            out[i] = x_hat
            state_x = np.where(ok, x_hat, np.nan)
            state_dx = np.where(had, dx_hat, 0.0)
            prev_raw = xi
    return out


def preprocess(seq: PoseSequence, cfg: dict) -> PoseSequence:
    """Gate low-confidence joints, interpolate short dropouts, smooth. Returns a new sequence.

    Smoothing runs on coordinates divided by image height so the filter
    parameters do not depend on video resolution. `conf` is left untouched.
    """
    p = cfg["preprocess"]
    h = float(seq.size[1])
    xy = seq.xy.astype(float) / h
    xy[seq.conf < p["min_conf"]] = np.nan
    xy = interpolate_gaps(xy, p["max_gap_frames"])
    e = p["one_euro"]
    xy = one_euro(xy, seq.t, e["min_cutoff"], e["beta"], e["d_cutoff"])
    return seq.with_xy(xy * h)
