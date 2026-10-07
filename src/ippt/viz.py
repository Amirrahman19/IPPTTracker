"""Annotated review video: near-side skeleton, live count, last decision."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pipeline import Analysis

BONES = [("ear", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
         ("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "toe"), ("ankle", "toe"), ("wrist", "index")]
GREEN, RED, WHITE = (80, 200, 80), (60, 60, 230), (255, 255, 255)


def annotate_video(video: str | Path, analysis: Analysis, out: str | Path, hold_s: float = 1.5) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pts = analysis.clean.side_points(analysis.profile.side)
    t = analysis.clean.t
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(t):
            break
        done = [r for r in analysis.reps if r.t_end is not None and r.t_end <= t[i]]
        valid = sum(r.valid and r.in_window for r in done)
        last = done[-1] if done else None
        for a, b in BONES:
            pa, pb = pts[a][i], pts[b][i]
            if not (np.isnan(pa).any() or np.isnan(pb).any()):
                cv2.line(frame, tuple(int(v) for v in pa), tuple(int(v) for v in pb), GREEN, 3)
        for p in pts.values():
            if not np.isnan(p[i]).any():
                cv2.circle(frame, tuple(int(v) for v in p[i]), 5, WHITE, -1)
        cv2.putText(frame, f"{analysis.profile.exercise}  valid: {valid}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 3)
        if last is not None and t[i] - last.t_end <= hold_s:
            colour = GREEN if last.valid else RED
            cv2.putText(frame, f"#{last.index} {last.explain()}"[:90], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
