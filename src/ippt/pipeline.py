"""End-to-end orchestration: pose track -> calibration -> features -> reps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import Profile, calibrate
from .config import load_pipeline_config, load_rules
from .features import compute_features, level_pose
from .reps import Rep, evaluate, summarize, visible_mask
from .schema import PoseSequence
from .signal import preprocess


@dataclass
class Analysis:
    profile: Profile
    reps: list[Rep]
    summary: dict
    features: dict[str, np.ndarray]
    clean: PoseSequence          # preprocessed pose track (what the checks actually saw)
    visible: np.ndarray

    def rep_log(self) -> pd.DataFrame:
        rows = []
        for r in self.reps:
            failed = {c.name: c for c in r.checks if not c.passed}
            rows.append({
                "rep": r.index, "t_start": r.t_start, "t_end": r.t_end, "valid": r.valid,
                "in_window": r.in_window, "reasons": ",".join(r.reasons),
                "detail": "; ".join(c.text() for c in failed.values()),
            })
        return pd.DataFrame(rows, columns=["rep", "t_start", "t_end", "valid", "in_window", "reasons", "detail"])


def calibrate_sequence(exercise: str, raw: PoseSequence, segments: dict[str, tuple[float, float]],
                       cfg: dict | None = None, rules: dict | None = None) -> Profile:
    cfg = cfg or load_pipeline_config()
    rules = rules or load_rules(exercise)
    return calibrate(exercise, preprocess(raw, cfg), segments, cfg, rules)


def analyze_sequence(exercise: str, raw: PoseSequence, profile: Profile, cfg: dict | None = None,
                     rules: dict | None = None, t_start: float | None = None) -> Analysis:
    cfg = cfg or load_pipeline_config()
    rules = rules or load_rules(exercise)
    if profile.exercise != exercise:
        raise ValueError(f"profile is for {profile.exercise!r}, not {exercise!r}")
    clean = preprocess(raw, cfg)
    pts = level_pose(clean, profile.leveler, profile.side)
    feats = compute_features(pts, profile)
    vis = visible_mask(clean, profile.side, rules)
    reps = evaluate(exercise, feats, clean.t, vis, profile, rules, t_start=t_start)
    return Analysis(profile, reps, summarize(reps), feats, clean, vis)
