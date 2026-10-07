"""Per-frame joint features in the ground-levelled, body-normalised frame.

Input points are in the Leveler frame (pixels, ground y=0, y up, head toward +x).
Every length is normalised by a calibrated segment length from the Profile, so
the features do not depend on camera distance or limb proportions.
"""

from __future__ import annotations

import numpy as np

from .geometry import Leveler, angle_deg, dist, signed_line_distance
from .schema import PoseSequence


def level_pose(seq: PoseSequence, leveler: Leveler, side: str) -> dict[str, np.ndarray]:
    """Near-side chain plus derived points, all in the levelled frame."""
    pts = {n: leveler.apply(p) for n, p in seq.side_points(side).items()}
    far = "r" if side == "l" else "l"
    near_sh = pts["shoulder"]
    far_sh = leveler.apply(seq.joint(f"{far}_shoulder"))
    pts["far_shoulder"] = far_sh
    # Neither backend has a neck point: use the shoulder midpoint (near shoulder if the far one is missing).
    with np.errstate(invalid="ignore"):
        neck = np.nanmean(np.stack([near_sh, far_sh]), axis=0)
    pts["neck"] = neck
    return pts


def body_offset(pts: dict[str, np.ndarray], torso_len: float) -> np.ndarray:
    """Signed hip offset from the shoulder-ankle line / torso length. Positive = hip above the line."""
    return signed_line_distance(pts["hip"], pts["ankle"], pts["shoulder"]) / torso_len


def compute_features(pts: dict[str, np.ndarray], prof) -> dict[str, np.ndarray]:
    L, base = prof.lengths, prof.baseline
    S, E, W = pts["shoulder"], pts["elbow"], pts["wrist"]
    H, K, A = pts["hip"], pts["knee"], pts["ankle"]
    heel, toe, ear, neck = pts["heel"], pts["toe"], pts["ear"], pts["neck"]
    f: dict[str, np.ndarray] = {}

    # Angles
    f["elbow_angle"] = angle_deg(S, E, W)
    f["knee_angle"] = angle_deg(H, K, A)
    f["hip_angle"] = angle_deg(S, H, K)
    f["neck_angle"] = angle_deg(ear, neck, H)
    # 0 = upper arm parallel to the ground, positive = shoulder above elbow.
    f["upper_arm_incline_deg"] = np.degrees(np.arctan2(S[:, 1] - E[:, 1], np.abs(S[:, 0] - E[:, 0])))
    # Angle of hip->shoulder vs the ground line: 0 lying flat, 90 upright, >90 leaning toward the knees.
    f["torso_angle_deg"] = np.degrees(np.arctan2(S[:, 1] - H[:, 1], S[:, 0] - H[:, 0]))

    # Body line (push-up): sag is negative, pike positive, relative to the participant's natural posture.
    f["body_dev"] = body_offset(pts, L["torso"]) - base["body_offset"]

    # Heights above the ground line, in body-relative units.
    f["shoulder_height"] = S[:, 1] / L["body"]
    f["hip_height"] = H[:, 1] / L["body"]
    f["knee_height"] = K[:, 1] / L["body"]
    # Knee vs the shoulder-ankle line: a knee on the ground reads as a large negative value.
    f["knee_dev"] = signed_line_distance(K, A, S) / L["torso"] - base["knee_offset"]

    # Hands
    f["wrist_dx"] = (W[:, 0] - S[:, 0]) / L["upper_arm"]
    f["wrist_ear_dist"] = dist(W, ear) / L["upper_arm"]

    # Feet anchoring (drift from the calibrated position, in foot lengths) and hip bridging.
    drift = np.stack([dist(pts[n], np.asarray(base[n])) for n in ("ankle", "heel", "toe")])
    f["foot_drift"] = np.fmax.reduce(drift, axis=0) / L["foot"]
    f["hip_rise"] = (H[:, 1] - base["hip_y"]) / L["torso"]
    return f

