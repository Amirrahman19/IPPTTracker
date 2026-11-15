"""Build the landing-page demo from one existing recording.

    python scripts/make_demo.py workout_vids_pushup__push-up_9.mp4

Reads data/interim/web/<clip> and its cached pose track, then writes
src/ippt/dashboard/static/demo/demo.mp4 and demo.json (smoothed skeleton per
frame, angle signal, rep timestamps). The landing page plays these in a loop
when no live camera is connected, so visitors see pose estimation + counting.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ippt.config import load_pipeline_config  # noqa: E402
from ippt.dashboard import recordings  # noqa: E402
from ippt.schema import PoseSequence  # noqa: E402
from ippt.signal import preprocess  # noqa: E402

OUT = ROOT / "src" / "ippt" / "dashboard" / "static" / "demo"


def main(name: str) -> None:
    src = recordings.media_path(name)
    res = recordings.analyse(name)
    if src is None or not res or not res["available"]:
        sys.exit(f"{name}: clip or cached pose track not found")
    seq = preprocess(PoseSequence.load(recordings._pose_path(name)), load_pipeline_config())
    w, h = seq.size
    joints = []
    for frame in seq.xy:
        joints.append([None if np.isnan(p[0]) else [round(float(p[0]) / w, 4), round(float(p[1]) / h, 4)]
                       for p in frame])
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, OUT / "demo.mp4")
    data = {
        "exercise": res["exercise"], "side": res["side"], "signal_label": res["signal_label"],
        "t": [round(float(t), 3) for t in seq.t], "joints": joints,
        "signal": res["signal"], "rep_times": res["rep_times"],
    }
    (OUT / "demo.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"{len(seq)} frames, {res['rep_count']} reps -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
