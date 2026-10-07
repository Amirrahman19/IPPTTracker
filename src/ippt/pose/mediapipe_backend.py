"""MediaPipe PoseLandmarker backend (Tasks API) adapted into the internal skeleton."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ..config import models_dir
from ..schema import JOINTS, MEDIAPIPE_INDEX, PoseSequence

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")
VARIANTS = ("lite", "full", "heavy")
_SRC = np.array([MEDIAPIPE_INDEX[j] for j in JOINTS])


def ensure_model(variant: str = "heavy") -> Path:
    """Return the local model path, downloading it on first use."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    path = models_dir() / f"pose_landmarker_{variant}.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL.format(variant=variant), tmp)
        tmp.rename(path)
    return path


class MediaPipePoseEstimator:
    def __init__(self, variant: str = "heavy", min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else ensure_model(variant)
        self._opts = dict(
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "MediaPipePoseEstimator":
        p = cfg["pose"]
        return cls(p["variant"], p["min_detection_confidence"], p["min_presence_confidence"],
                   p["min_tracking_confidence"])

    def _landmarker(self, video: bool):
        from mediapipe.tasks.python import BaseOptions, vision
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mode, num_poses=1, **self._opts,
        )
        return vision.PoseLandmarker.create_from_options(opts)

    @staticmethod
    def _to_arrays(result, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        xy = np.full((len(JOINTS), 2), np.nan)
        conf = np.zeros(len(JOINTS))
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            raw = np.array([[lm[i].x * w, lm[i].y * h] for i in _SRC])
            vis = np.array([lm[i].visibility if lm[i].visibility is not None else 1.0 for i in _SRC])
            xy, conf = raw, vis
        return xy, conf

    def estimate_image(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single image -> (xy (J, 2) pixels, conf (J,)). NaN / 0 when no person is found."""
        import mediapipe as mp
        h, w = bgr.shape[:2]
        with self._landmarker(video=False) as lm:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return self._to_arrays(lm.detect(img), w, h)

    def estimate_video(self, path: str | Path, max_frames: int | None = None, progress: bool = True) -> PoseSequence:
        import mediapipe as mp
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if max_frames:
            total = min(total or max_frames, max_frames)
        bar = None
        if progress:
            from tqdm import tqdm
            bar = tqdm(total=total, desc=Path(path).name, unit="f")
        ts, xys, confs = [], [], []
        with self._landmarker(video=True) as lm:
            i = 0
            while max_frames is None or i < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                t_ms = int(round(i * 1000.0 / fps))
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                xy, conf = self._to_arrays(lm.detect_for_video(img, t_ms), w, h)
                ts.append(i / fps)
                xys.append(xy)
                confs.append(conf)
                i += 1
                if bar:
                    bar.update(1)
        cap.release()
        if bar:
            bar.close()
        if not ts:
            raise ValueError(f"no frames read from {path}")
        return PoseSequence(np.array(ts), np.stack(xys), np.stack(confs), (w, h), str(path))
