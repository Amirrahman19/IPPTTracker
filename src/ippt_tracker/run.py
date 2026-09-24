"""CLI entry point for a single fixed-camera assessment session."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from .models import Exercise
from .overlay import draw_overlay
from .pose import YOLO26PoseEstimator
from .session import ExerciseSession, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track push-ups or sit-ups with YOLO26 Pose.")
    parser.add_argument("exercise", choices=[exercise.value for exercise in Exercise])
    parser.add_argument("--source", default="0", help="Webcam index (default: 0), video path, or stream URL.")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="YOLO26 pose checkpoint.")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"))
    parser.add_argument("--events", type=Path, default=None, help="YAML audit output; no video is saved by default.")
    parser.add_argument("--save-video", type=Path, default=None, help="Optional annotated MP4 output.")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window.")
    return parser.parse_args()


def _capture_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _default_event_path(exercise: Exercise) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{exercise.value}_{timestamp}_events.yaml"


def main() -> None:
    args = parse_args()
    exercise = Exercise(args.exercise)
    session = ExerciseSession(exercise, load_config(args.config))
    estimator = YOLO26PoseEstimator(args.model, device=args.device)
    capture = cv2.VideoCapture(_capture_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    video_writer: cv2.VideoWriter | None = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pose = estimator.infer(frame, frame_index, frame_index / fps)
            session.update(pose)
            annotated = draw_overlay(frame, session, pose)

            if args.save_video:
                if video_writer is None:
                    args.save_video.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(args.save_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                video_writer.write(annotated)

            if not args.no_display:
                cv2.imshow("IPPT Tracker — press q to finish", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    event_path = args.events or _default_event_path(exercise)
    session.write_events(event_path)
    print(f"Valid reps: {session.state.valid_reps}/{session.state.attempted_reps}")
    print(f"Audit events: {event_path}")


if __name__ == "__main__":
    main()
