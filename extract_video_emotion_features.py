from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
STRESS_WEIGHTS = {
    "angry": 0.80,
    "disgust": 0.68,
    "fear": 0.85,
    "happy": 0.18,
    "sad": 0.72,
    "surprise": 0.55,
    "neutral": 0.32,
}


def _center_crop(frame):
    h, w = frame.shape[:2]
    size = int(min(h, w) * 0.72)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    return frame[y1:y2, x1:x2]


def _empty_scores() -> dict[str, float]:
    return {name: 0.0 for name in EMOTIONS}


def _analyze_frame(deepface, frame) -> dict[str, float] | None:
    try:
        result = deepface.analyze(
            _center_crop(frame),
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        scores = result.get("emotion", {}) or {}
        out = {}
        for name in EMOTIONS:
            value = float(scores.get(name, 0.0))
            out[name] = value / 100.0 if value > 1.0 else value
        total = sum(out.values())
        if total > 0:
            out = {key: value / total for key, value in out.items()}
        return out
    except Exception:
        return None


def _aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        result = _empty_scores()
        result["emotion_confidence"] = 0.0
        result["dominant_emotion"] = "unknown"
        result["emotion_stress"] = 0.35
        result["frames_analyzed"] = 0
        return result
    arr = {name: float(np.mean([score.get(name, 0.0) for score in scores])) for name in EMOTIONS}
    dominant = max(arr, key=arr.get)
    confidence = float(arr[dominant])
    stress = float(sum(arr[name] * STRESS_WEIGHTS[name] for name in EMOTIONS))
    arr["emotion_confidence"] = confidence
    arr["dominant_emotion"] = dominant
    arr["emotion_stress"] = stress
    arr["frames_analyzed"] = len(scores)
    return arr


def extract_video_windows(
    video_path: Path,
    subject: str,
    task: str,
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    frames_per_window: int = 5,
) -> list[dict]:
    from deepface import DeepFace

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0
    rows = []
    starts = np.arange(0.0, max(0.0, duration - window_sec + 1.0), step_sec)
    for start_sec in starts:
        sample_times = np.linspace(start_sec + 4.0, start_sec + window_sec - 4.0, frames_per_window)
        frame_scores: list[dict[str, float]] = []
        for sample_time in sample_times:
            frame_idx = int(sample_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            scores = _analyze_frame(DeepFace, frame)
            if scores is not None:
                frame_scores.append(scores)
        row = {
            "subject": subject,
            "task": task,
            "window_start_sec": round(float(start_sec), 3),
            "window_end_sec": round(float(start_sec + window_sec), 3),
            "video_path": str(video_path),
        }
        row.update(_aggregate(frame_scores))
        rows.append(row)
    cap.release()
    return rows


def extract_dataset(
    dataset_roots: list[Path],
    out_path: Path,
    window_sec: float,
    step_sec: float,
    frames_per_window: int,
    limit_subjects: int | None = None,
) -> pd.DataFrame:
    all_rows: list[dict] = []
    subject_dirs: list[Path] = []
    for dataset_root in dataset_roots:
        subject_dirs.extend(sorted(path for path in dataset_root.iterdir() if path.is_dir()))
    if limit_subjects is not None:
        subject_dirs = subject_dirs[:limit_subjects]
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        for task in ["T1", "T2", "T3"]:
            video_path = subject_dir / f"vid_{subject}_{task}.avi"
            if not video_path.exists():
                continue
            cache_path = out_path.parent / (
                f".emotion_cache_{subject}_{task}_w{int(window_sec)}_s{int(step_sec)}_f{int(frames_per_window)}.json"
            )
            if cache_path.exists():
                all_rows.extend(json.loads(cache_path.read_text(encoding="utf-8")))
                print(f"Loaded cache for {subject} {task}")
                continue
            print(f"Extracting FER: {subject} {task}")
            rows = extract_video_windows(video_path, subject, task, window_sec, step_sec, frames_per_window)
            cache_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract window-level FER semantics from UBFC-Phys videos.")
    parser.add_argument("--dataset-root", type=str, required=True, nargs="+")
    parser.add_argument("--out", type=str, default="sample_data/ubfc_phys_emotion_features.csv")
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=30.0)
    parser.add_argument("--frames-per-window", type=int, default=5)
    parser.add_argument("--limit-subjects", type=int)
    args = parser.parse_args()

    df = extract_dataset(
        [Path(root) for root in args.dataset_root],
        Path(args.out),
        args.window_sec,
        args.step_sec,
        args.frames_per_window,
        args.limit_subjects,
    )
    print(f"Extracted {len(df)} emotion windows.")
    if not df.empty:
        print(df[["subject", "task", "window_start_sec", "dominant_emotion", "emotion_stress", "frames_analyzed"]].head())
        print(f"Saved features to {args.out}")


if __name__ == "__main__":
    main()
