from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.emotion import EmotionAnalyzer
from src.features import summarize_features
from src.fusion import EmotionContext, fuse
from src.ml_model import predict_with_model
from src.rppg import RPPGProcessor
from src.video import FaceMeshROIExtractor, _masked_mean_rgb


def _resolve_source(source: str) -> int | str:
    try:
        return int(source)
    except ValueError:
        return source


def _draw_panel(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    x, y = 18, 26
    width = 560
    height = 32 + 28 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (20, 24, 28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
        y += 28


def run_live(
    source: str,
    model_path: str | None,
    rppg_window_sec: float,
    max_buffer_sec: float,
    enable_fer: bool,
    fer_interval_frames: int,
) -> None:
    cap = cv2.VideoCapture(_resolve_source(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = fps if fps >= 5.0 else 30.0
    max_samples = int(max_buffer_sec * fps)
    rgb_series: deque[np.ndarray] = deque(maxlen=max_samples)
    hr_history: list[float] = []
    rppg = RPPGProcessor(fps=fps, window_sec=rppg_window_sec)
    roi_extractor = FaceMeshROIExtractor()
    emotion_analyzer = EmotionAnalyzer()
    emotion = EmotionContext()
    prev_gray = None
    motion_score = 0.0
    lighting_variation = 0.0
    frame_count = 0
    last_tick = time.perf_counter()
    live_fps = 0.0

    print("Live webcam vitals started. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_count += 1

            extraction = roi_extractor.extract(frame)
            mean_rgb = _masked_mean_rgb(extraction.roi_bgr, extraction.mask)
            if mean_rgb is not None:
                rgb_series.append(mean_rgb)
                gray = cv2.cvtColor(extraction.roi_bgr, cv2.COLOR_BGR2GRAY)
                if extraction.mask.shape == gray.shape and np.count_nonzero(extraction.mask) > 0:
                    values = gray[extraction.mask > 0]
                    lighting_variation = float(np.std(values) / 255.0)
                else:
                    lighting_variation = float(np.std(gray) / 255.0)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    diff = cv2.absdiff(gray, prev_gray)
                    motion_score = float(np.mean(diff) / 255.0)
                prev_gray = gray

            estimate = rppg.estimate(np.asarray(rgb_series, dtype=float)) if len(rgb_series) else None
            if estimate is not None and estimate.hr_bpm is not None:
                hr_history.append(float(estimate.hr_bpm))
                snr_db = float(estimate.snr_db)
                hr_text = f"{estimate.hr_bpm:.1f} BPM"
            else:
                snr_db = float("-inf")
                hr_text = "calibrating"

            features = summarize_features(
                hr_history[-1] if hr_history else None,
                hr_history,
                snr_db,
                motion_score,
                lighting_variation,
            )

            if enable_fer and frame_count % max(1, fer_interval_frames) == 0:
                emotion = emotion_analyzer.analyze(frame)

            rule_decision = fuse(features, emotion)
            model_prediction = predict_with_model(features, emotion, model_path=model_path)
            if model_prediction is not None and rule_decision.label != "Poor Signal Quality":
                label = model_prediction.label
                confidence = model_prediction.confidence
                model_name = model_prediction.model_name
            else:
                label = rule_decision.label
                confidence = rule_decision.confidence
                model_name = "rule baseline"

            now = time.perf_counter()
            elapsed = now - last_tick
            if elapsed > 0:
                live_fps = 0.9 * live_fps + 0.1 * (1.0 / elapsed) if live_fps else 1.0 / elapsed
            last_tick = now

            color = (80, 220, 120)
            if "Stress" in label:
                color = (80, 180, 255)
            if "Poor" in label or "Uncertain" in label or "calibrating" in hr_text:
                color = (80, 200, 255)

            lines = [
                ("OpenCV rPPG Live Vitals", (230, 240, 245)),
                (f"HR: {hr_text} | Signal quality: {features.signal_quality:.2f} | SNR: {snr_db:.1f} dB", color),
                (f"Stress: {label} | confidence: {confidence:.2f} | model: {model_name}", color),
                (f"Emotion: {emotion.dominant_emotion} ({emotion.confidence:.2f}) | arousal: {features.arousal_state}", (210, 220, 230)),
                (f"FPS: {live_fps:.1f} | motion: {motion_score:.3f} | lighting: {lighting_variation:.3f}", (210, 220, 230)),
            ]
            _draw_panel(frame, lines)
            cv2.imshow("OpenCV rPPG Live Vitals", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        roi_extractor.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV webcam live rPPG vital signs dashboard.")
    parser.add_argument("--source", default="0", help="Camera index or video file path.")
    parser.add_argument("--model", default="models/stress_xgb_video_rppg_online_qg.joblib")
    parser.add_argument("--rppg-window-sec", type=float, default=12.0)
    parser.add_argument("--max-buffer-sec", type=float, default=90.0)
    parser.add_argument("--enable-fer", action="store_true", help="Enable DeepFace emotion analysis during live mode.")
    parser.add_argument("--fer-interval-frames", type=int, default=90)
    args = parser.parse_args()

    model_path = args.model if Path(args.model).exists() else None
    run_live(
        source=args.source,
        model_path=model_path,
        rppg_window_sec=args.rppg_window_sec,
        max_buffer_sec=args.max_buffer_sec,
        enable_fer=args.enable_fer,
        fer_interval_frames=args.fer_interval_frames,
    )


if __name__ == "__main__":
    main()
