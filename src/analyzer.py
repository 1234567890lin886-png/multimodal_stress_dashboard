from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .emotion import EmotionAnalyzer
from .features import FeatureSummary, summarize_features
from .fusion import EmotionContext, FusionDecision, fuse
from .ml_model import ModelPrediction, predict_with_model
from .rppg import RPPGProcessor
from .synthetic import make_synthetic_session
from .video import VideoSignals, extract_video_signals


@dataclass
class AnalysisReport:
    source: str
    duration_sec: float
    fps: float
    features: FeatureSummary
    emotion: EmotionContext
    decision: FusionDecision
    rule_decision: FusionDecision
    model_prediction: Optional[ModelPrediction]
    timeline: pd.DataFrame

    def to_dict(self) -> dict:
        result = asdict(self)
        result["timeline"] = self.timeline.to_dict(orient="records")
        return result


def _rolling_estimates(rgb_series: np.ndarray, fps: float) -> tuple[list[float], list[float], list[float]]:
    processor = RPPGProcessor(fps=fps)
    hrs: list[float] = []
    snrs: list[float] = []
    stress_times: list[float] = []
    step = max(1, int(fps))
    for end in range(max(int(fps * 4), step), len(rgb_series) + 1, step):
        estimate = processor.estimate(rgb_series[:end])
        if estimate.hr_bpm is not None:
            hrs.append(float(estimate.hr_bpm))
            snrs.append(float(estimate.snr_db))
            stress_times.append(end / fps)
    return hrs, snrs, stress_times


def analyze_signals(
    source_name: str,
    fps: float,
    timestamps: np.ndarray,
    rgb_series: np.ndarray,
    motion_series: np.ndarray,
    lighting_series: np.ndarray,
    sample_frame_bgr: Optional[np.ndarray] = None,
    model_path: Optional[str | Path] = None,
) -> AnalysisReport:
    hrs, snrs, hr_times = _rolling_estimates(rgb_series, fps)
    latest_hr = hrs[-1] if hrs else None
    latest_snr = snrs[-1] if snrs else float("-inf")
    motion_score = float(np.median(motion_series[-min(len(motion_series), int(fps * 8)) :]))
    lighting_variation = float(np.std(lighting_series[-min(len(lighting_series), int(fps * 8)) :]))
    features = summarize_features(latest_hr, hrs, latest_snr, motion_score, lighting_variation)
    emotion = EmotionAnalyzer().analyze(sample_frame_bgr)
    rule_decision = fuse(features, emotion)
    model_prediction = predict_with_model(features, emotion, model_path=model_path)
    decision = rule_decision
    if model_prediction is not None and rule_decision.label != "Poor Signal Quality":
        gate_note = ""
        if model_prediction.physio_weight is not None and model_prediction.emotion_weight is not None:
            gate_note = (
                f" Adaptive fusion weights: physio={model_prediction.physio_weight:.2f}, "
                f"emotion={model_prediction.emotion_weight:.2f}."
            )
        decision = FusionDecision(
            label=model_prediction.label,
            confidence=model_prediction.confidence,
            stress_score=model_prediction.stress_score,
            rationale=(
                f"ML classifier ({model_prediction.model_name}) predicted {model_prediction.label} "
                f"with confidence {model_prediction.confidence:.2f}. "
                f"{gate_note} "
                f"Rule baseline: {rule_decision.label}. {rule_decision.rationale}"
            ),
        )
    timeline = pd.DataFrame({"time_sec": hr_times, "hr_bpm": hrs, "snr_db": snrs})
    return AnalysisReport(
        source=source_name,
        duration_sec=float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        fps=float(fps),
        features=features,
        emotion=emotion,
        decision=decision,
        rule_decision=rule_decision,
        model_prediction=model_prediction,
        timeline=timeline,
    )


def analyze_synthetic(model_path: Optional[str | Path] = None) -> AnalysisReport:
    session = make_synthetic_session()
    return analyze_signals(
        "synthetic-demo",
        session.fps,
        session.timestamps,
        session.rgb_series,
        session.motion_series,
        session.lighting_series,
        None,
        model_path=model_path,
    )


def analyze_video(
    video_path: str | Path,
    max_frames: int = 900,
    stride: int = 1,
    model_path: Optional[str | Path] = None,
) -> AnalysisReport:
    signals: VideoSignals = extract_video_signals(video_path, max_frames=max_frames, stride=stride)
    return analyze_signals(
        str(video_path),
        signals.fps,
        signals.timestamps,
        signals.rgb_series,
        signals.motion_series,
        signals.lighting_series,
        signals.sample_frame_bgr,
        model_path=model_path,
    )
