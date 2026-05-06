from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureSummary


STRESSY_EMOTIONS = {"angry", "fear", "sad", "disgust"}
CALM_EMOTIONS = {"happy", "neutral", "surprise"}


@dataclass
class EmotionContext:
    dominant_emotion: str = "neutral"
    confidence: float = 0.0
    stress_semantic: float = 0.35


@dataclass
class FusionDecision:
    label: str
    confidence: float
    stress_score: float
    rationale: str


def emotion_to_stress(emotion: str, confidence: float) -> float:
    name = (emotion or "neutral").lower()
    base = 0.35
    if name in STRESSY_EMOTIONS:
        base = 0.75
    elif name == "surprise":
        base = 0.55
    elif name == "happy":
        base = 0.18
    return float(0.35 + (base - 0.35) * np.clip(confidence, 0.0, 1.0))


def physiology_score(features: FeatureSummary) -> tuple[float, str]:
    if features.hr_bpm is None:
        return 0.5, "No reliable heart-rate estimate yet"
    if features.baseline_hr is None:
        return 0.5, "Baseline is still calibrating"

    score = 0.42
    reasons: list[str] = []
    if features.hr_delta_bpm >= 10.0:
        score += 0.28
        reasons.append(f"HR is +{features.hr_delta_bpm:.1f} bpm above baseline")
    elif features.hr_delta_bpm >= 6.0:
        score += 0.16
        reasons.append(f"HR is moderately elevated (+{features.hr_delta_bpm:.1f} bpm)")
    elif features.hr_delta_bpm <= -4.0:
        score -= 0.10
        reasons.append(f"HR is below baseline ({features.hr_delta_bpm:.1f} bpm)")

    if features.hrv_proxy < 1.8 and len(reasons) > 0:
        score += 0.10
        reasons.append("HRV proxy is low")
    elif features.hrv_proxy > 4.5:
        score -= 0.08
        reasons.append("HRV proxy is relatively high")

    if features.signal_quality < 0.45:
        score = 0.55 * score + 0.45 * 0.5
        reasons.append("signal quality is limited")

    return float(np.clip(score, 0.0, 1.0)), "; ".join(reasons) or "Physiology is near baseline"


def fuse(features: FeatureSummary, emotion: EmotionContext) -> FusionDecision:
    if features.signal_quality < 0.28:
        return FusionDecision(
            label="Poor Signal Quality",
            confidence=0.22,
            stress_score=0.5,
            rationale="Signal quality is too low for a stable decision; improve lighting and reduce motion.",
        )

    physio, physio_reason = physiology_score(features)
    emotion_score = emotion.stress_semantic
    physio_weight = 0.82 if features.signal_quality >= 0.55 else 0.68
    final_score = physio_weight * physio + (1.0 - physio_weight) * emotion_score
    disagreement = abs(physio - emotion_score)

    if disagreement >= 0.42:
        label = "Uncertain"
        confidence = 0.35
        reason = (
            f"Physiology and facial context disagree "
            f"(physio={physio:.2f}, emotion={emotion_score:.2f}). {physio_reason}."
        )
    elif final_score >= 0.70:
        label = "Elevated Physiological Stress"
        confidence = min(0.92, 0.52 + abs(final_score - 0.5))
        reason = f"Fused stress score is high ({final_score:.2f}). {physio_reason}."
    elif final_score >= 0.56:
        label = "Mild Stress"
        confidence = min(0.78, 0.42 + abs(final_score - 0.5))
        reason = f"Fused stress score is moderately elevated ({final_score:.2f}). {physio_reason}."
    elif final_score <= 0.34:
        label = "Relaxed"
        confidence = min(0.88, 0.52 + abs(final_score - 0.5))
        reason = f"Fused stress score is low ({final_score:.2f}). {physio_reason}."
    elif final_score <= 0.46:
        label = "Mostly Calm"
        confidence = 0.55
        reason = f"Fused stress score leans calm ({final_score:.2f}). {physio_reason}."
    else:
        label = "Uncertain"
        confidence = 0.38
        reason = f"Fused stress score is near the decision boundary ({final_score:.2f}). {physio_reason}."

    return FusionDecision(label=label, confidence=float(confidence), stress_score=float(final_score), rationale=reason)

