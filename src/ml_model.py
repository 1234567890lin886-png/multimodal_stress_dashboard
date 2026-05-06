from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import FeatureSummary
from .fusion import EmotionContext


FEATURE_NAMES = [
    "hr_bpm",
    "hr_delta_bpm",
    "hrv_proxy",
    "signal_quality",
    "motion_score",
    "lighting_variation",
    "snr_db",
    "emotion_stress",
    "emotion_confidence",
]

MULTIMODAL_FEATURE_NAMES = FEATURE_NAMES + [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

LABEL_STRESS_SCORES = {
    "Relaxed": 0.18,
    "Mostly Calm": 0.34,
    "Stress": 0.72,
    "Mild Stress": 0.62,
    "Elevated Physiological Stress": 0.84,
    "Uncertain": 0.50,
}


@dataclass
class ModelPrediction:
    label: str
    confidence: float
    stress_score: float
    model_name: str
    training_source: str
    physio_weight: Optional[float] = None
    emotion_weight: Optional[float] = None


def default_model_path() -> Path:
    gated_path = Path(__file__).resolve().parents[1] / "models" / "gated_fusion_mlp.joblib"
    if gated_path.exists():
        return gated_path
    xgb_path = Path(__file__).resolve().parents[1] / "models" / "stress_xgb.joblib"
    if xgb_path.exists():
        return xgb_path
    return Path(__file__).resolve().parents[1] / "models" / "stress_rf.joblib"


def feature_vector(features: FeatureSummary, emotion: EmotionContext) -> pd.DataFrame:
    hr = 0.0 if features.hr_bpm is None else float(features.hr_bpm)
    snr = -10.0 if not np.isfinite(features.snr_db) else float(features.snr_db)
    values = [
        hr,
        float(features.hr_delta_bpm),
        float(features.hrv_proxy),
        float(features.signal_quality),
        float(features.motion_score),
        float(features.lighting_variation),
        snr,
        float(emotion.stress_semantic),
        float(emotion.confidence),
    ]
    return pd.DataFrame([values], columns=FEATURE_NAMES)


def semantic_feature_frame(features: FeatureSummary, emotion: EmotionContext, feature_names: list[str]) -> pd.DataFrame:
    base = feature_vector(features, emotion)
    emotion_probs = {
        "angry": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "happy": 0.0,
        "sad": 0.0,
        "surprise": 0.0,
        "neutral": 0.0,
    }
    dominant = (emotion.dominant_emotion or "neutral").lower()
    if dominant in emotion_probs:
        emotion_probs[dominant] = float(np.clip(emotion.confidence, 0.0, 1.0))
    if sum(emotion_probs.values()) == 0.0:
        emotion_probs["neutral"] = 1.0
    for name, value in emotion_probs.items():
        base[name] = value
    for name in feature_names:
        if name not in base.columns:
            base[name] = 0.0
    return base[feature_names]


def load_model(model_path: Optional[str | Path] = None) -> Optional[dict]:
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.exists():
        return None
    try:
        import joblib

        bundle = joblib.load(path)
        if isinstance(bundle, dict) and ("model" in bundle or bundle.get("model_type") == "gated_fusion_mlp"):
            return bundle
    except Exception:
        return None
    return None


def predict_with_model(
    features: FeatureSummary,
    emotion: EmotionContext,
    model_path: Optional[str | Path] = None,
) -> Optional[ModelPrediction]:
    bundle = load_model(model_path)
    if bundle is None:
        return None

    if bundle.get("model_type") == "gated_fusion_mlp":
        return _predict_with_gated_model(bundle, features, emotion)

    model = bundle["model"]
    feature_names = list(bundle.get("feature_names", FEATURE_NAMES))
    x = semantic_feature_frame(features, emotion, feature_names)
    raw_label = model.predict(x)[0]
    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        label = str(label_encoder.inverse_transform(np.asarray([int(raw_label)]))[0])
    else:
        label = str(raw_label)
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        confidence = float(np.max(probs))
    stress_score = LABEL_STRESS_SCORES.get(label, 0.50)
    return ModelPrediction(
        label=label,
        confidence=confidence,
        stress_score=float(stress_score),
        model_name=str(bundle.get("model_name", "RandomForestClassifier")),
        training_source=str(bundle.get("training_source", "unknown")),
    )


def _predict_with_gated_model(
    bundle: dict,
    features: FeatureSummary,
    emotion: EmotionContext,
) -> Optional[ModelPrediction]:
    try:
        import tensorflow as tf

        from .gated_fusion_model import GatedFusionConfig, build_gated_fusion_model

        physio_features = list(bundle["physio_features"])
        emotion_features = list(bundle["emotion_features"])
        frame = semantic_feature_frame(features, emotion, physio_features + emotion_features)
        x_physio = bundle["physio_scaler"].transform(frame[physio_features])
        x_emotion = bundle["emotion_scaler"].transform(frame[emotion_features])
        config = GatedFusionConfig(**bundle["config"])
        model, gate_model = build_gated_fusion_model(config)
        weights_path = Path(bundle["weights_path"])
        if not weights_path.is_absolute():
            weights_path = Path(__file__).resolve().parents[1] / weights_path
        model.load_weights(weights_path)
        prob = float(model.predict([x_physio, x_emotion], verbose=0).reshape(-1)[0])
        gate_vector = gate_model.predict([x_physio, x_emotion], verbose=0)
        physio_weight = float(np.mean(gate_vector))
        emotion_weight = float(1.0 - physio_weight)
        label = "Stress" if prob >= 0.5 else "Relaxed"
        confidence = prob if label == "Stress" else 1.0 - prob
        stress_score = prob
        return ModelPrediction(
            label=label,
            confidence=float(confidence),
            stress_score=float(stress_score),
            model_name=str(bundle.get("model_name", "GatedFusionMLP")),
            training_source=str(bundle.get("training_source", "unknown")),
            physio_weight=physio_weight,
            emotion_weight=emotion_weight,
        )
    except Exception:
        return None
