from __future__ import annotations

from typing import Optional

import numpy as np

from .fusion import EmotionContext, emotion_to_stress


class EmotionAnalyzer:
    """Optional DeepFace wrapper with a neutral fallback."""

    def __init__(self) -> None:
        self.available = False
        self._deepface = None

    def _load_backend(self) -> None:
        if self._deepface is not None:
            return
        try:
            from deepface import DeepFace  # type: ignore

            self._deepface = DeepFace
            self.available = True
        except Exception:
            self.available = False

    def analyze(self, frame_bgr: Optional[np.ndarray]) -> EmotionContext:
        if frame_bgr is None:
            return EmotionContext("neutral", 0.0, emotion_to_stress("neutral", 0.0))
        self._load_backend()
        if not self.available:
            return EmotionContext("neutral", 0.0, emotion_to_stress("neutral", 0.0))
        try:
            result = self._deepface.analyze(
                frame_bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotion = str(result.get("dominant_emotion", "neutral"))
            scores = result.get("emotion", {}) or {}
            raw_conf = float(scores.get(emotion, 0.0))
            confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
            return EmotionContext(emotion, confidence, emotion_to_stress(emotion, confidence))
        except Exception:
            return EmotionContext("neutral", 0.0, emotion_to_stress("neutral", 0.0))
