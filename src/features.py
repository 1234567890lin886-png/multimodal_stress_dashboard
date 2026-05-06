from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FeatureSummary:
    hr_bpm: Optional[float]
    baseline_hr: Optional[float]
    hr_delta_bpm: float
    hrv_proxy: float
    signal_quality: float
    motion_score: float
    lighting_variation: float
    snr_db: float
    arousal_state: str


def compute_hrv_proxy(hr_series: np.ndarray) -> float:
    clean = np.asarray([x for x in hr_series if np.isfinite(x)], dtype=float)
    if len(clean) < 5:
        return 0.0
    diffs = np.diff(clean)
    rmssd_like = np.sqrt(np.mean(np.square(diffs))) if len(diffs) else 0.0
    return float(0.45 * np.std(clean) + 0.55 * rmssd_like)


def summarize_features(
    hr_bpm: Optional[float],
    hr_history: list[float],
    snr_db: float,
    motion_score: float,
    lighting_variation: float,
    baseline_hr: Optional[float] = None,
) -> FeatureSummary:
    clean_history = np.asarray([x for x in hr_history if np.isfinite(x)], dtype=float)
    if baseline_hr is None and len(clean_history) >= 6:
        baseline_hr = float(np.median(clean_history[: max(6, len(clean_history) // 3)]))

    hrv_proxy = compute_hrv_proxy(clean_history[-30:])
    hr_delta = float(hr_bpm - baseline_hr) if hr_bpm is not None and baseline_hr is not None else 0.0
    snr_norm = float(np.clip((snr_db + 4.0) / 14.0, 0.0, 1.0))
    motion_norm = float(np.clip(1.0 - motion_score, 0.0, 1.0))
    light_norm = float(np.clip(1.0 - lighting_variation, 0.0, 1.0))
    signal_quality = float(0.55 * snr_norm + 0.25 * motion_norm + 0.20 * light_norm)

    if hr_bpm is None or baseline_hr is None:
        arousal = "Calibrating"
    elif hr_delta >= 10.0 and hrv_proxy < 2.5:
        arousal = "High Arousal"
    elif hr_delta <= 3.0 and hrv_proxy >= 3.5:
        arousal = "Low Arousal"
    else:
        arousal = "Moderate Arousal"

    return FeatureSummary(
        hr_bpm=hr_bpm,
        baseline_hr=baseline_hr,
        hr_delta_bpm=hr_delta,
        hrv_proxy=float(hrv_proxy),
        signal_quality=signal_quality,
        motion_score=float(motion_score),
        lighting_variation=float(lighting_variation),
        snr_db=float(snr_db),
        arousal_state=arousal,
    )

