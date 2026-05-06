from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityGateConfig:
    min_signal_quality: float = 0.42
    min_snr_db: float = 0.0
    max_motion_score: float = 0.18
    max_lighting_variation: float = 0.12
    max_rppg_hr_error_bpm: float = 15.0
    min_rppg_hr_series_count: int = 1


def quality_gate_mask(
    frame: pd.DataFrame,
    config: QualityGateConfig | None = None,
    mode: str = "fyp",
) -> pd.Series:
    """Return rows that pass the FYP-style signal quality gate.

    mode="fyp" uses online-available quality signals. mode="training-reference"
    additionally uses rPPG-vs-BVP HR error when the reference column exists.
    """
    config = config or QualityGateConfig()
    mask = pd.Series(True, index=frame.index)

    if "hr_bpm" in frame.columns:
        mask &= np.isfinite(frame["hr_bpm"].astype(float))
    if "signal_quality" in frame.columns:
        mask &= frame["signal_quality"].astype(float) >= config.min_signal_quality
    if "snr_db" in frame.columns:
        mask &= frame["snr_db"].astype(float) >= config.min_snr_db
    if "motion_score" in frame.columns:
        mask &= frame["motion_score"].astype(float) <= config.max_motion_score
    if "lighting_variation" in frame.columns:
        mask &= frame["lighting_variation"].astype(float) <= config.max_lighting_variation
    if "rppg_hr_series_count" in frame.columns:
        mask &= frame["rppg_hr_series_count"].fillna(0).astype(float) >= config.min_rppg_hr_series_count

    if mode == "training-reference" and "rppg_hr_error_bpm" in frame.columns:
        err = frame["rppg_hr_error_bpm"].astype(float)
        mask &= err.isna() | (err <= config.max_rppg_hr_error_bpm)
    elif mode not in {"fyp", "training-reference"}:
        raise ValueError(f"Unknown quality gate mode: {mode}")

    return mask


def quality_gate_summary(
    frame: pd.DataFrame,
    mask: pd.Series,
    config: QualityGateConfig | None = None,
    mode: str = "fyp",
) -> dict:
    config = config or QualityGateConfig()
    total = int(len(frame))
    kept = int(mask.sum())
    rejected = total - kept
    reasons: dict[str, int] = {}

    def count(name: str, bad_mask: pd.Series) -> None:
        reasons[name] = int(bad_mask.fillna(False).sum())

    if "hr_bpm" in frame.columns:
        count("invalid_hr", ~np.isfinite(frame["hr_bpm"].astype(float)))
    if "signal_quality" in frame.columns:
        count("signal_quality_below_threshold", frame["signal_quality"].astype(float) < config.min_signal_quality)
    if "snr_db" in frame.columns:
        count("snr_below_threshold", frame["snr_db"].astype(float) < config.min_snr_db)
    if "motion_score" in frame.columns:
        count("motion_above_threshold", frame["motion_score"].astype(float) > config.max_motion_score)
    if "lighting_variation" in frame.columns:
        count(
            "lighting_variation_above_threshold",
            frame["lighting_variation"].astype(float) > config.max_lighting_variation,
        )
    if "rppg_hr_series_count" in frame.columns:
        count(
            "too_few_rppg_subwindows",
            frame["rppg_hr_series_count"].fillna(0).astype(float) < config.min_rppg_hr_series_count,
        )
    if mode == "training-reference" and "rppg_hr_error_bpm" in frame.columns:
        err = frame["rppg_hr_error_bpm"].astype(float)
        count("rppg_hr_error_above_threshold", err.notna() & (err > config.max_rppg_hr_error_bpm))

    return {
        "mode": mode,
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "kept_ratio": float(kept / total) if total else 0.0,
        "thresholds": config.__dict__,
        "rejection_reasons": reasons,
    }
