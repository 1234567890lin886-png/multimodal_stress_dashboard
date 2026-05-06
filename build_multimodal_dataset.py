from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EMOTION_COLUMNS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
    "emotion_confidence",
    "dominant_emotion",
    "emotion_stress",
    "frames_analyzed",
]


def _add_window_id(frame: pd.DataFrame, step_sec: float) -> pd.DataFrame:
    result = frame.copy()
    result["_window_id"] = np.rint(result["window_start_sec"].astype(float) / float(step_sec)).astype(int)
    return result


def _merge_exact(physio: pd.DataFrame, emotion: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "task", "window_start_sec", "window_end_sec"]
    keep = keys + [col for col in EMOTION_COLUMNS if col in emotion.columns]
    return physio.merge(emotion[keep], on=keys, how="left", suffixes=("", "_fer"))


def _merge_window_index(
    physio: pd.DataFrame,
    emotion: pd.DataFrame,
    step_sec: float,
    max_start_delta_sec: float,
) -> pd.DataFrame:
    physio_aligned = _add_window_id(physio, step_sec)
    emotion_aligned = _add_window_id(emotion, step_sec)
    emotion_aligned = emotion_aligned.rename(
        columns={
            "window_start_sec": "fer_window_start_sec",
            "window_end_sec": "fer_window_end_sec",
        }
    )
    keys = ["subject", "task", "_window_id"]
    keep = keys + [
        "fer_window_start_sec",
        "fer_window_end_sec",
    ] + [col for col in EMOTION_COLUMNS if col in emotion_aligned.columns]
    emotion_aligned = emotion_aligned[keep].sort_values(
        ["subject", "task", "_window_id", "frames_analyzed" if "frames_analyzed" in emotion_aligned.columns else keys[-1]]
    )
    emotion_aligned = emotion_aligned.drop_duplicates(keys, keep="last")
    merged = physio_aligned.merge(emotion_aligned, on=keys, how="left", suffixes=("", "_fer"))

    if "fer_window_start_sec" in merged.columns:
        delta = (merged["window_start_sec"].astype(float) - merged["fer_window_start_sec"].astype(float)).abs()
        too_far = delta > float(max_start_delta_sec)
        if too_far.any():
            fer_columns = [col for col in EMOTION_COLUMNS if col in merged.columns]
            fer_columns += ["fer_window_start_sec", "fer_window_end_sec"]
            for col in fer_columns:
                merged.loc[too_far, col] = np.nan
    return merged.drop(columns=["_window_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge physiological and FER window features.")
    parser.add_argument("--physio", type=str, default="sample_data/ubfc_phys_s51_s56_features.csv")
    parser.add_argument("--emotion", type=str, default="sample_data/ubfc_phys_emotion_features.csv")
    parser.add_argument("--out", type=str, default="sample_data/ubfc_phys_multimodal_features.csv")
    parser.add_argument(
        "--alignment",
        choices=["window-index", "exact"],
        default="window-index",
        help="Use window-index alignment to tolerate small timestamp drift between rPPG and FER windows.",
    )
    parser.add_argument("--step-sec", type=float, default=30.0)
    parser.add_argument("--max-start-delta-sec", type=float, default=1.0)
    args = parser.parse_args()

    physio = pd.read_csv(args.physio)
    emotion = pd.read_csv(args.emotion)
    if args.alignment == "exact":
        merged = _merge_exact(physio, emotion)
    else:
        merged = _merge_window_index(
            physio,
            emotion,
            step_sec=args.step_sec,
            max_start_delta_sec=args.max_start_delta_sec,
        )

    for col in ["emotion_stress", "emotion_confidence"]:
        fer_col = f"{col}_fer"
        if fer_col in merged.columns:
            merged[col] = merged[fer_col].fillna(merged[col])
            merged = merged.drop(columns=[fer_col])
    for col in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    if "dominant_emotion" in merged.columns:
        merged["dominant_emotion"] = merged["dominant_emotion"].fillna("unknown")
    if "frames_analyzed" in merged.columns:
        merged["frames_analyzed"] = merged["frames_analyzed"].fillna(0).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Merged {len(merged)} rows.")
    print(f"Rows with FER frames: {(merged.get('frames_analyzed', pd.Series([0]*len(merged))) > 0).sum()}")
    if "fer_window_start_sec" in merged.columns:
        matched = merged[merged.get("frames_analyzed", pd.Series([0] * len(merged))) > 0].copy()
        if not matched.empty:
            delta = (matched["window_start_sec"].astype(float) - matched["fer_window_start_sec"].astype(float)).abs()
            print(f"Mean FER/rPPG start delta: {delta.mean():.3f}s")
            print(f"Max FER/rPPG start delta: {delta.max():.3f}s")
    print(f"Saved multimodal dataset to {out_path}")


if __name__ == "__main__":
    main()
