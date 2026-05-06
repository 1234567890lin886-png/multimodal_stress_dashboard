from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from import_ubfc_phys import (
    BVP_FS,
    TASK_LABELS,
    _fft_hr_and_snr,
    _hrv_proxy_from_peaks,
    _read_self_report,
    _read_series,
    _read_subject_info,
)
from src.features import compute_hrv_proxy
from src.quality_gate import QualityGateConfig, quality_gate_mask, quality_gate_summary
from src.rppg import RPPGProcessor
from src.video import extract_video_signals


def _task_label(task: str, label_policy: str, anxiety: float | None) -> Optional[str]:
    if label_policy == "task-three-class":
        return TASK_LABELS[task]
    if label_policy == "task-binary":
        return "Relaxed" if task == "T1" else "Stress"
    if label_policy == "self-report-filtered":
        if anxiety is None or not np.isfinite(anxiety):
            return None
        if task == "T1" and anxiety <= 1.8:
            return "Relaxed"
        if task in {"T2", "T3"} and anxiety >= 1.8:
            return "Stress"
        return None
    raise ValueError(f"Unknown label policy: {label_policy}")


def _estimate_rppg_window(rgb_window: np.ndarray, fps: float) -> tuple[Optional[float], float, float, list[float]]:
    processor = RPPGProcessor(fps=fps, window_sec=min(12.0, max(4.0, len(rgb_window) / fps)))
    estimate = processor.estimate(rgb_window)

    hr_series: list[float] = []
    sub_window = int(max(fps * 12.0, fps * 4.0))
    sub_step = int(max(fps * 6.0, 1))
    if len(rgb_window) >= sub_window:
        for start in range(0, len(rgb_window) - sub_window + 1, sub_step):
            sub_processor = RPPGProcessor(fps=fps, window_sec=12.0)
            sub_estimate = sub_processor.estimate(rgb_window[start : start + sub_window])
            if sub_estimate.hr_bpm is not None:
                hr_series.append(float(sub_estimate.hr_bpm))
    if estimate.hr_bpm is not None and not hr_series:
        hr_series.append(float(estimate.hr_bpm))
    hrv_proxy = compute_hrv_proxy(np.asarray(hr_series, dtype=float))
    return estimate.hr_bpm, float(estimate.snr_db), hrv_proxy, hr_series


def _bvp_reference_for_window(
    subject_dir: Path,
    subject: str,
    task: str,
    start_sec: float,
    end_sec: float,
) -> tuple[Optional[float], Optional[float]]:
    bvp_path = subject_dir / f"bvp_{subject}_{task}.csv"
    if not bvp_path.exists():
        return None, None
    bvp = _read_series(bvp_path)
    start = int(start_sec * BVP_FS)
    end = int(end_sec * BVP_FS)
    bvp_window = bvp[start:end]
    hr, snr = _fft_hr_and_snr(bvp_window, BVP_FS)
    return hr, snr


def _signal_quality(snr_db: float, motion_score: float, lighting_variation: float) -> float:
    snr_norm = float(np.clip((snr_db + 4.0) / 18.0, 0.0, 1.0))
    motion_norm = float(np.clip(1.0 - motion_score, 0.0, 1.0))
    light_norm = float(np.clip(1.0 - lighting_variation, 0.0, 1.0))
    return float(0.58 * snr_norm + 0.24 * motion_norm + 0.18 * light_norm)


def _window_slices(num_samples: int, fps: float, window_sec: float, step_sec: float):
    win = int(window_sec * fps)
    step = int(step_sec * fps)
    for start in range(0, max(0, num_samples - win + 1), step):
        yield start, start + win


def extract_video_rppg_features_from_subject_dirs(
    subject_dirs: list[Path],
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    video_stride: int = 2,
    roi_refresh_frames: int = 1,
    label_policy: str = "task-binary",
    max_frames: Optional[int] = None,
    tasks: Optional[list[str]] = None,
    use_bvp_reference: bool = True,
) -> pd.DataFrame:
    raw_rows: list[dict] = []
    selected_tasks = tasks or ["T1", "T2", "T3"]
    for subject_dir in subject_dirs:
        info = _read_subject_info(subject_dir)
        self_scores = _read_self_report(subject_dir)
        for task in selected_tasks:
            video_path = subject_dir / f"vid_{info.subject}_{task}.avi"
            if not video_path.exists():
                continue
            anxiety = self_scores.get(task)
            label = _task_label(task, label_policy, anxiety)
            if label is None:
                continue
            print(f"Extracting video-rPPG: {info.subject} {task}")
            signals = extract_video_signals(
                video_path,
                max_frames=max_frames or 10_000_000,
                stride=video_stride,
                roi_refresh_frames=roi_refresh_frames,
            )
            for start, end in _window_slices(len(signals.rgb_series), signals.fps, window_sec, step_sec):
                start_sec = float(start / signals.fps)
                end_sec = float(start_sec + window_sec)
                rgb_window = signals.rgb_series[start:end]
                hr, snr, hrv_proxy, hr_series = _estimate_rppg_window(rgb_window, signals.fps)
                if hr is None:
                    continue
                motion_score = float(np.median(signals.motion_series[start:end])) if end > start else 0.0
                lighting_variation = float(np.std(signals.lighting_series[start:end])) if end > start else 0.0
                if use_bvp_reference:
                    bvp_hr, bvp_snr = _bvp_reference_for_window(subject_dir, info.subject, task, start_sec, end_sec)
                else:
                    bvp_hr, bvp_snr = None, None
                row = {
                    "subject": info.subject,
                    "gender": info.gender,
                    "difficulty": info.difficulty,
                    "task": task,
                    "window_start_sec": round(start_sec, 3),
                    "window_end_sec": round(end_sec, 3),
                    "self_report_anxiety": anxiety if anxiety is not None else np.nan,
                    "hr_bpm": float(hr),
                    "hrv_proxy": float(hrv_proxy),
                    "signal_quality": _signal_quality(snr, motion_score, lighting_variation),
                    "motion_score": motion_score,
                    "lighting_variation": lighting_variation,
                    "snr_db": float(snr),
                    "emotion_stress": 0.35,
                    "emotion_confidence": 0.0,
                    "rppg_hr_series_count": len(hr_series),
                    "label": label,
                    "label_source": label_policy,
                    "feature_source": "video_rppg",
                }
                if use_bvp_reference:
                    row.update(
                        {
                            "bvp_hr_bpm": bvp_hr if bvp_hr is not None else np.nan,
                            "bvp_snr_db": bvp_snr if bvp_snr is not None else np.nan,
                            "rppg_hr_error_bpm": abs(float(hr) - float(bvp_hr)) if bvp_hr is not None else np.nan,
                        }
                    )
                raw_rows.append(row)

    df = pd.DataFrame(raw_rows)
    if df.empty:
        return df
    baselines = (
        df[df["task"] == "T1"]
        .groupby("subject")
        .agg(
            baseline_hr=("hr_bpm", "median"),
            baseline_hrv=("hrv_proxy", "median"),
            baseline_signal_quality=("signal_quality", "median"),
        )
    )
    df = df.merge(baselines, on="subject", how="left")
    df["hr_delta_bpm"] = df["hr_bpm"] - df["baseline_hr"]
    df["hrv_delta"] = df["hrv_proxy"] - df["baseline_hrv"]
    df["signal_quality_delta"] = df["signal_quality"] - df["baseline_signal_quality"]
    ordered = [
        "subject",
        "gender",
        "difficulty",
        "task",
        "window_start_sec",
        "window_end_sec",
        "self_report_anxiety",
        "hr_bpm",
        "hr_delta_bpm",
        "hrv_proxy",
        "hrv_delta",
        "signal_quality",
        "signal_quality_delta",
        "motion_score",
        "lighting_variation",
        "snr_db",
        "emotion_stress",
        "emotion_confidence",
        "bvp_hr_bpm",
        "bvp_snr_db",
        "rppg_hr_error_bpm",
        "rppg_hr_series_count",
        "label",
        "label_source",
        "feature_source",
    ]
    return df[[col for col in ordered if col in df.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video-rPPG training features from UBFC-Phys videos.")
    parser.add_argument("--dataset-root", type=str, required=True, nargs="+")
    parser.add_argument("--out", type=str, default="sample_data/ubfc_phys_video_rppg_features.csv")
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=30.0)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--roi-refresh-frames", type=int, default=1)
    parser.add_argument(
        "--label-policy",
        choices=["task-binary", "task-three-class", "self-report-filtered"],
        default="task-binary",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--limit-subjects", type=int)
    parser.add_argument("--tasks", nargs="+", choices=["T1", "T2", "T3"])
    parser.add_argument("--quality-gate", choices=["none", "fyp", "training-reference"], default="none")
    parser.add_argument("--min-signal-quality", type=float, default=0.42)
    parser.add_argument("--min-snr-db", type=float, default=0.0)
    parser.add_argument("--max-motion-score", type=float, default=0.18)
    parser.add_argument("--max-lighting-variation", type=float, default=0.12)
    parser.add_argument("--max-rppg-hr-error-bpm", type=float, default=15.0)
    parser.add_argument(
        "--no-bvp-reference",
        action="store_true",
        help="Do not compute BVP reference HR/error columns. Use this for deployment-style online-gate experiments.",
    )
    args = parser.parse_args()

    subject_dirs: list[Path] = []
    for root in args.dataset_root:
        subject_dirs.extend(sorted(path for path in Path(root).iterdir() if path.is_dir()))
    if args.limit_subjects is not None:
        subject_dirs = subject_dirs[: args.limit_subjects]
    df = extract_video_rppg_features_from_subject_dirs(
        subject_dirs,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        video_stride=args.video_stride,
        roi_refresh_frames=args.roi_refresh_frames,
        label_policy=args.label_policy,
        max_frames=args.max_frames,
        tasks=args.tasks,
        use_bvp_reference=not args.no_bvp_reference,
    )
    if df.empty:
        raise SystemExit("No video-rPPG feature rows were extracted.")
    if args.quality_gate != "none":
        gate_config = QualityGateConfig(
            min_signal_quality=args.min_signal_quality,
            min_snr_db=args.min_snr_db,
            max_motion_score=args.max_motion_score,
            max_lighting_variation=args.max_lighting_variation,
            max_rppg_hr_error_bpm=args.max_rppg_hr_error_bpm,
        )
        mask = quality_gate_mask(df, gate_config, mode=args.quality_gate)
        summary = quality_gate_summary(df, mask, gate_config, mode=args.quality_gate)
        print(
            "Quality gate "
            f"({summary['mode']}): kept {summary['kept']}/{summary['total']} "
            f"({summary['kept_ratio']:.1%}), rejected {summary['rejected']}"
        )
        print(f"Quality gate rejection reasons: {summary['rejection_reasons']}")
        df = df.loc[mask].copy()
        if df.empty:
            raise SystemExit("Quality gate removed all video-rPPG rows.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Extracted {len(df)} video-rPPG windows from {df['subject'].nunique()} subjects.")
    print(df.groupby(["task", "label"]).size())
    if "rppg_hr_error_bpm" in df.columns and df["rppg_hr_error_bpm"].notna().any():
        mae = df["rppg_hr_error_bpm"].dropna().mean()
        print(f"rPPG vs BVP HR MAE: {mae:.2f} BPM")
    print(f"Saved video-rPPG features to {out_path}")


if __name__ == "__main__":
    main()
