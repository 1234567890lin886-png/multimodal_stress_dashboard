from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq


BVP_FS = 64.0
EDA_FS = 4.0
TASK_LABELS = {
    "T1": "Relaxed",
    "T2": "Mild Stress",
    "T3": "Elevated Physiological Stress",
}


@dataclass
class SubjectInfo:
    subject: str
    gender: str = "unknown"
    difficulty: str = "unknown"


def _read_series(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).iloc[:, 0].dropna().to_numpy(dtype=float)


def _read_subject_info(subject_dir: Path) -> SubjectInfo:
    subject = subject_dir.name
    path = subject_dir / f"info_{subject}.txt"
    if not path.exists():
        return SubjectInfo(subject=subject)
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    gender = lines[1] if len(lines) > 1 else "unknown"
    difficulty = lines[2] if len(lines) > 2 else "unknown"
    return SubjectInfo(subject=subject, gender=gender, difficulty=difficulty)


def _read_self_report(subject_dir: Path) -> dict[str, float]:
    subject = subject_dir.name
    candidates = list(subject_dir.glob(f"selfReportedAnx_{subject}.csv"))
    if not candidates:
        candidates = list(subject_dir.glob("selfReportedAnx_*.csv"))
    if not candidates:
        return {}
    df = pd.read_csv(candidates[0], header=None)
    scores: dict[str, float] = {}
    for idx, task in enumerate(["T1", "T2", "T3"]):
        if idx < len(df):
            values = pd.to_numeric(df.iloc[idx], errors="coerce").dropna()
            if len(values) > 0:
                scores[task] = float(values.mean())
    return scores


def _fft_hr_and_snr(bvp: np.ndarray, fs: float = BVP_FS) -> tuple[Optional[float], float]:
    if len(bvp) < int(fs * 10):
        return None, float("-inf")
    values = signal.detrend(np.asarray(bvp, dtype=float))
    low, high = 0.7, 3.5
    nyq = 0.5 * fs
    b_coeff, a_coeff = signal.butter(3, [low / nyq, high / nyq], btype="band")
    filtered = signal.filtfilt(b_coeff, a_coeff, values)
    windowed = filtered * np.hamming(len(filtered))
    freqs = rfftfreq(len(windowed) * 4, 1.0 / fs)
    spectrum = np.abs(rfft(windowed, n=len(windowed) * 4))
    valid = (freqs >= low) & (freqs <= high)
    if not np.any(valid):
        return None, float("-inf")
    vf = freqs[valid]
    vs = spectrum[valid]
    peak_idx = int(np.argmax(vs))
    hr = float(vf[peak_idx] * 60.0)
    snr = 10.0 * np.log10((vs[peak_idx] ** 2) / (np.median(vs**2) + 1e-10))
    if not 45.0 <= hr <= 180.0:
        return None, float(snr)
    return hr, float(snr)


def _hrv_proxy_from_peaks(bvp: np.ndarray, fs: float = BVP_FS) -> float:
    if len(bvp) < int(fs * 15):
        return 0.0
    values = signal.detrend(np.asarray(bvp, dtype=float))
    low, high = 0.7, 3.5
    nyq = 0.5 * fs
    b_coeff, a_coeff = signal.butter(3, [low / nyq, high / nyq], btype="band")
    filtered = signal.filtfilt(b_coeff, a_coeff, values)
    min_distance = int(0.36 * fs)
    peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=np.std(filtered) * 0.15)
    if len(peaks) < 4:
        return 0.0
    rr_ms = np.diff(peaks) / fs * 1000.0
    rr_ms = rr_ms[(rr_ms >= 350.0) & (rr_ms <= 1400.0)]
    if len(rr_ms) < 3:
        return 0.0
    sdnn = float(np.std(rr_ms))
    rmssd = float(np.sqrt(np.mean(np.square(np.diff(rr_ms))))) if len(rr_ms) > 1 else 0.0
    return float(np.clip((0.45 * sdnn + 0.55 * rmssd) / 25.0, 0.0, 8.0))


def _eda_window_features(eda: np.ndarray) -> tuple[float, float]:
    if len(eda) == 0:
        return 0.0, 0.0
    clean = np.asarray(eda, dtype=float)
    return float(np.mean(clean)), float(np.std(clean))


def _signal_quality(snr_db: float, eda_std: float) -> float:
    snr_norm = float(np.clip((snr_db + 4.0) / 18.0, 0.0, 1.0))
    eda_norm = float(np.clip(eda_std / 0.08, 0.0, 1.0))
    return float(np.clip(0.72 * snr_norm + 0.28 * (0.55 + 0.45 * eda_norm), 0.0, 1.0))


def _iter_windows(values: np.ndarray, fs: float, window_sec: float, step_sec: float):
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    for start in range(0, max(0, len(values) - win + 1), step):
        yield start, start + win


def extract_features(dataset_root: Path, window_sec: float = 60.0, step_sec: float = 30.0) -> pd.DataFrame:
    rows: list[dict] = []
    subject_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    return extract_features_from_subject_dirs(subject_dirs, window_sec, step_sec)


def extract_features_from_subject_dirs(
    subject_dirs: list[Path], window_sec: float = 60.0, step_sec: float = 30.0
) -> pd.DataFrame:
    rows: list[dict] = []
    baseline_by_subject: dict[str, float] = {}

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        bvp_path = subject_dir / f"bvp_{subject}_T1.csv"
        if not bvp_path.exists():
            continue
        bvp = _read_series(bvp_path)
        hrs = []
        for start, end in _iter_windows(bvp, BVP_FS, window_sec, step_sec):
            hr, _ = _fft_hr_and_snr(bvp[start:end], BVP_FS)
            if hr is not None:
                hrs.append(hr)
        if hrs:
            baseline_by_subject[subject] = float(np.median(hrs))

    for subject_dir in subject_dirs:
        info = _read_subject_info(subject_dir)
        self_scores = _read_self_report(subject_dir)
        baseline = baseline_by_subject.get(info.subject)
        for task in ["T1", "T2", "T3"]:
            bvp_path = subject_dir / f"bvp_{info.subject}_{task}.csv"
            eda_path = subject_dir / f"eda_{info.subject}_{task}.csv"
            if not bvp_path.exists() or not eda_path.exists():
                continue
            bvp = _read_series(bvp_path)
            eda = _read_series(eda_path)
            eda_ratio = EDA_FS / BVP_FS
            for start, end in _iter_windows(bvp, BVP_FS, window_sec, step_sec):
                hr, snr = _fft_hr_and_snr(bvp[start:end], BVP_FS)
                if hr is None:
                    continue
                hrv_proxy = _hrv_proxy_from_peaks(bvp[start:end], BVP_FS)
                eda_start = int(start * eda_ratio)
                eda_end = int(end * eda_ratio)
                eda_mean, eda_std = _eda_window_features(eda[eda_start:eda_end])
                hr_delta = float(hr - baseline) if baseline is not None else 0.0
                rows.append(
                    {
                        "subject": info.subject,
                        "gender": info.gender,
                        "difficulty": info.difficulty,
                        "task": task,
                        "window_start_sec": round(start / BVP_FS, 3),
                        "window_end_sec": round(end / BVP_FS, 3),
                        "self_report_anxiety": self_scores.get(task, np.nan),
                        "hr_bpm": hr,
                        "hr_delta_bpm": hr_delta,
                        "hrv_proxy": hrv_proxy,
                        "signal_quality": _signal_quality(snr, eda_std),
                        "motion_score": 0.05,
                        "lighting_variation": 0.05,
                        "snr_db": snr,
                        "emotion_stress": 0.35,
                        "emotion_confidence": 0.0,
                        "eda_mean": eda_mean,
                        "eda_std": eda_std,
                        "label": TASK_LABELS[task],
                        "label_source": "UBFC-Phys task protocol",
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UBFC-Phys BVP/EDA files into training features.")
    parser.add_argument("--dataset-root", type=str, required=True, nargs="+")
    parser.add_argument("--out", type=str, default="sample_data/ubfc_phys_features.csv")
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=30.0)
    args = parser.parse_args()

    subject_dirs: list[Path] = []
    for root in args.dataset_root:
        subject_dirs.extend(sorted(path for path in Path(root).iterdir() if path.is_dir()))
    df = extract_features_from_subject_dirs(subject_dirs, args.window_sec, args.step_sec)
    if df.empty:
        raise SystemExit("No UBFC-Phys feature rows were extracted. Check dataset path and file names.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Extracted {len(df)} windows from {df['subject'].nunique()} subjects.")
    print(df.groupby(["task", "label"]).size())
    print(f"Saved features to {out_path}")


if __name__ == "__main__":
    main()
