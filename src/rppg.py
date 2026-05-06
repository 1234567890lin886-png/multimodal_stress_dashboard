from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


@dataclass
class RPPGEstimate:
    hr_bpm: Optional[float]
    snr_db: float
    prominence: float
    bvp: np.ndarray


def pos_projection(rgb_window: np.ndarray) -> np.ndarray:
    channels = rgb_window.T.astype(np.float64)
    normalized = channels / (channels.mean(axis=1, keepdims=True) + 1e-6) - 1.0
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    projected = projection @ normalized
    alpha = np.std(projected[0]) / (np.std(projected[1]) + 1e-6)
    bvp = projected[0] + alpha * projected[1]
    return bvp - np.mean(bvp)


def chrom_projection(rgb_window: np.ndarray) -> np.ndarray:
    channels = rgb_window.T.astype(np.float64)
    normalized = channels / (channels.mean(axis=1, keepdims=True) + 1e-6) - 1.0
    x_signal = 3.0 * normalized[0] - 2.0 * normalized[1]
    y_signal = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]
    alpha = np.std(x_signal) / (np.std(y_signal) + 1e-6)
    bvp = x_signal - alpha * y_signal
    return bvp - np.mean(bvp)


def _bandpass(values: np.ndarray, fps: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = 0.5 * fps
    low = np.clip(low_hz / nyquist, 0.001, 0.99)
    high = np.clip(high_hz / nyquist, 0.001, 0.99)
    b_coeff, a_coeff = signal.butter(3, [low, high], btype="band")
    if len(values) < 16:
        return values
    return signal.filtfilt(b_coeff, a_coeff, signal.detrend(values))


def _parabolic_refine(freqs: np.ndarray, spectrum: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(spectrum) - 1:
        return float(freqs[index])
    left, center, right = spectrum[index - 1], spectrum[index], spectrum[index + 1]
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return float(freqs[index])
    offset = np.clip(0.5 * (left - right) / denom, -0.5, 0.5)
    step = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
    return float(freqs[index] + offset * step)


def estimate_hr_from_bvp(
    bvp: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    previous_hr: Optional[float] = None,
) -> tuple[Optional[float], float, float]:
    if len(bvp) < max(32, int(fps * 4)):
        return None, float("-inf"), 0.0

    filtered = _bandpass(np.asarray(bvp, dtype=np.float64), fps, low_hz, high_hz)
    windowed = filtered * np.hamming(len(filtered))
    fft_size = len(windowed) * 4
    freqs = rfftfreq(fft_size, 1.0 / fps)
    spectrum = np.abs(rfft(windowed, n=fft_size))
    valid = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(valid):
        return None, float("-inf"), 0.0

    vf = freqs[valid]
    vs = spectrum[valid]
    peaks, _ = signal.find_peaks(vs)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(vs))])

    peak_scores = vs[peaks] ** 2
    ordered = peaks[np.argsort(peak_scores)[::-1]][:6]
    candidates: list[tuple[float, float]] = []
    for peak in ordered:
        hr = _parabolic_refine(vf, vs, int(peak)) * 60.0
        if 45.0 <= hr <= 190.0:
            candidates.append((float(hr), float(vs[peak] ** 2)))
    if not candidates:
        return None, float("-inf"), 0.0

    max_power = max(power for _, power in candidates) + 1e-10
    best_hr, best_score = candidates[0][0], -1e9
    for hr, power in candidates:
        score = 2.5 * (power / max_power)
        if previous_hr is not None:
            score -= 0.08 * abs(hr - previous_hr)
        if hr >= 120.0:
            half_candidates = [p for h, p in candidates if abs(h - hr / 2.0) <= 6.0]
            if half_candidates and max(half_candidates) >= 0.45 * power:
                score -= 0.8
        if score > best_score:
            best_hr, best_score = hr, score

    best_index = int(np.argmax(vs))
    snr_db = 10.0 * np.log10((vs[best_index] ** 2) / (np.median(vs ** 2) + 1e-10))
    sorted_power = np.sort(peak_scores)[::-1]
    second = float(sorted_power[1]) if len(sorted_power) > 1 else float(np.median(vs ** 2) + 1e-10)
    prominence = float(sorted_power[0] / (second + 1e-10))
    return float(best_hr), float(snr_db), prominence


class RPPGProcessor:
    def __init__(self, fps: float = 30.0, window_sec: float = 12.0):
        self.fps = float(fps)
        self.window_sec = float(window_sec)
        self.previous_hr: Optional[float] = None

    def estimate(self, rgb_series: np.ndarray) -> RPPGEstimate:
        if len(rgb_series) < max(32, int(self.fps * 4)):
            return RPPGEstimate(None, float("-inf"), 0.0, np.array([]))
        window_len = min(len(rgb_series), int(self.fps * self.window_sec))
        rgb_window = np.asarray(rgb_series[-window_len:], dtype=np.float64)
        pos_bvp = pos_projection(rgb_window)
        pos_hr, pos_snr, pos_prom = estimate_hr_from_bvp(pos_bvp, self.fps, previous_hr=self.previous_hr)
        chrom_bvp = chrom_projection(rgb_window)
        chrom_hr, chrom_snr, chrom_prom = estimate_hr_from_bvp(chrom_bvp, self.fps, previous_hr=self.previous_hr)

        if chrom_hr is not None and (pos_hr is None or chrom_snr + 0.8 * chrom_prom > pos_snr + 0.8 * pos_prom):
            hr, snr, prom, bvp = chrom_hr, chrom_snr, chrom_prom, chrom_bvp
        else:
            hr, snr, prom, bvp = pos_hr, pos_snr, pos_prom, pos_bvp
        if hr is not None and snr >= -2.0:
            self.previous_hr = hr
        else:
            hr = None
        return RPPGEstimate(hr, float(snr), float(prom), bvp)

