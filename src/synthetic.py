from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticSession:
    fps: float
    timestamps: np.ndarray
    rgb_series: np.ndarray
    motion_series: np.ndarray
    lighting_series: np.ndarray
    expected_state: str


def make_synthetic_session(
    duration_sec: float = 60.0,
    fps: float = 30.0,
    baseline_hr: float = 74.0,
    elevated_hr: float = 96.0,
    stress_start_sec: float = 32.0,
    seed: int = 7,
) -> SyntheticSession:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_sec, 1.0 / fps)
    hr_curve = np.where(t < stress_start_sec, baseline_hr, elevated_hr)
    hr_curve += 1.8 * np.sin(2 * np.pi * t / 22.0)
    phase = np.cumsum(hr_curve / 60.0 / fps)
    pulse = np.sin(2 * np.pi * phase)
    drift = 0.015 * np.sin(2 * np.pi * t / 18.0)
    noise = rng.normal(0.0, 0.006, size=(len(t), 3))
    rgb = np.column_stack(
        [
            130.0 + 1.2 * pulse + 255 * drift,
            92.0 + 3.2 * pulse + 220 * drift,
            78.0 + 0.9 * pulse + 180 * drift,
        ]
    )
    rgb += 255.0 * noise
    rgb = np.clip(rgb, 0.0, 255.0)
    motion = np.where(t < stress_start_sec, 0.04, 0.09) + rng.normal(0.0, 0.01, size=len(t))
    lighting = 0.22 + 0.05 * np.sin(2 * np.pi * t / 15.0) + rng.normal(0.0, 0.01, size=len(t))
    return SyntheticSession(
        fps=fps,
        timestamps=t,
        rgb_series=rgb,
        motion_series=np.clip(motion, 0.0, 1.0),
        lighting_series=np.clip(lighting, 0.0, 1.0),
        expected_state="stress transition",
    )


def save_synthetic_csv(path: str) -> None:
    session = make_synthetic_session()
    df = pd.DataFrame(
        {
            "timestamp": session.timestamps,
            "r": session.rgb_series[:, 0],
            "g": session.rgb_series[:, 1],
            "b": session.rgb_series[:, 2],
            "motion": session.motion_series,
            "lighting": session.lighting_series,
        }
    )
    df.to_csv(path, index=False)

