from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.ml_model import FEATURE_NAMES, MULTIMODAL_FEATURE_NAMES
from src.quality_gate import QualityGateConfig, quality_gate_mask, quality_gate_summary


LABELS = ["Relaxed", "Mostly Calm", "Mild Stress", "Elevated Physiological Stress"]


def _label_from_features(hr_delta: float, hrv: float, quality: float, emotion_stress: float) -> str:
    if quality < 0.32:
        return "Uncertain"
    score = 0.42
    if hr_delta >= 12:
        score += 0.34
    elif hr_delta >= 7:
        score += 0.20
    elif hr_delta <= -4:
        score -= 0.12
    if hrv < 1.8 and hr_delta >= 5:
        score += 0.10
    elif hrv > 4.5:
        score -= 0.08
    score = 0.78 * score + 0.22 * emotion_stress
    if score >= 0.70:
        return "Elevated Physiological Stress"
    if score >= 0.55:
        return "Mild Stress"
    if score <= 0.32:
        return "Relaxed"
    return "Mostly Calm"


def make_training_data(n_samples: int = 1600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n_samples):
        baseline = rng.normal(74, 9)
        state = rng.choice(LABELS, p=[0.25, 0.28, 0.27, 0.20])
        if state == "Relaxed":
            hr_delta = rng.normal(-1.5, 2.8)
            hrv = rng.normal(5.2, 1.2)
            emotion_stress = rng.normal(0.22, 0.09)
        elif state == "Mostly Calm":
            hr_delta = rng.normal(2.0, 3.0)
            hrv = rng.normal(3.8, 1.0)
            emotion_stress = rng.normal(0.34, 0.10)
        elif state == "Mild Stress":
            hr_delta = rng.normal(8.5, 3.2)
            hrv = rng.normal(2.3, 0.8)
            emotion_stress = rng.normal(0.55, 0.14)
        else:
            hr_delta = rng.normal(16.0, 4.0)
            hrv = rng.normal(1.4, 0.5)
            emotion_stress = rng.normal(0.72, 0.13)

        signal_quality = rng.beta(5, 2)
        if rng.random() < 0.08:
            signal_quality = rng.uniform(0.18, 0.38)
        motion = np.clip(rng.normal(0.08 + (1 - signal_quality) * 0.25, 0.04), 0, 1)
        lighting = np.clip(rng.normal(0.06 + (1 - signal_quality) * 0.15, 0.035), 0, 1)
        snr = rng.normal(-2 + 16 * signal_quality, 2.2)
        emotion_conf = rng.uniform(0.35, 0.95)
        row = {
            "hr_bpm": np.clip(baseline + hr_delta, 45, 170),
            "hr_delta_bpm": hr_delta,
            "hrv_proxy": max(0.1, hrv),
            "signal_quality": signal_quality,
            "motion_score": motion,
            "lighting_variation": lighting,
            "snr_db": snr,
            "emotion_stress": np.clip(emotion_stress, 0, 1),
            "emotion_confidence": emotion_conf,
        }
        row["label"] = _label_from_features(
            row["hr_delta_bpm"],
            row["hrv_proxy"],
            row["signal_quality"],
            row["emotion_stress"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight stress classifier.")
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-in", type=str, help="Optional feature CSV with columns matching FEATURE_NAMES plus label.")
    parser.add_argument("--feature-set", choices=["base", "multimodal"], default="multimodal")
    parser.add_argument("--model-type", choices=["random-forest", "xgboost"], default="random-forest")
    parser.add_argument(
        "--label-mode",
        choices=["three-class", "binary"],
        default="three-class",
        help="Use original stress levels or merge Mild/Elevated into Stress.",
    )
    parser.add_argument("--model-out", type=str, default=None)
    parser.add_argument("--data-out", type=str, default="sample_data/training_features.csv")
    parser.add_argument("--quality-gate", choices=["none", "fyp", "training-reference"], default="none")
    parser.add_argument("--min-signal-quality", type=float, default=0.42)
    parser.add_argument("--min-snr-db", type=float, default=0.0)
    parser.add_argument("--max-motion-score", type=float, default=0.18)
    parser.add_argument("--max-lighting-variation", type=float, default=0.12)
    parser.add_argument("--max-rppg-hr-error-bpm", type=float, default=15.0)
    args = parser.parse_args()

    if args.data_in:
        df = pd.read_csv(args.data_in)
        selected_features = MULTIMODAL_FEATURE_NAMES if args.feature_set == "multimodal" else FEATURE_NAMES
        selected_features = [name for name in selected_features if name in df.columns]
        missing = [name for name in FEATURE_NAMES + ["label"] if name not in df.columns]
        if missing:
            raise SystemExit(f"Missing required columns in {args.data_in}: {missing}")
        df = df.dropna(subset=selected_features + ["label"]).copy()
    else:
        df = make_training_data(args.samples, args.seed)
        selected_features = FEATURE_NAMES

    gate_summary = None
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
        gate_summary = summary
        print(
            "Quality gate "
            f"({summary['mode']}): kept {summary['kept']}/{summary['total']} "
            f"({summary['kept_ratio']:.1%}), rejected {summary['rejected']}"
        )
        print(f"Quality gate rejection reasons: {summary['rejection_reasons']}")
        df = df.loc[mask].copy()
        if df.empty:
            raise SystemExit("Quality gate removed all rows; relax thresholds or inspect rPPG extraction.")
    x = df[selected_features]
    y = df["label"]
    if args.label_mode == "binary":
        y = y.replace(
            {
                "Mild Stress": "Stress",
                "Elevated Physiological Stress": "Stress",
                "Mostly Calm": "Relaxed",
            }
        )
    if "subject" in df.columns and df["subject"].nunique() >= 4:
        subjects = sorted(df["subject"].dropna().unique())
        train_subjects, test_subjects = train_test_split(subjects, test_size=0.34, random_state=args.seed)
        train_mask = df["subject"].isin(train_subjects)
        x_train, x_test = x[train_mask], x[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        print(f"Subject split: train={list(train_subjects)}, test={list(test_subjects)}")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=args.seed,
            stratify=y,
        )
    label_encoder = None
    if args.model_type == "xgboost":
        from xgboost import XGBClassifier

        label_encoder = LabelEncoder()
        y_train_model = label_encoder.fit_transform(y_train)
        y_test_model = label_encoder.transform(y_test)
        model = XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="multi:softprob" if len(label_encoder.classes_) > 2 else "binary:logistic",
            eval_metric="mlogloss" if len(label_encoder.classes_) > 2 else "logloss",
            random_state=args.seed,
        )
        weights = compute_sample_weight(class_weight="balanced", y=y_train_model)
        model.fit(x_train, y_train_model, sample_weight=weights)
        y_pred = label_encoder.inverse_transform(model.predict(x_test).astype(int))
        y_test_report = label_encoder.inverse_transform(y_test_model)
        model_name = "XGBClassifier"
    else:
        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=4,
            random_state=args.seed,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_test_report = y_test
        model_name = "RandomForestClassifier"
    print(classification_report(y_test_report, y_pred))

    data_path = Path(args.data_out)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    import joblib

    default_model_out = "models/stress_xgb.joblib" if args.model_type == "xgboost" else "models/stress_rf.joblib"
    model_path = Path(args.model_out or default_model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": selected_features,
            "model_name": model_name,
            "training_samples": int(len(df)),
            "training_source": args.data_in or "synthetic",
            "feature_set": args.feature_set,
            "label_mode": args.label_mode,
            "label_encoder": label_encoder,
            "quality_gate": gate_summary,
        },
        model_path,
    )
    print(f"Saved training data to {data_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
