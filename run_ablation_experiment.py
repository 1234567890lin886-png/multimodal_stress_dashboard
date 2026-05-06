from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


PHYSIO_FEATURES = [
    "hr_bpm",
    "hr_delta_bpm",
    "hrv_proxy",
    "signal_quality",
    "motion_score",
    "lighting_variation",
    "snr_db",
]

FER_FEATURES = [
    "emotion_stress",
    "emotion_confidence",
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

FEATURE_SETS = {
    "rppg_only": PHYSIO_FEATURES,
    "fer_only": FER_FEATURES,
    "multimodal": PHYSIO_FEATURES + FER_FEATURES,
}


def _binary_labels(labels: pd.Series) -> pd.Series:
    return labels.replace(
        {
            "Mild Stress": "Stress",
            "Elevated Physiological Stress": "Stress",
            "Mostly Calm": "Relaxed",
        }
    )


def _prepare(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    data = frame.copy()
    for name in features:
        if name not in data.columns:
            data[name] = 0.0
    x = data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = _binary_labels(data["label"])
    return x, y


def _fit_xgb(x: pd.DataFrame, y: pd.Series, seed: int):
    from xgboost import XGBClassifier

    encoder = LabelEncoder()
    y_model = encoder.fit_transform(y)
    model = XGBClassifier(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        objective="binary:logistic" if len(encoder.classes_) == 2 else "multi:softprob",
        eval_metric="logloss" if len(encoder.classes_) == 2 else "mlogloss",
        random_state=seed,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_model)
    model.fit(x, y_model, sample_weight=weights)
    return model, encoder


def _predict(model, encoder: LabelEncoder, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict(x).astype(int)
    return encoder.inverse_transform(raw)


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    labels = ["Relaxed", "Stress"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "relaxed_f1": float(f1_score(y_true, y_pred, labels=["Relaxed"], average="macro")),
        "stress_f1": float(f1_score(y_true, y_pred, labels=["Stress"], average="macro")),
        "confusion_matrix": matrix.tolist(),
    }


def _fer_coverage(frame: pd.DataFrame) -> str:
    if "frames_analyzed" not in frame.columns:
        return "0/0"
    covered = int((frame["frames_analyzed"].fillna(0) > 0).sum())
    total = int(len(frame))
    return f"{covered}/{total} ({covered / total:.1%})" if total else "0/0"


def _save_model_bundle(
    model_dir: Path,
    name: str,
    model,
    encoder: LabelEncoder,
    features: list[str],
    train_data: str,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": features,
            "model_name": "XGBClassifier",
            "training_samples": None,
            "training_source": train_data,
            "feature_set": name,
            "label_mode": "binary",
            "label_encoder": encoder,
        },
        model_dir / f"ablation_xgb_{name}.joblib",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rPPG/FER/multimodal ablation experiments.")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--external-data", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="outputs/ablation_results.csv")
    parser.add_argument("--out-md", type=str, default="models/ABLATION_EXPERIMENT.md")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_data)
    external_df = pd.read_csv(args.external_data)
    subjects = sorted(train_df["subject"].dropna().unique())
    train_subjects, test_subjects = train_test_split(subjects, test_size=0.34, random_state=args.seed)
    internal_train = train_df[train_df["subject"].isin(train_subjects)].copy()
    internal_test = train_df[~train_df["subject"].isin(train_subjects)].copy()

    rows: list[dict] = []
    md_lines = [
        "# Ablation Experiment",
        "",
        "Task: binary `Relaxed` vs `Stress` classification.",
        "",
        "## Data",
        "",
        f"- Training/internal data: `{args.train_data}`",
        f"- External test data: `{args.external_data}`",
        f"- Internal split train subjects: `{list(train_subjects)}`",
        f"- Internal split test subjects: `{list(test_subjects)}`",
        f"- Training FER coverage: {_fer_coverage(train_df)}",
        f"- External FER coverage: {_fer_coverage(external_df)}",
        "",
        "## Results",
        "",
        "| Feature Set | Internal Accuracy | Internal Macro F1 | External Accuracy | External Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, features in FEATURE_SETS.items():
        x_train, y_train = _prepare(internal_train, features)
        x_internal, y_internal = _prepare(internal_test, features)
        internal_model, internal_encoder = _fit_xgb(x_train, y_train, args.seed)
        internal_pred = _predict(internal_model, internal_encoder, x_internal)
        internal_metrics = _metrics(y_internal, internal_pred)

        x_full, y_full = _prepare(train_df, features)
        x_external, y_external = _prepare(external_df, features)
        external_model, external_encoder = _fit_xgb(x_full, y_full, args.seed)
        external_pred = _predict(external_model, external_encoder, x_external)
        external_metrics = _metrics(y_external, external_pred)
        _save_model_bundle(Path(args.model_dir), name, external_model, external_encoder, features, args.train_data)

        rows.append(
            {
                "feature_set": name,
                "features": ", ".join(features),
                "internal_accuracy": internal_metrics["accuracy"],
                "internal_macro_f1": internal_metrics["macro_f1"],
                "internal_relaxed_f1": internal_metrics["relaxed_f1"],
                "internal_stress_f1": internal_metrics["stress_f1"],
                "internal_confusion_matrix": internal_metrics["confusion_matrix"],
                "external_accuracy": external_metrics["accuracy"],
                "external_macro_f1": external_metrics["macro_f1"],
                "external_relaxed_f1": external_metrics["relaxed_f1"],
                "external_stress_f1": external_metrics["stress_f1"],
                "external_confusion_matrix": external_metrics["confusion_matrix"],
            }
        )
        md_lines.append(
            f"| `{name}` | {internal_metrics['accuracy']:.3f} | {internal_metrics['macro_f1']:.3f} "
            f"| {external_metrics['accuracy']:.3f} | {external_metrics['macro_f1']:.3f} |"
        )

    result = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    md_lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `rppg_only` measures how much the quality-gated physiological signal contributes.",
            "- `fer_only` measures whether facial emotion semantics alone carry stress information.",
            "- `multimodal` measures whether combining rPPG and FER improves over rPPG alone.",
            "",
            "Confusion matrices are saved in the CSV in `[Relaxed, Stress]` label order.",
        ]
    )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(result[["feature_set", "internal_accuracy", "internal_macro_f1", "external_accuracy", "external_macro_f1"]])
    print(f"Saved ablation CSV to {out_csv}")
    print(f"Saved ablation report to {out_md}")


if __name__ == "__main__":
    main()
