from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.gated_fusion_model import GatedFusionConfig, build_gated_fusion_model


PHYSIO_FEATURES = [
    "hr_bpm",
    "hr_delta_bpm",
    "hrv_proxy",
    "signal_quality",
    "motion_score",
    "lighting_variation",
    "snr_db",
]

EMOTION_FEATURES = [
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


def _binary_labels(labels: pd.Series) -> pd.Series:
    return labels.replace(
        {
            "Mild Stress": "Stress",
            "Elevated Physiological Stress": "Stress",
            "Mostly Calm": "Relaxed",
        }
    )


def _split_by_subject(df: pd.DataFrame, seed: int):
    subjects = sorted(df["subject"].dropna().unique())
    train_subjects, test_subjects = train_test_split(subjects, test_size=0.34, random_state=seed)
    train_mask = df["subject"].isin(train_subjects)
    return train_mask, train_subjects, test_subjects


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a gated semantic fusion MLP.")
    parser.add_argument("--data-in", type=str, default="sample_data/ubfc_phys_s11_s20_s51_s56_multimodal_features_f3.csv")
    parser.add_argument("--model-out", type=str, default="models/gated_fusion_mlp.joblib")
    parser.add_argument("--weights-out", type=str, default="models/gated_fusion_mlp.weights.h5")
    parser.add_argument("--metrics-out", type=str, default="models/gated_fusion_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.20)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    df = pd.read_csv(args.data_in)
    required = PHYSIO_FEATURES + EMOTION_FEATURES + ["label", "subject"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    df = df.dropna(subset=required).copy()
    labels = _binary_labels(df["label"])
    y = labels.map({"Relaxed": 0, "Stress": 1}).to_numpy(dtype=np.float32)

    train_mask, train_subjects, test_subjects = _split_by_subject(df, args.seed)
    print(f"Subject split: train={list(train_subjects)}, test={list(test_subjects)}")

    physio_scaler = StandardScaler()
    emotion_scaler = StandardScaler()
    x_physio_train = physio_scaler.fit_transform(df.loc[train_mask, PHYSIO_FEATURES])
    x_emotion_train = emotion_scaler.fit_transform(df.loc[train_mask, EMOTION_FEATURES])
    x_physio_test = physio_scaler.transform(df.loc[~train_mask, PHYSIO_FEATURES])
    x_emotion_test = emotion_scaler.transform(df.loc[~train_mask, EMOTION_FEATURES])
    y_train = y[train_mask.to_numpy()]
    y_test = y[(~train_mask).to_numpy()]

    config = GatedFusionConfig(
        physio_dim=len(PHYSIO_FEATURES),
        emotion_dim=len(EMOTION_FEATURES),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    )
    model, gate_model = build_gated_fusion_model(config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    classes = np.array([0, 1])
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.astype(int))
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=24,
            restore_best_weights=True,
            mode="max",
        )
    ]
    model.fit(
        [x_physio_train, x_emotion_train],
        y_train,
        validation_split=0.20,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    probs = model.predict([x_physio_test, x_emotion_test], verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    print(classification_report(y_test.astype(int), preds, target_names=["Relaxed", "Stress"]))
    acc = float(accuracy_score(y_test.astype(int), preds))
    gates = gate_model.predict([x_physio_test, x_emotion_test], verbose=0)
    physio_weight = float(np.mean(gates))
    emotion_weight = float(1.0 - physio_weight)
    print(f"accuracy={acc:.3f}")
    print(f"mean_physio_gate={physio_weight:.3f}, mean_emotion_gate={emotion_weight:.3f}")

    weights_path = Path(args.weights_out)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)

    bundle = {
        "model_type": "gated_fusion_mlp",
        "model_name": "GatedFusionMLP",
        "weights_path": str(weights_path),
        "physio_features": PHYSIO_FEATURES,
        "emotion_features": EMOTION_FEATURES,
        "physio_scaler": physio_scaler,
        "emotion_scaler": emotion_scaler,
        "config": config.__dict__,
        "training_source": args.data_in,
        "label_mode": "binary",
        "training_subjects": list(train_subjects),
        "test_subjects": list(test_subjects),
        "accuracy": acc,
        "mean_physio_gate": physio_weight,
        "mean_emotion_gate": emotion_weight,
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    metrics = {
        "accuracy": acc,
        "classification_report": classification_report(
            y_test.astype(int),
            preds,
            target_names=["Relaxed", "Stress"],
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test.astype(int), preds).tolist(),
        "mean_physio_gate": physio_weight,
        "mean_emotion_gate": emotion_weight,
        "train_subjects": list(train_subjects),
        "test_subjects": list(test_subjects),
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved gated model bundle to {model_path}")
    print(f"Saved gated weights to {weights_path}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()

