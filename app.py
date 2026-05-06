from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analyzer import AnalysisReport, analyze_synthetic, analyze_video


st.set_page_config(page_title="Multimodal Stress Dashboard", layout="wide")


def available_models() -> dict[str, str | None]:
    candidates: dict[str, str | None] = {"Auto": None}
    paths = {
        "XGBoost Online Gate": Path("models/stress_xgb_video_rppg_online_qg.joblib"),
        "XGBoost Video-rPPG Reference Gate": Path("models/stress_xgb_video_rppg.joblib"),
        "Gated Fusion MLP": Path("models/gated_fusion_mlp.joblib"),
        "XGBoost Early Fusion": Path("models/stress_xgb.joblib"),
        "Random Forest Baseline": Path("models/stress_rf.joblib"),
    }
    for name, path in paths.items():
        if path.exists():
            candidates[name] = str(path)
    return candidates


def render_metric_cards(report: AnalysisReport) -> None:
    features = report.features
    decision = report.decision
    cols = st.columns(5)
    cols[0].metric("Stress Label", decision.label)
    cols[1].metric("HR", "N/A" if features.hr_bpm is None else f"{features.hr_bpm:.1f} BPM")
    cols[2].metric("Baseline", "N/A" if features.baseline_hr is None else f"{features.baseline_hr:.1f} BPM")
    cols[3].metric("Signal Quality", f"{features.signal_quality:.2f}")
    cols[4].metric("Confidence", f"{decision.confidence:.2f}")


def render_report(report: AnalysisReport) -> None:
    st.subheader("Session Summary")
    render_metric_cards(report)

    left, right = st.columns([1.35, 1.0])
    with left:
        if not report.timeline.empty:
            fig = px.line(report.timeline, x="time_sec", y="hr_bpm", markers=True, title="Estimated Heart Rate")
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="BPM", height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough signal duration for a rolling heart-rate estimate.")

    with right:
        st.write("**Decision Rationale**")
        st.write(report.decision.rationale)
        if report.model_prediction is not None:
            st.write("**ML Classifier**")
            st.write(
                {
                    "model": report.model_prediction.model_name,
                    "training_source": report.model_prediction.training_source,
                    "prediction": report.model_prediction.label,
                    "confidence": round(report.model_prediction.confidence, 3),
                    "rule_baseline": report.rule_decision.label,
                }
            )
            if report.model_prediction.physio_weight is not None:
                st.write("**Adaptive Fusion Weights**")
                st.write(
                    {
                        "physiological_modality": round(report.model_prediction.physio_weight, 3),
                        "facial_emotion_modality": round(report.model_prediction.emotion_weight or 0.0, 3),
                    }
                )
        else:
            st.write("**ML Classifier**")
            st.write("No trained model found. Run `python train_classifier.py` to enable it.")
        st.write("**Emotion Context**")
        st.write(
            {
                "dominant_emotion": report.emotion.dominant_emotion,
                "emotion_confidence": round(report.emotion.confidence, 3),
                "emotion_stress_score": round(report.emotion.stress_semantic, 3),
            }
        )
        st.write("**Physiological Features**")
        st.write(
            {
                "hr_delta_bpm": round(report.features.hr_delta_bpm, 2),
                "hrv_proxy": round(report.features.hrv_proxy, 2),
                "snr_db": round(report.features.snr_db, 2),
                "motion_score": round(report.features.motion_score, 3),
                "lighting_variation": round(report.features.lighting_variation, 3),
                "arousal_state": report.features.arousal_state,
            }
        )

    payload = json.dumps(report.to_dict(), indent=2)
    st.download_button("Download JSON Report", payload, file_name="stress_report.json", mime="application/json")


st.title("Multimodal Stress Monitoring Dashboard")
st.caption("rPPG + HRV proxy + optional FER + uncertainty-aware semantic fusion")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Choose source", ["Synthetic demo", "Upload video", "Local video path"])
    model_options = available_models()
    selected_model = st.selectbox("Model", list(model_options.keys()))
    model_path = model_options[selected_model]
    max_frames = st.slider("Max frames", min_value=120, max_value=1800, value=900, step=60)
    stride = st.slider("Frame stride", min_value=1, max_value=10, value=6)
    run = st.button("Analyze", type="primary")

if mode == "Synthetic demo":
    st.info("Synthetic demo simulates a baseline period followed by elevated physiological arousal.")
    if run:
        with st.spinner("Analyzing synthetic session..."):
            render_report(analyze_synthetic(model_path=model_path))
elif mode == "Upload video":
    uploaded = st.file_uploader("Upload a short face video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None and run:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Extracting rPPG and stress features..."):
                render_report(analyze_video(tmp_path, max_frames=max_frames, stride=stride, model_path=model_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    elif run:
        st.warning("Upload a video before analyzing.")
else:
    local_path = st.text_input("Local video path", value=r"H:\s51_to_s56\s54\vid_s54_T2.avi")
    if run:
        path = Path(local_path.strip().strip('"'))
        if not path.exists():
            st.warning(f"Video path does not exist: {path}")
        else:
            with st.spinner("Extracting rPPG and stress features from local file..."):
                render_report(analyze_video(path, max_frames=max_frames, stride=stride, model_path=model_path))
