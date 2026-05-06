# Multimodal Stress Monitoring Dashboard

Privacy-preserving stress and emotion monitoring demo inspired by a final-year project on semantic EVM, rPPG, and facial expression recognition. This repository is intentionally standalone: it does not import or modify the original FYP codebase.

## What It Does

- Accepts a video file or a synthetic demo signal.
- Extracts camera-based rPPG features with FaceMesh cheek ROI, skin-mask refinement, and POS/CHROM RGB projection.
- Estimates HR, HRV proxy, signal quality, motion and lighting stability.
- Adds optional facial emotion context when DeepFace is installed.
- Produces an uncertainty-aware stress label.
- Shows a Streamlit dashboard with time-series plots, feature cards, rationales, and a downloadable JSON report.

## Why This Project Fits AI Roles

This project demonstrates:

- Multimodal AI signal fusion
- rPPG and lightweight physiological feature extraction
- Explainable decision logic
- Uncertainty-aware classification
- Privacy-preserving edge-style processing
- A usable dashboard rather than a notebook-only prototype

## Quick Start

```bash
cd multimodal_stress_dashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If you do not have a test video, use the built-in synthetic demo in the sidebar.

Optional facial emotion recognition:

```bash
pip install -r requirements-optional.txt
```

Without the optional dependency, the dashboard still runs and uses neutral facial context.

## CLI Demo

```bash
python run_demo.py --synthetic --out outputs/sample_report.json
```

Analyze a video:

```bash
python run_demo.py --video path/to/video.mp4 --out outputs/video_report.json
```

## Train The Lightweight Classifier

Train a Random Forest classifier on synthetic physiological and emotion features:

```bash
python train_classifier.py
```

This creates:

```text
models/stress_rf.joblib
sample_data/training_features.csv
```

Once the model file exists, both the dashboard and CLI use it for the final stress label while still showing the rule-based baseline for transparency.

## Train From UBFC-Phys

If you have UBFC-Phys subject folders with `bvp_*`, `eda_*`, and task files:

```bash
python import_ubfc_phys.py --dataset-root H:\s51_to_s56 --out sample_data/ubfc_phys_s51_s56_features.csv
python train_classifier.py --data-in sample_data/ubfc_phys_s51_s56_features.csv
```

The importer uses BVP and EDA files to create window-level features. Labels are derived from the UBFC-Phys task protocol:

- `T1` -> `Relaxed`
- `T2` -> `Mild Stress`
- `T3` -> `Elevated Physiological Stress`

This gives the project a real physiological dataset path while keeping the video-based rPPG dashboard separate.

To train a true multimodal semantic fusion model, extract facial emotion semantics from the UBFC-Phys videos and merge them with BVP/EDA features:

```bash
python extract_video_emotion_features.py --dataset-root H:\s51_to_s56 --out sample_data/ubfc_phys_emotion_features.csv --frames-per-window 1
python build_multimodal_dataset.py --physio sample_data/ubfc_phys_s51_s56_features.csv --emotion sample_data/ubfc_phys_emotion_features.csv --out sample_data/ubfc_phys_multimodal_features.csv
python train_classifier.py --data-in sample_data/ubfc_phys_multimodal_features.csv --feature-set multimodal
```

This trains the classifier on both physiological features and FER semantic probabilities.

For higher cross-subject accuracy on the small `s51-s56` subset, train a binary model:

```bash
python train_classifier.py --data-in sample_data/ubfc_phys_multimodal_features.csv --feature-set multimodal --label-mode binary
```

Binary mode maps `T1` to `Relaxed` and `T2/T3` to `Stress`. This is often more reliable than forcing `Mild Stress` vs `Elevated Physiological Stress` on a six-subject subset.

Multiple UBFC-Phys folders can be passed at once:

```bash
python import_ubfc_phys.py --dataset-root H:\s11_to_s20 H:\s51_to_s56 --out sample_data/ubfc_phys_all_features.csv
python extract_video_emotion_features.py --dataset-root H:\s11_to_s20 H:\s51_to_s56 --out sample_data/ubfc_phys_all_emotion_features.csv --frames-per-window 5
```

Current 16-subject binary multimodal model:

```text
training data: H:\s11_to_s20 + H:\s51_to_s56
windows: 240
model: XGBoost
FER frames per window: 3 for the current saved XGBoost model; 5 is recommended for the next full retraining run.
quality gate: training-reference mode, kept 240/240 rows
subject-level accuracy: 0.76
```

## Adaptive Semantic Fusion

The project also includes a Gated Fusion MLP:

```bash
python train_gated_fusion.py --data-in sample_data/ubfc_phys_s11_s20_s51_s56_multimodal_features_f3.csv
```

This model encodes physiological features and facial emotion semantics separately, then learns sample-level modality weights before final classification:

```text
physio features -> physio encoder
emotion features -> emotion encoder
both embeddings -> learned gate -> fused representation -> classifier
```

Current subject-level result:

```text
Gated Fusion MLP accuracy: 0.71
XGBoost early-fusion baseline: 0.76
```

The Gated Fusion MLP is kept for adaptive semantic fusion and explainability because it outputs physiological vs facial-emotion weights for each prediction. XGBoost remains the stronger performance baseline on the current small dataset.

The dashboard sidebar lets you choose between:

```text
Auto
Gated Fusion MLP
XGBoost Early Fusion
Random Forest Baseline
```


## Project Structure

```text
multimodal_stress_dashboard/
  app.py
  run_demo.py
  train_classifier.py
  import_ubfc_phys.py
  extract_video_emotion_features.py
  build_multimodal_dataset.py
  models/
    stress_rf.joblib
  requirements.txt
  src/
    analyzer.py
    emotion.py
    features.py
    fusion.py
    rppg.py
    synthetic.py
    video.py
  sample_data/
    synthetic_session.csv
  outputs/
    .gitkeep
```

## Output Labels

- `Relaxed`
- `Mostly Calm`
- `Mild Stress`
- `Elevated Physiological Stress`
- `Uncertain`
- `Poor Signal Quality`

The app is for wellness and technical demonstration only. It is not a medical diagnostic tool.



