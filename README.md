# OpenCV rPPG Live Vitals Dashboard

OpenCV-based webcam and video dashboard for camera-only vital-sign monitoring.

The project turns ordinary face video into:

```text
heart rate
HRV proxy
stress prediction
facial emotion context
signal quality
motion / lighting robustness indicators
```

It is designed as a deployable dashboard rather than paper-only code: local video paths, large AVI support, pre-trained lightweight models, Streamlit reports, and an OpenCV live webcam mode are included.

## What It Is

```text
Computer Vision + Signal Processing + Machine Learning + Multimodal Fusion
```

- **Computer Vision:** FaceMesh cheek ROI, skin-mask refinement, frame quality monitoring.
- **Signal Processing:** POS/CHROM rPPG, FFT HR estimation, HRV proxy, SNR.
- **Machine Learning:** XGBoost, Random Forest baseline, Gated Fusion MLP.
- **Multimodal Fusion:** rPPG physiology + FER emotion semantics.

## Product Features

- Streamlit dashboard for synthetic demos, uploaded videos, and local video paths.
- OpenCV webcam live mode for real-time HR/stress overlays.
- Local-path analysis for large `.avi` files without browser upload failures.
- Online quality gate that uses only video-available features.
- Optional DeepFace FER for facial emotion context.
- Pre-trained XGBoost and Gated Fusion models included.
- Reproducible UBFC-Phys import, alignment, and ablation scripts.

## Quick Start

```powershell
Set-Location D:\project\multimodal_stress_dashboard
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\Scripts\streamlit.exe" run app.py
```

Open:

```text
http://localhost:8501
```

Recommended dashboard settings:

```text
Input: Local video path
Model: XGBoost Online Gate
Frame stride: 6
Max frames: 900-1200
```

For large UBFC `.avi` files, use `Local video path` instead of uploading through the browser.

Example:

```text
H:\s51_to_s56\s54\vid_s54_T2.avi
```

## OpenCV Webcam Live Mode

Run the real-time OpenCV overlay:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" live_webcam.py --source 0
```

Enable facial emotion recognition during live mode:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" live_webcam.py --source 0 --enable-fer
```

Use a video file as the source:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" live_webcam.py --source "H:\s51_to_s56\s54\vid_s54_T2.avi"
```

Press `q` to quit.

Live mode displays:

```text
HR
stress label
confidence
emotion
signal quality
SNR
motion score
lighting score
FPS
```

## CLI Video Analysis

Synthetic demo:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" run_demo.py --synthetic --model models\stress_xgb_video_rppg_online_qg.joblib --out outputs\online_xgb_demo.json
```

Analyze a local video:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" run_demo.py --video "H:\s51_to_s56\s54\vid_s54_T2.avi" --model models\stress_xgb_video_rppg_online_qg.joblib --out outputs\video_report.json --stride 6 --max-frames 1200
```

## Current Models

Recommended deployment-style model:

```text
models/stress_xgb_video_rppg_online_qg.joblib
```

Why this model:

```text
Uses online quality gate only
Does not use BVP ground truth for filtering
Uses aligned 5-frame FER semantics
Works with webcam/video-only inference
```

Other models:

```text
models/stress_xgb_video_rppg.joblib        # BVP-reference-gated experiment
models/gated_fusion_mlp.joblib             # adaptive modality weights
models/stress_rf.joblib                    # baseline
models/ablation_xgb_multimodal.joblib      # ablation model
```

## Evaluation Snapshot

Online-gate setting, no BVP ground-truth filtering:

```text
Training data: H:\s11_to_s20 + H:\s51_to_s56
External test: H:\s1_to_s10
FER coverage: 100%
External XGBoost accuracy: 0.8267
```

Online-gate ablation:

```text
rPPG-only external accuracy: 0.820
FER-only external accuracy: 0.687
Multimodal external accuracy: 0.867
```

Reports:

```text
models/ONLINE_GATE_EXPERIMENT.md
models/ABLATION_EXPERIMENT_F5_ALIGNED_ONLINE_QG.md
SIGNAL_ALIGNMENT.md
```

## Training And Experiments

Extract video-rPPG features with online gate:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" import_ubfc_phys_video_rppg.py --dataset-root H:\s11_to_s20 H:\s51_to_s56 --out sample_data\ubfc_phys_video_rppg_features_online_qg.csv --window-sec 60 --step-sec 30 --video-stride 6 --label-policy task-binary --quality-gate fyp --no-bvp-reference
```

Extract 5-frame FER semantics:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" extract_video_emotion_features.py --dataset-root H:\s11_to_s20 H:\s51_to_s56 --out sample_data\ubfc_phys_s11_s20_s51_s56_emotion_features_f5.csv --frames-per-window 5
```

Merge with window-index alignment:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" build_multimodal_dataset.py --physio sample_data\ubfc_phys_video_rppg_features_online_qg.csv --emotion sample_data\ubfc_phys_s11_s20_s51_s56_emotion_features_f5.csv --out sample_data\ubfc_phys_video_rppg_multimodal_features_f5_aligned_online_qg.csv --alignment window-index --step-sec 30
```

Train XGBoost:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" train_classifier.py --data-in sample_data\ubfc_phys_video_rppg_multimodal_features_f5_aligned_online_qg.csv --feature-set multimodal --label-mode binary --model-type xgboost --quality-gate fyp --model-out models\stress_xgb_video_rppg_online_qg.joblib
```

Run ablation:

```powershell
& "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe" run_ablation_experiment.py --train-data sample_data\ubfc_phys_video_rppg_multimodal_features_f5_aligned_online_qg.csv --external-data sample_data\ubfc_phys_s1_s10_video_rppg_multimodal_features_f5_aligned_online_qg.csv --out-csv outputs\ablation_f5_aligned_online_qg_results.csv --out-md models\ABLATION_EXPERIMENT_F5_ALIGNED_ONLINE_QG.md
```

## Edge AI Direction

This project is intentionally lightweight:

```text
OpenCV
MediaPipe FaceMesh
POS/CHROM rPPG
XGBoost
optional DeepFace FER
```

That makes it suitable for edge-device experiments:

```text
Raspberry Pi
QuecPi
Jetson Nano
USB webcam
SBC camera
mobile camera stream
```

For edge deployment, use the OpenCV live mode first and disable FER if CPU is limited:

```powershell
python live_webcam.py --source 0
```

## Project Structure

```text
app.py                          # Streamlit dashboard
live_webcam.py                  # OpenCV webcam live mode
run_demo.py                     # CLI video/synthetic analysis
src/video.py                    # FaceMesh ROI and RGB signal extraction
src/rppg.py                     # POS/CHROM rPPG and HR estimation
src/ml_model.py                 # XGBoost / Gated MLP prediction
src/quality_gate.py             # online and reference quality gates
import_ubfc_phys_video_rppg.py  # video-rPPG dataset importer
extract_video_emotion_features.py
build_multimodal_dataset.py
run_ablation_experiment.py
models/
sample_data/
```

## Safety Notice

This is a wellness and technical demonstration system. It is not a medical device and must not be used for diagnosis, treatment, or emergency monitoring.
