# Video-rPPG Training Changes

This document records the changes made after reviewing the main sources of noise in the previous training setup:

- T2/T3 task-label noise
- facial expression vs physiological disagreement
- DeepFace frame-level misclassification
- large subject-to-subject physiological variation
- train/test domain gap between sensor BVP training features and dashboard video-rPPG inference features

## What Changed

### 1. Added Video-rPPG Feature Extraction

New script:

```text
import_ubfc_phys_video_rppg.py
```

Purpose:

```text
UBFC-Phys vid_*.avi
    -> video ROI RGB series
    -> POS/CHROM rPPG
    -> HR / HRV proxy / SNR / signal quality
    -> stress-training feature CSV
```

This replaces the old assumption that model training should rely on clean sensor BVP features. The goal is to make training features match the dashboard inference path:

```text
video -> rPPG -> features -> classifier
```

### 1.1 Copied FYP ROI Logic Into This Standalone Project

The original FYP source remains unchanged. The standalone project now adapts the FYP preprocessing idea inside:

```text
src/video.py
```

The previous video signal extractor used a simple center face-like crop. It now uses:

```text
MediaPipe FaceMesh
  -> cheek ROI landmarks
  -> optional forehead landmarks
  -> convex-hull ROI mask
  -> YCrCb + HSV skin-mask refinement
  -> masked RGB mean series
```

Fallback behavior:

```text
FaceMesh unavailable or face not detected -> center ROI
```

This improves the rPPG input quality because the pulse signal is extracted from skin regions instead of background, hair, clothes, or the whole face rectangle.

### 2. Kept BVP Only for Validation

The new video-rPPG importer still reads `bvp_*.csv`, but only to compute reference columns:

```text
bvp_hr_bpm
bvp_snr_db
rppg_hr_error_bpm
```

These columns are for rPPG quality evaluation, not for video-only model inference.

### 2.1 Added FYP-style Quality Gate

The standalone project now has a reusable quality gate:

```text
src/quality_gate.py
```

It adapts the FYP decision thresholds into training-time filtering:

```text
signal_quality >= 0.42
snr_db >= 0.0
motion_score <= 0.18
lighting_variation <= 0.12
```

For real video-rPPG training data, `training-reference` mode also uses the UBFC-Phys BVP reference to remove windows with:

```text
rppg_hr_error_bpm > 15
```

This extra BVP-based filter is only for dataset construction/training. The deployed dashboard cannot use BVP because it receives only video.

### 3. Added Subject Baseline Normalization

The new video-rPPG features include subject-level T1 baseline deltas:

```text
hr_delta_bpm
hrv_delta
signal_quality_delta
```

This reduces subject-to-subject differences such as naturally high or low resting heart rate.

### 4. Added Cleaner Label Policy Options

The new importer supports:

```text
--label-policy task-binary
--label-policy task-three-class
--label-policy self-report-filtered
```

Recommended for deployment-style training:

```text
--label-policy task-binary
```

Recommended for cleaner research experiments:

```text
--label-policy self-report-filtered
```

`self-report-filtered` keeps only clearer samples:

```text
T1 with low anxiety -> Relaxed
T2/T3 with higher anxiety -> Stress
ambiguous samples -> dropped
```

### 5. Increased FER Sampling Recommendation

`extract_video_emotion_features.py` now defaults to:

```text
--frames-per-window 5
```

The current saved XGBoost model was trained with 3 FER frames per window. The next full retraining run should use 5 frames per window for more stable facial-emotion semantics.

### 6. Preserved Existing Baselines

The existing models are kept:

```text
models/stress_xgb.joblib
models/gated_fusion_mlp.joblib
models/stress_rf.joblib
```

The dashboard can still choose between:

```text
Gated Fusion MLP
XGBoost Early Fusion
Random Forest Baseline
```

## Recommended New Full Training Flow

Run these from:

```powershell
cd D:\project\multimodal_stress_dashboard
```

### Step 1: Extract video-rPPG physiological features

For all current UBFC-Phys folders:

```powershell
& 'D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe' import_ubfc_phys_video_rppg.py `
  --dataset-root H:\s11_to_s20 H:\s51_to_s56 `
  --out sample_data\ubfc_phys_video_rppg_features.csv `
  --video-stride 6 `
  --label-policy task-binary `
  --quality-gate training-reference
```

For stricter labels:

```powershell
& 'D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe' import_ubfc_phys_video_rppg.py `
  --dataset-root H:\s11_to_s20 H:\s51_to_s56 `
  --out sample_data\ubfc_phys_video_rppg_features_self_report.csv `
  --video-stride 6 `
  --label-policy self-report-filtered `
  --quality-gate training-reference
```

### Step 2: Extract facial emotion semantic features

```powershell
& 'D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe' extract_video_emotion_features.py `
  --dataset-root H:\s11_to_s20 H:\s51_to_s56 `
  --out sample_data\ubfc_phys_emotion_features_f5.csv `
  --frames-per-window 5
```

### Step 3: Merge video-rPPG and emotion features

```powershell
& 'D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe' build_multimodal_dataset.py `
  --physio sample_data\ubfc_phys_video_rppg_features.csv `
  --emotion sample_data\ubfc_phys_emotion_features_f5.csv `
  --out sample_data\ubfc_phys_video_rppg_multimodal_features_f5.csv
```

### Step 4: Train XGBoost video-rPPG model

```powershell
& 'D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe' train_classifier.py `
  --data-in sample_data\ubfc_phys_video_rppg_multimodal_features_f5.csv `
  --feature-set multimodal `
  --label-mode binary `
  --model-type xgboost `
  --quality-gate training-reference `
  --model-out models\stress_xgb_video_rppg.joblib
```

### Step 5: Train Gated Fusion MLP video-rPPG model

```powershell
& 'D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe' train_gated_fusion.py `
  --data-in sample_data\ubfc_phys_video_rppg_multimodal_features_f5.csv `
  --model-out models\gated_fusion_mlp_video_rppg.joblib `
  --weights-out models\gated_fusion_mlp_video_rppg.weights.h5 `
  --metrics-out models\gated_fusion_video_rppg_metrics.json
```

## Current Status

Implemented:

- video-rPPG feature extraction script
- FaceMesh cheek ROI + skin-mask extraction copied/adapted from the FYP logic
- reusable FYP-style quality gate for importer/trainer
- BVP-as-reference validation columns
- subject baseline deltas
- label-policy controls
- FER default increased to 5 frames per window
- documented full retraining flow

Not yet completed:

- full video-rPPG extraction over all raw AVI files
- final video-rPPG-trained model replacement

Reason:

The raw UBFC-Phys `.avi` files are very large, and OpenCV decoding is slow on this machine. The full video-rPPG extraction should be run as an offline job. The previous saved production model remains the XGBoost model trained on BVP-derived physiological features plus FER semantics.

## Why This Matters

The previous model had this mismatch:

```text
training: sensor BVP -> HR/HRV features
dashboard: video -> rPPG -> HR/HRV features
```

The new flow fixes that by training on:

```text
training: video -> rPPG -> HR/HRV features
dashboard: video -> rPPG -> HR/HRV features
```

This makes the deployed dashboard model conceptually more consistent and closer to the FYP goal of camera-based stress monitoring.

## Further AI/Model Options To Improve Accuracy

Recommended order:

1. **XGBoost on video-rPPG + FER features**

   Keep XGBoost as the strongest current baseline, but retrain it on video-rPPG features extracted by the new FaceMesh ROI path. This directly targets the largest current weakness: the old model was trained from sensor BVP features.

2. **Quality-filtered training**

   Use BVP only during training-time validation to remove bad windows:

   ```text
   high rPPG_hr_error_bpm
   low snr_db
   high motion_score
   high lighting_variation
   ```

   This can improve classifier accuracy because the model sees fewer mislabeled/noisy physiological features.

3. **Late-fusion XGBoost**

   Train:

   ```text
   rPPG model -> stress probability
   FER model -> stress probability
   meta model -> final stress probability
   ```

   This is useful because facial emotion and physiological stress often disagree. Late fusion lets the system learn when to trust one modality more.

4. **Gated Fusion MLP**

   Keep this for explainability. It may not beat XGBoost on small data, but it outputs dynamic modality weights:

   ```text
   physio_weight
   emotion_weight
   ```

   That is valuable for explaining "why" a prediction was made.

5. **Deep rPPG models**

   Later, compare hand-crafted POS/CHROM features with deep video-rPPG models:

   ```text
   DeepPhys
   PhysNet
   TS-CAN
   EfficientPhys
   ```

   These need more compute and careful preprocessing, so they are a second-stage improvement rather than the one-week baseline.
