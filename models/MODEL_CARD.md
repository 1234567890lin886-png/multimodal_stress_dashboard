# Stress Classifier Model Card

## Current Model

`stress_xgb.joblib` is an XGBoost classifier trained from local UBFC-Phys subsets:

```text
H:\s11_to_s20
H:\s51_to_s56
```

The converter extracts window-level features from BVP and EDA files and maps the UBFC-Phys task protocol to labels:

- `T1` -> `Relaxed`
- `T2` -> `Mild Stress`
- `T3` -> `Elevated Physiological Stress`

## Important Caveat

The current subset has sixteen subjects, so cross-subject generalization is still limited but more stable than the earlier six-subject version. The model is useful as a real-data multimodal training baseline and portfolio demonstration, not as a medical or production stress detector.

## Latest Evaluation

Subject-level split was used, meaning test subjects are unseen during training.

XGBoost early-fusion binary model (`Relaxed` vs `Stress`):

```text
subjects: 16
windows: 240
FER frames per window: 3
quality gate: training-reference mode, kept 240/240 rows
accuracy: 0.76
Relaxed recall: 0.63
Stress recall: 0.82
```

Three-class XGBoost comparison model (`Relaxed`, `Mild Stress`, `Elevated Physiological Stress`):

```text
accuracy: 0.44
```

Gated Fusion MLP adaptive semantic-fusion model:

```text
accuracy: 0.71
mean physio gate: 0.50
mean emotion gate: 0.50
```

The gated model is less accurate than XGBoost on the current small dataset, but it is useful for explaining sample-level modality weighting.

## Reproduce

```bash
python import_ubfc_phys.py --dataset-root H:\s11_to_s20 H:\s51_to_s56 --out sample_data/ubfc_phys_s11_s20_s51_s56_features.csv
python extract_video_emotion_features.py --dataset-root H:\s11_to_s20 H:\s51_to_s56 --out sample_data/ubfc_phys_s11_s20_s51_s56_emotion_features_f3.csv --frames-per-window 3
python build_multimodal_dataset.py --physio sample_data/ubfc_phys_s11_s20_s51_s56_features.csv --emotion sample_data/ubfc_phys_s11_s20_s51_s56_emotion_features_f3.csv --out sample_data/ubfc_phys_s11_s20_s51_s56_multimodal_features_f3.csv
python train_classifier.py --data-in sample_data/ubfc_phys_s11_s20_s51_s56_multimodal_features_f3.csv --feature-set multimodal --label-mode binary --model-type xgboost --quality-gate training-reference
python train_gated_fusion.py --data-in sample_data/ubfc_phys_s11_s20_s51_s56_multimodal_features_f3.csv
```

## Quality Gate

The XGBoost trainer now supports the FYP-style quality gate:

```text
signal_quality >= 0.42
snr_db >= 0.0
motion_score <= 0.18
lighting_variation <= 0.12
```

For video-rPPG training data, `training-reference` mode also removes windows with:

```text
rppg_hr_error_bpm > 15
```

The current BVP-derived training CSV already passed these thresholds, so the retrained model kept all 240 rows and retained the previous 0.76 subject-level accuracy. The gate is expected to matter more after full video-rPPG feature extraction.
