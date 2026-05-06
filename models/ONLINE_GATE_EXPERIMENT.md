# Online Quality Gate Experiment

This experiment removes BVP-reference filtering from both training and external testing.

## Goal

Previous video-rPPG experiments used `training-reference` quality gating:

```text
rppg_hr_error_bpm <= 15
```

That is useful for offline dataset cleaning, but it is not available in deployment because a real dashboard only receives video.

This experiment uses online quality gating only:

```text
signal_quality >= 0.42
snr_db >= 0.0
motion_score <= 0.18
lighting_variation <= 0.12
```

BVP reference columns were not computed or saved in the online-gate datasets.

## Data

Training source:

```text
H:\s11_to_s20
H:\s51_to_s56
```

External test source:

```text
H:\s1_to_s10
```

Window setup:

```text
window_sec = 60
step_sec = 30
video_stride = 6
FER frames per window = 5
alignment = window-index alignment
```

## Online Gate Coverage

Training set:

```text
raw windows: 240
kept by online gate: 240
rejected: 0
FER coverage after alignment: 240/240
```

External test set:

```text
raw windows: 150
kept by online gate: 150
rejected: 0
FER coverage after alignment: 150/150
```

## Main XGBoost Result

Model:

```text
models\stress_xgb_video_rppg_online_qg.joblib
```

This model follows the standard `train_classifier.py` subject split, so it is trained on the internal training subjects and tested on held-out internal subjects.

Internal subject split:

```text
accuracy: 0.77
Relaxed recall: 0.67
Stress recall: 0.82
```

External `s1-s10` test with the saved split-trained model:

```text
rows: 150
accuracy: 0.8267
Relaxed F1: 0.73
Stress F1: 0.87
confusion matrix [Relaxed, Stress]:
[[36, 14],
 [12, 88]]
```

Predictions:

```text
outputs\s1_s10_external_test_predictions_online_qg.csv
```

## Ablation Result

For external ablation, each feature-set model is trained on all 16 training subjects and tested on the 10 external subjects.

Report:

```text
models\ABLATION_EXPERIMENT_F5_ALIGNED_ONLINE_QG.md
```

Results:

| Feature Set | Internal Accuracy | Internal Macro F1 | External Accuracy | External Macro F1 |
|---|---:|---:|---:|---:|
| rPPG-only | 0.867 | 0.841 | 0.820 | 0.800 |
| FER-only | 0.611 | 0.551 | 0.687 | 0.659 |
| Multimodal | 0.767 | 0.740 | 0.867 | 0.851 |

External confusion matrices in `[Relaxed, Stress]` order:

```text
rPPG-only:
[[38, 12],
 [15, 85]]

FER-only:
[[30, 20],
 [27, 73]]

Multimodal:
[[41,  9],
 [11, 89]]
```

## Interpretation

With online-only quality gating, rPPG remains the strongest single modality. FER alone is weaker but still above chance, and multimodal fusion improves external generalization:

```text
rPPG-only external accuracy: 0.820
Multimodal external accuracy: 0.867
Gain: +0.047
```

This is the more deployment-realistic result because it does not use BVP ground truth for window filtering.
