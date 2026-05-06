# Fusion Experiment Summary

Dataset:

```text
H:\s11_to_s20 + H:\s51_to_s56
16 subjects
240 aligned 60-second windows
FER frames per window: 3
Quality gate: training-reference, kept 240/240 rows
Evaluation: subject-level split
Task: Relaxed vs Stress
```

## Results

| Model | Fusion Type | Accuracy | Notes |
|---|---:|---:|---|
| XGBoost | Early fusion | 0.76 | Retrained with FYP-style quality gate; strongest current performance baseline |
| Gated Fusion MLP | Adaptive semantic fusion | 0.71 | Provides learned physiological/emotion modality weights |
| Random Forest | Early fusion | 0.72 | Previous baseline with 1-frame FER |

## Interpretation

XGBoost performs best on the current small tabular dataset. The Gated Fusion MLP is kept because it implements the project's semantic-fusion idea: physiological and facial-emotion features are encoded separately, then combined through a learned gate that exposes modality weights for each prediction.
