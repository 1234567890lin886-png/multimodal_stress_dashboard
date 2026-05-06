# Ablation Experiment

Task: binary `Relaxed` vs `Stress` classification.

## Data

- Training/internal data: `sample_data\ubfc_phys_video_rppg_multimodal_features_f5_aligned_online_qg.csv`
- External test data: `sample_data\ubfc_phys_s1_s10_video_rppg_multimodal_features_f5_aligned_online_qg.csv`
- Internal split train subjects: `['s19', 's20', 's13', 's56', 's15', 's18', 's51', 's53', 's14', 's17']`
- Internal split test subjects: `['s11', 's12', 's16', 's55', 's54', 's52']`
- Training FER coverage: 240/240 (100.0%)
- External FER coverage: 150/150 (100.0%)

## Results

| Feature Set | Internal Accuracy | Internal Macro F1 | External Accuracy | External Macro F1 |
|---|---:|---:|---:|---:|
| `rppg_only` | 0.867 | 0.841 | 0.820 | 0.800 |
| `fer_only` | 0.611 | 0.551 | 0.687 | 0.659 |
| `multimodal` | 0.767 | 0.740 | 0.867 | 0.851 |

## Interpretation Guide

- `rppg_only` measures how much the quality-gated physiological signal contributes.
- `fer_only` measures whether facial emotion semantics alone carry stress information.
- `multimodal` measures whether combining rPPG and FER improves over rPPG alone.

Confusion matrices are saved in the CSV in `[Relaxed, Stress]` label order.