# Ablation Experiment

Task: binary `Relaxed` vs `Stress` classification.

## Data

- Training/internal data: `sample_data\ubfc_phys_video_rppg_multimodal_features_f5_aligned_qg.csv`
- External test data: `sample_data\ubfc_phys_s1_s10_video_rppg_multimodal_features_f5_aligned_qg.csv`
- Internal split train subjects: `['s19', 's20', 's13', 's56', 's15', 's18', 's51', 's53', 's14', 's17']`
- Internal split test subjects: `['s11', 's12', 's16', 's55', 's54', 's52']`
- Training FER coverage: 159/159 (100.0%)
- External FER coverage: 97/97 (100.0%)

## Results

| Feature Set | Internal Accuracy | Internal Macro F1 | External Accuracy | External Macro F1 |
|---|---:|---:|---:|---:|
| `rppg_only` | 0.918 | 0.910 | 0.825 | 0.821 |
| `fer_only` | 0.607 | 0.588 | 0.722 | 0.700 |
| `multimodal` | 0.820 | 0.812 | 0.845 | 0.838 |

## Interpretation Guide

- `rppg_only` measures how much the quality-gated physiological signal contributes.
- `fer_only` measures whether facial emotion semantics alone carry stress information.
- `multimodal` measures whether combining rPPG and FER improves over rPPG alone.

Confusion matrices are saved in the CSV in `[Relaxed, Stress]` label order.