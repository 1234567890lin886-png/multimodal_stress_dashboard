# Signal Alignment Notes

This project uses three time-related signal sources:

```text
1. Video-rPPG physiological features
2. UBFC-Phys BVP reference signal
3. Facial expression recognition (FER) semantic features
```

The goal of alignment is to make sure all modalities describe the same time window before training or evaluating the stress classifier.

## 1. Video-rPPG Windowing

Video frames are decoded from each `vid_*.avi` file. The video pipeline extracts:

```text
video frame
  -> FaceMesh cheek ROI
  -> skin-mask RGB mean
  -> POS/CHROM rPPG
  -> HR / HRV proxy / SNR / signal quality
```

For training and testing, features are summarized in sliding windows:

```text
window_sec = 60
step_sec = 30
video_stride = 6
```

Because the source video FPS is approximately `35.138`, after `stride=6` the effective FPS is not exactly an integer:

```text
35.138 / 6 = 5.8563 FPS
```

Therefore, rPPG windows can start at times such as:

```text
0.000
29.882
59.764
89.647
119.529
```

These are effectively the same as `0, 30, 60, 90, 120` second windows, with small timestamp drift from FPS rounding.

## 2. BVP Reference Alignment

The UBFC-Phys BVP signal is not used as an input at deployment time. It is used only during dataset construction to evaluate whether the video-rPPG HR estimate is reliable.

For each video-rPPG window:

```text
rPPG window_start_sec / window_end_sec
  -> same time interval in bvp_*.csv
  -> reference BVP HR
  -> rppg_hr_error_bpm = abs(rPPG HR - BVP HR)
```

This allows the training-time quality gate to remove windows where the camera-based HR estimate is clearly wrong.

## 3. FER Windowing

FER features are extracted from the same `vid_*.avi` files. Each FER window samples several frames inside the window and averages DeepFace emotion probabilities:

```text
video window
  -> sample N frames
  -> DeepFace emotion probabilities
  -> mean angry/disgust/fear/happy/sad/surprise/neutral
  -> emotion_stress
```

The current recommended setting is:

```text
frames_per_window = 5
```

This is more stable than 3 frames because it reduces the impact of a single bad facial-expression frame.

## 4. Why Exact Timestamp Matching Failed

The old merge used exact keys:

```text
subject + task + window_start_sec + window_end_sec
```

That caused many missed FER matches:

```text
rPPG: 29.882 - 89.882
FER:  30.000 - 90.000
```

These are the same semantic window, but exact floating-point matching treats them as different. This is why many rows had no FER features even though FER extraction worked.

## 5. Current Alignment Method

The current merge uses a window index:

```text
window_id = round(window_start_sec / step_sec)
```

Then it merges on:

```text
subject + task + window_id
```

Example:

```text
rPPG start 29.882 / 30 = 0.996 -> window_id 1
FER  start 30.000 / 30 = 1.000 -> window_id 1
```

This correctly aligns the modalities while still enforcing a safety check:

```text
abs(rPPG_window_start_sec - FER_window_start_sec) <= 1.0 second
```

If the start-time difference is larger than the tolerance, the FER values are not used for that row.

## 6. Label Alignment

Labels come from the UBFC-Phys task protocol:

```text
T1 -> Relaxed
T2 -> Stress
T3 -> Stress
```

The label is assigned to every window from that task. This is a protocol-level label, not a second-by-second self-report label, so it can contain noise. This is one reason binary classification is more reliable than three-class stress classification here.

## 7. Quality Gate Position

The quality gate is applied after video-rPPG feature extraction:

```text
video-rPPG features
  -> BVP reference HR comparison
  -> quality gate
  -> aligned with FER features
  -> model training/evaluation
```

The gate keeps windows that satisfy:

```text
signal_quality >= 0.42
snr_db >= 0.0
motion_score <= 0.18
lighting_variation <= 0.12
rppg_hr_error_bpm <= 15   # training/reference datasets only
```

The deployed dashboard cannot use `rppg_hr_error_bpm` because it does not have BVP. That column is only for dataset cleaning and evaluation.

## 8. Why This Matters For Multimodal AI

After alignment, each row represents:

```text
same subject
same task
same 60-second time window
same protocol label
quality-gated rPPG features
averaged FER semantic features
```

This makes the multimodal fusion fairer. Without this fix, the model was mostly learning from physiological features because FER values were missing for many rows due to timestamp mismatch.
