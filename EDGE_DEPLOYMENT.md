# Edge Deployment Notes

This project can be packaged as a lightweight edge AI rPPG system because the default pipeline does not require a heavy deep-learning rPPG model.

## Recommended Edge Stack

```text
Camera
  -> OpenCV capture
  -> MediaPipe FaceMesh ROI
  -> POS/CHROM rPPG
  -> online quality gate
  -> XGBoost stress classifier
  -> local dashboard / overlay
```

## Target Devices

Good starting targets:

```text
Windows laptop + webcam
Raspberry Pi 4/5 + USB camera
QuecPi + camera
Jetson Nano / Orin Nano
mini PC
Android phone camera stream
```

## CPU-Friendly Mode

Use:

```powershell
python live_webcam.py --source 0
```

This uses:

```text
OpenCV
FaceMesh
POS/CHROM
XGBoost
```

Avoid `--enable-fer` on weak CPUs because DeepFace can reduce FPS.

## Performance Tips

- Use 640x480 or 720p camera input.
- Keep face centered and stable.
- Use soft frontal lighting.
- Avoid strong backlight.
- Use `Frame stride = 6` in the Streamlit dashboard for large videos.
- Prefer local file path input over browser upload for large `.avi` files.

## Deployment-Friendly Claims

Good project description:

```text
OpenCV-based real-time rPPG vital-sign dashboard with online signal-quality gating and lightweight ML stress classification.
```

Avoid overclaiming:

```text
medical diagnosis
clinical-grade heart-rate monitoring
guaranteed stress detection
```

## Future Product Upgrades

Useful next steps:

```text
learned quality gate
late fusion model
mobile browser camera stream
Raspberry Pi installation script
Docker image
ONNX/TFLite lightweight FER
optional deep rPPG inference backend
```
