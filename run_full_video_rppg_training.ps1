$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\project\multimodal_stress_dashboard"
$Python = "D:\vs\SharedcomponentsTtoolSdk\Python39_64\python.exe"
$LogDir = Join-Path $ProjectRoot "outputs"
$LogPath = Join-Path $LogDir "full_video_rppg_training.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Start-Transcript -Path $LogPath -Force

Write-Host "Step 1/3: Extract video-rPPG features with FaceMesh ROI and quality gate"
& $Python import_ubfc_phys_video_rppg.py `
  --dataset-root H:\s11_to_s20 H:\s51_to_s56 `
  --out sample_data\ubfc_phys_video_rppg_features_qg.csv `
  --window-sec 60 `
  --step-sec 30 `
  --video-stride 6 `
  --roi-refresh-frames 1 `
  --label-policy task-binary `
  --quality-gate training-reference

Write-Host "Step 2/3: Merge video-rPPG physiology with existing FER semantic features"
& $Python build_multimodal_dataset.py `
  --physio sample_data\ubfc_phys_video_rppg_features_qg.csv `
  --emotion sample_data\ubfc_phys_s11_s20_s51_s56_emotion_features_f3.csv `
  --out sample_data\ubfc_phys_video_rppg_multimodal_features_f3_qg.csv

Write-Host "Step 3/3: Train XGBoost on video-rPPG multimodal features"
& $Python train_classifier.py `
  --data-in sample_data\ubfc_phys_video_rppg_multimodal_features_f3_qg.csv `
  --feature-set multimodal `
  --label-mode binary `
  --model-type xgboost `
  --quality-gate training-reference `
  --model-out models\stress_xgb_video_rppg.joblib `
  --data-out sample_data\ubfc_phys_video_rppg_multimodal_features_f3_qg_training.csv

Write-Host "Done. Log saved to $LogPath"
Stop-Transcript
