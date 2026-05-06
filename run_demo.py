from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analyzer import analyze_synthetic, analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal stress analysis.")
    parser.add_argument("--synthetic", action="store_true", help="Run the built-in synthetic demo.")
    parser.add_argument("--video", type=str, help="Path to a video file.")
    parser.add_argument("--out", type=str, default="outputs/report.json", help="Output JSON path.")
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--model", type=str, help="Optional model bundle path.")
    args = parser.parse_args()

    if not args.synthetic and not args.video:
        raise SystemExit("Use --synthetic or --video path/to/file.mp4")

    report = (
        analyze_synthetic(model_path=args.model)
        if args.synthetic
        else analyze_video(args.video, args.max_frames, args.stride, model_path=args.model)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print(f"Source: {report.source}")
    print(f"Label: {report.decision.label}")
    print(f"Confidence: {report.decision.confidence:.2f}")
    if report.model_prediction is not None:
        print(f"Model: {report.model_prediction.model_name}")
        print(f"Training source: {report.model_prediction.training_source}")
        if report.model_prediction.physio_weight is not None:
            print(f"Physio weight: {report.model_prediction.physio_weight:.2f}")
            print(f"Emotion weight: {report.model_prediction.emotion_weight:.2f}")
        print(f"Rule baseline: {report.rule_decision.label}")
    else:
        print("Model: not found; used rule-based fusion")
    print(f"HR: {report.features.hr_bpm}")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
