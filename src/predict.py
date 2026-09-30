"""
Command-line inference: image, folder of images, or video -> annotated output + compliance report.

Examples:
    python src/predict.py --source samples/site.jpg
    python src/predict.py --source samples/ --model models/best_int8.onnx
    python src/predict.py --source site_cam.mp4 --required helmet vest
"""

import argparse
import json
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppe_utils import check_compliance, detections_from_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image, folder, or video")
    ap.add_argument("--model", default="models/best.onnx")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--required", nargs="+", default=["helmet", "vest"])
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    model = YOLO(a.model, task="detect")
    report = []
    for r in model.predict(a.source, conf=a.conf, imgsz=640, stream=True,
                           save=True, project=a.out, name="predict", exist_ok=True,
                           verbose=False):
        _, summary = check_compliance(detections_from_result(r), a.required)
        summary["file"] = os.path.basename(r.path)
        report.append(summary)
        print(f"{summary['file']:<30} {summary['frame_status']:<10} "
              f"persons={summary['persons']} violations={summary['violations']}")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "compliance_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAnnotated outputs + compliance_report.json saved in {a.out}/")


if __name__ == "__main__":
    main()
