"""
PPE Compliance Checker — Gradio web app.

Usage:
    python app/app.py                       # uses models/best.onnx (falls back to best.pt)
    python app/app.py --model models/best_int8.onnx
    python app/app.py --share               # public link (e.g. from Colab)
"""

import argparse
import os
import sys
import tempfile
import time

import cv2
import gradio as gr
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ppe_utils import check_compliance, detections_from_result  # noqa: E402

GEAR_OPTIONS = ["helmet", "vest", "gloves", "boots", "goggles"]


def pick_model(path=None):
    candidates = [path] if path else [
        os.path.join(ROOT, "models", "best.onnx"),
        os.path.join(ROOT, "models", "best.pt"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            print(f"Loading model: {p}")
            return YOLO(p, task="detect"), os.path.basename(p)
    raise FileNotFoundError("No model found. Train with the notebook first "
                            "(it saves models/best.pt and models/best.onnx).")


def annotate(model, frame_bgr, conf, required):
    """Run detection + compliance on one BGR frame. Returns (annotated_bgr, summary, ms)."""
    t0 = time.perf_counter()
    r = model.predict(frame_bgr, conf=conf, imgsz=640, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    persons, summary = check_compliance(detections_from_result(r), required)

    im = r.plot(line_width=2)
    for s in persons:
        x1, y1, x2, y2 = map(int, s.box)
        color = (0, 200, 0) if s.compliant else (0, 0, 255)
        label = "OK" if s.compliant else "MISSING: " + ", ".join(s.missing)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 4)
        cv2.putText(im, label, (x1, min(y2 + 24, im.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    status = summary["frame_status"]
    banner = {"COMPLIANT": (0, 160, 0), "VIOLATION": (0, 0, 220)}.get(status, (90, 90, 90))
    cv2.rectangle(im, (0, 0), (im.shape[1], 42), banner, -1)
    cv2.putText(im, f"{status} | persons: {summary['persons']}  violations: {summary['violations']}",
                (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im, summary, ms


def build_ui(model, model_name):
    def run_image(img_rgb, conf, required):
        if img_rgb is None:
            return None, "Upload an image first."
        required = required or ["helmet"]
        im, s, ms = annotate(model, img_rgb[:, :, ::-1].copy(), conf, required)
        report = (f"### Frame status: **{s['frame_status']}**\n"
                  f"- Persons detected: {s['persons']}\n"
                  f"- Compliant: {s['compliant']}\n"
                  f"- Violations: {s['violations']}\n"
                  f"- Required PPE: {', '.join(required)}\n"
                  f"- Inference: {ms:.1f} ms ({model_name})")
        return im[:, :, ::-1], report

    def run_video(video_path, conf, required, max_frames):
        if not video_path:
            return None, "Upload a video first."
        required = required or ["helmet"]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(tempfile.mkdtemp(), "ppe_result.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = viol = 0
        total_ms = 0.0
        while n < int(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            im, s, ms = annotate(model, frame, conf, required)
            writer.write(im)
            n += 1
            total_ms += ms
            viol += s["frame_status"] == "VIOLATION"
        cap.release(); writer.release()
        if n == 0:
            return None, "Could not read frames from this video."
        report = (f"### Video summary\n"
                  f"- Frames processed: {n}\n"
                  f"- Frames with violations: {viol} ({100 * viol / n:.0f}%)\n"
                  f"- Avg inference: {total_ms / n:.1f} ms/frame ({model_name})")
        return out_path, report

    with gr.Blocks(title="PPE Compliance Checker") as demo:
        gr.Markdown("# 🦺 PPE Compliance Checker\nDetects workers and flags missing "
                    "safety gear using a fine-tuned YOLO11 model.")
        with gr.Row():
            conf = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence threshold")
            required = gr.CheckboxGroup(GEAR_OPTIONS, value=["helmet", "vest"], label="Required PPE")
        with gr.Tab("Image"):
            with gr.Row():
                inp = gr.Image(type="numpy", label="Site image")
                out = gr.Image(type="numpy", label="Result")
            rep = gr.Markdown()
            gr.Button("Check compliance", variant="primary").click(
                run_image, [inp, conf, required], [out, rep])
        with gr.Tab("Video"):
            with gr.Row():
                vin = gr.Video(label="Site video")
                vout = gr.Video(label="Result")
            max_frames = gr.Slider(30, 900, value=300, step=30, label="Max frames to process")
            vrep = gr.Markdown()
            gr.Button("Process video", variant="primary").click(
                run_video, [vin, conf, required, max_frames], [vout, vrep])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to .onnx or .pt model")
    ap.add_argument("--share", action="store_true", help="create a public share link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    mdl, name = pick_model(args.model)
    build_ui(mdl, name).launch(share=args.share, server_port=args.port)
