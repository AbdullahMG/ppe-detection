"""
Shared utilities for the PPE Compliance Detection project.

- Compliance logic: assigns detected PPE items to detected persons and decides
  whether each person is compliant.
- Per-image scoring: compares predictions to ground truth to pick success and
  failure examples.
- ONNX INT8 static quantization with a calibration reader that matches YOLO
  preprocessing (letterbox to 640, RGB, /255, NCHW).
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Compliance logic
# ---------------------------------------------------------------------------

PERSON_CLASS = "person"
# Positive gear classes and the matching "missing gear" classes in Construction-PPE
POSITIVE_TO_NEGATIVE = {
    "helmet": "no_helmet",
    "vest": None,          # dataset has no "no_vest" class -> absence means missing
    "gloves": "no_gloves",
    "boots": "no_boots",
    "goggles": "no_goggle",
}
DEFAULT_REQUIRED = ("helmet", "vest")


@dataclass
class Detection:
    name: str
    conf: float
    box: tuple  # (x1, y1, x2, y2) in pixels


@dataclass
class PersonStatus:
    box: tuple
    gear: list = field(default_factory=list)
    missing: list = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return not self.missing


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _inside_ratio(item, person):
    """Fraction of the item's box that lies inside the person's box."""
    a = _area(item)
    return _inter(item, person) / a if a > 0 else 0.0


def check_compliance(detections, required=DEFAULT_REQUIRED, inside_thr=0.5):
    """
    Assign each PPE item to the person box that contains most of it, then
    decide compliance per person.

    A person is NON-compliant if:
      * a "no_<gear>" box is assigned to them for a required gear, or
      * a required gear item is not detected on them at all.

    Returns (list[PersonStatus], summary dict).
    """
    required = tuple(r.lower() for r in required)
    persons = [d for d in detections if d.name.lower() == PERSON_CLASS]
    items = [d for d in detections if d.name.lower() != PERSON_CLASS]

    statuses = [PersonStatus(box=p.box) for p in persons]
    unassigned_violations = []

    for it in items:
        best_i, best_r = -1, 0.0
        for i, p in enumerate(persons):
            r = _inside_ratio(it.box, p.box)
            if r > best_r:
                best_i, best_r = i, r
        if best_i >= 0 and best_r >= inside_thr:
            statuses[best_i].gear.append(it.name.lower())
        elif it.name.lower().startswith("no_"):
            unassigned_violations.append(it.name.lower())

    for s in statuses:
        for req in required:
            neg = POSITIVE_TO_NEGATIVE.get(req)
            if (neg and neg in s.gear) or req not in s.gear:
                s.missing.append(req)

    n_ok = sum(s.compliant for s in statuses)
    summary = {
        "persons": len(statuses),
        "compliant": n_ok,
        "violations": len(statuses) - n_ok,
        "unassigned_violation_boxes": unassigned_violations,
        "frame_status": "VIOLATION"
        if (len(statuses) - n_ok) > 0 or unassigned_violations
        else ("COMPLIANT" if statuses else "NO PERSON"),
    }
    return statuses, summary


def detections_from_result(result):
    """Convert an Ultralytics Results object into a list[Detection]."""
    names = result.names
    out = []
    if result.boxes is None:
        return out
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    for b, c, s in zip(xyxy, cls, conf):
        out.append(Detection(name=names[int(c)], conf=float(s), box=tuple(map(float, b))))
    return out


# ---------------------------------------------------------------------------
# Per-image scoring (for choosing success / failure examples)
# ---------------------------------------------------------------------------

def load_yolo_labels(label_path, img_w, img_h):
    """Read a YOLO txt label file -> list of (cls_id, (x1,y1,x2,y2))."""
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, xc, yc, w, h = int(p[0]), *map(float, p[1:5])
            x1, y1 = (xc - w / 2) * img_w, (yc - h / 2) * img_h
            x2, y2 = (xc + w / 2) * img_w, (yc + h / 2) * img_h
            gts.append((c, (x1, y1, x2, y2)))
    return gts


def image_f1(preds, gts, iou_thr=0.5):
    """
    Greedy class-aware matching.
    preds: list of (cls_id, conf, box); gts: list of (cls_id, box).
    Returns dict with tp, fp, fn, f1.
    """
    preds = sorted(preds, key=lambda x: -x[1])
    used = set()
    tp = 0
    for c, _, b in preds:
        best_j, best_iou = -1, iou_thr
        for j, (gc, gb) in enumerate(gts):
            if j in used or gc != c:
                continue
            v = iou(b, gb)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            used.add(best_j)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}


# ---------------------------------------------------------------------------
# ONNX INT8 static quantization
# ---------------------------------------------------------------------------

def letterbox(img_bgr, size=640, color=114):
    """Resize with unchanged aspect ratio and pad to size x size (like YOLO)."""
    import cv2

    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess_for_onnx(img_bgr, size=640):
    x = letterbox(img_bgr, size)[:, :, ::-1]            # BGR -> RGB
    x = x.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, 0..1
    return np.ascontiguousarray(x[None])                 # add batch dim


def _make_calib_reader(image_dir, input_name, n=100, size=640, seed=0):
    import cv2
    from onnxruntime.quantization import CalibrationDataReader

    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(image_dir, ext))
    )
    random.Random(seed).shuffle(paths)
    paths = paths[:n]

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(paths)

        def get_next(self):
            p = next(self._it, None)
            if p is None:
                return None
            img = cv2.imread(p)
            return {input_name: preprocess_for_onnx(img, size)}

    return _Reader(), len(paths)


def quantize_onnx_int8(fp32_path, int8_path, calib_image_dir, n_calib=100,
                       size=640, exclude_substrings=("/model.23/",)):
    """
    Static INT8 (QDQ) quantization with ONNX Runtime.

    exclude_substrings: node-name fragments to keep in FP32. For YOLO11n/YOLOv8n
    the detection head is module index 23; keeping it in FP32 protects box
    regression accuracy at almost no speed cost.

    The Ultralytics metadata (class names, stride, imgsz, task) is copied into
    the INT8 model so `YOLO("best_int8.onnx")` can load and validate it.
    """
    import onnx
    from onnxruntime.quantization import (QuantFormat, QuantType,
                                          quantize_static)

    src = onnx.load(fp32_path)
    input_name = src.graph.input[0].name

    prep_path = fp32_path.replace(".onnx", "_prep.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(fp32_path, prep_path, skip_symbolic_shape=True)
    except Exception as e:  # pre-processing is optional
        print(f"[quantize] pre-process skipped ({e}); using original model")
        prep_path = fp32_path

    nodes = [n.name for n in onnx.load(prep_path).graph.node]
    exclude = [n for n in nodes if any(s in n for s in exclude_substrings)]

    reader, used = _make_calib_reader(calib_image_dir, input_name, n_calib, size)
    if used == 0:
        raise FileNotFoundError(f"No calibration images found in {calib_image_dir}")

    quantize_static(
        model_input=prep_path,
        model_output=int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=exclude,
    )

    # copy Ultralytics metadata (names, stride, imgsz, task ...)
    q = onnx.load(int8_path)
    existing = {p.key for p in q.metadata_props}
    for p in src.metadata_props:
        if p.key not in existing:
            q.metadata_props.add(key=p.key, value=p.value)
    onnx.save(q, int8_path)

    if prep_path != fp32_path and os.path.exists(prep_path):
        os.remove(prep_path)
    print(f"[quantize] {used} calibration images, {len(exclude)} nodes kept FP32")
    return int8_path
