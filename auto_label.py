"""
auto_label.py — Multi-class draft label generator
===================================================
Generates YOLO-format draft annotation (.txt) files for every image in
dataset/images/ using two models:

  • best.pt   (custom tyre model) → detects tyres and heuristically assigns:
        class 0  moving_tyre      small/medium box in lower half of frame
        class 1  stationary_tyre  small/medium box, upper or central area
        class 2  tyre_group       large box (area > GROUP_AREA_RATIO of frame)

  • yolo11n.pt (COCO pretrained)  → detects persons → class 3  person

Output label files are written to:
  dataset/labels/train/<stem>.txt   (for images already in train/)
  dataset/labels/val/<stem>.txt     (for images already in val/)
  dataset/labels/<stem>.txt         (for root-level images not yet split)

The files are DRAFTS.  You must:
  1. Open them in a labelling tool (Roboflow, CVAT, LabelImg).
  2. Correct/delete wrong boxes and add any missed ones.
  3. Then run:  python prepare_dataset.py
  4. Then run:  python train_multiclass.py

Usage:
  python auto_label.py
  python auto_label.py --images dataset/images --tyre-model runs/detect/models/tyre_yolo11/weights/best.pt
  python auto_label.py --conf 0.20 --group-ratio 0.08
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Class IDs (must match dataset/data.yaml) ─────────────────────────────────
CLS_MOVING_TYRE     = 0
CLS_STATIONARY_TYRE = 1
CLS_TYRE_GROUP      = 2
CLS_PERSON          = 3

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_TYRE_MODEL   = "runs/detect/models/tyre_yolo11/weights/best.pt"
FALLBACK_TYRE_MODEL  = "yolo11n.pt"
DEFAULT_PERSON_MODEL = "yolo11n.pt"   # COCO model; class 0 = person in COCO
PERSON_COCO_CLASS    = 0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Heuristic: decide tyre class from box size + position ────────────────────
def classify_tyre(
    cx_rel: float,          # normalised centroid x (0–1)
    cy_rel: float,          # normalised centroid y (0–1)
    area_rel: float,        # bbox_area / frame_area (0–1)
    group_area_ratio: float,
    motion_zone_y: float,   # y fraction below which we expect rolling tyres
) -> int:
    """
    Heuristic class assignment for a detected tyre bounding box.

    Rules (in priority order):
      1. Large box  →  tyre_group   (rack / stack)
      2. Box centre in lower zone  →  moving_tyre
      3. Everything else  →  stationary_tyre
    """
    if area_rel >= group_area_ratio:
        return CLS_TYRE_GROUP
    if cy_rel >= motion_zone_y:
        return CLS_MOVING_TYRE
    return CLS_STATIONARY_TYRE


def yolo_line(cls_id: int, cx: float, cy: float, w: float, h: float) -> str:
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def find_label_path(image_path: Path, images_root: Path) -> Path:
    """Mirror the image path into the labels/ directory."""
    try:
        rel = image_path.relative_to(images_root)
    except ValueError:
        rel = Path(image_path.name)
    labels_root = images_root.parent.parent / "labels"
    label_path = labels_root / rel.with_suffix(".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    return label_path


def collect_images(images_root: Path) -> list[Path]:
    """Collect all images under images_root (including train/ and val/ subdirs)."""
    paths: list[Path] = []
    for p in sorted(images_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return paths


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate draft YOLO labels (multi-class)")
    parser.add_argument("--images",       default="dataset/images",
                        help="Root folder containing images (or train/val sub-folders)")
    parser.add_argument("--tyre-model",   default=DEFAULT_TYRE_MODEL,
                        help="Path to custom tyre detection model")
    parser.add_argument("--person-model", default=DEFAULT_PERSON_MODEL,
                        help="Path to COCO model used for person detection")
    parser.add_argument("--conf",         type=float, default=0.20,
                        help="Minimum detection confidence (default 0.20)")
    parser.add_argument("--group-ratio",  type=float, default=0.08,
                        help="Box area / frame area threshold above which a tyre box "
                             "is classified as tyre_group (default 0.08 = 8%%)")
    parser.add_argument("--motion-zone-y", type=float, default=0.55,
                        help="Normalised Y above which tyres are moving_tyre (default 0.55)")
    parser.add_argument("--imgsz",        type=int, default=1280,
                        help="Inference image size (default 1280)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Overwrite existing label files (default: skip)")
    args = parser.parse_args()

    images_root = Path(args.images).resolve()
    if not images_root.exists():
        raise SystemExit(f"Images folder not found: {images_root}")

    # ── Load tyre model ──────────────────────────────────────────────────────
    tyre_model_path = args.tyre_model
    if not os.path.exists(tyre_model_path):
        print(f"[WARN] Tyre model '{tyre_model_path}' not found — falling back to '{FALLBACK_TYRE_MODEL}'")
        tyre_model_path = FALLBACK_TYRE_MODEL
    print(f"[INFO] Tyre model  : {tyre_model_path}")
    tyre_model = YOLO(tyre_model_path)

    # ── Load person model ────────────────────────────────────────────────────
    print(f"[INFO] Person model: {args.person_model}")
    person_model = YOLO(args.person_model)

    images = collect_images(images_root)
    print(f"[INFO] Found {len(images)} images under {images_root}")
    if not images:
        raise SystemExit("No images found — check --images path.")

    written = skipped = 0

    for idx, img_path in enumerate(images, 1):
        label_path = find_label_path(img_path, images_root)

        if label_path.exists() and not args.overwrite:
            skipped += 1
            if idx % 50 == 0:
                print(f"  [{idx}/{len(images)}] Skipping (label exists): {img_path.name}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        fh, fw = frame.shape[:2]
        frame_area = fh * fw
        lines: list[str] = []

        # ── Tyre detection ───────────────────────────────────────────────────
        tyre_results = tyre_model(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
        for result in tyre_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw = x2 - x1
                bh = y2 - y1
                cx_abs = x1 + bw / 2
                cy_abs = y1 + bh / 2

                # Normalise
                cx_n = cx_abs / fw
                cy_n = cy_abs / fh
                bw_n = bw / fw
                bh_n = bh / fh
                area_n = (bw * bh) / frame_area

                cls_id = classify_tyre(
                    cx_n, cy_n, area_n,
                    group_area_ratio=args.group_ratio,
                    motion_zone_y=args.motion_zone_y,
                )
                lines.append(yolo_line(cls_id, cx_n, cy_n, bw_n, bh_n))

        # ── Person detection (COCO model, class 0 = person) ──────────────────
        person_results = person_model(
            frame,
            conf=args.conf,
            imgsz=640,      # COCO model runs at 640
            classes=[PERSON_COCO_CLASS],
            verbose=False,
        )
        for result in person_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw = x2 - x1
                bh = y2 - y1
                cx_n = (x1 + bw / 2) / fw
                cy_n = (y1 + bh / 2) / fh
                bw_n = bw / fw
                bh_n = bh / fh
                lines.append(yolo_line(CLS_PERSON, cx_n, cy_n, bw_n, bh_n))

        label_path.write_text("\n".join(lines), encoding="utf-8")
        written += 1

        if idx % 10 == 0 or idx == len(images):
            print(f"  [{idx}/{len(images)}] Labelled: {img_path.name}  "
                  f"({len(lines)} boxes — "
                  f"{sum(1 for l in lines if l.startswith('0'))} moving, "
                  f"{sum(1 for l in lines if l.startswith('1'))} stationary, "
                  f"{sum(1 for l in lines if l.startswith('2'))} group, "
                  f"{sum(1 for l in lines if l.startswith('3'))} person)")

    print(f"\n[DONE] Written {written} label files, skipped {skipped}.")
    print("Next steps:")
    print("  1. Review/correct labels in Roboflow or CVAT")
    print("  2. python prepare_dataset.py   ← re-splits train/val")
    print("  3. python train_multiclass.py  ← trains the 4-class model")


if __name__ == "__main__":
    main()
