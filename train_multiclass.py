"""
train_multiclass.py — High-accuracy 4-class YOLOv11 training script
=====================================================================
Classes trained (must match dataset/data.yaml):
  0  tyre            — single tyre (stationary or resting)
  1  moving_tyre     — single tyre being actively rolled by a worker
  2  loading_vehicle — forklift, pallet jack, truck, etc.
  3  person          — warehouse worker

Design choices for maximum accuracy:
  • yolo11m.pt base  : medium backbone — ~40% more accurate than nano,
                       fits within 4 GB VRAM at batch=8, imgsz=1280.
  • imgsz=1280       : matches source video resolution (2940×1664 → 1280).
  • AdamW optimizer  : generally faster convergence than SGD on small datasets.
  • Cosine LR decay  : prevents LR spikes; standard YOLO best practice.
  • Heavy augmentation tuned for warehouse CCTV footage.
  • class_weights    : moving_tyre (rarest class) upweighted 3× via fl_gamma.
  • Two-phase training:
      Phase 1 (freeze backbone, 20 epochs) — warm up the new head quickly.
      Phase 2 (full fine-tune, 180 epochs) — full model convergence.
  • Early stopping (patience=40) — avoids overfitting on small datasets.
  • Label smoothing (0.1) — helps when auto-labelled data has noise.
  • Overlap mask + copy_paste — maximises variety on small datasets.

Usage:
  python train_multiclass.py
  python train_multiclass.py --epochs 200 --batch 4 --model yolo11l.pt
  python train_multiclass.py --no-freeze   # skip phase-1 freeze
  python train_multiclass.py --resume      # resume last checkpoint

Requirements:
  pip install ultralytics torch torchvision pyyaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_multiclass")


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_YAML        = "dataset/data.yaml"
PROJECT_DIR      = "models"          # Ultralytics prepends 'runs/detect/' automatically
RUN_NAME         = "tyre_multiclass"

# MUST match dataset/data.yaml names in exact order
EXPECTED_CLASSES = ["tyre", "moving_tyre", "loading_vehicle", "person"]
NUM_CLASSES      = len(EXPECTED_CLASSES)

# Model preference order (largest available within VRAM budget wins)
MODEL_PREFERENCE = [
    "yolo11l.pt",   # large  — best accuracy, needs ~8 GB VRAM
    "yolo11m.pt",   # medium — recommended (RTX 3050 4 GB at batch=8)
    "yolo11s.pt",   # small  — if VRAM is very tight
    "yolo11n.pt",   # nano   — CPU fallback
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_best_model(preferred: str | None) -> str:
    """Return the best available pretrained model file."""
    if preferred:
        if os.path.exists(preferred):
            return preferred
        logger.warning("Requested model '%s' not found locally (will auto-download).", preferred)
        return preferred  # Ultralytics will download it

    for name in MODEL_PREFERENCE:
        if os.path.exists(name):
            logger.info("Found local model: %s", name)
            return name

    logger.info("No local model found — will download yolo11m.pt")
    return "yolo11m.pt"


def validate_data_yaml(path: str) -> dict:
    """Confirm data.yaml has exactly 4 classes in the correct order."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {path}\n"
            "Please ensure dataset/data.yaml exists."
        )
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nc    = cfg.get("nc", 0)
    names = cfg.get("names", [])

    if nc != NUM_CLASSES or names != EXPECTED_CLASSES:
        raise ValueError(
            f"data.yaml must have nc={NUM_CLASSES} and names={EXPECTED_CLASSES}.\n"
            f"  Found nc={nc}, names={names}\n"
            f"Please update dataset/data.yaml to match exactly."
        )
    logger.info("data.yaml OK — nc=%d, names=%s", nc, names)
    return cfg


def check_label_files(data_cfg: dict) -> None:
    """Report label coverage per split and warn if less than 50% are labelled."""
    for split in ("train", "val"):
        img_dir = Path(data_cfg.get(split, f"dataset/{split}/images"))
        if not img_dir.exists():
            # Try alternate layout dataset/images/{split}
            alt = Path("dataset") / "images" / split
            if alt.exists():
                img_dir = alt
            else:
                logger.warning("Image directory not found: %s", img_dir)
                continue

        label_dir = img_dir.parent.parent / "labels" / img_dir.name
        if not label_dir.exists():
            label_dir = img_dir.parent / "labels"

        images = [
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        labelled = sum(
            1 for p in images
            if (label_dir / p.with_suffix(".txt").name).exists()
            and (label_dir / p.with_suffix(".txt").name).stat().st_size > 0
        )

        pct = 100 * labelled / max(len(images), 1)
        logger.info(
            "%s split: %d images, %d labelled (%.0f%%)",
            split, len(images), labelled, pct,
        )
        if pct < 50:
            logger.warning(
                "Less than 50%% of %s images have non-empty labels. "
                "Review/correct labels before training.",
                split,
            )


# ── Augmentation ──────────────────────────────────────────────────────────────
#
# Tuned for warehouse CCTV footage (overhead/angled camera, mixed lighting):
#
# hsv_h=0.015      — slight hue shift; indoor lighting has warm/cool casts
# hsv_s=0.7        — saturation jitter; cameras auto-adjust white balance
# hsv_v=0.4        — brightness jitter; shadows and reflections on floor
# degrees=10        — tyres lean slightly when rolling; workers tilt
# translate=0.1    — partial detections at frame edges
# scale=0.7        — tyres appear at very different distances from CCTV
# shear=2.0        — mild shear for perspective variation
# perspective=0.0005 — wide-angle barrel distortion
# fliplr=0.5       — bidirectional rolling; symmetric architecture
# flipud=0.0       — do NOT flip vertically (gravity is fixed)
# mosaic=1.0       — 4-image mosaic; critical for small tyre detection
# mixup=0.2        — softens class boundary (moving vs stationary tyre)
# copy_paste=0.3   — paste moving tyres onto new backgrounds (key for rare class)
# erasing=0.4      — simulate occlusion: workers partly blocking tyres
# label_smoothing  — handles noisy auto-labels from auto_label.py
# ─────────────────────────────────────────────────────────────────────────────
AUGMENTATION = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.7,
    shear=0.5,         # reduced from 2.0 — large shear causes RAM spikes on big segments
    perspective=0.0005,
    fliplr=0.5,
    flipud=0.0,
    mosaic=1.0,
    mixup=0.0,         # disabled — high VRAM cost on 4 GB GPU
    copy_paste=0.0,    # DISABLED — requires segmentation masks; causes RAM OOM on detection datasets
    erasing=0.4,
    label_smoothing=0.1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 4-class tyre detection model (YOLOv11)")
    p.add_argument("--model",     default=None,
                   help="Pretrained model (e.g. yolo11s.pt). Auto-selects if omitted.")
    p.add_argument("--data",      default=DATA_YAML,
                   help=f"Dataset YAML (default: {DATA_YAML})")
    p.add_argument("--epochs",    type=int, default=200,
                   help="Total training epochs (default: 200)")
    p.add_argument("--batch",     type=int, default=4,
                   help="Batch size (default: 4; safe for 4 GB VRAM)")
    p.add_argument("--imgsz",     type=int, default=640,
                   help="Training image size (default: 640; use 1280 only on 8+ GB VRAM)")
    p.add_argument("--freeze",    type=int, default=20,
                   help="Epochs to freeze backbone for phase-1 warmup (0 = skip)")
    p.add_argument("--patience",  type=int, default=40,
                   help="Early stopping patience in epochs (default: 40)")
    p.add_argument("--workers",   type=int, default=0,
                   help="DataLoader workers (0 = main thread; safe on Windows)")
    p.add_argument("--device",    default=None,
                   help="Device: 'cpu', '0', '0,1', … Auto-detects if omitted.")
    p.add_argument("--no-freeze", dest="no_freeze", action="store_true",
                   help="Skip phase-1 frozen backbone warm-up (single phase)")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from last checkpoint if available")
    return p.parse_args()


# ── Training phases ───────────────────────────────────────────────────────────
def phase1_frozen(model, args: argparse.Namespace, device: str) -> None:
    """
    Phase 1 — frozen backbone warm-up.
    Trains ONLY the detection head for `args.freeze` epochs so it adapts
    to 4 classes before the backbone weights are updated.
    Uses AdamW + warm LR for fast initial convergence.
    """
    logger.info("=== Phase 1: Frozen backbone warm-up (%d epochs) ===", args.freeze)
    model.train(
        data=args.data,
        epochs=args.freeze,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        freeze=10,               # freeze first 10 backbone layers
        patience=args.patience,
        workers=args.workers,
        project=PROJECT_DIR,
        name=RUN_NAME + "_phase1",
        exist_ok=True,
        plots=True,
        verbose=True,
        val=True,
        save=True,
        **AUGMENTATION,
    )
    logger.info("Phase 1 complete.")


def phase2_finetune(
    model,
    args: argparse.Namespace,
    device: str,
    remaining_epochs: int,
    model_path: str,
) -> None:
    """
    Phase 2 — full fine-tuning with cosine LR decay.
    All layers are unfrozen; lower LR to avoid disrupting the phase-1 head.
    """
    logger.info("=== Phase 2: Full fine-tune (%d epochs) ===", remaining_epochs)
    model.train(
        model=model_path,          # explicit path avoids KeyError: 'model' on re-train
        data=args.data,
        epochs=remaining_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer="AdamW",
        lr0=0.0001,               # lower LR for fine-tuning after phase-1
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0,          # no warmup needed in phase 2
        cos_lr=True,              # cosine LR schedule
        patience=args.patience,
        workers=args.workers,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,           # checkpoint every 10 epochs
        val=True,
        verbose=True,
        **AUGMENTATION,
    )
    logger.info("Phase 2 complete.")


def single_phase(model, args: argparse.Namespace, device: str) -> None:
    """Single-phase training — used when --no-freeze is set."""
    logger.info("=== Single-phase training (%d epochs) ===", args.epochs)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        patience=args.patience,
        workers=args.workers,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
        **AUGMENTATION,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Validate dataset ─────────────────────────────────────────────────────
    logger.info("Validating dataset config: %s", args.data)
    data_cfg = validate_data_yaml(args.data)
    check_label_files(data_cfg)

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        device = "0"
        logger.info(
            "GPU detected: %s  (%.1f GB VRAM)",
            torch.cuda.get_device_name(0), vram_gb,
        )
        if vram_gb < 4 and args.batch > 4:
            logger.warning(
                "VRAM %.1f GB < 4 GB — reducing batch to 4 to avoid OOM.", vram_gb
            )
            args.batch = 4
        if vram_gb < 8 and args.imgsz > 1280:
            logger.warning(
                "VRAM %.1f GB may be insufficient for imgsz=%d — consider --imgsz 640.",
                vram_gb, args.imgsz,
            )
    else:
        device = "cpu"
        logger.warning("No GPU detected — training on CPU. This will be very slow.")
        logger.warning(
            "Tip: use --imgsz 640 --batch 4 --epochs 50 for a quick CPU test run."
        )

    # ── Model selection ───────────────────────────────────────────────────────
    model_path = find_best_model(args.model)
    logger.info("Base model: %s", model_path)
    logger.info("Classes (%d): %s", NUM_CLASSES, EXPECTED_CLASSES)

    # Import after validation so errors surface early
    from ultralytics import YOLO

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume:
        resume_path = (
            Path("runs") / "detect" / PROJECT_DIR / RUN_NAME / "weights" / "last.pt"
        )
        if resume_path.exists():
            logger.info("Resuming from: %s", resume_path)
            model = YOLO(str(resume_path))
            single_phase(model, args, device)
            _print_summary(args)
            return
        else:
            logger.warning(
                "--resume: no checkpoint found at %s — starting fresh.", resume_path
            )

    model = YOLO(model_path)

    # ── Two-phase or single-phase training ───────────────────────────────────
    if args.freeze > 0 and not args.no_freeze:
        phase1_frozen(model, args, device)

        # Locate phase-1 best weights
        phase1_dir = Path("runs") / "detect" / PROJECT_DIR / (RUN_NAME + "_phase1") / "weights"
        phase1_weights = (
            phase1_dir / "best.pt"
            if (phase1_dir / "best.pt").exists()
            else phase1_dir / "last.pt"
            if (phase1_dir / "last.pt").exists()
            else None
        )

        if phase1_weights:
            logger.info("Loading phase-1 weights for phase-2: %s", phase1_weights)
            model = YOLO(str(phase1_weights))
            reload_path = str(phase1_weights)
        else:
            logger.warning("Phase-1 weights not found — continuing with original model.")
            reload_path = model_path

        remaining = max(1, args.epochs - args.freeze)
        phase2_finetune(model, args, device, remaining, reload_path)
    else:
        single_phase(model, args, device)

    _print_summary(args)


def _print_summary(args: argparse.Namespace) -> None:
    best_pt     = Path("runs") / "detect" / PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    results_csv = Path("runs") / "detect" / PROJECT_DIR / RUN_NAME / "results.csv"

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best weights : %s", best_pt)
    logger.info("  Results CSV  : %s", results_csv)
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Update config.yaml → model.path: '%s'", best_pt)
    logger.info("  2. Ensure config.yaml → model.classes: [1]  # only moving_tyre triggers counting")
    logger.info("  3. python app.py --no-zone-setup")
    logger.info("  4. Open http://localhost:5050 and verify class-coloured boxes")

    if best_pt.exists():
        try:
            with open(results_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last     = rows[-1]
                map50    = last.get("metrics/mAP50(B)", "n/a")
                map50_95 = last.get("metrics/mAP50-95(B)", "n/a")
                logger.info(
                    "Final metrics  →  mAP50: %s   mAP50-95: %s", map50, map50_95
                )
        except Exception:
            pass


if __name__ == "__main__":
    main()
