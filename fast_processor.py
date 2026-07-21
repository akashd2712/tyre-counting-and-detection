"""
fast_processor.py — GPU-accelerated tyre counting pipeline
===========================================================
Drop-in alternative to app.py that processes video significantly faster by:

  • Forcing CUDA device (falls back to CPU gracefully)
  • FP16 half-precision: model weights converted ONCE via model.model.half()
    (avoids the deprecated 'half=' per-call kwarg warning)
  • Frame-skip: only run inference every N frames; skipped frames reuse the
    most recent detection set for annotation (no counting on skipped frames)
  • Three-thread pipeline:  Reader ──> Inference (main) ──> Writer
  • Batch accumulation: collect B frames then flush with per-frame tracking calls
  • CUDA warmup on the first batch to prime GPU CUDA graphs

All existing modules (TyreCounter, HomographyProjector, MotionFilter,
annotate_frame, web-server helpers …) are imported UNCHANGED.  No existing
file is modified.

Usage
-----
  python fast_processor.py [same flags as app.py] [extra flags below]

Extra flags
-----------
  --frame_skip  N   Process 1 frame out of every N (default 2).
                    Skipped frames reuse the previous batch's detections.
  --batch_size  B   Number of frames to accumulate before calling YOLO (default 4).
  --no-warmup       Skip the GPU warmup pass.
  --queue_size  Q   Frame-queue depth between threads (default 32).
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ── Reuse ALL existing project modules unchanged ───────────────────────────────
from counting.counter import TyreCounter
from utils.homography import HomographyProjector
from utils.motion import MotionFilter
from utils.roi import apply_roi
from utils.visualization import annotate_frame
from utils.web_server import (
    clear_zone_result,
    get_zone_result,
    set_zone_frame,
    start_server,
    update_frame,
    update_stats,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("fast_processor")

WEB_PORT = 5050

# ── Class indices (same as app.py) ─────────────────────────────────────────────
CLASS_TYRE            = 0
CLASS_MOVING_TYRE     = 1
CLASS_LOADING_VEHICLE = 2
CLASS_PERSON          = 3

# ── Sentinel that signals threads to stop ─────────────────────────────────────
_STOP = object()


# ══════════════════════════════════════════════════════════════════════════════
#  Config / CLI helpers  (mirrors app.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU-accelerated YOLOv11 tyre counting (fast_processor)"
    )
    # ── Same flags as app.py ──────────────────────────────────────────────────
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--source",        default=None)
    p.add_argument("--model",         default=None)
    p.add_argument("--show",          action="store_true")
    p.add_argument("--no-zone-setup", action="store_true",
                   help="Skip interactive zone setup; use config.yaml directly.")
    p.add_argument("--port",          type=int, default=WEB_PORT,
                   help=f"Web UI port (default: {WEB_PORT}).")
    # ── Extra speed flags ─────────────────────────────────────────────────────
    p.add_argument("--frame_skip",  type=int, default=2,
                   help="Process 1 in every N frames (default: 2). "
                        "Skipped frames reuse previous detections for annotation only.")
    p.add_argument("--batch_size",  type=int, default=4,
                   help="Frames to accumulate before flushing to YOLO (default: 4).")
    p.add_argument("--no-warmup",   action="store_true",
                   help="Skip GPU warmup pass.")
    p.add_argument("--queue_size",  type=int, default=32,
                   help="Inter-thread queue depth (default: 32).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  GPU utilities
# ══════════════════════════════════════════════════════════════════════════════

def _select_device(config_device: str) -> str:
    """
    Return the best available device string for YOLO.
    Prefers CUDA even if config says 'cpu', but respects an explicit 'cpu'.
    """
    if config_device.lower() == "cpu":
        logger.info("Device forced to CPU via config.")
        return "cpu"

    if torch.cuda.is_available():
        idx  = config_device if config_device.isdigit() else "0"
        name = torch.cuda.get_device_name(int(idx))
        vram = torch.cuda.get_device_properties(int(idx)).total_memory / 1024 ** 3
        logger.info("🚀  GPU detected: %s  (%.1f GB VRAM) — using CUDA:%s",
                    name, vram, idx)
        return idx
    else:
        logger.warning(
            "⚠️  CUDA not available — falling back to CPU.  "
            "Install PyTorch with CUDA support for GPU acceleration."
        )
        return "cpu"


def _apply_half_precision(model: YOLO, device: str) -> bool:
    """
    Convert model weights to FP16 IN PLACE (done once, not per-call).
    Avoids the deprecated 'half=' per-call keyword-argument warning.
    Returns True if FP16 was applied.
    """
    if device == "cpu" or not torch.cuda.is_available():
        return False
    try:
        model.model.half()
        logger.info("✅  Model weights converted to FP16 (half precision).")
        return True
    except Exception as exc:
        logger.warning("FP16 conversion failed (%s) — using FP32.", exc)
        return False


def _warmup(model: YOLO, device: str, imgsz: int) -> None:
    """
    Run one dummy inference to prime CUDA graphs / cuDNN auto-tuner.
    The model weights are already in FP16 at this point, so no extra kwarg needed.
    """
    logger.info("🔥  Warming up GPU …")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, imgsz=imgsz, device=device, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    logger.info("    Warmup done.")


# ══════════════════════════════════════════════════════════════════════════════
#  Zone helpers (identical to app.py)
# ══════════════════════════════════════════════════════════════════════════════

def _wait_for_zone_setup(cap: cv2.VideoCapture, config: dict, port: int) -> dict:
    saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()
    if ret:
        set_zone_frame(frame)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
    except Exception:
        pass

    ez = config.get("entrance_zone", {})
    logger.info(
        "Zone setup: http://localhost:%d/zone_setup?ez_top=%s&ez_bot=%s&line_y=%s",
        port, ez.get("y_min", 0.60), ez.get("y_max", 1.00), ez.get("line_y", 0.80),
    )
    logger.info("Open the URL above then click Confirm.")

    clear_zone_result()
    while True:
        result = get_zone_result()
        if result is not None:
            break
        time.sleep(0.2)

    if result.get("skipped"):
        logger.info("Zone setup skipped — using config.yaml values.")
        return config

    logger.info("Zone setup confirmed: %s", result)
    new_config = dict(config)
    new_config["entrance_zone"] = result
    return new_config


def _best_vehicle_bbox(detections: list) -> Optional[Tuple]:
    """Return pixel bbox of the largest loading_vehicle detection."""
    vehicles = [d for d in detections if d.get("class_id") == CLASS_LOADING_VEHICLE]
    if not vehicles:
        return None
    return max(vehicles, key=lambda d: (
        (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
    ))["bbox"]


def _derive_vehicle_zone(vehicle_bbox, frame_h: int, frame_w: int,
                          gate_width_ratio: float, line_ratio: float,
                          orientation: str = "vertical") -> dict:
    vx1, vy1, vx2, vy2 = vehicle_bbox
    vw = vx2 - vx1
    vh = vy2 - vy1

    if orientation == "vertical":
        gate_w = vw * gate_width_ratio
        gx1 = max(0, vx1 - gate_w / 2)
        gx2 = min(frame_w, vx1 + gate_w / 2)
        gy1 = vy1 + vh * 0.05
        gy2 = vy2 - vh * 0.05
    else:
        vcx = (vx1 + vx2) / 2.0
        gate_w = vw * gate_width_ratio
        gx1 = max(0, vcx - gate_w / 2)
        gx2 = min(frame_w, vcx + gate_w / 2)
        gy1 = vy1
        gy2 = vy1 + vh * 0.65

    return {
        "enabled":          True,
        "x_min":            gx1 / frame_w,
        "x_max":            gx2 / frame_w,
        "y_min":            gy1 / frame_h,
        "y_max":            gy2 / frame_h,
        "line_ratio":       line_ratio,
        "line_orientation": orientation,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Detection parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_results(results, x_off: int, y_off: int,
                   conf_thresh: float, projector: HomographyProjector) -> List[dict]:
    """Convert raw YOLO result objects into the same dict format used by app.py."""
    all_dets: List[dict] = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls_id   = int(box.cls[0])   if box.cls  is not None else 0
            conf_val = float(box.conf[0]) if box.conf is not None else 0.0
            if conf_val < conf_thresh:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 += x_off; y1 += y_off; x2 += x_off; y2 += y_off
            tid = int(box.id[0]) if box.id is not None else None
            all_dets.append({
                "bbox":          (x1, y1, x2, y2),
                "class_id":      cls_id,
                "conf":          conf_val,
                "track_id":      tid,
                "bottom_center": projector.project_bbox_bottom((x1, y1, x2, y2)),
            })
    return all_dets


# ══════════════════════════════════════════════════════════════════════════════
#  Thread: Frame Reader
# ══════════════════════════════════════════════════════════════════════════════

def _reader_thread(
    cap_factory,
    frame_queue: queue.Queue,
    frame_skip: int,
    playlist: Optional[List[Path]],
    loop_video: bool,
    stop_event: threading.Event,
) -> None:
    """
    Reads frames from the video source and pushes
    (frame_idx, frame, is_inference_frame) tuples into frame_queue.
    """
    cap, playlist_idx, _ = cap_factory()
    frame_idx = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if playlist is not None:
                    playlist_idx = (playlist_idx + 1) % len(playlist)
                    current_video = playlist[playlist_idx]
                    cap.release()
                    cap = cv2.VideoCapture(str(current_video))
                    logger.info("[Reader] Next video: %s", current_video.name)
                    continue
                elif loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_idx += 1
            is_inference = (frame_idx % frame_skip == 0) or (frame_idx == 1)

            try:
                frame_queue.put((frame_idx, frame, is_inference), timeout=2.0)
            except queue.Full:
                logger.debug("[Reader] Queue full — dropping frame %d", frame_idx)

    finally:
        cap.release()
        frame_queue.put(_STOP)
        logger.info("[Reader] Thread finished.")


# ══════════════════════════════════════════════════════════════════════════════
#  Thread: Writer / Annotator
# ══════════════════════════════════════════════════════════════════════════════

def _writer_thread(
    result_queue: queue.Queue,
    writer: Optional[cv2.VideoWriter],
    show_window: bool,
    stop_event: threading.Event,
) -> None:
    """Pulls annotated frames from result_queue and writes / displays them."""
    try:
        while not stop_event.is_set():
            try:
                item = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            annotated_frame = item
            update_frame(annotated_frame)

            if writer is not None:
                writer.write(annotated_frame)

            if show_window:
                cv2.imshow("Fast Tyre Counter", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

    finally:
        if show_window:
            cv2.destroyAllWindows()
        logger.info("[Writer] Thread finished.")


# ══════════════════════════════════════════════════════════════════════════════
#  Core per-frame processing  (mirrors app.py logic exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _process_inference_frame(
    *,
    fidx: int,
    orig_frame: np.ndarray,
    all_dets: List[dict],
    config: dict,
    projector: HomographyProjector,
    counter: TyreCounter,
    motion_filter: MotionFilter,
    frame_h: int,
    frame_w: int,
    coord_h: int,
    coord_w: int,
    # mutable vehicle-tracking state (passed by reference via list wrapper)
    state: dict,
    auto_zone: bool,
    gate_width_ratio: float,
    no_vehicle_patience: int,
    line_ratio: float,
    orientation: str,
    stationary_suppress_frames: int,
    track_stationary_counts: dict,
    conf_thresh: float,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Full per-frame processing identical to app.py's main loop body.
    Returns (annotated_frame, valid_dets).
    state dict keys: smoothed_vbbox, frames_since_vehicle, counting_paused, initialized
    """
    smoothed_vbbox      = state["smoothed_vbbox"]
    frames_since_vehicle = state["frames_since_vehicle"]
    counting_paused     = state["counting_paused"]

    counter.tick()

    # ── First-frame initialisation ─────────────────────────────────────────────
    if not state["initialized"]:
        counter.initialize_from_detections(all_dets, coord_h, coord_w)
        state["initialized"] = True

    # ── Vehicle detection + EMA smooth + auto-zone ─────────────────────────────
    raw_vbbox = _best_vehicle_bbox(all_dets)

    if raw_vbbox is not None:
        frames_since_vehicle = 0
        counting_paused      = False

        if smoothed_vbbox is None:
            smoothed_vbbox = raw_vbbox
        else:
            a  = 0.35
            sv = smoothed_vbbox
            rv = raw_vbbox
            smoothed_vbbox = tuple(
                int(a * rv[i] + (1 - a) * sv[i]) for i in range(4)
            )

        if auto_zone:
            new_ez = _derive_vehicle_zone(
                smoothed_vbbox, frame_h, frame_w,
                gate_width_ratio, line_ratio, orientation,
            )
            config["entrance_zone"] = new_ez

            if orientation == "vertical":
                vx_proj, vy_proj = projector.project_point(
                    smoothed_vbbox[0], smoothed_vbbox[3]
                )
            else:
                vx_proj, vy_proj = projector.project_bbox_bottom(smoothed_vbbox)
            counter.fix_line_from_vehicle(
                (vx_proj, vy_proj), coord_h, coord_w,
                offset_px=0, orientation=orientation,
            )
    else:
        frames_since_vehicle += 1
        if frames_since_vehicle >= no_vehicle_patience:
            counting_paused  = True
            smoothed_vbbox   = None

    # ── Zone pixel bounds ──────────────────────────────────────────────────────
    ez       = config.get("entrance_zone", {})
    ez_en    = ez.get("enabled", False)
    ez_y1_px = int(ez.get("y_min", 0.0) * frame_h)
    ez_y2_px = int(ez.get("y_max", 1.0) * frame_h)
    ez_x1_px = int(ez.get("x_min", 0.0) * frame_w)
    ez_x2_px = int(ez.get("x_max", 1.0) * frame_w)
    frame_area = frame_h * frame_w

    # ── Per-detection filtering + counting ────────────────────────────────────
    valid_dets: List[dict] = []

    for det in all_dets:
        cls_id          = det["class_id"]
        x1, y1, x2, y2 = det["bbox"]
        bbox_area       = (x2 - x1) * (y2 - y1)
        tid             = det["track_id"]
        cx              = (x1 + x2) / 2.0
        cy              = (y1 + y2) / 2.0

        # Always show vehicles
        if cls_id == CLASS_LOADING_VEHICLE:
            valid_dets.append(det)
            continue

        # Size filter
        if not motion_filter.is_valid_size(bbox_area, frame_area):
            continue

        # Motion / stationary suppression
        cx_proj, cy_proj    = det["bottom_center"]
        prev_center_proj    = counter.get_previous_center(tid) if tid is not None else None
        is_moving           = motion_filter.should_process(
            prev_center_proj, (cx_proj, cy_proj), bbox_area, frame_area
        )

        if tid is not None:
            if is_moving:
                track_stationary_counts[tid] = 0
            else:
                track_stationary_counts[tid] = track_stationary_counts.get(tid, 0) + 1

        if track_stationary_counts.get(tid, 0) >= stationary_suppress_frames:
            if tid is not None:
                counter.reset_track(tid)
            continue

        valid_dets.append(det)

        # Spatial zone check
        in_zone = True
        if ez_en:
            in_zone = (ez_y1_px <= cy <= ez_y2_px and ez_x1_px <= cx <= ez_x2_px)

        # Count only tyres
        if cls_id in (CLASS_TYRE, CLASS_MOVING_TYRE) and tid is not None:
            event = counter.process_track(
                track_id=tid,
                cx_px=cx_proj, cy_px=cy_proj,
                frame_h=coord_h, frame_w=coord_w,
                in_zone=in_zone,
                is_vehicle_present=not counting_paused,
            )
            if event:
                logger.info(
                    "Frame %d: %s — track %d  (count=%d)",
                    fidx, event.upper(), tid, counter.current_count,
                )

    # ── Annotate ───────────────────────────────────────────────────────────────
    annotated = orig_frame.copy()
    annotate_frame(
        annotated, valid_dets, counter, config,
        vehicle_bbox=smoothed_vbbox if auto_zone else None,
        fixed_line_pos=counter.line_pos,
        counting_paused=counting_paused,
        projector=projector,
    )
    if counting_paused and auto_zone:
        cv2.putText(
            annotated, "COUNTING PAUSED — no vehicle detected",
            (10, frame_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
            (0, 100, 255), 2,
        )

    # ── Write back mutable state ───────────────────────────────────────────────
    state["smoothed_vbbox"]       = smoothed_vbbox
    state["frames_since_vehicle"] = frames_since_vehicle
    state["counting_paused"]      = counting_paused

    return annotated, valid_dets


# ══════════════════════════════════════════════════════════════════════════════
#  Main inference loop  (runs on main thread — owns PyTorch / CUDA context)
# ══════════════════════════════════════════════════════════════════════════════

def _run_inference_loop(
    frame_queue:      queue.Queue,
    result_queue:     queue.Queue,
    model:            YOLO,
    config:           dict,
    projector:        HomographyProjector,
    counter:          TyreCounter,
    motion_filter:    MotionFilter,
    frame_h:          int,
    frame_w:          int,
    conf_thresh:      float,
    imgsz:            int,
    device:           str,
    batch_size:       int,
    tracker_cfg:      str,
    tracking_enabled: bool,
    stop_event:       threading.Event,
    fps_report_interval: int = 30,
) -> None:
    """
    Pulls frames from frame_queue.
    • Inference frames:  accumulated into a batch, then flushed via model.track()
      or model() one-at-a-time (tracker requires sequential calls).
      Full vehicle-zone + counting logic runs on each inference frame.
    • Skipped frames:    last valid_dets reused for annotation only; no counting.
    """
    auto_zone            = config.get("auto_zone_from_vehicle", True)
    gate_width_ratio     = config.get("gate_width_ratio", 0.40)
    no_vehicle_patience  = config.get("no_vehicle_patience", 30)
    ez_cfg               = config.get("entrance_zone", {})
    line_ratio           = ez_cfg.get("line_ratio", 0.50)
    orientation          = ez_cfg.get("line_orientation", "vertical")
    cnt_cfg              = config["counting"]
    stationary_suppress_frames = cnt_cfg.get("stationary_suppress_frames", 10)

    coord_w = (
        config.get("homography", {}).get("map_width",  frame_w)
        if projector.enabled else frame_w
    )
    coord_h = (
        config.get("homography", {}).get("map_height", frame_h)
        if projector.enabled else frame_h
    )

    # ── Mutable vehicle-tracking state (passed into _process_inference_frame) ──
    vehicle_state = {
        "smoothed_vbbox":       None,
        "frames_since_vehicle": 0,
        "counting_paused":      False,
        "initialized":          False,
    }
    track_stationary_counts: dict = {}

    # ── Pending inference batch ────────────────────────────────────────────────
    # Each item: (frame_idx, orig_frame, roi_frame, x_off, y_off)
    batch_items: List[Tuple] = []

    # Last annotated valid_dets reused for skipped frames
    cached_valid_dets: List[dict] = []
    cached_smoothed_vbbox         = None

    # FPS measurement
    fps_t0          = time.time()
    total_frames    = 0        # ALL frames (inference + skipped)
    fps_value       = 0.0

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _flush_batch() -> None:
        nonlocal cached_valid_dets, cached_smoothed_vbbox, total_frames, fps_t0, fps_value

        if not batch_items:
            return

        t_inf = time.time()

        for (fidx, orig_frame, roi_frame, x_off, y_off) in batch_items:
            # ── YOLO call (per-frame because tracker is stateful) ──────────────
            if tracking_enabled:
                results = model.track(
                    roi_frame,
                    conf=conf_thresh,
                    imgsz=imgsz,
                    classes=None,
                    stream=False,
                    tracker=tracker_cfg,
                    persist=True,
                    device=device,
                    verbose=False,
                )
            else:
                results = model(
                    roi_frame,
                    conf=conf_thresh,
                    imgsz=imgsz,
                    classes=None,
                    stream=False,
                    device=device,
                    verbose=False,
                )

            all_dets = _parse_results(results, x_off, y_off, conf_thresh, projector)

            annotated, valid_dets = _process_inference_frame(
                fidx=fidx,
                orig_frame=orig_frame,
                all_dets=all_dets,
                config=config,
                projector=projector,
                counter=counter,
                motion_filter=motion_filter,
                frame_h=frame_h,
                frame_w=frame_w,
                coord_h=coord_h,
                coord_w=coord_w,
                state=vehicle_state,
                auto_zone=auto_zone,
                gate_width_ratio=gate_width_ratio,
                no_vehicle_patience=no_vehicle_patience,
                line_ratio=line_ratio,
                orientation=orientation,
                stationary_suppress_frames=stationary_suppress_frames,
                track_stationary_counts=track_stationary_counts,
                conf_thresh=conf_thresh,
            )

            # Cache latest results for skipped frames
            cached_valid_dets    = valid_dets
            cached_smoothed_vbbox = vehicle_state["smoothed_vbbox"]

            total_frames += 1

            # FPS reporting
            if total_frames % fps_report_interval == 0:
                elapsed  = max(time.time() - fps_t0, 0.001)
                fps_value = fps_report_interval / elapsed
                fps_t0    = time.time()
                logger.info(
                    "[Inference] Frame %-6d | FPS %.1f | Count %d | GPU %s",
                    fidx, fps_value, counter.current_count,
                    "CUDA:" + device if device != "cpu" else "CPU",
                )

            # Stats to web UI
            state_snap = counter.get_state()
            update_stats({
                "current_count": state_snap["current_count"],
                "entry_count":   state_snap["entry_count"],
                "exit_count":    state_snap["exit_count"],
                "fps":           round(fps_value, 1),
                "frame_idx":     fidx,
            })

            try:
                result_queue.put(annotated, timeout=2.0)
            except queue.Full:
                logger.debug("[Inference] Result queue full — dropping frame %d", fidx)

        inf_ms = (time.time() - t_inf) * 1000
        logger.debug("Flushed %d frames — total %.1f ms (%.1f ms/frame)",
                     len(batch_items), inf_ms, inf_ms / max(len(batch_items), 1))
        batch_items.clear()

    # ── Main pull loop ─────────────────────────────────────────────────────────
    try:
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _STOP:
                _flush_batch()
                break

            fidx, frame, is_inference = item
            roi_frame, x_off, y_off = apply_roi(frame, config.get("roi", {}))

            if is_inference:
                # Accumulate into batch; flush when batch is full
                batch_items.append((fidx, frame, roi_frame, x_off, y_off))
                if len(batch_items) >= batch_size:
                    _flush_batch()

            else:
                # Skipped frame: annotate with cached dets (no counting)
                total_frames += 1
                annotated = frame.copy()
                annotate_frame(
                    annotated, cached_valid_dets, counter, config,
                    vehicle_bbox=cached_smoothed_vbbox if auto_zone else None,
                    fixed_line_pos=counter.line_pos,
                    counting_paused=vehicle_state["counting_paused"],
                    projector=projector,
                )
                try:
                    result_queue.put(annotated, timeout=1.0)
                except queue.Full:
                    pass

    finally:
        result_queue.put(_STOP)
        logger.info("[Inference] Thread finished.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args   = parse_args()
    config = load_config(args.config)

    # ── Model ──────────────────────────────────────────────────────────────────
    model_path     = args.model or config["model"].get("path", "yolo11n.pt")
    fallback_model = config["model"].get("fallback_path", "yolo11n.pt")

    if not os.path.exists(model_path):
        logger.warning("Primary model '%s' not found — trying fallback '%s'",
                       model_path, fallback_model)
        model_path = fallback_model
    if not os.path.exists(model_path):
        logger.warning("Fallback '%s' not found — using 'yolo11n.pt'", model_path)
        model_path = "yolo11n.pt"

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    # ── GPU device selection ───────────────────────────────────────────────────
    config_device = config["model"].get("device", "0")
    device        = _select_device(config_device)
    conf_thresh   = config["model"].get("conf",  0.35)
    imgsz         = config["model"].get("imgsz", 640)
    tracker_cfg   = config["tracking"].get("tracker", "custom_bytetrack.yaml")
    tracking_en   = config["tracking"].get("enabled", True)

    # ── FP16: convert model weights ONCE — no per-call 'half=' kwarg needed ───
    fp16_active = _apply_half_precision(model, device)

    logger.info(
        "Device: %s | FP16: %s | imgsz: %d | conf: %.2f",
        device, fp16_active, imgsz, conf_thresh,
    )
    logger.info(
        "Frame-skip: %d | Batch size: %d | Queue depth: %d",
        args.frame_skip, args.batch_size, args.queue_size,
    )

    # ── Source ─────────────────────────────────────────────────────────────────
    source      = args.source or config["source"].get("input", "0")
    show_window = args.show   or config["source"].get("show_window", False)
    source_path = Path(source)
    extensions  = tuple(config["source"].get("extensions",
                         [".mov", ".mp4", ".avi", ".mkv"]))
    loop_video  = config["source"].get("loop", False)

    playlist: Optional[List[Path]] = None

    if source_path.is_dir():
        playlist = sorted(
            p for p in source_path.iterdir() if p.suffix.lower() in extensions
        )
        if not playlist:
            raise RuntimeError(f"No video files in: {source}")
        logger.info("Folder source — %d video(s):", len(playlist))
        for p in playlist:
            logger.info("  %s", p.name)

    def _open_cap():
        _pi  = 0
        _cv  = playlist[0] if playlist else source_path
        _cap = cv2.VideoCapture(str(_cv) if playlist else source)
        if not _cap.isOpened() and not playlist:
            try:
                cam_idx = int(source)
                logger.warning("Retrying with CAP_DSHOW for camera %d", cam_idx)
                _cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            except (ValueError, TypeError):
                pass
        if not _cap.isOpened():
            raise RuntimeError(f"Unable to open source: {source}")
        return _cap, _pi, _cv

    init_cap, _, _ = _open_cap()
    frame_w = int(init_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(init_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = int(init_cap.get(cv2.CAP_PROP_FPS)) or 25
    init_cap.release()
    logger.info("Source: %dx%d @ %d FPS", frame_w, frame_h, fps)

    # ── Web server ─────────────────────────────────────────────────────────────
    start_server(port=args.port)
    logger.info("Web UI: http://localhost:%d", args.port)

    # ── Zone setup ─────────────────────────────────────────────────────────────
    auto_zone_mode = config.get("auto_zone_from_vehicle", True)
    if not auto_zone_mode and not args.no_zone_setup:
        probe_cap, _, _ = _open_cap()
        config = _wait_for_zone_setup(probe_cap, config, args.port)
        probe_cap.release()
    else:
        logger.info("Auto-zone mode active — zone locks to first detected vehicle.")

    # ── GPU Warmup ─────────────────────────────────────────────────────────────
    if not args.no_warmup and device != "cpu":
        _warmup(model, device, imgsz)

    # ── Output writer ──────────────────────────────────────────────────────────
    output_path = Path(config["source"].get("output", "output/tyre_counting_fast.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[cv2.VideoWriter] = None
    if output_path.suffix.lower() == ".mp4":
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (frame_w, frame_h),
        )
        logger.info("Output video: %s", output_path)

    # ── Pipeline components ────────────────────────────────────────────────────
    counter       = TyreCounter(config)
    projector     = HomographyProjector(config)
    cnt_cfg       = config["counting"]
    motion_filter = MotionFilter(
        min_movement_px    = cnt_cfg.get("min_movement_px",    5),
        min_box_area_ratio = cnt_cfg.get("min_box_area_ratio", 0.0005),
        max_box_area_ratio = cnt_cfg.get("max_box_area_ratio", 0.20),
    )

    # ── Queues & threads ───────────────────────────────────────────────────────
    frame_q  = queue.Queue(maxsize=args.queue_size)
    result_q = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()

    reader_t = threading.Thread(
        target=_reader_thread,
        args=(_open_cap, frame_q, args.frame_skip, playlist, loop_video, stop_event),
        name="FrameReader",
        daemon=True,
    )
    reader_t.start()

    writer_t = threading.Thread(
        target=_writer_thread,
        args=(result_q, writer, show_window, stop_event),
        name="FrameWriter",
        daemon=True,
    )
    writer_t.start()

    # ── Inference loop (main thread — owns CUDA context) ──────────────────────
    logger.info("🏎️  Fast processor running …  (Ctrl-C to stop)")
    try:
        _run_inference_loop(
            frame_queue      = frame_q,
            result_queue     = result_q,
            model            = model,
            config           = config,
            projector        = projector,
            counter          = counter,
            motion_filter    = motion_filter,
            frame_h          = frame_h,
            frame_w          = frame_w,
            conf_thresh      = conf_thresh,
            imgsz            = imgsz,
            device           = device,
            batch_size       = args.batch_size,
            tracker_cfg      = tracker_cfg,
            tracking_enabled = tracking_en,
            stop_event       = stop_event,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        stop_event.set()
    except Exception:
        logger.exception("Unexpected error in inference loop.")
        stop_event.set()

    # ── Shutdown ───────────────────────────────────────────────────────────────
    stop_event.set()
    reader_t.join(timeout=5)
    writer_t.join(timeout=5)

    if writer is not None:
        writer.release()

    state = counter.get_state()
    logger.info("─── Session Summary ───────────────────────")
    logger.info("Final count  : %d", state["current_count"])
    logger.info("Total entries: %d", state["entry_count"])
    logger.info("Total exits  : %d", state["exit_count"])
    logger.info("───────────────────────────────────────────")


if __name__ == "__main__":
    main()
