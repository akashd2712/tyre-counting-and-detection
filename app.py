import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

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

logger = logging.getLogger("tyre_counter")

WEB_PORT = 5050

# ── 4-class model indices ──────────────────────────────────────────────────────
CLASS_TYRE            = 0   # stationary tyre — drawn but never counted
CLASS_MOVING_TYRE     = 1   # rolling tyre    — the ONLY class that gets counted
CLASS_LOADING_VEHICLE = 2   # truck / forklift — defines the counting zone
CLASS_PERSON          = 3   # worker           — ignored


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv11 tyre counting system")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None)
    p.add_argument("--model",  default=None)
    p.add_argument("--show",   action="store_true")
    p.add_argument("--no-zone-setup", action="store_true",
                   help="Skip interactive zone setup; use config.yaml directly.")
    p.add_argument("--port", type=int, default=WEB_PORT,
                   help=f"Web UI port (default: {WEB_PORT}).")
    return p.parse_args()


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


# ── Vehicle-zone helpers ───────────────────────────────────────────────────────

def _best_vehicle_bbox(detections: list) -> tuple | None:
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
    """
    Compute a NARROW entrance zone. For a vertical plane (side loading),
    this is a vertical slice covering the vehicle's height.
    """
    vx1, vy1, vx2, vy2 = vehicle_bbox
    vcx = (vx1 + vx2) / 2.0
    vcy = (vy1 + vy2) / 2.0
    vw = vx2 - vx1
    vh = vy2 - vy1

    if orientation == "vertical":
        # Vertical gate (side loading)
        # Center the gate on the left edge (vx1) where the label is located
        gate_w = vw * gate_width_ratio
        gx1 = max(0, vx1 - gate_w / 2)
        gx2 = min(frame_w, vx1 + gate_w / 2)
        gy1 = vy1 + vh * 0.05
        gy2 = vy2 - vh * 0.05
    else:
        # Horizontal gate (top loading or downward movement)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args   = parse_args()
    config = load_config(args.config)

    # ── Model ─────────────────────────────────────────────────────────────────
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

    # ── Source ────────────────────────────────────────────────────────────────
    source      = args.source or config["source"].get("input", "0")
    show_window = args.show   or config["source"].get("show_window", False)
    source_path = Path(source)
    extensions  = tuple(config["source"].get("extensions", [".mov", ".mp4", ".avi", ".mkv"]))

    if source_path.is_dir():
        playlist = sorted(p for p in source_path.iterdir()
                          if p.suffix.lower() in extensions)
        if not playlist:
            raise RuntimeError(f"No video files in: {source}")
        logger.info("Folder source — %d video(s):", len(playlist))
        for p in playlist:
            logger.info("  %s", p.name)
        playlist_idx  = 0
        current_video = playlist[0]
        cap = cv2.VideoCapture(str(current_video))
    else:
        playlist      = None
        playlist_idx  = 0
        current_video = source_path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            try:
                cam_idx = int(source)
                logger.warning("Retrying with CAP_DSHOW for camera %d", cam_idx)
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            except (ValueError, TypeError):
                pass

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    logger.info("Source: %dx%d @ %d FPS  [%s]", frame_w, frame_h, fps, current_video.name)

    # ── Web server ────────────────────────────────────────────────────────────
    start_server(port=args.port)
    logger.info("Web UI: http://localhost:%d", args.port)

    # ── Zone setup ────────────────────────────────────────────────────────────
    auto_zone = config.get("auto_zone_from_vehicle", True)
    if not auto_zone and not args.no_zone_setup:
        logger.info("Waiting for manual zone setup via web UI …")
        config = _wait_for_zone_setup(cap, config, args.port)
    else:
        logger.info("Auto-zone mode active — zone locks to first detected vehicle.")

    # ── Output video writer ───────────────────────────────────────────────────
    output_path = Path(config["source"].get("output", "output/tyre_counting.mp4"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if output_path.suffix.lower() == ".mp4":
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (frame_w, frame_h),
        )
        logger.info("Output video: %s", output_path)

    # ── Counter & motion filter ───────────────────────────────────────────────
    counter = TyreCounter(config)
    projector = HomographyProjector(config)
    
    cnt_cfg = config["counting"]
    motion_filter = MotionFilter(
        min_movement_px      = cnt_cfg.get("min_movement_px", 5),
        min_box_area_ratio   = cnt_cfg.get("min_box_area_ratio", 0.0005),
        max_box_area_ratio   = cnt_cfg.get("max_box_area_ratio", 0.20),
    )

    stationary_suppress_frames = cnt_cfg.get("stationary_suppress_frames", 10)
    track_stationary_counts: dict = {}

    # ── Vehicle tracking state ────────────────────────────────────────────────
    auto_zone         = config.get("auto_zone_from_vehicle", True)
    gate_width_ratio  = config.get("gate_width_ratio", 0.40)
    no_vehicle_patience = config.get("no_vehicle_patience", 30)
    
    ez_config = config.get("entrance_zone", {})
    line_ratio = ez_config.get("line_ratio", 0.50)
    orientation = ez_config.get("line_orientation", "vertical")

    # EMA-smoothed vehicle bbox for stable zone
    smoothed_vbbox: tuple | None = None
    vehicle_smooth_alpha = 0.35
    frames_since_vehicle = 0
    counting_paused      = False

    # Inference settings
    conf_thresh = config["model"].get("conf", 0.35)
    imgsz       = config["model"].get("imgsz", 640)
    device      = config["model"].get("device", "cpu")
    tracker_cfg = config["tracking"].get("tracker", "custom_bytetrack.yaml")

    loop_video        = config["source"].get("loop", False)
    frame_idx         = 0
    fps_counter_start = time.time()
    fps_value         = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if playlist is not None:
                    playlist_idx  = (playlist_idx + 1) % len(playlist)
                    current_video = playlist[playlist_idx]
                    cap.release()
                    cap = cv2.VideoCapture(str(current_video))
                    logger.info("Next video: %s", current_video.name)
                    continue
                elif loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_idx += 1
            counter.tick()
            t_start = time.time()

            roi_frame, x_off, y_off = apply_roi(frame, config.get("roi", {}))

            # ── Inference ─────────────────────────────────────────────────────
            if config["tracking"].get("enabled", True):
                results = model.track(
                    roi_frame,
                    conf=conf_thresh,
                    imgsz=imgsz,
                    classes=None,      # detect all 4 classes
                    stream=False,
                    tracker=tracker_cfg,
                    persist=True,
                    device=device,
                )
            else:
                results = model(roi_frame, conf=conf_thresh, imgsz=imgsz,
                                classes=None, stream=False, device=device)

            # ── Coordinate spaces ─────────────────────────────────────────────
            coord_w = config.get("homography", {}).get("map_width", frame_w) if projector.enabled else frame_w
            coord_h = config.get("homography", {}).get("map_height", frame_h) if projector.enabled else frame_h

            # ── Parse detections ──────────────────────────────────────────────
            all_dets = []
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
                        "bbox":     (x1, y1, x2, y2),
                        "class_id": cls_id,
                        "conf":     conf_val,
                        "track_id": tid,
                        "bottom_center": projector.project_bbox_bottom((x1, y1, x2, y2)),
                    })

            if frame_idx == 1:
                counter.initialize_from_detections(all_dets, coord_h, coord_w)

            # ── Vehicle detection + zone locking ──────────────────────────────
            raw_vbbox = _best_vehicle_bbox(all_dets)

            if raw_vbbox is not None:
                frames_since_vehicle = 0
                counting_paused      = False

                # EMA smooth
                if smoothed_vbbox is None:
                    smoothed_vbbox = raw_vbbox
                else:
                    a   = vehicle_smooth_alpha
                    sv  = smoothed_vbbox
                    rv  = raw_vbbox
                    smoothed_vbbox = tuple(int(a * rv[i] + (1 - a) * sv[i])
                                           for i in range(4))

                # Auto-zone: update config entrance zone using SMOOTHED bbox (in 2D space)
                if auto_zone:
                    new_ez = _derive_vehicle_zone(
                        smoothed_vbbox, frame_h, frame_w,
                        gate_width_ratio, line_ratio, orientation
                    )
                    config["entrance_zone"] = new_ez

                    # Fix the counting line ONCE from the smoothed vehicle position
                    # Align the counting line with the left edge of the vehicle (vx1) instead of the center
                    if orientation == "vertical":
                        vx_proj, vy_proj = projector.project_point(smoothed_vbbox[0], smoothed_vbbox[3])
                    else:
                        vx_proj, vy_proj = projector.project_bbox_bottom(smoothed_vbbox)
                    counter.fix_line_from_vehicle((vx_proj, vy_proj), coord_h, coord_w, offset_px=0, orientation=orientation)

            else:
                frames_since_vehicle += 1
                if frames_since_vehicle >= no_vehicle_patience:
                    counting_paused  = True
                    smoothed_vbbox   = None

            # ── Zone pixel bounds ─────────────────────────────────────────────
            ez        = config.get("entrance_zone", {})
            ez_en     = ez.get("enabled", False)
            ez_y1_px  = int(ez.get("y_min", 0.0) * frame_h)
            ez_y2_px  = int(ez.get("y_max", 1.0) * frame_h)
            ez_x1_px  = int(ez.get("x_min", 0.0) * frame_w)
            ez_x2_px  = int(ez.get("x_max", 1.0) * frame_w)

            frame_area = frame_h * frame_w

            # ── Per-detection processing ──────────────────────────────────────
            valid_dets = []

            for det in all_dets:
                cls_id    = det["class_id"]
                x1, y1, x2, y2 = det["bbox"]
                bbox_area = (x2 - x1) * (y2 - y1)
                tid       = det["track_id"]
                cx        = (x1 + x2) / 2.0
                cy        = (y1 + y2) / 2.0

                # Always pass vehicle through to overlay
                if cls_id == CLASS_LOADING_VEHICLE:
                    valid_dets.append(det)
                    continue

                # Size filter
                if not motion_filter.is_valid_size(bbox_area, frame_area):
                    continue

                # Motion / stationary suppression (calculated in PROJECTED space)
                cx_proj, cy_proj = det["bottom_center"]
                prev_center_proj = counter.get_previous_center(tid) if tid is not None else None
                
                is_moving = motion_filter.should_process(
                    prev_center_proj,
                    (cx_proj, cy_proj), bbox_area, frame_area
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
                    in_zone = (ez_y1_px <= cy <= ez_y2_px and
                               ez_x1_px <= cx <= ez_x2_px)

                # ── Count ONLY tyres that passed motion filter ─────────────────────────
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
                            frame_idx, event.upper(), tid, counter.current_count,
                        )

            counter.tick()

            # ── Annotate & stream ─────────────────────────────────────────────
            annotate_frame(
                frame, valid_dets, counter, config,
                vehicle_bbox=smoothed_vbbox if auto_zone else None,
                fixed_line_pos=counter.line_pos,
                counting_paused=counting_paused,
                projector=projector,
            )

            if counting_paused and auto_zone:
                cv2.putText(
                    frame, "COUNTING PAUSED — no vehicle detected",
                    (10, frame_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    (0, 100, 255), 2,
                )

            update_frame(frame)

            if frame_idx % 10 == 0:
                fps_value         = 10.0 / max(time.time() - fps_counter_start, 0.001)
                fps_counter_start = time.time()

            state = counter.get_state()
            update_stats({
                "current_count": state["current_count"],
                "entry_count":   state["entry_count"],
                "exit_count":    state["exit_count"],
                "fps":           round(fps_value, 1),
                "frame_idx":     frame_idx,
            })

            if writer is not None:
                writer.write(frame)

            if show_window:
                cv2.imshow("Tyre Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.time() - t_start
            logger.debug("Frame %d: %d dets, %.1f ms", frame_idx, len(all_dets), elapsed * 1000)
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception:
        logger.exception("Unexpected error.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

        state = counter.get_state()
        logger.info("--- Session Summary ---")
        logger.info("Frames processed : %d", frame_idx)
        logger.info("Final count      : %d", state["current_count"])
        logger.info("Total entries    : %d", state["entry_count"])
        logger.info("Total exits      : %d", state["exit_count"])


if __name__ == "__main__":
    main()
