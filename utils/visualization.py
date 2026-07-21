import cv2
import numpy as np

# ── 4-class model colour map ───────────────────────────────────────────────────
#   0  tyre            → yellow        (stationary — drawn but NOT counted)
#   1  moving_tyre     → bright green  (the ONLY class that gets counted)
#   2  loading_vehicle → magenta       (defines the auto-zone)
#   3  person          → sky blue      (never counted)
CLASS_COLORS = {
    0: (0,   210, 255),    # tyre            — yellow  (BGR)
    1: (0,   220,  60),    # moving_tyre     — green
    2: (200,   0, 200),    # loading_vehicle — magenta
    3: (220, 130,   0),    # person          — sky blue (BGR)
}
CLASS_NAMES = {
    0: "tyre",
    1: "moving_tyre",
    2: "loading_vehicle",
    3: "person",
}
_FALLBACK_COLOR = (0, 255, 0)
_FALLBACK_LABEL = "object"


def _class_style(class_id: int):
    return (
        CLASS_COLORS.get(class_id, _FALLBACK_COLOR),
        CLASS_NAMES.get(class_id, _FALLBACK_LABEL),
    )


# ── Zone / line drawing ────────────────────────────────────────────────────────

def draw_vehicle_zone(
    frame: np.ndarray,
    vehicle_bbox,
    config: dict,
    fixed_line_pos: float,
) -> None:
    """
    Draw:
      - The full vehicle bounding box (magenta)
      - The narrow counting gate (green rectangle)
      - The FIXED counting line (cyan — never moves after first lock)
    """
    h, w = frame.shape[:2]

    if vehicle_bbox is None:
        # ── Waiting overlay ───────────────────────────────────────────────────
        msg = "Waiting for loading vehicle..."
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x0 = (w - tw) // 2
        y0 = h // 2
        cv2.rectangle(frame, (x0 - 12, y0 - th - 12), (x0 + tw + 12, y0 + 12), (20, 20, 20), -1)
        cv2.putText(frame, msg, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        return

    # Full vehicle box (magenta outline)
    vx1, vy1, vx2, vy2 = vehicle_bbox
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), CLASS_COLORS[2], 2)
    cv2.putText(frame, "LOADING VEHICLE", (vx1 + 4, max(vy1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS[2], 2)

    # Narrow counting gate (green rectangle from entrance_zone config)
    ez = config.get("entrance_zone", {})
    orientation = ez.get("line_orientation", "vertical")
    
    if ez.get("enabled", False):
        gx1 = int(ez.get("x_min", 0.0) * w)
        gx2 = int(ez.get("x_max", 1.0) * w)
        gy1 = int(ez.get("y_min", 0.0) * h)
        gy2 = int(ez.get("y_max", 1.0) * h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 200, 100), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 220, 120), 2)
        cv2.putText(frame, "COUNTING GATE (AUTO)", (gx1 + 4, gy1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 120), 2)

        # ── FIXED counting line ────────────────────────────────────────────────
        buf_px = config.get("counting", {}).get("buffer_zone_px", 30)

        if orientation == "vertical":
            line_pos_px = int(fixed_line_pos * w)
            cv2.line(frame, (line_pos_px, gy1), (line_pos_px, gy2), (0, 255, 255), 3)
            cv2.putText(frame, "FIXED LINE", (line_pos_px + 4, gy1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if buf_px > 0:
                bov = frame.copy()
                cv2.rectangle(bov,
                              (line_pos_px - buf_px, gy1),
                              (line_pos_px + buf_px, gy2),
                              (0, 255, 255), -1)
                cv2.addWeighted(bov, 0.08, frame, 0.92, 0, frame)
        else:
            line_pos_px = int(fixed_line_pos * h)
            cv2.line(frame, (gx1, line_pos_px), (gx2, line_pos_px), (0, 255, 255), 3)
            cv2.putText(frame, "FIXED LINE", (gx1 + 4, line_pos_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if buf_px > 0:
                bov = frame.copy()
                cv2.rectangle(bov,
                              (gx1, line_pos_px - buf_px),
                              (gx2, line_pos_px + buf_px),
                              (0, 255, 255), -1)
                cv2.addWeighted(bov, 0.08, frame, 0.92, 0, frame)


def draw_entrance_zone(frame: np.ndarray, config: dict, fixed_line_pos: float) -> None:
    """Manual (non-auto) zone drawing — used when auto_zone is off."""
    ez = config.get("entrance_zone", {})
    if not ez.get("enabled", False):
        return

    h, w = frame.shape[:2]
    x_min = int(ez.get("x_min", 0.00) * w)
    x_max = int(ez.get("x_max", 1.00) * w)
    y_min = int(ez.get("y_min", 0.60) * h)
    y_max = int(ez.get("y_max", 1.00) * h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 200, 100), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 220, 120), 2)
    cv2.putText(frame, "ENTRY ZONE", (x_min + 6, y_min + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

    # Fixed counting line
    orientation = ez.get("line_orientation", "vertical")
    if orientation == "diagonal":
        p1 = ez.get("point1", [0.0, 0.0])
        p2 = ez.get("point2", [1.0, 1.0])
        x1, y1 = int(p1[0] * w), int(p1[1] * h)
        x2, y2 = int(p2[0] * w), int(p2[1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, "DIAGONAL LINE", (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        buf_px = config.get("counting", {}).get("buffer_zone_px", 30)
        if buf_px > 0:
            bov = frame.copy()
            # Draw a thick line to represent the buffer zone
            cv2.line(bov, (x1, y1), (x2, y2), (0, 255, 255), buf_px * 2)
            cv2.addWeighted(bov, 0.08, frame, 0.92, 0, frame)
    elif orientation == "vertical":
        line_pos_px = int(fixed_line_pos * w)
        cv2.line(frame, (line_pos_px, y_min), (line_pos_px, y_max), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (line_pos_px + 6, y_min + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    else:
        line_pos_px = int(fixed_line_pos * h)
        cv2.line(frame, (x_min, line_pos_px), (x_max, line_pos_px), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (x_min + 6, line_pos_px - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def draw_roi_boundary(frame: np.ndarray, config: dict) -> None:
    roi = config.get("roi", {})
    if not roi.get("enabled", False):
        return
    h, w = frame.shape[:2]
    x1 = int(w * roi.get("x1", 0.06))
    y1 = int(h * roi.get("y1", 0.20))
    x2 = int(w * roi.get("x2", 0.94))
    y2 = int(h * roi.get("y2", 0.94))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(frame, "ROI", (x1 + 5, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def _draw_legend(frame: np.ndarray, counter_state: dict) -> None:
    h, w = frame.shape[:2]
    line_label = f"FIXED line pos={counter_state.get('line_pos', 0):.3f}"
    if counter_state.get('orientation') == 'diagonal':
        line_label = "DIAGONAL line"

    items = [
        (CLASS_COLORS[1], "moving_tyre  ← COUNTED"),
        (CLASS_COLORS[0], "tyre (stationary)"),
        (CLASS_COLORS[2], "loading_vehicle (auto-zone)"),
        (CLASS_COLORS[3], "person"),
        ((0, 255, 255), f"{line_label}  "
                        f"{'LOCKED' if counter_state.get('line_fixed') or counter_state.get('orientation') == 'diagonal' else 'pending'}"),
    ]
    x0   = 10
    y0   = h - 10 - len(items) * 22
    bsz  = 14
    for color, text in items:
        cv2.rectangle(frame, (x0, y0), (x0 + bsz, y0 + bsz), color, -1)
        cv2.putText(frame, text, (x0 + bsz + 6, y0 + bsz - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        y0 += 22


def annotate_frame(
    frame: np.ndarray,
    detections: list,
    counter,
    config: dict,
    vehicle_bbox=None,
    fixed_line_pos: float = 0.5,
    counting_paused: bool = False,
    projector=None,
) -> None:
    """Draw all overlays, bounding boxes, HUD, legend onto frame in place."""
    h, w = frame.shape[:2]

    draw_roi_boundary(frame, config)

    auto_zone = config.get("auto_zone_from_vehicle", True)
    if auto_zone:
        draw_vehicle_zone(frame, vehicle_bbox, config, fixed_line_pos)
    else:
        draw_entrance_zone(frame, config, fixed_line_pos)

    # ── HUD ───────────────────────────────────────────────────────────────────
    state = counter.get_state()

    # Dark background pill for readability
    cv2.rectangle(frame, (0, 0), (230, 110), (10, 10, 10), -1)
    cv2.addWeighted(frame[:110, :230], 0.6, frame[:110, :230], 0.4, 0, frame[:110, :230])

    status_color = (0, 80, 200) if counting_paused else (0, 220, 60)
    cv2.putText(frame, f"COUNT: {state['current_count']}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)
    cv2.putText(frame, f"IN:  {state['entry_count']}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"OUT: {state['exit_count']}",
                (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    # Frame counter top-right
    cv2.putText(frame, f"Frame: {state.get('frame_idx', '')}",
                (w - 170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # ── Bounding boxes ────────────────────────────────────────────────────────
    for det in detections:
        cls_id = det.get("class_id", 0)

        # Loading vehicle is drawn by draw_vehicle_zone()
        if cls_id == 2:
            continue

        x1, y1, x2, y2 = det["bbox"]
        conf     = det["conf"]
        track_id = det.get("track_id")
        color, cls_label = _class_style(cls_id)

        # Thicker border for moving_tyre
        thickness = 3 if cls_id == 1 else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        parts = [cls_label]
        if track_id is not None:
            parts.append(str(track_id))
        parts.append(f"{conf:.2f}")
        label = " ".join(parts)

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        ly = max(y1 - 2, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + bl), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

    # ── Legend ────────────────────────────────────────────────────────────────
    _draw_legend(frame, state)

    # ── Homography PiP (Bird's Eye View) ──────────────────────────────────────
    if projector is not None and projector.enabled:
        pip_size = 200
        map_w = config.get("homography", {}).get("map_width", 500)
        map_h = config.get("homography", {}).get("map_height", 500)
        
        # Create a dark PiP background
        pip = np.zeros((pip_size, pip_size, 3), dtype=np.uint8)
        pip[:] = (20, 25, 20)
        
        # Draw the counting line in PiP
        scale_x = pip_size / map_w
        scale_y = pip_size / map_h
        
        orientation = config.get("entrance_zone", {}).get("line_orientation", "vertical")
        if orientation == "vertical":
            lx = int(fixed_line_pos * map_w * scale_x)
            cv2.line(pip, (lx, 0), (lx, pip_size), (0, 255, 255), 2)
        else:
            ly = int(fixed_line_pos * map_h * scale_y)
            cv2.line(pip, (0, ly), (pip_size, ly), (0, 255, 255), 2)
            
        # Draw vehicle position in PiP if available
        if vehicle_bbox:
            vx, vy = projector.project_bbox_bottom(vehicle_bbox)
            cv2.circle(pip, (int(vx * scale_x), int(vy * scale_y)), 6, CLASS_COLORS[2], -1)
            
        # Draw detections in PiP
        for det in detections:
            if "bottom_center" in det:
                px, py = det["bottom_center"]
                cls_id = det.get("class_id", 0)
                color, _ = _class_style(cls_id)
                cv2.circle(pip, (int(px * scale_x), int(py * scale_y)), 4, color, -1)
                
        cv2.putText(pip, "BIRD'S EYE VIEW", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Overlay PiP in bottom-right
        frame[h - pip_size - 10:h - 10, w - pip_size - 10:w - 10] = pip
        cv2.rectangle(frame, (w - pip_size - 10, h - pip_size - 10), (w - 10, h - 10), (100, 100, 100), 2)
