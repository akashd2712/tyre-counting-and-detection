"""
Direct tyre detection and counting on dataset frames.

Bypasses the full pipeline's motion/crossing logic and instead:
1. Runs YOLO detection with SAHI-style sliced inference on each frame
2. Counts the number of unique tyres detected per frame
3. Streams results to the web UI at http://localhost:5050
"""

import glob
import os
import re
import time

import cv2
import numpy as np
from ultralytics import YOLO

from utils.web_server import start_server, update_frame, update_stats
from utils.visualization import annotate_frame


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(s))]


def sahi_detect(model, frame, slice_h=640, slice_w=640, overlap=0.2, conf=0.20):
    """
    Run YOLO on overlapping tiles of the frame and merge results with NMS.
    This catches small/medium tyres that get lost at low resolution.
    """
    h, w = frame.shape[:2]
    step_h = int(slice_h * (1 - overlap))
    step_w = int(slice_w * (1 - overlap))

    all_boxes = []
    all_confs = []
    all_classes = []

    # Also run on full frame for large objects
    for scale_frame, x_off, y_off in _generate_slices(frame, h, w, slice_h, slice_w, step_h, step_w):
        results = model(scale_frame, conf=conf, imgsz=640, verbose=False)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Map back to full-frame coordinates
                x1 += x_off
                y1 += y_off
                x2 += x_off
                y2 += y_off
                all_boxes.append([x1, y1, x2, y2])
                all_confs.append(float(box.conf[0]))
                all_classes.append(int(box.cls[0]))

    if not all_boxes:
        return []

    # NMS to merge overlapping detections from adjacent tiles
    boxes_np = np.array(all_boxes, dtype=np.float32)
    confs_np = np.array(all_confs, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes_np.tolist(), confs_np.tolist(), conf, 0.4
    )

    detections = []
    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            x1, y1, x2, y2 = [int(v) for v in boxes_np[idx]]
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_id": all_classes[idx],
                "conf": confs_np[idx],
                "track_id": idx,  # use NMS index as pseudo-ID
            })

    return detections


def _generate_slices(frame, h, w, slice_h, slice_w, step_h, step_w):
    """Yield (crop, x_offset, y_offset) for overlapping tiles + full frame."""
    # Full frame (rescaled)
    yield frame, 0, 0

    # Tiles
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            y2 = min(y + slice_h, h)
            x2 = min(x + slice_w, w)
            crop = frame[y:y2, x:x2]
            if crop.shape[0] < 64 or crop.shape[1] < 64:
                continue
            yield crop, x, y


def main():
    print("Loading model...")
    model = YOLO("runs/detect/models/tyre_yolo11/weights/best.pt")

    img_dir = "dataset/images"
    pattern = os.path.join(img_dir, "frame_[0-9]*.jpg")
    files = sorted(glob.glob(pattern), key=natural_sort_key)
    print(f"Found {len(files)} frames")

    # Start web server
    start_server(port=5050)
    print("Web UI: http://localhost:5050")
    print("Processing frames...")

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # Track cumulative counts
    total_tyres_seen = 0
    max_in_single_frame = 0
    fps_start = time.time()

    for i, fpath in enumerate(files):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        start = time.time()

        # SAHI-style sliced detection
        detections = sahi_detect(model, frame, slice_h=640, slice_w=640, overlap=0.25, conf=0.20)

        n_det = len(detections)
        max_in_single_frame = max(max_in_single_frame, n_det)
        total_tyres_seen += n_det

        # Draw detections on frame
        h, w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"tyre {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Stats overlay
        cv2.putText(frame, f"Tyres in frame: {n_det}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Max seen: {max_in_single_frame}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i+1}/{len(files)}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Stream to web
        update_frame(frame)

        elapsed = time.time() - start
        if (i + 1) % 10 == 0:
            fps_val = 10.0 / max(time.time() - fps_start, 0.001)
            fps_start = time.time()
        else:
            fps_val = 1.0 / max(elapsed, 0.001)

        update_stats({
            "current_count": n_det,
            "entry_count": max_in_single_frame,
            "exit_count": total_tyres_seen,
            "fps": round(fps_val, 1),
            "frame_idx": i + 1,
        })

        # Save annotated frame
        out_path = os.path.join(out_dir, f"det_{os.path.basename(fpath)}")
        cv2.imwrite(out_path, frame)

        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)  # ~20 fps cap for web viewing

    print(f"\n--- Results ---")
    print(f"Frames processed: {len(files)}")
    print(f"Max tyres in single frame: {max_in_single_frame}")
    print(f"Total tyre detections: {total_tyres_seen}")
    print(f"Annotated frames saved to: {out_dir}/")
    print("Web server still running at http://localhost:5050 (Ctrl+C to stop)")

    # Keep server alive to view last frame
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
