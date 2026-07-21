# Product Requirements Document (PRD)
## Smart Tyre Detection and Bidirectional Counting System

**Version:** 2.0  
**Date:** July 2026  
**Status:** ✅ Implemented & Running

---

## 1. Product Overview

The **Smart Tyre Detection and Bidirectional Counting System** is a computer vision application designed to automate tracking and counting of individual rolling tyres in warehouse environments. Warehouse workers roll tyres across a camera's field of view; the system detects each tyre, tracks it across frames, and counts it when it crosses a user-defined virtual line.

The system uses a **custom-trained YOLOv11 model** (class: `tyre`) for detection and **ByteTrack** for multi-object tracking. A **Flask-based web dashboard** streams the annotated video live and displays real-time entry/exit counts. Before processing begins, operators draw their own entry zone and counting line directly in the browser — no code editing required.

---

## 2. Objectives & Goals

| Goal | Status |
|------|--------|
| Detect only tyres (not workers, machinery, or background) | ✅ Custom `best.pt` model trained on tyre dataset |
| Track rolling tyres across frames without double-counting | ✅ ByteTrack integration |
| Count entries and exits bidirectionally | ✅ `counting/counter.py` state machine |
| Let operators define zones interactively | ✅ Web-based zone drawing tool (`/zone_setup`) |
| Stream annotated video to a live web dashboard | ✅ MJPEG stream via Flask (`utils/web_server.py`) |
| Filter out stationary tyre stacks | ✅ Motion filter + size filter in `app.py` |
| Loop video for continuous operation / testing | ✅ `source.loop: true` in `config.yaml` |

---

## 3. Use Cases

- **Warehouse Loading Docks:** Count tyres as workers roll them into or out of shipping containers.
- **Dispatch Verification:** Confirm the correct number of tyres leave or enter a storage bay.
- **Inventory Audits:** Run recorded CCTV footage through the system to reconcile daily movement.

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         app.py  (Core Controller)                │
│                                                                  │
│  VideoCapture ──► ROI Crop ──► YOLO Inference ──► ByteTrack     │
│                                     │                            │
│                              Motion Filter                       │
│                              Size Filter                         │
│                              Zone Spatial Filter (x/y bounds)   │
│                                     │                            │
│                           counting/counter.py                    │
│                     (Line-crossing state machine)                │
│                                     │                            │
│                        utils/visualization.py                    │
│                   (Bounding boxes, zone, counts HUD)             │
│                                     │                            │
│                         utils/web_server.py                      │
│               (Flask MJPEG stream + zone setup UI)               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. File & Module Reference

### Root Files

| File | Purpose |
|------|---------|
| [`app.py`](app.py) | Main entry point. Orchestrates all modules. |
| [`config.yaml`](config.yaml) | All runtime settings (model, source, zones, thresholds). |
| [`train_multiclass.py`](train_multiclass.py) | **Primary** high-accuracy 5-class YOLOv11 training script. |
| [`train.py`](train.py) | Legacy YOLO fine-tuning script. |
| [`train_fixed.py`](train_fixed.py) | Legacy fixed training script. |
| [`auto_label.py`](auto_label.py) | Automatically generates draft YOLO annotations using a pre-trained model. |
| [`convert_dataset.py`](convert_dataset.py) | Converts exported COCO JSON annotations to YOLOv11 TXT format. |
| [`extract_more_frames.py`](extract_more_frames.py) | Extracts additional raw image frames from source videos for the dataset. |
| [`prepare_dataset.py`](prepare_dataset.py) | Splits images into train/val sets. |
| [`detect_dataset.py`](detect_dataset.py) | Runs inference on the dataset and generates YOLO labels. |
| [`make_video.py`](make_video.py) | Assembles annotated frames into an output video. |
| [`setup_zones.py`](setup_zones.py) | Standalone OpenCV zone-drawing tool (alternative to web UI). |
| [`requirements.txt`](requirements.txt) | Python dependencies. |

### `counting/` Module

| File | Purpose |
|------|---------|
| [`counting/counter.py`](counting/counter.py) | `TyreCounter` state machine — tracks each tyre's centroid history, detects line-crossing, increments/decrements inventory. |

### `utils/` Module

| File | Purpose |
|------|---------|
| [`utils/roi.py`](utils/roi.py) | Crops the frame to the configured Region of Interest. |
| [`utils/motion.py`](utils/motion.py) | Computes centroid movement between frames to filter stationary objects. |
| [`utils/visualization.py`](utils/visualization.py) | Draws bounding boxes, HUD (entry/exit counts, FPS), entry-zone rectangle, and configurable counting line (horizontal / vertical / diagonal). |
| [`utils/web_server.py`](utils/web_server.py) | Flask app serving the live MJPEG stream (`/`), zone setup UI (`/zone_setup`), and zone-confirm endpoint (`/zone_confirm`). |
| [`utils/zone_setup.py`](utils/zone_setup.py) | Legacy OpenCV-based interactive zone drawing. |

### Model Weights

| File | Description |
|------|-------------|
| `runs/detect/models/tyre_yolo11/weights/best.pt` | **Primary model** — custom YOLOv11 trained on warehouse tyre footage. Single class: `tyre`. |
| `yolo11n.pt` | Fallback COCO model (used only if `best.pt` is unavailable). |

---

## 6. Configuration Reference (`config.yaml`)

```yaml
model:
  path: "runs/detect/models/tyre_yolo11/weights/best.pt"  # custom tyre model
  fallback_path: "yolo11n.pt"
  conf: 0.30          # detection confidence threshold
  imgsz: 1280         # inference image size
  device: "0"         # "0" = GPU 0, "cpu" = CPU
  classes: [0]        # class 0 = tyre in the custom model

source:
  input: "dataset/dataset_video.mp4"  # video file or RTSP URL or camera index
  output: "output/tyre_counting.mp4"
  show_window: false  # OpenCV window (off = headless, use web UI)
  loop: true          # loop video file continuously

roi:
  enabled: false      # set to true + add x1/y1/x2/y2 to restrict frame region

counting:
  entry_direction: "down"      # direction that counts as an entry
  min_movement_px: 3           # centroid must move this many px between frames
  min_box_area_ratio: 0.0005   # ignore detections smaller than this (% of frame)
  max_box_area_ratio: 0.25     # ignore detections larger than this (% of frame)
  cooldown_frames: 4           # suppress re-count for N frames after crossing
  stationary_suppress_frames: 15

entrance_zone:
  enabled: true
  y_min: 0.30               # top edge of the zone (ratio of frame height)
  line_y: 0.50              # legacy absolute line position
  line_orientation: horizontal  # horizontal | vertical | diagonal
  line_ratio: 0.5           # line position within zone (0.0 = top, 1.0 = bottom)
  x_min: 0.00               # left edge of the zone
  x_max: 1.00               # right edge of the zone

tracking:
  enabled: true
  persist: true
  tracker: "bytetrack.yaml"
  iou: 0.5
  conf: 0.25
```

---

## 7. Key Features & Functional Requirements

### 7.1 Custom Tyre Detection
- **Model:** YOLOv11 fine-tuned on real warehouse footage. Multi-class architecture (5 classes):
  - `moving_tyre` (Class 0): Actively rolled/pushed. The **only** class that triggers counting.
  - `stationary_tyre` (Class 1): Resting on floor. Ignored.
  - `tyre_group` (Class 2): Racks/stacks. Ignored.
  - `person` (Class 3): Warehouse worker. Ignored.
  - `loading_vehicle` (Class 4): Forklift, pallet jack, truck. Ignored.
- **Inference size:** 1280 px (high-resolution for accurate small-object detection).
- **Confidence threshold:** 0.25 (configurable).
- **Fallback:** Auto-downloads `yolo11n.pt` if custom weights are missing.

### 7.2 Interactive Zone Drawing (Web UI)
- Before processing starts, the operator navigates to **`http://localhost:5050/zone_setup`**.
- **Step 1 — Draw Entry Zone:** Click and drag a free rectangle over any part of the frame.
- **Step 2 — Place Counting Line:** Click inside the rectangle; a line appears. Orientation can be set to **horizontal**, **vertical**, or **diagonal** using toolbar buttons.
- **Step 3 — Confirm:** Zones are written to `config.yaml` and the counting loop begins.
- The rectangle and line can be resized, repositioned, and reset at any time.

### 7.3 Spatial Zone Filter
- Detections whose centroid falls **outside** the drawn rectangle (`x_min/x_max/y_min/y_max`) are silently discarded before reaching the counter.
- Eliminates false counts from tyres visible in other parts of the frame.

### 7.4 Motion & Size Filters
- **Motion filter (`utils/motion.py`):** Tracks centroid history; ignores detections that haven't moved ≥ `min_movement_px` across recent frames. Filters out stationary stacked tyres.
- **Size filter:** Bounding boxes below `min_box_area_ratio` (noise) or above `max_box_area_ratio` (background stacks) are rejected.
- **Stationary suppression:** A track silent for `stationary_suppress_frames` frames is reset.

### 7.5 Bidirectional Counting
- A virtual line (inside the entry zone) divides the zone into two halves.
- When a tracked tyre's centroid crosses the line **top-to-bottom** (`entry_direction: down`) → **+1 entry, inventory +1**.
- Crossing **bottom-to-top** → **+1 exit, inventory −1**.
- A `cooldown_frames` guard prevents flicker-counts on jitter.

### 7.6 Configurable Counting Line Orientation
- **Horizontal** (default): Standard left-to-right line; ideal for camera looking down a corridor.
- **Vertical**: Line runs top-to-bottom; suited for a side-on camera angle.
- **Diagonal**: Corner-to-corner of the zone; for angled camera setups.
- Position within the zone is controlled by `line_ratio` (0.0 – 1.0).

### 7.7 Live Web Dashboard
- **URL:** `http://localhost:5050`
- Serves an MJPEG stream annotated with:
  - Green bounding boxes + tyre confidence score + track ID
  - Entry zone rectangle overlay (semi-transparent green tint)
  - Counting line (cyan) with label
  - HUD: Total Count · Entries · Exits · FPS
- Auto-refreshes; no installation needed — just open in any browser.

### 7.8 Video Loop Support
- When `source.loop: true`, the video rewinds and repeats indefinitely.
- Useful for testing with a fixed dataset video or for 24/7 monitoring scenarios.

---

## 8. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.9+ |
| Detection | Ultralytics YOLO (YOLOv11) | ≥ 8.3.0 |
| Tracking | ByteTrack (via Ultralytics) | built-in |
| Deep Learning | PyTorch | ≥ 2.2.0 |
| Computer Vision | OpenCV | ≥ 4.10.0 |
| Web Server | Flask (via `utils/web_server.py`) | included |
| Configuration | PyYAML | ≥ 6.0.0 |
| Supervision | Supervision | ≥ 0.21.0 |

---

## 9. Non-Functional Requirements

| Requirement | Target | Current State |
|-------------|--------|---------------|
| Detection accuracy | Tyre only (no workers/stacks) | ✅ Custom model, class = `tyre` |
| CPU inference speed | Acceptable for 10 FPS source | ✅ ~120–200 ms/frame on CPU |
| GPU inference speed | ≥ 25 FPS real-time | Ready (set `device: "0"` in config) |
| Graceful fallback | Auto-download base model if custom unavailable | ✅ `fallback_path` in config |
| Headless operation | No OpenCV window needed | ✅ `show_window: false` + web UI |
| Modular codebase | Separate ML / counting / UI | ✅ Enforced by module structure |

---

## 10. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the system
python app.py

# 3. Open browser
# → Zone Setup:   http://localhost:5050/zone_setup
# → Live Feed:    http://localhost:5050

# 4. (Optional) Retrain the model on new footage
python train_multiclass.py
```

---

## 11. Roadmap / Future Enhancements

- [ ] **Multi-class detection:** Distinguish tyre sizes or types (car vs. truck tyre).
- [ ] **RTSP / IP camera support:** Accept live camera streams (change `source.input` to RTSP URL).
- [ ] **Database logging:** Save each crossing event (timestamp, direction, track ID) to SQLite or PostgreSQL.
- [ ] **Multi-camera support:** Run multiple streams concurrently from a single backend.
- [ ] **Alert / notification:** Send a webhook or email when count thresholds are crossed.
- [ ] **GPU acceleration:** Verify RTX-series inference for real-time 30 FPS pipeline.
- [ ] **Zone presets:** Save and reload named zone configurations for different camera positions.
