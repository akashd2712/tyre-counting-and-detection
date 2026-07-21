"""
Interactive Zone Drawing Tool
==============================
Run this ONCE before starting app.py to define your zones by drawing
directly on the first video frame with your mouse.

Usage:
    python setup_zones.py

Controls:
    [E]          → Switch to Entry Zone mode (click + drag rectangle)
    [L]          → Switch to Counting Line mode (click to place horizontal line)
    [C]          → Confirm & save zones to config.yaml
    [R]          → Reset all zones
    [Q] / ESC    → Quit without saving
"""

from __future__ import annotations

import sys
import yaml
import cv2
import numpy as np

CONFIG_PATH = "config.yaml"

# ─── colours ────────────────────────────────────────────────────────────────
COL_ZONE   = (0, 220, 120)   # green  – entry zone
COL_LINE   = (0, 220, 255)   # cyan   – counting line
COL_HANDLE = (255, 255, 255) # white  – drag handles
COL_TEXT   = (255, 255, 255)
COL_DIM    = (120, 120, 120)
COL_BG     = (20, 20, 20)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ─── state ───────────────────────────────────────────────────────────────────
class ZoneDrawer:
    """Holds all mutable drawing state."""

    MODES = ["entry_zone", "counting_line"]

    def __init__(self, frame: np.ndarray):
        self.base   = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.mode   = "entry_zone"   # current drawing mode

        # Entry zone polygon (pixel coords)
        self.polygon: list[tuple[int, int]] = []
        self._poly_done = False

        # Counting line (pixel y)
        self.line_y: int | None = None

    # ── mouse callback ────────────────────────────────────────────────────────
    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))

        if self.mode == "entry_zone":
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self._poly_done:
                    self.polygon.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.polygon) > 2:
                    self._poly_done = True

        elif self.mode == "counting_line":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.line_y = y

    # ── render ────────────────────────────────────────────────────────────────
    def render(self) -> np.ndarray:
        frame = self.base.copy()

        # ── entry zone ──────────────────────────────────────────────────────
        if self.polygon:
            overlay = frame.copy()
            pts = np.array(self.polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            if self._poly_done:
                cv2.fillPoly(overlay, [pts], COL_ZONE)
                cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
                cv2.polylines(frame, [pts], True, COL_ZONE, 2)
            else:
                cv2.polylines(frame, [pts], False, COL_ZONE, 2)

            for cx, cy in self.polygon:
                cv2.circle(frame, (cx, cy), 6, COL_ZONE, -1)
                
            if self._poly_done:
                x1, y1 = self.polygon[0]
                cv2.putText(frame, "ENTRY ZONE", (x1 + 6, y1 + 22),
                            FONT, 0.55, COL_ZONE, 2)

        # ── counting line ────────────────────────────────────────────────────
        if self.line_y is not None:
            ly = self.line_y
            cv2.line(frame, (0, ly), (self.w, ly), COL_LINE, 2)
            cv2.circle(frame, (self.w // 2, ly), 8, COL_LINE, -1)
            cv2.putText(frame, f"COUNTING LINE  (y={round(ly / self.h, 3)})",
                        (10, ly - 8), FONT, 0.50, COL_LINE, 1)

        # ── instruction panel ────────────────────────────────────────────────
        self._draw_panel(frame)
        return frame

    def _draw_panel(self, frame: np.ndarray) -> None:
        panel_w, panel_h = 310, 195
        # dark background
        bg = frame.copy()
        cv2.rectangle(bg, (0, 0), (panel_w, panel_h), COL_BG, -1)
        cv2.addWeighted(bg, 0.72, frame, 0.28, 0, frame)

        mode_label = "ENTRY ZONE (drag)" if self.mode == "entry_zone" \
                     else "COUNTING LINE (click)"
        lines = [
            ("ZONE SETUP TOOL",  (0, 200, 255), 2),
            ("",                 COL_DIM,         1),
            (f"Mode : {mode_label}", COL_TEXT,    1),
            ("",                 COL_DIM,         1),
            ("[E]  Entry zone — click pts, right-click to finish", COL_ZONE, 1),
            ("[L]  Counting line — click to place", COL_LINE, 1),
            ("",                 COL_DIM,         1),
            ("[C]  Confirm & save to config.yaml", (80, 200, 80), 2),
            ("[R]  Reset all zones",              COL_DIM,       1),
            ("[Q]  Quit without saving",          COL_DIM,       1),
        ]
        for i, (txt, col, thick) in enumerate(lines):
            cv2.putText(frame, txt, (10, 18 + i * 18),
                        FONT, 0.40, col, thick)

        # status badges top-right
        ez_ok   = self._poly_done and len(self.polygon) > 2
        line_ok = self.line_y is not None
        self._badge(frame, self.w - 190, 16, "ZONE",  ez_ok)
        self._badge(frame, self.w -  95, 16, "LINE", line_ok)

    @staticmethod
    def _badge(frame, x, y, label, ok):
        col = (40, 180, 40) if ok else (80, 80, 80)
        cv2.rectangle(frame, (x, y - 12), (x + 80, y + 6), col, -1)
        cv2.putText(frame, f"{'✓' if ok else '○'} {label}",
                    (x + 6, y + 2), FONT, 0.38, (255, 255, 255), 1)

    # ── export ────────────────────────────────────────────────────────────────
    def as_config(self) -> dict:
        """Return entrance_zone dict in config.yaml format."""
        if not self._poly_done or len(self.polygon) < 3:
            # Fallback
            poly = [[0.0, 0.60], [1.0, 0.60], [1.0, 1.0], [0.0, 1.0]]
            y1, y2 = int(0.60 * self.h), self.h
            x_min, x_max, y_min, y_max = 0.0, 1.0, 0.60, 1.0
        else:
            poly = [[round(x / self.w, 4), round(y / self.h, 4)] for x, y in self.polygon]
            ys = [p[1] for p in self.polygon]
            xs = [p[0] for p in self.polygon]
            y1, y2 = min(ys), max(ys)
            x_min = min(xs) / self.w
            x_max = max(xs) / self.w
            y_min = y1 / self.h
            y_max = y2 / self.h

        ly = self.line_y if self.line_y is not None else (y1 + y2) // 2

        # Clamp line inside zone
        ly = max(y1, min(ly, y2))

        return {
            "enabled": True,
            "polygon": poly,
            "x_min": round(x_min, 4),
            "x_max": round(x_max, 4),
            "y_min": round(y_min, 4),
            "y_max": round(y_max, 4),
            "line_y": round(ly / self.h, 4),
        }


# ─── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    # Load config to get the video source
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    source = config["source"].get("input", "0")
    try:
        source = int(source)   # camera index
    except ValueError:
        pass   # file path string

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[setup_zones] ERROR: Cannot open source: {source}")
        sys.exit(1)

    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("[setup_zones] ERROR: Could not read first frame.")
        sys.exit(1)

    h, w = first_frame.shape[:2]
    drawer = ZoneDrawer(first_frame)

    win = "Zone Setup  —  [E] Entry Zone   [L] Counting Line   [C] Confirm   [Q] Quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1400), min(h, 800))
    cv2.moveWindow(win, 80, 40)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(win, drawer.on_mouse)

    print("\n[setup_zones] Window opened.")
    print("  Draw your ENTRY ZONE first (click points, right-click to finish).")
    print("  Press [L] to switch to counting-line mode, click to place the line.")
    print("  Press [C] to save & exit.\n")

    saved = False
    while True:
        cv2.imshow(win, drawer.render())
        key = cv2.waitKey(20) & 0xFF

        if key == ord("e"):
            drawer.mode = "entry_zone"
            print("[setup_zones] Mode → ENTRY ZONE")

        elif key == ord("l"):
            drawer.mode = "counting_line"
            print("[setup_zones] Mode → COUNTING LINE")

        elif key == ord("r"):
            drawer.polygon = []
            drawer._poly_done = False
            drawer.line_y = None
            print("[setup_zones] Zones reset.")

        elif key == ord("c"):
            ez_ok   = drawer._poly_done and len(drawer.polygon) > 2
            line_ok = drawer.line_y is not None
            if not ez_ok:
                print("[setup_zones] ⚠ Draw the ENTRY ZONE first (click points, right-click to finish).")
                continue
            if not line_ok:
                print("[setup_zones] ⚠ Place the COUNTING LINE first (press [L] then click).")
                continue

            # Write back to config.yaml
            config["entrance_zone"] = drawer.as_config()
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print("\n[setup_zones] ✓ Zones saved to config.yaml:")
            for k, v in config["entrance_zone"].items():
                print(f"   {k}: {v}")
            print("\n[setup_zones] Now run:  python app.py --no-zone-setup\n")
            saved = True
            break

        elif key in (ord("q"), 27):   # Q or ESC
            print("[setup_zones] Quit — no changes saved.")
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if saved:
        # Optionally launch app.py immediately
        answer = input("Launch app.py now? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "app.py", "--no-zone-setup"])


if __name__ == "__main__":
    main()
