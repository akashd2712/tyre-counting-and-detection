"""
Interactive zone setup tool.

Usage:
    from utils.zone_setup import run_zone_setup

    config = run_zone_setup(cap, config)

How the UI works
----------------
A single frozen frame from the camera is shown.  The user:

  1. Drags the top/bottom green borders to define the ENTRANCE ZONE band.
  2. Drags the cyan line inside the band to set the COUNTING LINE.
  3. Press  [c]  to confirm and continue.
     Press  [r]  to reset to the values in config.yaml.
     Press  [q] / ESC  to skip and keep the config.yaml values.

All values are stored as ratios (0-1) so they are resolution-independent.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# State shared across mouse callbacks
# ---------------------------------------------------------------------------
class _ZoneState:
    """Mutable state bundle passed to the OpenCV mouse callback."""

    IDLE = "idle"
    DRAG_TOP = "drag_top"
    DRAG_BOTTOM = "drag_bottom"
    DRAG_LINE = "drag_line"

    def __init__(self, h: int, w: int, default_ez: dict):
        self.h = h
        self.w = w

        self.ez_top: int = int(default_ez.get("y_min", 0.60) * h)
        self.ez_bot: int = int(default_ez.get("y_max", 1.00) * h)
        self.line_y: int = int(default_ez.get("line_y", 0.80) * h)

        self.mode: str = self.IDLE
        self._drag_start_y: int = 0

    def _clamp(self, v: int) -> int:
        return max(0, min(self.h - 1, v))

    def as_ratios(self) -> dict:
        return {
            "enabled": True,
            "y_min": round(self.ez_top / self.h, 4),
            "y_max": round(self.ez_bot / self.h, 4),
            "line_y": round(self.line_y / self.h, 4),
        }

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        y = self._clamp(y)

        if event == cv2.EVENT_LBUTTONDOWN:
            dists = {
                self.DRAG_TOP: abs(y - self.ez_top),
                self.DRAG_BOTTOM: abs(y - self.ez_bot),
                self.DRAG_LINE: abs(y - self.line_y),
            }
            nearest, dist = min(dists.items(), key=lambda kv: kv[1])
            self.mode = nearest if dist <= 20 else self.IDLE
            self._drag_start_y = y

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.mode == self.DRAG_TOP:
                self.ez_top = min(y, self.ez_bot - 10)
                self.line_y = max(self.line_y, self.ez_top)
                self.line_y = min(self.line_y, self.ez_bot)
            elif self.mode == self.DRAG_BOTTOM:
                self.ez_bot = max(y, self.ez_top + 10)
                self.line_y = max(self.line_y, self.ez_top)
                self.line_y = min(self.line_y, self.ez_bot)
            elif self.mode == self.DRAG_LINE:
                self.line_y = max(self.ez_top, min(y, self.ez_bot))

        elif event == cv2.EVENT_LBUTTONUP:
            self.mode = self.IDLE


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_ui(base: np.ndarray, state: _ZoneState) -> np.ndarray:
    """Return an annotated copy of *base* with the current zone handles."""
    frame = base.copy()
    h, w = frame.shape[:2]

    # Semi-transparent entrance zone tint
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, state.ez_top), (w, state.ez_bot), (0, 200, 100), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # Top border handle
    cv2.line(frame, (0, state.ez_top), (w, state.ez_top), (0, 220, 120), 2)
    cv2.circle(frame, (w // 2, state.ez_top), 8, (0, 255, 150), -1)
    cv2.putText(frame, "ENTRANCE TOP  (drag)", (w // 2 + 12, state.ez_top - 6),
                _FONT, 0.5, (0, 255, 150), 1)

    # Bottom border handle
    cv2.line(frame, (0, state.ez_bot), (w, state.ez_bot), (0, 220, 120), 2)
    cv2.circle(frame, (w // 2, state.ez_bot), 8, (0, 255, 150), -1)
    cv2.putText(frame, "ENTRANCE BOTTOM  (drag)", (w // 2 + 12, state.ez_bot + 16),
                _FONT, 0.5, (0, 255, 150), 1)

    # Counting line handle
    cv2.line(frame, (0, state.line_y), (w, state.line_y), (0, 255, 255), 2)
    cv2.circle(frame, (w // 2, state.line_y), 8, (0, 200, 255), -1)
    cv2.putText(frame, "COUNTING LINE  (drag)", (w // 2 + 12, state.line_y - 6),
                _FONT, 0.5, (0, 200, 255), 1)

    # Instruction panel (top-left)
    instructions = [
        "ZONE SETUP",
        "",
        "Drag handles to position zones:",
        "  GREEN lines  = Entrance zone top / bottom",
        "  CYAN  line   = Counting line",
        "",
        "[ C ]  Confirm & start counting",
        "[ R ]  Reset to config.yaml defaults",
        "[ Q ]  Skip (use config.yaml as-is)",
    ]
    box_w, box_h = 330, 195
    bg = frame.copy()
    cv2.rectangle(bg, (0, 0), (box_w, box_h), (20, 20, 20), -1)
    cv2.addWeighted(bg, 0.70, frame, 0.30, 0, frame)

    for i, line in enumerate(instructions):
        color = (0, 220, 255) if i == 0 else (255, 255, 255)
        thickness = 2 if i == 0 else 1
        cv2.putText(frame, line, (10, 18 + i * 19), _FONT, 0.44, color, thickness)

    # Live ratio readout (bottom bar)
    ratios = state.as_ratios()
    info = (
        f"ez_top={ratios['y_min']:.3f}   "
        f"ez_bot={ratios['y_max']:.3f}   "
        f"line_y={ratios['line_y']:.3f}"
    )
    cv2.putText(frame, info, (10, h - 10), _FONT, 0.42, (200, 200, 200), 1)

    return frame


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_zone_setup(cap: cv2.VideoCapture, config: dict) -> dict:
    """
    Show an interactive zone-setup window.

    Parameters
    ----------
    cap    : Already-opened VideoCapture (seek position is not changed).
    config : Current config dict (used to seed default handle positions).

    Returns
    -------
    Updated config dict with the new entrance_zone values applied.
    If the user skips, the original config is returned unchanged.
    """
    # Grab one frame to use as the background image
    saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, base_frame = cap.read()
    if not ret:
        print("[zone_setup] Could not read a frame — skipping zone setup.")
        return config

    # Try to restore stream position (works for file sources)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
    except Exception:
        pass

    h, w = base_frame.shape[:2]
    default_ez = config.get("entrance_zone", {})
    state = _ZoneState(h, w, default_ez)

    win = "Zone Setup  —  [C] Confirm   [R] Reset   [Q] Skip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))

    # Force the window to appear at the top-left of the primary monitor
    cv2.moveWindow(win, 100, 50)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)

    cv2.setMouseCallback(win, state.on_mouse)

    # Paint the very first frame immediately so the window is visible at once
    cv2.imshow(win, _draw_ui(base_frame, state))
    cv2.waitKey(1)

    result_ratios: Optional[dict] = None

    while True:
        cv2.imshow(win, _draw_ui(base_frame, state))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("c"):
            result_ratios = state.as_ratios()
            print("[zone_setup] Zones confirmed:")
            for k, v in result_ratios.items():
                print(f"  {k}: {v}")
            break

        elif key == ord("r"):
            state = _ZoneState(h, w, default_ez)
            cv2.setMouseCallback(win, state.on_mouse)

        elif key == ord("q") or key == 27:   # Q or ESC
            print("[zone_setup] Skipped — using config.yaml values.")
            break

    cv2.destroyWindow(win)
    cv2.waitKey(1)   # flush the destroy event

    if result_ratios is not None:
        new_config = dict(config)
        new_config["entrance_zone"] = result_ratios
        return new_config

    return config
