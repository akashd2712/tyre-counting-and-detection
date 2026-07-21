import cv2
import numpy as np


def apply_roi(frame: np.ndarray, roi_config: dict) -> tuple[np.ndarray, int, int]:
    if not roi_config.get("enabled", True):
        return frame, 0, 0

    height, width = frame.shape[:2]
    x1 = int(width * roi_config.get("x1", 0.06))
    y1 = int(height * roi_config.get("y1", 0.20))
    x2 = int(width * roi_config.get("x2", 0.94))
    y2 = int(height * roi_config.get("y2", 0.94))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return frame[y1:y2, x1:x2], x1, y1
