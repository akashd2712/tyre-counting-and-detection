class MotionFilter:
    def __init__(self, min_movement_px: float = 8.0, min_box_area_ratio: float = 0.001, max_box_area_ratio: float = 0.05):
        self.min_movement_px = min_movement_px
        self.min_box_area_ratio = min_box_area_ratio
        self.max_box_area_ratio = max_box_area_ratio

    def is_valid_size(self, bbox_area: float, frame_area: float) -> bool:
        """Check if the bounding box is a valid size for a single tyre."""
        if frame_area <= 0:
            return True
        ratio = bbox_area / frame_area
        return self.min_box_area_ratio <= ratio <= self.max_box_area_ratio

    def should_process(self, prev_center, current_center, bbox_area: float, frame_area: float = 0) -> bool:
        """Determine whether a detection should be processed.

        Filters out:
        - Detections whose bounding box is too small or too large relative to the frame
          (noise / false positives / full-stack boxes).
        - Detections that have not moved enough since the last frame
          (stationary objects or vibration).
        """
        if not self.is_valid_size(bbox_area, frame_area):
            return False

        if prev_center is None or current_center is None:
            return True
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]
        movement = (dx * dx + dy * dy) ** 0.5
        return movement >= self.min_movement_px
