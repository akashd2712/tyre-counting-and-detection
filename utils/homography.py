import numpy as np
import cv2

class HomographyProjector:
    def __init__(self, config: dict):
        self.enabled = config.get("homography", {}).get("enabled", False)
        self.matrix = None
        
        matrix_list = config.get("homography", {}).get("matrix")
        if self.enabled and matrix_list:
            self.matrix = np.array(matrix_list, dtype=np.float32)
            
    def project_point(self, x: float, y: float) -> tuple[float, float]:
        """
        Projects a 2D image coordinate (x, y) into the top-down 3D bird's-eye view space.
        Returns the mapped (x', y') coordinate.
        """
        if not self.enabled or self.matrix is None:
            return x, y
            
        pts = np.array([[[x, y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.matrix)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def project_bbox_bottom(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """
        Takes a bounding box (x1, y1, x2, y2) and projects its bottom-center point.
        The bottom-center represents where the object touches the ground plane.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = float(y2)  # Bottom of the bounding box
        return self.project_point(cx, cy)
