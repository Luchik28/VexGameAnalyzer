"""Pixel <-> field-inch mapping for one camera setup.

A Calibration wraps a single 3x3 homography H mapping image pixels to the
canonical field frame (games/base.py). Because H maps the floor plane,
robot positions must be taken at the bounding box's ground-contact point
(bottom-center), not the box center.
"""

import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Calibration:
    H: np.ndarray  # 3x3, pixel -> field inches
    reproj_err_in: float = 0.0

    def to_field(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) pixel points -> (N,2) field inches."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_pixel(self, pts_in: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_in, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def to_json(self) -> str:
        return json.dumps(self.H.flatten().tolist())

    @classmethod
    def from_json(cls, s: str, reproj_err_in: float = 0.0) -> "Calibration":
        return cls(np.array(json.loads(s), dtype=np.float64).reshape(3, 3), reproj_err_in)


def fit(pixel_pts: np.ndarray, field_pts: np.ndarray) -> Calibration:
    """Least-squares homography from >= 4 correspondences; error = RMS of
    reprojection in field inches over the given points."""
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64)
    field_pts = np.asarray(field_pts, dtype=np.float64)
    H, _mask = cv2.findHomography(pixel_pts, field_pts, method=0)
    if H is None:
        raise ValueError("homography fit failed")
    cal = Calibration(H)
    err = np.sqrt(np.mean(np.sum((cal.to_field(pixel_pts) - field_pts) ** 2, axis=1)))
    cal.reproj_err_in = float(err)
    return cal


def draw_field_grid(frame: np.ndarray, cal: Calibration, field_size: float,
                    landmarks=(), tile: float = 23.4) -> np.ndarray:
    """Overlay the projected tile grid + landmarks for visual verification."""
    vis = frame.copy()
    n = int(round(field_size / tile))
    for i in range(n + 1):
        c = i * field_size / n
        for a, b in (((c, 0), (c, field_size)), ((0, c), (field_size, c))):
            p = cal.to_pixel(np.array([a, b])).astype(int)
            cv2.line(vis, tuple(p[0]), tuple(p[1]), (0, 255, 0), 1, cv2.LINE_AA)
    for lm in landmarks:
        p = cal.to_pixel(np.array([[lm.x, lm.y]])).astype(int)[0]
        cv2.circle(vis, tuple(p), 5, (0, 0, 255), -1)
        cv2.putText(vis, lm.name, (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return vis
