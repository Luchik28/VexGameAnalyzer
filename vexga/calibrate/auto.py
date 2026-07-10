"""Automatic field calibration from a single frame.

The field floor is a large low-saturation gray quadrilateral surrounded by
darker walls and patterned venue carpet, so:
1. mask gray pixels (low saturation, mid brightness), keep the largest
   connected component, take its convex hull,
2. reduce the hull to 4 corners (dominant-direction quad fit),
3. orient the quad: the red alliance wall is the floor edge nearest the red
   park zone (a saturated red blob hugging the floor boundary),
4. fit the homography to the canonical frame.

Returns (Calibration, debug_info). Fails (returns None) on frames without a
dominant floor quad (ranking screens, closeups), which doubles as a junk-
frame detector.
"""

import cv2
import numpy as np

from vexga.calibrate.homography import Calibration, fit
from vexga.games.base import GameConfig

GRAY_S_MAX = 60
GRAY_V_MIN, GRAY_V_MAX = 60, 210
MIN_FLOOR_FRAC = 0.12   # floor must fill this fraction of the frame
RED_LO_1, RED_HI_1 = (0, 100, 90), (10, 255, 255)
RED_LO_2, RED_HI_2 = (160, 60, 120), (180, 255, 255)
BLUE_LO, BLUE_HI = (98, 100, 70), (125, 255, 255)


def _floor_quad(frame: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < GRAY_S_MAX)
            & (hsv[:, :, 2] > GRAY_V_MIN) & (hsv[:, :, 2] < GRAY_V_MAX)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < MIN_FLOOR_FRAC * h * w:
        return None
    comp = (labels == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    # Iteratively coarsen the polygon until 4 points remain.
    eps = 0.01 * cv2.arcLength(hull, True)
    for _ in range(20):
        quad = cv2.approxPolyDP(hull, eps, True)
        if len(quad) <= 4:
            break
        eps *= 1.4
    if len(quad) != 4:
        return None
    return quad.reshape(4, 2).astype(np.float64)


def _order_quad(quad: np.ndarray, frame: np.ndarray) -> np.ndarray | None:
    """Order corners as (red-near, blue-near, blue-far, red-far) using the
    image positions of the red/blue park-zone blobs on the side walls."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def biggest_blob(mask: np.ndarray) -> np.ndarray | None:
        n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] < 40:
            return None
        return cent[i]

    # Only look inside the floor quad for park-zone plastic.
    poly_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly_mask, [quad.astype(np.int32)], 255)
    red_m = (cv2.inRange(hsv, RED_LO_1, RED_HI_1) | cv2.inRange(hsv, RED_LO_2, RED_HI_2)) & poly_mask
    blue_m = cv2.inRange(hsv, BLUE_LO, BLUE_HI) & poly_mask
    # Park zones are the large red/blue structures near the left/right floor
    # edges; blocks are small, so a strong erosion leaves only the zones.
    k = np.ones((5, 5), np.uint8)
    red_c = biggest_blob(cv2.morphologyEx(red_m, cv2.MORPH_OPEN, k, iterations=2))
    blue_c = biggest_blob(cv2.morphologyEx(blue_m, cv2.MORPH_OPEN, k, iterations=2))
    if red_c is None or blue_c is None:
        return None

    # Sort corners: two nearest the bottom of the frame are "near".
    by_y = quad[np.argsort(quad[:, 1])]
    far = by_y[:2][np.argsort(by_y[:2, 0])]      # left, right
    near = by_y[2:][np.argsort(by_y[2:, 0])]     # left, right
    red_left = red_c[0] < blue_c[0]
    if red_left:
        ordered = np.array([near[0], near[1], far[1], far[0]])
    else:
        ordered = np.array([near[1], near[0], far[0], far[1]])
    return ordered


def auto_calibrate(frame: np.ndarray, game: GameConfig) -> Calibration | None:
    quad = _floor_quad(frame)
    if quad is None:
        return None
    ordered = _order_quad(quad, frame)
    if ordered is None:
        return None
    f = game.field_size
    field = np.array([(0, 0), (f, 0), (f, f), (0, f)], dtype=np.float64)
    cal = fit(ordered, field)
    # Sanity: reprojected field center must be inside the quad.
    center_px = cal.to_pixel(np.array([[f / 2, f / 2]]))[0]
    if not cv2.pointPolygonTest(quad.astype(np.float32), tuple(center_px), False) >= 0:
        return None
    return cal
