"""Auto pre-labeling for the assisted-labeling loop.

Two sources, merged into YOLO label files next to each image:
- Blocks: HSV color segmentation inside the calibrated field region. Push
  Back blocks are saturated red/blue plastic; robots also carry red/blue
  plates, so block candidates are size-filtered using the homography (a block
  is ~4" across on the floor plane).
- Robots: a COCO-pretrained YOLO11 run with low confidence; anything big and
  moving on the field that isn't wall/human. COCO has no "robot" class, so we
  accept a set of classes it typically fires on for robots (or use raw
  objectness via low conf) and let the human fix the rest.

Pre-labels are a starting point; every frame still gets human review.
"""

from pathlib import Path

import cv2
import numpy as np

from vexga.calibrate.homography import Calibration
from vexga.games.base import GameConfig

# HSV ranges for red/blue game plastic, tuned on 720p stream frames. Venue
# white balance washes "red" blocks toward pink (measured H 160-178,
# S 50-200, V 150+ on the Kalahari stream), so the red mask spans both the
# true-red band and the pink-magenta band at a low saturation floor.
RED_LO_1, RED_HI_1 = (0, 100, 120), (10, 255, 255)
RED_LO_2, RED_HI_2 = (160, 50, 140), (180, 255, 255)
BLUE_LO, BLUE_HI = (98, 100, 70), (125, 255, 255)

# COCO classes a pretrained model tends to fire on for VEX robots.
COCO_ROBOT_CLASSES = {"suitcase", "tv", "microwave", "oven", "laptop", "keyboard",
                      "cell phone", "book", "toaster", "refrigerator", "car", "truck"}


def field_mask(shape: tuple[int, int], cal: Calibration, field_size: float,
               margin_in: float = 6.0) -> np.ndarray:
    """Binary mask of the field floor in image pixels (slightly shrunk)."""
    m = margin_in
    corners_in = np.array([(m, m), (field_size - m, m),
                           (field_size - m, field_size - m), (m, field_size - m)])
    poly = cal.to_pixel(corners_in).astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def block_scale_px(cal: Calibration, cx: float, cy: float, block_in: float = 4.0) -> float:
    """Approximate pixel size of a block at image point (cx, cy)."""
    f = cal.to_field(np.array([[cx, cy]]))[0]
    a = cal.to_pixel(np.array([f, f + [block_in, 0]]))
    b = cal.to_pixel(np.array([f, f + [0, block_in]]))
    return float((np.linalg.norm(a[1] - a[0]) + np.linalg.norm(b[1] - b[0])) / 2)


def detect_blocks(frame: np.ndarray, cal: Calibration, game: GameConfig
                  ) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return [(class_name, xyxy)] block candidates via color segmentation."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fmask = field_mask(frame.shape, cal, game.field_size)
    out = []
    for cls, mask in (
        ("block_red", cv2.inRange(hsv, RED_LO_1, RED_HI_1) | cv2.inRange(hsv, RED_LO_2, RED_HI_2)),
        ("block_blue", cv2.inRange(hsv, BLUE_LO, BLUE_HI)),
    ):
        mask &= fmask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            s = block_scale_px(cal, cx, cy)
            # One block: roughly s x s px; accept 0.4-2.5x to allow tilt/partial
            # occlusion but reject robot plates (elongated) and clusters (huge).
            if not (0.4 * s <= max(w, h) <= 2.5 * s):
                continue
            if area < 0.15 * s * s or max(w, h) / max(1, min(w, h)) > 2.5:
                continue
            out.append((cls, (float(x), float(y), float(x + w), float(y + h))))
    return out


class RobotPrelabeler:
    def __init__(self, conf: float = 0.15) -> None:
        from ultralytics import YOLO

        self.model = YOLO("yolo11m.pt")  # COCO weights, downloaded on first use
        self.conf = conf

    def detect(self, frame: np.ndarray, cal: Calibration, game: GameConfig
               ) -> list[tuple[str, tuple[float, float, float, float]]]:
        fmask = field_mask(frame.shape, cal, game.field_size, margin_in=-4.0)
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        names = res.names
        out = []
        for b in res.boxes:
            if names[int(b.cls)] not in COCO_ROBOT_CLASSES | {"person"}:
                continue
            x0, y0, x1, y1 = map(float, b.xyxy[0])
            foot = (int((x0 + x1) / 2), int(min(y1, frame.shape[0] - 1)))
            if fmask[foot[1], foot[0]] == 0:
                continue  # ground point off-field (referees, audience)
            if names[int(b.cls)] == "person":
                continue  # people lean over walls; skip even on-field hits
            # Alliance color from plate pixels inside the box.
            crop = frame[int(y0):int(y1), int(x0):int(x1)]
            cls = "robot_red" if _redness(crop) >= 0 else "robot_blue"
            out.append((cls, (x0, y0, x1, y1)))
        return out


def _redness(crop: np.ndarray) -> float:
    """>0 if red plate pixels outnumber blue ones."""
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = int((cv2.inRange(hsv, RED_LO_1, RED_HI_1) | cv2.inRange(hsv, RED_LO_2, RED_HI_2)).sum())
    blue = int(cv2.inRange(hsv, BLUE_LO, BLUE_HI).sum())
    return float(red - blue)


def to_yolo_line(cls_id: int, xyxy: tuple[float, float, float, float],
                 img_w: int, img_h: int) -> str:
    x0, y0, x1, y1 = xyxy
    cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
    w, h = (x1 - x0) / img_w, (y1 - y0) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prelabel_dataset(dataset_dir: Path, cal: Calibration, game: GameConfig) -> int:
    """Write YOLO label files for every image in the dataset. Returns count."""
    robot = RobotPrelabeler()
    cls_ids = {c: i for i, c in enumerate(game.detect_classes)}
    n = 0
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(img_dir.glob("*.jpg")):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            dets = detect_blocks(frame, cal, game) + robot.detect(frame, cal, game)
            lines = [to_yolo_line(cls_ids[c], box, w, h) for c, box in dets if c in cls_ids]
            (lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
    return n
