"""Motion-based robot detector for dry runs (no trained weights needed).

Robots are the only large moving objects inside the field during a match, so
a median background model + frame differencing finds them reliably from a
static camera. Blob color (red/blue plate pixels) sets the alliance; a tiny
greedy nearest-neighbor associator provides persistent track ids so the
downstream slot logic works unchanged.

The fine-tuned YOLO detector replaces this for production; this exists so
the whole tracking -> storage -> viewer -> analytics path can be validated
before any labeling has happened.
"""

import cv2
import numpy as np

from vexga.calibrate.homography import Calibration
from vexga.detect.prelabel import _redness, detect_blocks, field_mask
from vexga.games.base import GameConfig


def background_model(video_path: str, start_ts: float, end_ts: float,
                     n: int = 25) -> np.ndarray:
    """Median of n frames spread across the span (robots average out)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for ts in np.linspace(start_ts, end_ts, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


class MotionRobotDetector:
    def __init__(self, bg: np.ndarray, cal: Calibration, game: GameConfig,
                 with_blocks: bool = True) -> None:
        self.bg = bg
        self.cal = cal
        self.game = game
        self.with_blocks = with_blocks
        self.fmask = field_mask((*bg.shape, 3), cal, game.field_size, margin_in=-2.0)
        # robot pixel scale at field center (~18" footprint)
        c = game.field_size / 2
        p = cal.to_pixel(np.array([[c - 9, c], [c + 9, c]]))
        self.robot_px = float(np.linalg.norm(p[1] - p[0]))
        self._tracks: dict[int, tuple[float, float]] = {}  # id -> last center
        self._next_id = 1

    def _associate(self, centers: list[tuple[float, float]]) -> list[int]:
        """Greedy NN: match new centers to previous ones within ~1 robot."""
        ids = [-1] * len(centers)
        free = dict(self._tracks)
        order = sorted(
            ((np.hypot(cx - px, cy - py), i, tid)
             for i, (cx, cy) in enumerate(centers)
             for tid, (px, py) in free.items()),
            key=lambda x: x[0],
        )
        used_i, used_t = set(), set()
        for d, i, tid in order:
            if d > self.robot_px * 1.5 or i in used_i or tid in used_t:
                continue
            ids[i] = tid
            used_i.add(i)
            used_t.add(tid)
        for i in range(len(centers)):
            if ids[i] < 0:
                ids[i] = self._next_id
                self._next_id += 1
        self._tracks = {tid: centers[i] for i, tid in enumerate(ids)}
        return ids

    def __call__(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.bg)
        _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        mask &= self.fmask
        k = max(3, int(self.robot_px * 0.12) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _lbl, stats, _cent = cv2.connectedComponentsWithStats(mask)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # robot-sized: between 0.35x and 1.8x nominal footprint, solid-ish
            if not (0.35 * self.robot_px <= max(w, h) <= 1.8 * self.robot_px):
                continue
            if area < 0.15 * self.robot_px ** 2:
                continue
            boxes.append((float(x), float(y), float(x + w), float(y + h)))
        centers = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes]
        ids = self._associate(centers)
        out = []
        for (x0, y0, x1, y1), tid in zip(boxes, ids):
            crop = frame[int(y0):int(y1), int(x0):int(x1)]
            cls = "robot_red" if _redness(crop) >= 0 else "robot_blue"
            out.append((cls, tid, 0.5, (x0, y0, x1, y1)))
        if self.with_blocks:
            for cls, box in detect_blocks(frame, self.cal, self.game):
                out.append((cls, None, 0.5, box))
        return out
