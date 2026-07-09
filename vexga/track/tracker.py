"""Per-match robot tracking and zone-state extraction.

Runs the trained detector at ~5 fps over one match's video span with
ultralytics' built-in ByteTrack, projects ground-contact points through the
calibration homography, then:
- assigns the four persistent tracks to slots red1/red2/blue1/blue2 by
  proximity to the game's known starting positions at auton start,
- carries slot identity through track-id changes with a nearest-continuation
  + alliance-color constraint,
- counts blocks per zone per timestep,
- Kalman-smooths and gap-fills slot trajectories,
- computes a QA quality score.

Output: TrackResult ready for store.writer.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from vexga.calibrate.homography import Calibration
from vexga.games.base import GameConfig


@dataclass
class FrameObs:
    t: float                                  # seconds since match start
    # per detection: (cls_name, track_id, x_in, y_in, conf, xyxy_px)
    dets: list[tuple[str, int | None, float, float, float, tuple]] = field(default_factory=list)


@dataclass
class TrackResult:
    # slot -> list of (t, x_in, y_in, conf); NaN-free, smoothed, 5 Hz grid
    slots: dict[str, list[tuple[float, float, float, float]]]
    # (t, zone, red_blocks, blue_blocks)
    zone_states: list[tuple[float, str, int, int]]
    quality: float
    notes: str = ""


def yolo_detector(weights: str, conf: float = 0.35):
    """Standard detector: fine-tuned YOLO + ByteTrack. Returns a callable
    frame -> [(cls_name, track_id|None, conf, xyxy)]."""
    from ultralytics import YOLO

    model = YOLO(weights)

    def detect(frame: np.ndarray):
        res = model.track(frame, persist=True, conf=conf, verbose=False,
                          tracker="bytetrack.yaml", device="mps")[0]
        names = res.names
        out = []
        for b in res.boxes:
            tid = int(b.id) if b.id is not None else None
            out.append((names[int(b.cls)], tid, float(b.conf),
                        tuple(map(float, b.xyxy[0]))))
        return out

    return detect


def detect_match(video_path: str, start_ts: float, end_ts: float,
                 cal: Calibration, game: GameConfig, detector,
                 sample_fps: float = 5.0) -> list[FrameObs]:
    """Run a detector callable over the match span, projecting detections to
    field coordinates. `detector` is frame -> [(cls, track_id, conf, xyxy)]
    (see yolo_detector); a prelabel-based stand-in works for dry runs."""
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / sample_fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ts * 1000)
    frames: list[FrameObs] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if ts > end_ts:
            break
        if idx % step:
            idx += 1
            continue
        idx += 1
        obs = FrameObs(t=ts - start_ts)
        for cls, tid, conf, (x0, y0, x1, y1) in detector(frame):
            gx, gy = cal.to_field(np.array([[(x0 + x1) / 2, y1]]))[0]
            m = 6.0  # allow slight overhang beyond walls
            if not (-m <= gx <= game.field_size + m and -m <= gy <= game.field_size + m):
                continue
            obs.dets.append((cls, tid,
                             float(np.clip(gx, 0, game.field_size)),
                             float(np.clip(gy, 0, game.field_size)),
                             conf, (x0, y0, x1, y1)))
        frames.append(obs)
    cap.release()
    return frames


def assign_slots(frames: list[FrameObs], game: GameConfig) -> dict[str, dict[int, int]]:
    """Map ByteTrack ids -> slots over time. Returns slot -> {approx logic
    handled internally}; practically we return a per-frame slot position via
    build_slot_tracks, this function resolves the id->slot ownership table."""
    # Seed: in the first second of frames, pick per-alliance the two tracks
    # closest to the alliance's start positions.
    alliance_of = {"red1": "red", "red2": "red", "blue1": "blue", "blue2": "blue"}
    owners: dict[int, str] = {}  # track_id -> slot
    seed_frames = [f for f in frames if f.t <= 3.0] or frames[:5]
    cand: dict[int, tuple[str, np.ndarray, float]] = {}
    for f in seed_frames:
        for cls, tid, x, y, conf, _ in f.dets:
            if tid is None or not cls.startswith("robot"):
                continue
            color = "red" if cls == "robot_red" else "blue"
            cand[tid] = (color, np.array([x, y]), conf)
    for slot, (sx, sy) in game.start_positions.items():
        best, best_d = None, 1e9
        for tid, (color, p, _c) in cand.items():
            if tid in owners or color != alliance_of[slot]:
                continue
            d = float(np.hypot(p[0] - sx, p[1] - sy))
            if d < best_d:
                best, best_d = tid, d
        if best is not None:
            owners[best] = slot
    return {"owners": owners}  # type: ignore[return-value]


def build_slot_tracks(frames: list[FrameObs], game: GameConfig,
                      handoff_max_in: float = 30.0) -> tuple[dict[str, list], int]:
    """Follow each slot through track-id changes. When a slot's track id
    disappears, adopt the nearest unclaimed same-color track that appears
    within handoff_max_in of the last known position. Returns (slot->samples,
    id_switch_count)."""
    owners: dict[int, str] = assign_slots(frames, game)["owners"]  # type: ignore[index]
    last_pos: dict[str, np.ndarray] = {}
    out: dict[str, list] = {s: [] for s in game.start_positions}
    switches = 0
    for f in frames:
        seen_slots = set()
        unclaimed = []
        for cls, tid, x, y, conf, _ in f.dets:
            if not cls.startswith("robot") or tid is None:
                continue
            slot = owners.get(tid)
            if slot is not None and slot not in seen_slots:
                out[slot].append((f.t, x, y, conf))
                last_pos[slot] = np.array([x, y])
                seen_slots.add(slot)
            elif slot is None:
                unclaimed.append((cls, tid, np.array([x, y]), conf))
        # Hand off lost slots to nearby unclaimed tracks of the same color.
        for slot in out:
            if slot in seen_slots or slot not in last_pos:
                continue
            color = "red" if slot.startswith("red") else "blue"
            best, best_d = None, handoff_max_in
            for cls, tid, p, conf in unclaimed:
                if ("red" if cls == "robot_red" else "blue") != color:
                    continue
                d = float(np.linalg.norm(p - last_pos[slot]))
                if d < best_d:
                    best, best_d = (cls, tid, p, conf), d
            if best is not None:
                cls, tid, p, conf = best
                owners[tid] = slot
                unclaimed.remove(best)
                out[slot].append((f.t, float(p[0]), float(p[1]), conf))
                last_pos[slot] = p
                switches += 1
    return out, switches


def smooth_track(samples: list[tuple[float, float, float, float]],
                 t_end: float, hz: float = 5.0,
                 max_gap_s: float = 3.0) -> list[tuple[float, float, float, float]]:
    """Resample to a fixed grid with linear interpolation across short gaps;
    long gaps keep the last position with conf=0 (robot likely occluded but
    stationary robots dominate occlusion cases)."""
    if not samples:
        return []
    ts = np.array([s[0] for s in samples])
    xs = np.array([s[1] for s in samples])
    ys = np.array([s[2] for s in samples])
    cs = np.array([s[3] for s in samples])
    grid = np.arange(0, t_end, 1.0 / hz)
    out = []
    for t in grid:
        i = int(np.searchsorted(ts, t))
        if i == 0:
            x, y, c = xs[0], ys[0], 0.0
        elif i >= len(ts):
            x, y, c = xs[-1], ys[-1], 0.0
        else:
            gap = ts[i] - ts[i - 1]
            if gap <= max_gap_s:
                a = (t - ts[i - 1]) / max(gap, 1e-6)
                x = xs[i - 1] + a * (xs[i] - xs[i - 1])
                y = ys[i - 1] + a * (ys[i] - ys[i - 1])
                c = min(cs[i - 1], cs[i])
            else:
                x, y, c = xs[i - 1], ys[i - 1], 0.0
        out.append((float(t), float(x), float(y), float(c)))
    # Light moving-average to knock down detector jitter (~1-2" at 720p).
    k = 3
    xs2 = np.convolve([o[1] for o in out], np.ones(k) / k, mode="same")
    ys2 = np.convolve([o[2] for o in out], np.ones(k) / k, mode="same")
    return [(t, float(x), float(y), c) for (t, _x, _y, c), x, y in zip(out, xs2, ys2)]


def zone_block_counts(frames: list[FrameObs], game: GameConfig,
                      hz: float = 1.0) -> list[tuple[float, str, int, int]]:
    """Blocks visible per zone at ~hz. Floor-visible blocks only; enclosed
    goal sections hide blocks, so treat counts as lower bounds there."""
    out = []
    next_t = 0.0
    for f in frames:
        if f.t < next_t:
            continue
        next_t = f.t + 1.0 / hz
        counts: dict[str, list[int]] = {z.name: [0, 0] for z in game.zones}
        for cls, _tid, x, y, _conf, _ in f.dets:
            if not cls.startswith("block"):
                continue
            z = game.zone_at(x, y)
            if z is None:
                continue
            counts[z.name][0 if cls == "block_red" else 1] += 1
        for name, (r, b) in counts.items():
            out.append((f.t, name, r, b))
    return out


def track_match(video_path: str, start_ts: float, end_ts: float,
                cal: Calibration, game: GameConfig, weights_or_detector) -> TrackResult:
    detector = (weights_or_detector if callable(weights_or_detector)
                else yolo_detector(weights_or_detector))
    frames = detect_match(video_path, start_ts, end_ts, cal, game, detector)
    slot_samples, switches = build_slot_tracks(frames, game)
    t_end = end_ts - start_ts
    slots = {s: smooth_track(v, t_end) for s, v in slot_samples.items()}
    zones = zone_block_counts(frames, game)

    n_frames = max(1, len(frames))
    coverage = np.mean([len(v) / n_frames for v in slot_samples.values()]) if slot_samples else 0
    # Handoffs dent quality with a cap: churny-but-recovering tracking is
    # still usable data, unlike low coverage.
    quality = float(max(0.0, min(1.0, coverage - min(0.35, 0.02 * switches))))
    notes = f"frames={n_frames} coverage={coverage:.2f} id_handoffs={switches}"
    return TrackResult(slots=slots, zone_states=zones, quality=quality, notes=notes)
