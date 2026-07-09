"""Interactive corner-click calibration tool (user-run, opens a cv2 window).

Usage:
    .venv/bin/python -m vexga.calibrate.tool data/videos/<id>.mp4 --ts 3600

Click the four INTERIOR field-floor corners in this order:
    1. red-side  near corner   (red alliance wall, closest to camera bottom)
    2. blue-side near corner
    3. blue-side far corner
    4. red-side  far corner
"Red side" is the wall with the red park zone / red alliance station.
Clicking in this order fixes the canonical orientation regardless of which
side of the arena the camera is on.

Keys:  u = undo last click,  a = accept,  q = abort.
After 4 clicks a projected tile grid + landmarks overlay appears; accept only
if it visually hugs the tile lines. The homography is stored in the DB keyed
by (video_id, from_ts).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from vexga.calibrate.homography import draw_field_grid, fit
from vexga.games.base import get_game
from vexga.store.db import connect

CORNER_ORDER = ["red-near", "blue-near", "blue-far", "red-far"]


def grab_frame(video: str, ts: float) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"cannot read frame at {ts}s from {video}")
    return frame


def calibrate_interactive(video: str, ts: float, game_name: str = "pushback"):
    game = get_game(game_name)
    f = game.field_size
    field_corners = np.array([(0, 0), (f, 0), (f, f), (0, f)], dtype=np.float64)

    frame = grab_frame(video, ts)
    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    win = "calibrate - click corners (u=undo, a=accept, q=abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cal = None
    while True:
        if len(clicks) == 4 and cal is None:
            cal = fit(np.array(clicks, dtype=np.float64), field_corners)
        vis = (draw_field_grid(frame, cal, f, game.landmarks) if cal is not None
               else frame.copy())
        for i, (x, y) in enumerate(clicks):
            cv2.circle(vis, (x, y), 6, (0, 255, 255), 2)
            cv2.putText(vis, CORNER_ORDER[i], (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
        nxt = CORNER_ORDER[len(clicks)] if len(clicks) < 4 else "review grid, a=accept"
        cv2.putText(vis, f"next: {nxt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and clicks:
            clicks.pop()
            cal = None
        elif k == ord("a") and cal is not None:
            cv2.destroyAllWindows()
            return cal
        elif k == ord("q"):
            cv2.destroyAllWindows()
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ts", type=float, default=0.0, help="video timestamp (s) to calibrate at")
    ap.add_argument("--game", default="pushback")
    args = ap.parse_args()

    cal = calibrate_interactive(args.video, args.ts, args.game)
    if cal is None:
        print("aborted")
        return
    print(f"reprojection RMS over corners: {cal.reproj_err_in:.2f} in")
    video_id = Path(args.video).stem
    con = connect()
    con.execute(
        "INSERT OR REPLACE INTO calibrations (video_id, from_ts, homography, reproj_err_in) VALUES (?,?,?,?)",
        (video_id, args.ts, cal.to_json(), cal.reproj_err_in),
    )
    con.commit()
    print(f"saved calibration for video {video_id} from ts {args.ts}")


if __name__ == "__main__":
    main()
