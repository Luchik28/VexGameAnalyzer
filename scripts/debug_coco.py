import time, cv2, numpy as np
t0=time.time()
from vexga.store.db import connect
from vexga.games.base import get_game
from vexga.track.process import calibration_for
from ultralytics import YOLO
from vexga.config import MODELS
print(f"imports {time.time()-t0:.0f}s", flush=True)
con = connect(); game = get_game("pushback")
cal = calibration_for(con, "G5qJkqAOAJE_10000_12000", 0)
model = YOLO(str(MODELS / "yolo11m.pt"))
cap = cv2.VideoCapture("data/videos/G5qJkqAOAJE_10000_12000.mp4")
for ts in (790, 830, 870):
    cap.set(cv2.CAP_PROP_POS_MSEC, ts*1000)
    ok, frame = cap.read()
    t1=time.time()
    res = model.track(frame, persist=True, conf=0.10, verbose=False, tracker="bytetrack.yaml", device="mps")[0]
    print(f"--- ts={ts}: {len(res.boxes)} raw dets ({time.time()-t1:.1f}s)", flush=True)
    for b in res.boxes:
        x0,y0,x1,y1 = map(float, b.xyxy[0])
        g = cal.to_field(np.array([[(x0+x1)/2, y1]]))[0]
        print(f"  {res.names[int(b.cls)]:12s} conf={float(b.conf):.2f} id={b.id if b.id is None else int(b.id)} field=({g[0]:.0f},{g[1]:.0f})", flush=True)
print("DONE", flush=True)
