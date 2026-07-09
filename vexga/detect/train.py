"""Train / evaluate the detector on MPS.

    .venv/bin/python -m vexga.detect.train data/datasets/pushback_v1/dataset.yaml

YOLO11s @ 960 px. On an M1 (16GB) budget ~2-5 h for 100 epochs on ~1500
images - run overnight. Best weights land in models/<run>/weights/best.pt.
"""

import argparse

from vexga.config import MODELS


def train(data_yaml: str, run: str = "pushback_v1", epochs: int = 100,
          imgsz: int = 960, model: str = "yolo11s.pt") -> None:
    from ultralytics import YOLO

    y = YOLO(model)
    y.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device="mps",
        batch=8,          # fits 16GB unified memory at 960px
        project=str(MODELS),
        name=run,
        patience=25,
        cache=False,      # datasets are small; avoid RAM pressure
        plots=True,
    )
    metrics = y.val(data=data_yaml, imgsz=imgsz, device="mps")
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_yaml")
    ap.add_argument("--run", default="pushback_v1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--model", default="yolo11s.pt")
    a = ap.parse_args()
    train(a.data_yaml, a.run, a.epochs, a.imgsz, a.model)
