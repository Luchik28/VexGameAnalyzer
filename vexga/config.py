"""Project-wide paths and settings. All data lives under data/ (gitignored)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
VIDEOS = DATA / "videos"
FRAMES = DATA / "frames"
DATASETS = DATA / "datasets"
DB_PATH = DATA / "db" / "vexga.sqlite"
EXPORTS = DATA / "exports"
MODELS = ROOT / "models"

ROBOTEVENTS_TOKEN = os.environ.get("ROBOTEVENTS_TOKEN", "")

for _p in (VIDEOS, FRAMES, DATASETS, DB_PATH.parent, EXPORTS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
