"""Find and download VEX tournament livestream VODs with yt-dlp.

Downloads go to data/videos/<video_id>.mp4 at 720p (sufficient for detection,
keeps 8h VODs ~3-4 GB). A section like "1:00:00-1:30:00" can be given during
development to avoid multi-GB pulls. Metadata sidecars (<id>.info.json) keep
the title/duration for later event matching.
"""

import json
import subprocess
import sys
from pathlib import Path

from vexga.config import VIDEOS, ffmpeg_exe

def _node_exe() -> str | None:
    import glob
    import shutil

    found = shutil.which("node")
    if found:
        return found
    versions = sorted(glob.glob(str(Path.home() / ".nvm/versions/node/v*/bin/node")))
    return versions[-1] if versions else None


# -4: YouTube stalls indefinitely over IPv6 on some networks.
# --js-runtimes node: needed (with the yt-dlp-ejs pip package) to solve
# YouTube's JS challenges; without it most formats are missing or DRM-flagged.
_NODE = _node_exe()
YTDLP = [sys.executable, "-m", "yt_dlp", "-4"] + (
    ["--js-runtimes", f"node:{_NODE}", "--remote-components", "ejs:github"]
    if _NODE else []
)
FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"yt-dlp failed ({out.returncode}): {out.stderr.strip()[-500:]}")
    return out


def video_id(url: str) -> str:
    out = _run(YTDLP + ["--print", "id", "--no-download", url])
    return out.stdout.strip().splitlines()[-1]


def search(query: str, limit: int = 20) -> list[dict]:
    """yt-dlp flat search; returns [{id, title, duration, channel, url}]."""
    out = _run(YTDLP + ["--flat-playlist", "--dump-json", f"ytsearch{limit}:{query}"])
    results = []
    for line in out.stdout.splitlines():
        j = json.loads(line)
        results.append({
            "id": j.get("id"),
            "title": j.get("title"),
            "duration": j.get("duration"),
            "channel": j.get("channel") or j.get("uploader"),
            "url": f"https://www.youtube.com/watch?v={j.get('id')}",
        })
    return results


def download(url: str, section: str | None = None, force: bool = False) -> Path:
    """Download a VOD (or a HH:MM:SS-HH:MM:SS section of it). Returns the
    local mp4 path. Skips if already present unless force."""
    vid = video_id(url)
    suffix = f"_{section.replace(':', '').replace('-', '_')}" if section else ""
    dest = VIDEOS / f"{vid}{suffix}.mp4"
    if dest.exists() and not force:
        return dest
    cmd = YTDLP + [
        "-f", FORMAT,
        "--ffmpeg-location", ffmpeg_exe(),
        "--write-info-json",
        "-o", str(dest.with_suffix("")) + ".%(ext)s",
        "--no-part" if section else "--continue",
        url,
    ]
    if section:
        cmd += ["--download-sections", f"*{section}", "--force-keyframes-at-cuts"]
    subprocess.run(cmd, check=True)  # stream output; failures raise
    if not dest.exists():  # yt-dlp may emit .mkv/.webm if mp4 mux fails
        candidates = list(VIDEOS.glob(f"{vid}{suffix}.*"))
        media = [c for c in candidates if c.suffix in (".mp4", ".mkv", ".webm")]
        if not media:
            raise FileNotFoundError(f"download produced no media file for {url}")
        dest = media[0]
    return dest
