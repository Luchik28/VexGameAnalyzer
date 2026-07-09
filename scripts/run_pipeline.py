"""Batch pipeline: for each VOD url -> download -> segment -> clip -> delete.

Space-aware: the machine has only ~6 GB free, so full VODs are transient and
only per-match clips persist. Aborts a download if free space drops below
MIN_FREE_GB.

    PYTHONPATH=. .venv/bin/python -u scripts/run_pipeline.py
"""

import shutil
import sys
import time
import traceback

from vexga.acquire.clips import extract_match_clips
from vexga.acquire.youtube import download
from vexga.segment.ingest import segment_and_store
from vexga.store.db import connect

URLS = [
    "https://www.youtube.com/watch?v=G5qJkqAOAJE",  # Sahara Day 2 (overlay verified)
    "https://www.youtube.com/watch?v=mSLJO6ITDqc",
    "https://www.youtube.com/watch?v=dtWuzN5-J_Q",
    "https://www.youtube.com/watch?v=S-80sdud56E",
]
MIN_FREE_GB = 4.5


def free_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def main() -> None:
    con = connect()
    for url in URLS:
        t0 = time.time()
        print(f"\n=== {url} (free: {free_gb():.1f} GB) ===", flush=True)
        if free_gb() < MIN_FREE_GB:
            print("ABORT: not enough free disk for a full VOD download", flush=True)
            sys.exit(2)
        try:
            path = download(url)
            print(f"downloaded {path.name} ({path.stat().st_size/1e9:.1f} GB, "
                  f"{(time.time()-t0)/60:.0f} min)", flush=True)
            vid = path.stem
            ids = segment_and_store(str(path))
            print(f"segmented: {len(ids)} matches ({(time.time()-t0)/60:.0f} min total)",
                  flush=True)
            n = extract_match_clips(con, vid, delete_source=True)
            print(f"clipped {n} matches, VOD deleted (free: {free_gb():.1f} GB)", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAILED on {url}, continuing with next", flush=True)
    # The old dev slice of Sahara Day 2 is superseded by full-VOD matches.
    con.execute("DELETE FROM score_timeline WHERE match_id IN"
                " (SELECT id FROM matches WHERE video_id='G5qJkqAOAJE_10000_12000')")
    con.execute("DELETE FROM matches WHERE video_id='G5qJkqAOAJE_10000_12000'")
    con.commit()
    rows = con.execute(
        "SELECT v.source_id AS src, COUNT(*) n, SUM(m.ocr_red_score IS NOT NULL) scored,"
        " SUM(m.breakdown IS NOT NULL) cards FROM matches m"
        " JOIN videos v ON v.id = m.video_id GROUP BY v.source_id"
    ).fetchall()
    print("\n=== PIPELINE SUMMARY ===", flush=True)
    for r in rows:
        print(f"  {r['src']}: {r['n']} matches, {r['scored']} with OCR scores,"
              f" {r['cards']} with result-card breakdowns", flush=True)


if __name__ == "__main__":
    main()
