"""Finish the interrupted batch: segment VOD 4, clip VODs 3+4, summarize.
Idempotent: segmentation replaces any partial rows; clip extraction only
touches matches still pointing at a source VOD."""

import time
import traceback

from vexga.acquire.clips import extract_match_clips
from vexga.segment.ingest import segment_and_store
from vexga.store.db import connect

con = connect()

# VOD 4: segment (download already on disk), then clip.
try:
    t0 = time.time()
    done = con.execute(
        "SELECT COUNT(*) c FROM matches WHERE video_id='S-80sdud56E'"
        " OR video_id IN (SELECT id FROM videos WHERE source_id='S-80sdud56E')"
    ).fetchone()["c"]
    if not done:
        ids = segment_and_store("data/videos/S-80sdud56E.mp4")
        print(f"segmented S-80sdud56E: {len(ids)} matches"
              f" ({(time.time()-t0)/60:.0f} min)", flush=True)
    extract_match_clips(con, "S-80sdud56E", delete_source=True)
except Exception:
    traceback.print_exc()

# VOD 3: clips only (segmented earlier; extraction was SIGKILLed).
try:
    extract_match_clips(con, "dtWuzN5-J_Q", delete_source=True)
except Exception:
    traceback.print_exc()

# Supersede the old dev slice.
con.execute("DELETE FROM score_timeline WHERE match_id IN"
            " (SELECT id FROM matches WHERE video_id='G5qJkqAOAJE_10000_12000')")
con.execute("DELETE FROM matches WHERE video_id='G5qJkqAOAJE_10000_12000'")
con.commit()

print("\n=== PIPELINE SUMMARY ===", flush=True)
for r in con.execute(
    "SELECT COALESCE(v.source_id, m.video_id) src, COUNT(*) n,"
    " SUM(m.ocr_red_score IS NOT NULL) scored, SUM(m.breakdown IS NOT NULL) cards"
    " FROM matches m LEFT JOIN videos v ON v.id = m.video_id GROUP BY 1"
):
    print(f"  {r['src']}: {r['n']} matches, {r['scored']} with OCR scores,"
          f" {r['cards']} result cards", flush=True)
