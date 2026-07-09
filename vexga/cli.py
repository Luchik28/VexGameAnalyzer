"""Command-line entry points.

Examples:
    python -m vexga.cli download https://www.youtube.com/watch?v=XXXX --section 1:00:00-1:20:00
    python -m vexga.cli segment data/videos/XXXX.mp4
    python -m vexga.cli matches                      # list segmented matches
    python -m vexga.cli join-re --event 58913 --division 1
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="vexga")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download a VOD (or section) to data/videos")
    d.add_argument("url")
    d.add_argument("--section", help="HH:MM:SS-HH:MM:SS slice for development")

    s = sub.add_parser("segment", help="OCR-segment a VOD into matches")
    s.add_argument("video")
    s.add_argument("--event", type=int, help="RobotEvents event id")
    s.add_argument("--division")
    s.add_argument("--period", type=float, default=1.0)

    m = sub.add_parser("matches", help="list matches in the DB")
    m.add_argument("--video")

    j = sub.add_parser("join-re", help="join matches against RobotEvents results")
    j.add_argument("--event", type=int, required=True)
    j.add_argument("--division", type=int, required=True)

    ds = sub.add_parser("dataset", help="extract frames + pre-labels + Label Studio tasks")
    ds.add_argument("--name", default="pushback_v1")
    ds.add_argument("--frames", type=int, default=1500)
    ds.add_argument("--game", default="pushback")

    tr = sub.add_parser("track", help="run tracker over segmented matches")
    tr.add_argument("--weights", required=True)
    tr.add_argument("--video")
    tr.add_argument("--game", default="pushback")
    tr.add_argument("--redo", action="store_true", help="reprocess matches that already have tracks")

    ex = sub.add_parser("export", help="export parquet + viewer JSONs")
    ex.add_argument("--match", type=int, help="export a single match JSON")

    args = ap.parse_args()

    if args.cmd == "download":
        from vexga.acquire.youtube import download

        print("saved:", download(args.url, section=args.section))

    elif args.cmd == "segment":
        from vexga.segment.ingest import segment_and_store

        ids = segment_and_store(args.video, args.event, args.division, args.period)
        print(f"stored {len(ids)} matches")

    elif args.cmd == "matches":
        from vexga.store.db import connect

        con = connect()
        q = "SELECT id, video_id, name, video_start_ts, video_end_ts, red1, red2, blue1, blue2, red_score, blue_score, notes FROM matches"
        rows = con.execute(q + (" WHERE video_id = ?" if args.video else ""),
                           (args.video,) if args.video else ()).fetchall()
        for r in rows:
            dur = (r["video_end_ts"] - r["video_start_ts"]) if r["video_end_ts"] else 0
            print(f"#{r['id']:>4} {r['name'] or '?':>7}  video={r['video_id']} "
                  f"start={r['video_start_ts']:.0f}s dur={dur:.0f}s "
                  f"red={r['red1']},{r['red2']} blue={r['blue1']},{r['blue2']} "
                  f"score={r['red_score']}-{r['blue_score']} {r['notes'] or ''}")

    elif args.cmd == "join-re":
        from vexga.segment.ingest import join_robotevents
        from vexga.store.db import connect

        joined, unjoined = join_robotevents(connect(), args.event, args.division)
        print(f"joined {joined}, unjoined {unjoined}")

    elif args.cmd == "dataset":
        from vexga.detect.dataset import make_dataset
        from vexga.detect.labelstudio import yolo_to_tasks
        from vexga.detect.prelabel import prelabel_dataset
        from vexga.games.base import get_game
        from vexga.store.db import connect
        from vexga.track.process import calibration_for

        game = get_game(args.game)
        out = make_dataset(args.name, game, args.frames)
        con = connect()
        vid = con.execute("SELECT id FROM videos LIMIT 1").fetchone()
        cal = calibration_for(con, vid["id"], 0) if vid else None
        if cal is None:
            print("no calibration yet - skipping pre-labels (rerun after vexga.calibrate.tool)")
        else:
            n = prelabel_dataset(out, cal, game)
            print(f"pre-labeled {n} frames")
        yolo_to_tasks(out, game)

    elif args.cmd == "track":
        from vexga.track.process import process_matches

        process_matches(args.weights, args.game, args.video, only_missing=not args.redo)

    elif args.cmd == "export":
        from vexga.store.export import export_match_json, to_parquet

        if args.match:
            print("wrote", export_match_json(args.match))
        else:
            to_parquet()
            from vexga.store.db import connect

            con = connect()
            for r in con.execute("SELECT id FROM matches WHERE EXISTS"
                                 " (SELECT 1 FROM robot_tracks rt WHERE rt.match_id = matches.id)"):
                export_match_json(r["id"])
            print("viewer JSONs in data/exports/matches/")


if __name__ == "__main__":
    main()
