"""Sample long ball-tracking gaps for video review (EDA stratum 1).

Gaps come from ball_gaps.find_gaps on graded frames; gaps cut by a half boundary are dropped
(their true length is unknown). Eligible = length >= the q-quantile of gap lengths, or, with
--min-s/--max-s, min <= seconds < max. Gaps overlapping an --exclude range are dropped. The
sample is spread over the halves as evenly as the eligible gaps allow.

Usage (needs ball_gaps.py next to this file; stdlib + numpy only)
    # competition match: paths from games.asset_paths(), graded frames from
    # games.half_video_blocks(GAME); --exclude defaults to the public GT window and the
    # second-half pre-kickoff frames
    PYTHONPATH=/path/to/team-repo python eda/scripts/sample_long_gaps.py OUT_DIR

    # any other game in the raw ATD format
    python sample_long_gaps.py OUT_DIR --raw RAW.txt --frames 0-69660,69661-143760

Writes OUT_DIR/long_gap_sample.csv. That file is FRAME-INDEXED by design: never commit it.
stdout carries only aggregate counts.
"""

import argparse
import csv
import os
import sys

import numpy as np

import ball_gaps as bg
from ball_gaps import parse_frames  # noqa: F401  (re-exported for tests)

FPS = bg.FPS
PAD_S = 3                       # seconds of video context either side
CSV_NAME = "long_gap_sample.csv"
COLUMNS = ["half", "start_frame", "end_frame", "gap_s", "view_from", "view_to", "category",
           "notes"]
# BAR-ATH, public: the published GT window (DATA.md) and half 2 before the kickoff at 113742
# (starter/01_smart_tagging.ipynb); half 2 is graded from 110808.
DEFAULT_EXCLUDE = [(11185, 18684), (110808, 113741)]


def ts(frame, fps=FPS):
    """Video frame -> h:mm:ss.ss of the video, for seeking."""
    s = frame / float(fps)
    return "{}:{:02d}:{:05.2f}".format(int(s // 3600), int(s % 3600 // 60), s % 60)


def parse_ranges(text):
    """'a-b,c-d' -> [(a, b), (c, d)] inclusive."""
    return [(a, b) for _, a, b in bg.parse_frames(text)]


def quotas(available, k):
    """{half: count}: hand out k one at a time to the half with the smallest quota that still
    has unused gaps; ties go to the lower half."""
    q = {h: 0 for h in available}
    for _ in range(k):
        open_ = [h for h in sorted(q) if q[h] < available[h]]
        if not open_:
            break
        h = min(open_, key=lambda x: (q[x], x))
        q[h] += 1
    return q


def analyse(raw_path, blocks, n=8, quantile=0.9, min_s=None, max_s=None, seed=2027,
            exclude=(), fps=FPS):
    """Gaps, eligibility and the sample, as a dict. Nothing printed here."""
    ball, _ = bg.read_ball(raw_path)
    gaps = bg.find_gaps(~np.isfinite(ball).all(axis=1), blocks)
    candidates = [g for g in gaps if not g["edge"]]

    if min_s is not None or max_s is not None:
        lo = (min_s or 0.0) * fps
        hi = np.inf if max_s is None else max_s * fps
        eligible = [g for g in candidates if lo <= g["length"] < hi]
        threshold = lo
    else:
        lengths = [g["length"] for g in candidates]
        threshold = float(np.quantile(lengths, quantile)) if lengths else np.nan
        eligible = [g for g in candidates if g["length"] >= threshold]
    eligible = [g for g in eligible
                if not any(a <= g["end"] and g["start"] <= b for a, b in exclude)]

    by_half = {h: [g for g in eligible if g["half"] == h] for h, _, _ in blocks}
    q = quotas({h: len(v) for h, v in by_half.items()}, min(n, len(eligible)))
    rng = np.random.default_rng(seed)
    sample = []
    for h, _, _ in blocks:
        if q[h]:
            idx = rng.choice(len(by_half[h]), size=q[h], replace=False)
            sample.extend(by_half[h][i] for i in idx)
    sample.sort(key=lambda g: g["start"])
    return {"gaps": gaps, "n_gaps": len(gaps), "n_candidates": len(candidates),
            "threshold_frames": threshold, "eligible": eligible, "sample": sample,
            "quotas": q}


def rows(sample, fps=FPS, pad_s=PAD_S):
    pad = int(round(pad_s * fps))
    return [[g["half"], g["start"], g["end"], round(g["length"] / float(fps), 1),
             ts(max(0, g["start"] - pad), fps), ts(g["end"] + pad, fps), "", ""]
            for g in sample]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (must be outside the repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH", help="games.py game id (default BAR-ATH)")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames per half, e.g. '0-69660,69661-143760'; "
                        "required with --raw")
    p.add_argument("--n", type=int, default=8, help="gaps to sample (default 8)")
    p.add_argument("--quantile", type=float, default=0.9,
                   help="eligible = length >= this quantile (default 0.9)")
    p.add_argument("--min-s", type=float, default=None, help="eligible from (s, inclusive)")
    p.add_argument("--max-s", type=float, default=None, help="eligible below (s, exclusive)")
    p.add_argument("--exclude", type=parse_ranges, default=None,
                   help="drop gaps overlapping these inclusive video-frame ranges, 'a-b,c-d'; "
                        "default without --raw: 11185-18684,110808-113741; with --raw: none")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--csv-name", default=CSV_NAME,
                   help="output file name inside OUT_DIR (default {})".format(CSV_NAME))
    return p


def options_from(argv):
    """analyse() keyword arguments from CLI options (without out_dir/--raw/--frames)."""
    a = build_parser().parse_args(["_"] + list(argv))
    return {"n": a.n, "quantile": a.quantile, "min_s": a.min_s, "max_s": a.max_s,
            "seed": a.seed, "exclude": a.exclude or [], "fps": a.fps}


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if os.path.basename(a.csv_name) != a.csv_name or not a.csv_name.endswith(".csv"):
        p.error("--csv-name must be a plain file name ending in .csv")
    bg.check_out_dir(a.out_dir, example="~/eda_out/long_gaps")

    if a.raw:
        if a.frames is None:
            p.error("--frames is required with --raw")
        raw, blocks = a.raw, a.frames
        exclude = a.exclude or []
    else:
        games = bg._games()
        try:
            raw = games.require(["atd_raw"], game=a.game)[0]
        except (IOError, OSError, KeyError) as err:
            sys.exit(str(err))
        blocks = a.frames or games.half_video_blocks(a.game)
        exclude = DEFAULT_EXCLUDE if a.exclude is None else a.exclude

    res = analyse(raw, blocks, n=a.n, quantile=a.quantile, min_s=a.min_s, max_s=a.max_s,
                  seed=a.seed, exclude=exclude, fps=a.fps)

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, a.csv_name)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows(res["sample"], a.fps))

    # Aggregates only.
    rule = ("range [{}, {}) s".format(a.min_s or 0, "inf" if a.max_s is None else a.max_s)
            if a.min_s is not None or a.max_s is not None
            else "q={} -> {:.1f} s".format(a.quantile, res["threshold_frames"] / a.fps))
    print("gaps: {} ({} not cut by a half edge); eligible ({}, {} exclude range(s)): {}".format(
        res["n_gaps"], res["n_candidates"], rule, len(exclude), len(res["eligible"])))
    print("sampled {} (per half {}) -> {} (frame-indexed: do not commit)".format(
        len(res["sample"]), ", ".join("{}: {}".format(h, v) for h, v in res["quotas"].items()),
        path))
    return res


if __name__ == "__main__":
    main()
