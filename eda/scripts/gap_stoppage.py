"""Do long ball-missing gaps look like stoppages? (clause d of the draft dead-ball rule)

For each ball-missing gap (ball_gaps.find_gaps), the mean over its frames of three signals from
signal_profiles.py: the share of players below walking pace, the homography reprojection error
and the number of tracked player slots. Only aggregates are written: per gap-length bin
[0,2) [2,5) [5,10) [10,20) [20,35) [35,60) [60,inf) s, the n, median and quartiles of each
per-gap mean. Optional --clusters k: k-means on standardised (ln length, slow share,
reprojection error), reporting cluster sizes and centres only.

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/scripts/gap_stoppage.py OUT_DIR [--clusters 2]

    # any other game in the raw ATD format
    python gap_stoppage.py OUT_DIR --raw RAW.txt --frames 0-69660,69661-143760 \\
        [--homography H.csv] --pitch 105 68

Writes OUT_DIR/gap_stoppage_summary.csv (long format: analysis, scope, bin, metric, value,
unit). Definitions and fixture answers: eda/tests/fixtures/README.md ("signal_profiles.py and
gap_stoppage.py"). No per-gap rows anywhere.
"""

import argparse
import math
import os

import numpy as np

import ball_gaps as bg
import ball_stationary as bs
import common
import signal_profiles as spf

FPS = bg.FPS
PITCH = bg.PITCH
BINS_S = [(0.0, 2.0, "0-2s"), (2.0, 5.0, "2-5s"), (5.0, 10.0, "5-10s"), (10.0, 20.0, "10-20s"),
          (20.0, 35.0, "20-35s"), (35.0, 60.0, "35-60s"), (60.0, math.inf, ">=60s")]
METRICS = (("slow_players", "share of players"), ("reproj_error", "px"),
           ("n_tracked", "slots"))
N_INIT = 10
SEED = 0
CSV_NAME = "gap_stoppage_summary.csv"


def bin_label(length_frames, fps=FPS):
    s = length_frames / float(fps)
    for lo, hi, label in BINS_S:
        if lo <= s < hi:
            return label
    raise ValueError(length_frames)


def gap_means(gaps, values):
    """Float array (len(gaps),): mean of `values` over each gap's frames where defined."""
    out = np.full(len(gaps), np.nan)
    if values is None:
        return out
    for j, g in enumerate(gaps):
        seg = values[g["start"]:g["end"] + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[j] = float(seg.mean())
    return out


def kmeans(x, k, seed=SEED, n_init=N_INIT, iters=100):
    """(labels, centres, inertia): Lloyd's algorithm, best of n_init seeded starts."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centres = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(iters):
            d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            new = np.array([x[labels == c].mean(axis=0) if np.any(labels == c) else centres[c]
                            for c in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = float(((x - centres[labels]) ** 2).sum())
        if best is None or inertia < best[2] - 1e-12:
            best = (labels, centres, inertia)
    return best


def clusters(lengths_s, slow, reproj, k, seed=SEED):
    """{n_used, inertia, clusters: [{size, length_s, slow_players, reproj_error}]} sorted by
    centre length; None if fewer usable gaps than k."""
    feats = np.column_stack([np.log(lengths_s), slow, reproj])
    ok = np.isfinite(feats).all(axis=1)
    x = feats[ok]
    if len(x) < k:
        return None
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    labels, centres, inertia = kmeans((x - mu) / sd, k, seed)
    out = []
    for c in range(k):
        member = x[labels == c]
        centre = member.mean(axis=0)
        out.append({"size": int(len(member)), "length_s": float(np.exp(centre[0])),
                    "slow_players": float(centre[1]), "reproj_error": float(centre[2])})
    out.sort(key=lambda c: c["length_s"])
    return {"n_used": int(len(x)), "inertia": inertia, "clusters": out}


def analyse(raw_path, blocks, homography_path=None, fps=FPS, pitch=PITCH, lag=bs.LAG,
            walk_ms=spf.WALK_MS, n_clusters=0, seed=SEED):
    ball, _ = bg.read_ball(raw_path)
    missing = ~np.isfinite(ball).all(axis=1)
    gaps = bg.find_gaps(missing, blocks)
    n = max(len(ball), max(b for _, _, b in blocks) + 1)
    half_of = common.graded_half_of(blocks, n)
    slow, tracked = common.player_signals(raw_path, half_of, pitch, fps, lag, walk_ms)
    reproj = common.read_reprojection(homography_path, n) if homography_path else None
    means = {"slow_players": gap_means(gaps, slow), "reproj_error": gap_means(gaps, reproj),
             "n_tracked": gap_means(gaps, tracked)}
    labels = [bin_label(g["length"], fps) for g in gaps]
    res = {"blocks": list(blocks), "fps": fps, "pitch": tuple(pitch), "lag": lag,
           "walk_ms": walk_ms, "n_gaps": len(gaps), "homography": homography_path is not None,
           "bins": {}, "clusters": None, "n_clusters": n_clusters, "seed": seed}
    for _, _, label in [(0, 0, "all")] + BINS_S:
        idx = [j for j, l in enumerate(labels) if label in ("all", l)]
        cell = {"n_gaps": len(idx),
                "length_s": common.percentiles([gaps[j]["length"] / float(fps) for j in idx],
                                               (50, 25, 75))}
        for m, _ in METRICS:
            v = means[m][idx]
            v = v[np.isfinite(v)]
            cell[m] = {"n": int(len(v)), "q": common.percentiles(v, (50, 25, 75))}
        res["bins"][label] = cell
    if n_clusters and n_clusters > 0:
        if reproj is None:
            res["clusters"] = {"skipped": "no homography"}
        else:
            lengths = np.array([g["length"] / float(fps) for g in gaps])
            c = clusters(lengths, means["slow_players"], means["reproj_error"], n_clusters, seed)
            res["clusters"] = c if c is not None else {"skipped": "fewer usable gaps than k"}
    return res


def summary_rows(res):
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    add("params", "all", "", "fps", res["fps"], "frames/s")
    add("params", "all", "", "pitch_length", res["pitch"][0], "m")
    add("params", "all", "", "pitch_width", res["pitch"][1], "m")
    add("params", "all", "", "lag", res["lag"], "frames")
    add("params", "all", "", "walk_ms", res["walk_ms"], "m/s")
    add("params", "all", "", "clusters", res["n_clusters"], "count")
    add("params", "all", "", "seed", res["seed"], "")
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")
    for label, cell in res["bins"].items():
        add("gap_bin", "all", label, "n_gaps", cell["n_gaps"], "count")
        for q, v in zip(("median", "p25", "p75"), cell["length_s"]):
            add("gap_bin", "length", label, q, v, "s")
        for m, unit in METRICS:
            add("gap_bin", m, label, "n", cell[m]["n"], "count")
            for q, v in zip(("median", "p25", "p75"), cell[m]["q"]):
                add("gap_bin", m, label, q, v, unit)
    c = res["clusters"]
    if c and "clusters" in c:
        add("cluster", "all", "", "n_used", c["n_used"], "count")
        add("cluster", "all", "", "inertia", c["inertia"], "")
        for i, cl in enumerate(c["clusters"], 1):
            sc = "cluster {}".format(i)
            add("cluster", sc, "", "size", cl["size"], "count")
            add("cluster", sc, "", "length", cl["length_s"], "s")
            add("cluster", sc, "", "slow_players", cl["slow_players"], "share of players")
            add("cluster", sc, "", "reproj_error", cl["reproj_error"], "px")
    return rows


def _f(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, csv_path):
    print("{} gaps; per bin: n, then median [p25, p75] of the per-gap mean slow_players, "
          "reproj_error, n_tracked{}".format(
              res["n_gaps"], "" if res["homography"] else " (no homography: reproj empty)"))
    for label, cell in res["bins"].items():
        if label == "all":
            continue
        cols = []
        for m, _ in METRICS:
            q = cell[m]["q"]
            cols.append("{} [{}, {}]".format(_f(q[0]), _f(q[1]), _f(q[2])))
        print("  {:<8}{:>6}  {:<26}{:<26}{}".format(label, cell["n_gaps"], *cols))
    c = res["clusters"]
    if c:
        if "skipped" in c:
            print("clusters: skipped, {}".format(c["skipped"]))
        else:
            print("clusters ({} gaps): ".format(c["n_used"]) + "; ".join(
                "n={} length {} s, slow {}, reproj {}".format(
                    cl["size"], _f(cl["length_s"], "{:.1f}"), _f(cl["slow_players"]),
                    _f(cl["reproj_error"])) for cl in c["clusters"]))
    print("wrote {}".format(csv_path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (outside the repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--homography", help="homography CSV; default games.asset_paths()"
                   "['homography'] when --raw is not given")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--lag", type=int, default=bs.LAG, help="speed over this many frames (default 5)")
    p.add_argument("--walk-ms", type=float, default=spf.WALK_MS,
                   help="slow players: below this speed, m/s (default 2)")
    p.add_argument("--clusters", type=int, default=0,
                   help="k-means with this many clusters (default 0 = off)")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/gap_stoppage")
    if args.lag < 1:
        p.error("--lag must be at least 1")
    if args.raw:
        if args.frames is None:
            p.error("--frames is required with --raw")
        raw, hom, blocks = args.raw, args.homography, args.frames
    else:
        paths = common.game_paths(args.game, ["atd_raw"],
                                  [] if args.homography else ["homography"])
        raw, hom = paths["atd_raw"], args.homography or paths.get("homography")
        blocks = args.frames or bg._games().half_video_blocks(args.game)
    res = analyse(raw, blocks, hom, fps=args.fps, pitch=tuple(args.pitch), lag=args.lag,
                  walk_ms=args.walk_ms, n_clusters=args.clusters, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, CSV_NAME)
    bg.write_csv(summary_rows(res), csv_path)
    print_summary(res, csv_path)
    return res


if __name__ == "__main__":
    main()
