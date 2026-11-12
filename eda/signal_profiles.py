"""Per-frame signals profiled around set-piece tag starts (and ends), with a shift null.

Generalises view A of tag_gaps.py from "ball missing" to any per-frame signal. For SET PIECES
and each set-piece Type, the pooled mean of the signal in 1 s bins from -30 s to +30 s around
each tag's start (and end), against the same tags shifted circularly within each half
(null mean, p5, p95). Signals (one value per graded frame, NaN = undefined):

    ball_missing   1 if the ball is missing (reproduces tag_gaps view A exactly)
    ball_outside   1 if the visible ball is out of the pitch by more than --margin-m
    ball_still     1 if the frame is in a ball_stationary.py still run of >= --min-still-s
    slow_players   share of tracked players below --walk-ms (derived speed over --lag frames)
    reproj_error   homography reprojection_error where valid (skipped without a homography)
    n_tracked      number of player slots with a position

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/signal_profiles.py OUT_DIR

    # any other game in the raw ATD format (profiles need --tags)
    python signal_profiles.py OUT_DIR --raw RAW.txt --frames 0-69660,69661-143760 \\
        [--tags TAGS.xml] [--homography H.csv] --pitch 105 68

Writes OUT_DIR/signal_profiles_summary.csv (long format: analysis, scope, bin, metric, value,
unit); scope is "<signal>:<group>". Definitions and fixture answers: tests/fixtures/README.md
("signal_profiles.py and gap_stoppage.py"). Aggregates only.
"""

import argparse
import math
import os
import warnings

import numpy as np

import ball_gaps as bg
import ball_outside as bo
import ball_stationary as bs
import common
import tag_gaps as tg

FPS = bg.FPS
PITCH = bg.PITCH
WALK_MS = 2.0            # brisk walking pace; jogging starts around 2-3 m/s
SIGNALS = ("ball_missing", "ball_outside", "ball_still", "slow_players", "reproj_error",
           "n_tracked")
UNITS = {"ball_missing": "share of frames", "ball_outside": "share of frames",
         "ball_still": "share of frames", "slow_players": "share of players",
         "reproj_error": "px", "n_tracked": "slots"}
CSV_NAME = "signal_profiles_summary.csv"


def compute_signals(raw_path, blocks, homography_path=None, fps=FPS, pitch=PITCH, lag=bs.LAG,
                    walk_ms=WALK_MS, margin_m=bo.MARGIN_M, still_ms=bs.STILL_MS,
                    min_still_s=bs.MIN_STILL_S, still_gap=bs.STILL_GAP):
    """{signal: float array by video frame}, all of length (last graded frame + 1)."""
    ball, _ = bg.read_ball(raw_path)
    n = max(len(ball), max(b for _, _, b in blocks) + 1)
    if len(ball) < n:
        ball = np.vstack([ball, np.full((n - len(ball), 2), np.nan)])
    visible = np.isfinite(ball).all(axis=1)
    half_of = common.graded_half_of(blocks, n)
    graded = half_of > 0
    out = {"ball_missing": np.where(graded, (~visible).astype(float), np.nan)}

    ex, ey = bo.excess(ball, pitch)
    with np.errstate(invalid="ignore"):
        outside = np.maximum(ex, ey) > margin_m
    out["ball_outside"] = np.where(graded & visible, outside.astype(float), np.nan)

    speed = bs.ball_speed(ball, half_of, pitch, fps, lag)
    still = bs.still_mask(speed, visible, lag, still_ms) & graded
    min_frames = int(math.ceil(min_still_s * fps - 1e-9))
    in_run = np.zeros(n, dtype=bool)
    for r in bs.merge_runs(common.find_runs(still, blocks), still_gap):
        if r["length"] >= min_frames:
            in_run[r["start"]:r["end"] + 1] = True
    out["ball_still"] = np.where(graded & visible, in_run.astype(float), np.nan)

    out["slow_players"], out["n_tracked"] = common.player_signals(raw_path, half_of, pitch, fps,
                                                                  lag, walk_ms)
    out["reproj_error"] = None
    if homography_path:
        out["reproj_error"] = np.where(graded, common.read_reprojection(homography_path, n),
                                       np.nan)
    return out


def profile(values, groups, blocks, fps=FPS, window_s=tg.WINDOW_S, bin_s=tg.BIN_S,
            offsets=None, end=False):
    """{group: {n, n_dropped, bins_s, sum, counted, value[, null_mean, null_p5, null_p95]}} as
    tag_gaps.view_a, with the pooled mean of the signal as the value."""
    window_f, bin_f = int(round(window_s * fps)), int(round(bin_s * fps))
    halves = common.signal_halves(values, blocks)
    nb = 2 * window_f // bin_f
    bins_s = [float((k - nb // 2) * bin_s) for k in range(nb)]
    out = {}
    for name, tags in groups.items():
        anchors, dropped = tg.graded_anchors(tags, blocks, end=end)
        total, counted = tg.profile_counts(halves, anchors, window_f, bin_f)
        res = {"n": len(anchors), "n_dropped": dropped, "bins_s": bins_s,
               "sum": total.tolist(), "counted": counted.tolist(),
               "value": tg._share(total, counted).tolist()}
        if offsets is not None and anchors:
            t_null, c_null = tg.profile_counts(halves, anchors, window_f, bin_f, offsets)
            val = tg._share(t_null, c_null)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res["null_mean"] = np.nanmean(val, axis=0).tolist()
                p5, p95 = np.nanpercentile(val, [5, 95], axis=0)
            res["null_p5"], res["null_p95"] = p5.tolist(), p95.tolist()
        out[name] = res
    return out


def analyse(raw_path, blocks, tags_path=None, homography_path=None, fps=FPS, pitch=PITCH,
            lag=bs.LAG, walk_ms=WALK_MS, margin_m=bo.MARGIN_M, still_ms=bs.STILL_MS,
            min_still_s=bs.MIN_STILL_S, window_s=tg.WINDOW_S, bin_s=tg.BIN_S, ends=False,
            null_n=bg.NULL_N, null_seed=bg.NULL_SEED, null_min_shift_s=bg.NULL_MIN_SHIFT_S,
            offsets=None):
    """All numbers, as a dict. `offsets` overrides the drawn null (tests)."""
    signals = compute_signals(raw_path, blocks, homography_path, fps, pitch, lag, walk_ms,
                              margin_m, still_ms, min_still_s)
    res = {"blocks": list(blocks), "fps": fps, "pitch": tuple(pitch), "lag": lag,
           "walk_ms": walk_ms, "margin_m": margin_m, "still_ms": still_ms,
           "min_still_s": min_still_s, "window_s": window_s, "bin_s": bin_s, "ends": ends,
           "null_n": null_n, "null_seed": null_seed, "null_min_shift_s": null_min_shift_s,
           "signals": [s for s in SIGNALS if signals[s] is not None], "baseline": {},
           "tags": tags_path is not None, "null_skipped": None, "profiles": {}}
    for s in res["signals"]:
        v = signals[s]
        ok = np.isfinite(v)
        res["baseline"][s] = (float(v[ok].mean()) if ok.any() else math.nan, int(ok.sum()))
    if not tags_path:
        return res
    groups = common.set_piece_groups(tg.read_tags(tags_path, fps))
    res["n_tags"] = {k: (len(v), tg.graded_anchors(v, blocks)[1]) for k, v in groups.items()}
    if offsets is None:
        offsets, res["null_skipped"] = common.draw_null(blocks, null_n, null_seed,
                                                        null_min_shift_s, fps)
    for anchor in ("start", "end") if ends else ("start",):
        for s in res["signals"]:
            res["profiles"][(anchor, s)] = profile(signals[s], groups, blocks, fps, window_s,
                                                   bin_s, offsets, end=anchor == "end")
    return res


def summary_rows(res):
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    for name, value, unit in (("fps", res["fps"], "frames/s"),
                              ("pitch_length", res["pitch"][0], "m"),
                              ("pitch_width", res["pitch"][1], "m"),
                              ("lag", res["lag"], "frames"), ("walk_ms", res["walk_ms"], "m/s"),
                              ("margin", res["margin_m"], "m"),
                              ("still_ms", res["still_ms"], "m/s"),
                              ("min_still", res["min_still_s"], "s"),
                              ("window", res["window_s"], "s"), ("bin", res["bin_s"], "s"),
                              ("null_n", res["null_n"], "count"),
                              ("null_seed", res["null_seed"], ""),
                              ("null_min_shift", res["null_min_shift_s"], "s")):
        add("params", "all", "", name, value, unit)
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")
    for s in res["signals"]:
        mean, n = res["baseline"][s]
        add("baseline", s, "", "mean", mean, UNITS[s])
        add("baseline", s, "", "defined_frames", n, "frames")
    for name, (n, dropped) in res.get("n_tags", {}).items():
        add("tags", name, "", "n_tags", n, "count")
        add("tags", name, "", "n_dropped", dropped, "count")
    for (anchor, s), groups in res["profiles"].items():
        for name, a in groups.items():
            sc = "{}:{}".format(s, name)
            add("profile_" + anchor, sc, "", "n_instances", a["n"], "count")
            for k, left in enumerate(a["bins_s"]):
                b = "{:g}".format(left)
                add("profile_" + anchor, sc, b, "value", a["value"][k], UNITS[s])
                add("profile_" + anchor, sc, b, "counted", a["counted"][k], "frames")
                for m in ("null_mean", "null_p5", "null_p95"):
                    if m in a:
                        add("profile_" + anchor, sc, b, m, a[m][k], UNITS[s])
    return rows


def _f(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, csv_path, pre_s=10.0):
    print("baseline over graded frames: " + "; ".join(
        "{} {} (n={})".format(s, _f(res["baseline"][s][0]), res["baseline"][s][1])
        for s in res["signals"]))
    if "reproj_error" not in res["signals"]:
        print("note: no homography file, reproj_error skipped")
    if not res["tags"]:
        print("no Smart Tagging file: profiles skipped")
        print("wrote {}".format(csv_path))
        return
    if res["null_skipped"]:
        print("  null skipped: {}".format(res["null_skipped"]))
    sp = "code:" + tg.SET_PIECE_CODE
    for s in res["signals"]:
        a = res["profiles"][("start", s)][sp]
        idx = [k for k, left in enumerate(a["bins_s"]) if -pre_s <= left < 0]
        tot = sum(a["sum"][k] for k in idx)
        cnt = sum(a["counted"][k] for k in idx)
        null = float(np.nanmean([a["null_mean"][k] for k in idx])) if "null_mean" in a \
            else math.nan
        print("  {:<13} {:g} s before SET PIECES starts (n={}): {} vs null {}".format(
            s, pre_s, a["n"], _f(tot / cnt if cnt else math.nan), _f(null)))
    print("wrote {}".format(csv_path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (outside the repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--tags", help="Smart Tagging XML; default games.asset_paths()"
                   "['smart_tagging'] when --raw is not given")
    p.add_argument("--homography", help="homography CSV; default games.asset_paths()"
                   "['homography'] when --raw is not given")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--lag", type=int, default=bs.LAG, help="speed over this many frames (default 5)")
    p.add_argument("--walk-ms", type=float, default=WALK_MS,
                   help="slow_players: below this speed, m/s (default 2)")
    p.add_argument("--margin-m", type=float, default=bo.MARGIN_M,
                   help="ball_outside margin, m (default 1)")
    p.add_argument("--still-ms", type=float, default=bs.STILL_MS,
                   help="ball_still speed threshold, m/s (default 0.5)")
    p.add_argument("--min-still-s", type=float, default=bs.MIN_STILL_S,
                   help="ball_still minimum run, s (default 1)")
    p.add_argument("--window", type=float, default=tg.WINDOW_S,
                   help="seconds either side of the anchor (default 30)")
    p.add_argument("--bin", type=float, default=tg.BIN_S, help="bin width, s (default 1)")
    p.add_argument("--ends", action="store_true", help="also profile around tag ends")
    p.add_argument("--null-n", type=int, default=bg.NULL_N, help="shifts (default 1000; 0 = off)")
    p.add_argument("--null-seed", type=int, default=bg.NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=bg.NULL_MIN_SHIFT_S,
                   help="smallest shift in seconds (default 60)")
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/signal_profiles")
    if int(round(args.window * args.fps)) % int(round(args.bin * args.fps)):
        p.error("--window must be a multiple of --bin")
    if args.lag < 1:
        p.error("--lag must be at least 1")
    if args.raw:
        if args.frames is None:
            p.error("--frames is required with --raw")
        raw, tags, hom, blocks = args.raw, args.tags, args.homography, args.frames
    else:
        optional = [n for n, given in (("smart_tagging", args.tags), ("homography", args.homography))
                    if not given]
        paths = common.game_paths(args.game, ["atd_raw"], optional)
        raw = paths["atd_raw"]
        tags = args.tags or paths.get("smart_tagging")
        hom = args.homography or paths.get("homography")
        blocks = args.frames or bg._games().half_video_blocks(args.game)
    res = analyse(raw, blocks, tags, hom, fps=args.fps, pitch=tuple(args.pitch), lag=args.lag,
                  walk_ms=args.walk_ms, margin_m=args.margin_m, still_ms=args.still_ms,
                  min_still_s=args.min_still_s, window_s=args.window, bin_s=args.bin,
                  ends=args.ends, null_n=args.null_n, null_seed=args.null_seed,
                  null_min_shift_s=args.null_min_shift)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, CSV_NAME)
    bg.write_csv(summary_rows(res), csv_path)
    print_summary(res, csv_path)
    return res


if __name__ == "__main__":
    main()
