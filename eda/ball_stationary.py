"""Ball stationary at a restart location (clause c of the draft dead-ball rule).

On ball-visible graded frames, a still run is a stretch where the ball's derived speed (over
`--lag` frames) stays below `--still-ms` for at least `--min-still-s`, with gaps of up to
`--still-gap` frames bridged. At each minimum length (main plus `--min-still-also`):

  location   each run is placed by its median position: centre spot, penalty spot, corner arc,
             goal area, touchline band or elsewhere (IFAB marks; symmetric, no attacking
             direction needed); counts, share of graded frames and length bins per class
  link       does a run end within +-W s of a set-piece start? SET PIECES and per Type, per
             location, with the tag_gaps circular-shift null (lift, p)
  crosstab   location x Type of the nearest set-piece start within W
  kickoff    centre-spot runs whose last frame has each team in its own half, per half
             (needs the ATD metadata for the slot teams)

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/ball_stationary.py OUT_DIR

    # any other game in the raw ATD format
    python ball_stationary.py OUT_DIR --raw RAW.txt --meta META.xml \\
        --frames 0-69660,69661-143760 --pitch 105 68

Writes OUT_DIR/ball_stationary_summary.csv (long format: analysis, scope, bin, metric, value,
unit). scope is "s<min still>:<location>". Definitions and fixture answers:
tests/fixtures/README.md ("ball_stationary.py"). Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import ball_gaps as bg
import common
import tag_gaps as tg

FPS = bg.FPS
PITCH = bg.PITCH
LAG = 5                  # frames: speed over 0.2 s, so per-frame jitter does not read as motion
STILL_MS = 0.5           # a ball being placed or waiting; rolling balls are well above this
MIN_STILL_S = 1.0
MIN_STILL_ALSO = (0.5, 2.0)
STILL_GAP = 3            # frames of noise or missing ball bridged inside a run
SPOT_TOL_M = 1.0         # about the ball-tracking error
BAND_M = 2.0             # throw-ins: ball on or just off the touchline
LINK_WINDOW_S = 5.0
LOCATIONS = ("centre_spot", "penalty_spot", "corner_arc", "goal_area", "touchline_band",
             "elsewhere")
BINS_S = [(0.0, 1.0, "<1s"), (1.0, 2.0, "1-2s"), (2.0, 5.0, "2-5s"), (5.0, 10.0, "5-10s"),
          (10.0, math.inf, ">=10s")]
CSV_NAME = "ball_stationary_summary.csv"

# IFAB, metres
PENALTY_SPOT_M = 11.0
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 3.66 + 5.5
CORNER_ARC_M = 1.0


# ---------------------------------------------------------------- still runs

def ball_speed(ball, half_of, pitch=PITCH, fps=FPS, lag=LAG):
    """Float array by frame: speed (m/s) over [f - lag, f]; NaN unless the ball is visible on
    both and both are graded in the same half."""
    n = len(ball)
    speed = np.full(n, np.nan)
    if n <= lag:
        return speed
    hv = half_of[:n] if len(half_of) >= n else np.concatenate(
        [half_of, np.zeros(n - len(half_of), dtype=int)])
    v = common.step_speed(ball[:-lag, :2], ball[lag:, :2], pitch, fps) / lag
    ok = (hv[lag:] > 0) & (hv[lag:] == hv[:-lag]) & np.isfinite(v)
    speed[lag:][ok] = v[ok]
    return speed


def still_mask(speed, visible, lag=LAG, still_ms=STILL_MS):
    """Frames covered by a still window [f - lag, f] (speed[f] < still_ms) and visible."""
    with np.errstate(invalid="ignore"):
        ends = (speed < still_ms).astype(np.int64)
    # frame i is covered if some end g in [i, i + lag] is still
    cs = np.concatenate([[0], np.cumsum(ends)])
    i = np.arange(len(ends))
    covered = (cs[np.minimum(i + lag + 1, len(ends))] - cs[i]) > 0
    return covered & visible


def merge_runs(runs, max_gap):
    """Merge consecutive runs of the same half separated by at most max_gap frames."""
    out = []
    for r in runs:
        if out and out[-1]["half"] == r["half"] and r["start"] - out[-1]["end"] - 1 <= max_gap:
            out[-1] = dict(out[-1], end=r["end"], length=r["end"] - out[-1]["start"] + 1)
        else:
            out.append(dict(r))
    return out


def classify(x_m, y_m, pitch=PITCH, tol=SPOT_TOL_M, band=BAND_M):
    """Location class of a point in metres (origin at a corner)."""
    length, width = pitch
    dx = min(x_m, length - x_m)              # to the nearer goal line, negative outside
    dy = min(y_m, width - y_m)               # to the nearer touchline, negative outside
    off_centre = abs(y_m - width / 2.0)
    if math.hypot(x_m - length / 2.0, off_centre) <= tol:
        return "centre_spot"
    if math.hypot(dx - PENALTY_SPOT_M, off_centre) <= tol:
        return "penalty_spot"
    if math.hypot(dx, dy) <= CORNER_ARC_M + tol:
        return "corner_arc"
    if dx <= GOAL_AREA_DEPTH_M + tol and off_centre <= GOAL_AREA_HALF_WIDTH_M + tol:
        return "goal_area"
    if abs(dy) <= band:
        return "touchline_band"
    return "elsewhere"


def length_label(frames, fps=FPS):
    s = frames / float(fps)
    for lo, hi, label in BINS_S:
        if lo <= s < hi:
            return label
    raise ValueError(frames)


# ---------------------------------------------------------------- kick-off

def kickoff_ok(slots, teams, pitch=PITCH, tol=SPOT_TOL_M):
    """Each team on its own side of halfway (within tol metres), at least one player each."""
    if slots is None or teams is None:
        return False
    k = min(len(slots), len(teams))
    pts, tm = slots[:k], teams[:k]
    ok = np.isfinite(pts[:, :2]).all(axis=1)
    home, away = pts[ok & (tm == 0), 0], pts[ok & (tm == 1), 0]
    if len(home) == 0 or len(away) == 0:
        return False
    t = tol / pitch[0]
    left = np.all(home <= 0.5 + t) and np.all(away >= 0.5 - t)
    right = np.all(home >= 0.5 - t) and np.all(away <= 0.5 + t)
    return bool(left or right)


# ---------------------------------------------------------------- driver

def analyse(raw_path, blocks, tags_path=None, meta_path=None, fps=FPS, pitch=PITCH, lag=LAG,
            still_ms=STILL_MS, min_still_s=MIN_STILL_S, min_still_also=MIN_STILL_ALSO,
            still_gap=STILL_GAP, spot_tol=SPOT_TOL_M, band=BAND_M, link_window_s=LINK_WINDOW_S,
            null_n=bg.NULL_N, null_seed=bg.NULL_SEED, null_min_shift_s=bg.NULL_MIN_SHIFT_S):
    """All numbers, as a dict. No per-frame or per-run data leaves this function."""
    ball, _ = bg.read_ball(raw_path)
    visible = np.isfinite(ball).all(axis=1)
    half_of = common.graded_half_of(blocks, len(ball))
    speed = ball_speed(ball, half_of, pitch, fps, lag)
    still = still_mask(speed, visible, lag, still_ms) & (half_of > 0)
    all_runs = merge_runs(common.find_runs(still, blocks), still_gap)
    for r in all_runs:
        seg = ball[r["start"]:r["end"] + 1]
        seg = seg[np.isfinite(seg).all(axis=1)]
        mx, my = np.median(seg[:, 0]) * pitch[0], np.median(seg[:, 1]) * pitch[1]
        r["location"] = classify(mx, my, pitch, spot_tol, band)

    graded_total = sum(b - a + 1 for _, a, b in blocks)
    w_frames = int(round(link_window_s * fps))
    mins = [min_still_s] + [m for m in min_still_also if m != min_still_s]
    res = {"blocks": list(blocks), "fps": fps, "pitch": tuple(pitch), "lag": lag,
           "still_ms": still_ms, "mins": mins, "still_gap": still_gap, "spot_tol": spot_tol,
           "band": band, "link_window_s": link_window_s, "null_n": null_n,
           "null_seed": null_seed, "null_min_shift_s": null_min_shift_s,
           "graded_frames": graded_total, "visible_frames": int(np.sum(visible & (half_of > 0))),
           "still_frames": int(still.sum()), "tags": tags_path is not None,
           "meta": meta_path is not None, "null_skipped": None, "by_min": {}}

    sp_groups = offsets = None
    if tags_path:
        sp_groups = common.set_piece_groups(tg.read_tags(tags_path, fps))
        offsets, res["null_skipped"] = common.draw_null(blocks, null_n, null_seed,
                                                        null_min_shift_s, fps)
        res["n_tags"] = {k: len(v) for k, v in sp_groups.items()}

    kick = {}
    if meta_path:
        ends = [r["end"] for r in all_runs if r["location"] == "centre_spot"]
        slots = bg.read_slots(raw_path, ends)
        specs = bg.read_slot_teams(meta_path)
        kick = {f: kickoff_ok(slots.get(f), bg.teams_for(specs, f), pitch, spot_tol)
                for f in ends}

    for m in mins:
        min_frames = int(math.ceil(m * fps - 1e-9))
        runs = [r for r in all_runs if r["length"] >= min_frames]
        res["by_min"][m] = summarise(runs, blocks, sp_groups, offsets, w_frames, fps, kick,
                                     meta_path is not None)
    return res


def summarise(runs, blocks, sp_groups, offsets, w_frames, fps, kick, has_meta):
    groups = {"all": list(range(len(runs)))}
    for loc in LOCATIONS:
        groups[loc] = [j for j, r in enumerate(runs) if r["location"] == loc]
    out = {"locations": {}, "link": None, "crosstab": None, "kickoff": None}
    for loc, idx in groups.items():
        lengths = {label: 0 for _, _, label in BINS_S}
        for j in idx:
            lengths[length_label(runs[j]["length"], fps)] += 1
        out["locations"][loc] = {"n_runs": len(idx),
                                 "frames": int(sum(runs[j]["length"] for j in idx)),
                                 "lengths": lengths}
    if sp_groups is not None:
        points = common.link_points(runs)
        out["link"] = {}
        for name, tags in sp_groups.items():
            anchors, _ = tg.graded_anchors(tags, blocks)
            hits = tg.link_hits(points, anchors, blocks, w_frames)
            matrix = tg.link_null_matrix(points, anchors, blocks, offsets, w_frames) \
                if offsets is not None and points else None
            out["link"][name] = {loc: tg.null_stats(hits, matrix, idx)
                                 for loc, idx in groups.items()}
        sp = sp_groups.get("code:" + tg.SET_PIECE_CODE, [])
        cross = {loc: {} for loc in groups}
        for r in runs:
            t = common.nearest_type(r["end"], r["half"], sp, blocks, w_frames)
            for loc in ("all", r["location"]):
                cross[loc][t] = cross[loc].get(t, 0) + 1
        out["crosstab"] = cross
    if has_meta:
        out["kickoff"] = {}
        for h, _, _ in blocks:
            centre = [r for r in runs if r["location"] == "centre_spot" and r["half"] == h]
            out["kickoff"][h] = {"centre_runs": len(centre),
                                 "kickoff_runs": sum(bool(kick.get(r["end"])) for r in centre)}
    return out


# ---------------------------------------------------------------- output

def scope(m, loc):
    return "s{:g}:{}".format(m, loc)


def summary_rows(res):
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    add("params", "all", "", "fps", res["fps"], "frames/s")
    add("params", "all", "", "pitch_length", res["pitch"][0], "m")
    add("params", "all", "", "pitch_width", res["pitch"][1], "m")
    add("params", "all", "", "lag", res["lag"], "frames")
    add("params", "all", "", "still_ms", res["still_ms"], "m/s")
    for i, m in enumerate(res["mins"]):
        add("params", "all", "main" if i == 0 else "also", "min_still", m, "s")
    add("params", "all", "", "still_gap", res["still_gap"], "frames")
    add("params", "all", "", "spot_tol", res["spot_tol"], "m")
    add("params", "all", "", "band", res["band"], "m")
    add("params", "all", "", "link_window", res["link_window_s"], "s")
    add("params", "all", "", "null_n", res["null_n"], "count")
    add("params", "all", "", "null_seed", res["null_seed"], "")
    add("params", "all", "", "null_min_shift", res["null_min_shift_s"], "s")
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")
    add("overview", "all", "", "graded_frames", res["graded_frames"], "frames")
    add("overview", "all", "", "visible_frames", res["visible_frames"], "frames")
    add("overview", "all", "", "still_frames", res["still_frames"], "frames")
    for name, n in res.get("n_tags", {}).items():
        add("tags", name, "", "n_tags", n, "count")
    units = {"n_gaps": "count", "real_count": "count", "p_emp": "p", "lift": "ratio"}
    for m, r in res["by_min"].items():
        for loc, st in r["locations"].items():
            sc = scope(m, loc)
            add("location", sc, "", "n_runs", st["n_runs"], "count")
            add("location", sc, "", "frames", st["frames"], "frames")
            add("location", sc, "", "share", bg._rate(st["frames"], res["graded_frames"]),
                "share of graded frames")
            for label, n in st["lengths"].items():
                add("runs", sc, label, "n_runs", n, "count")
            if r["link"] is not None:
                for name, by_loc in r["link"].items():
                    s = by_loc[loc]
                    for metric in ("n_gaps", "real_count", "real_share", "null_mean", "null_p5",
                                   "null_p95", "p_emp", "lift"):
                        if metric in s:
                            add("link", sc, name, "n_runs" if metric == "n_gaps" else metric,
                                s[metric], units.get(metric, "share of runs"))
            if r["crosstab"] is not None:
                for t, n in sorted(r["crosstab"][loc].items()):
                    add("crosstab", sc, t, "n_runs", n, "count")
        if r["kickoff"] is not None:
            for h, k in r["kickoff"].items():
                sc = scope(m, "centre_spot")
                add("kickoff", sc, "half {}".format(h), "centre_runs", k["centre_runs"], "count")
                add("kickoff", sc, "half {}".format(h), "kickoff_runs", k["kickoff_runs"], "count")
    return rows


def _f(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, csv_path):
    print("{} graded frames, ball visible on {}, still on {} (speed < {:g} m/s over {} frames)"
          .format(res["graded_frames"], res["visible_frames"], res["still_frames"],
                  res["still_ms"], res["lag"]))
    for i, (m, r) in enumerate(res["by_min"].items()):
        loc = r["locations"]
        print("{} >= {:g} s: {} runs, {} of graded frames; ".format(
            "main" if i == 0 else "also", m, loc["all"]["n_runs"],
            _f(bg._rate(loc["all"]["frames"], res["graded_frames"]), "{:.3f}")) + ", ".join(
                "{} {}".format(l, loc[l]["n_runs"]) for l in LOCATIONS))
    main = res["by_min"][res["mins"][0]]
    if not res["tags"]:
        print("no Smart Tagging file: link and crosstab skipped")
    else:
        if res["null_skipped"]:
            print("  null skipped: {}".format(res["null_skipped"]))
        link = main["link"]["code:" + tg.SET_PIECE_CODE]
        print("  runs ending within {:g} s of a SET PIECES start, by location (real/n, lift): "
              .format(res["link_window_s"]) + "; ".join(
                  "{} {}/{} {}".format(l, link[l]["real_count"], link[l]["n_gaps"],
                                       _f(link[l].get("lift"), "{:.1f}"))
                  for l in ("all",) + LOCATIONS if link[l]["n_gaps"]))
        tops = []
        for l in LOCATIONS:
            cells = sorted(main["crosstab"][l].items(), key=lambda kv: -kv[1])
            if cells:
                tops.append("{} {} {}/{}".format(l, cells[0][0], cells[0][1],
                                                 main["locations"][l]["n_runs"]))
        print("  most common nearest Type per location: " + ("; ".join(tops) or "-"))
    if main["kickoff"] is None:
        print("kick-off check: no ATD metadata, skipped")
    else:
        print("kick-off check (centre-spot runs with each team in its own half): " + ", ".join(
            "half {} {}/{}".format(h, k["kickoff_runs"], k["centre_runs"])
            for h, k in main["kickoff"].items()))
    print("wrote {}".format(csv_path))


def _floats(text):
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (outside the repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--meta", help="ATD metadata XML (kick-off check); default games.asset_paths()"
                   "['atd_meta'] when --raw is not given")
    p.add_argument("--tags", help="Smart Tagging XML; default games.asset_paths()"
                   "['smart_tagging'] when --raw is not given")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--lag", type=int, default=LAG, help="speed over this many frames (default 5)")
    p.add_argument("--still-ms", type=float, default=STILL_MS,
                   help="still below this speed, m/s (default 0.5)")
    p.add_argument("--min-still-s", type=float, default=MIN_STILL_S,
                   help="main minimum run length, s (default 1)")
    p.add_argument("--min-still-also", type=_floats, default=list(MIN_STILL_ALSO),
                   help="extra minimum lengths, comma-separated s (default 0.5,2)")
    p.add_argument("--still-gap", type=int, default=STILL_GAP,
                   help="bridge up to this many non-still frames inside a run (default 3)")
    p.add_argument("--spot-tol-m", type=float, default=SPOT_TOL_M,
                   help="tolerance around the pitch marks, m (default 1)")
    p.add_argument("--band-m", type=float, default=BAND_M,
                   help="touchline band half-width, m (default 2)")
    p.add_argument("--link-window", type=float, default=LINK_WINDOW_S,
                   help="a set-piece start within this many s of the run end links (default 5)")
    p.add_argument("--null-n", type=int, default=bg.NULL_N, help="shifts (default 1000; 0 = off)")
    p.add_argument("--null-seed", type=int, default=bg.NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=bg.NULL_MIN_SHIFT_S,
                   help="smallest shift in seconds (default 60)")
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/ball_stationary")
    if args.lag < 1:
        p.error("--lag must be at least 1")
    if args.raw:
        if args.frames is None:
            p.error("--frames is required with --raw")
        raw, meta, tags, blocks = args.raw, args.meta, args.tags, args.frames
    else:
        paths = common.game_paths(args.game, ["atd_raw", "atd_meta"],
                                  [] if args.tags else ["smart_tagging"])
        raw, meta = paths["atd_raw"], args.meta or paths["atd_meta"]
        tags = args.tags or paths.get("smart_tagging")
        blocks = args.frames or bg._games().half_video_blocks(args.game)
    res = analyse(raw, blocks, tags, meta, fps=args.fps, pitch=tuple(args.pitch), lag=args.lag,
                  still_ms=args.still_ms, min_still_s=args.min_still_s,
                  min_still_also=args.min_still_also, still_gap=args.still_gap,
                  spot_tol=args.spot_tol_m, band=args.band_m, link_window_s=args.link_window,
                  null_n=args.null_n, null_seed=args.null_seed,
                  null_min_shift_s=args.null_min_shift)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, CSV_NAME)
    bg.write_csv(summary_rows(res), csv_path)
    print_summary(res, csv_path)
    return res


if __name__ == "__main__":
    main()
