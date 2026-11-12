"""Ball outside the pitch (clause a of the draft dead-ball rule), ball-visible graded frames only.

At each margin M (metres beyond the line; `--margin-m` plus `--margins-also`):

  share      ball-visible graded frames that are out, split into goal lines and touchlines
  runs       maximal out-of-bounds runs: count and length distribution (half-open bins in s)
  follows    what comes right after a run: the ball goes missing, comes back inside, half ends
  link       does a set-piece tag start within +-W s of the run's last frame? Per SET PIECES and
             per Type, by line, with the tag_gaps circular-shift null (lift, p)
  crosstab   line crossed x Type of the nearest set-piece start within W

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/scripts/ball_outside.py OUT_DIR

    # any other game in the raw ATD format (link and crosstab need --tags)
    python ball_outside.py OUT_DIR --raw RAW.txt --frames 0-69660,69661-143760 --pitch 105 68

Writes OUT_DIR/ball_outside_summary.csv (long format: analysis, scope, bin, metric, value,
unit). scope is "m<margin>:<line>" (line: all, goal_line, touchline). Definitions and fixture
answers: eda/tests/fixtures/README.md ("ball_outside.py"). Aggregates only.
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
MARGIN_M = 1.0              # ball-tracking error near the lines is about a metre
MARGINS_ALSO = (0.0, 2.0)
LINK_WINDOW_S = 5.0
LINES = ("all", "goal_line", "touchline")
FOLLOWS = ("missing", "inside", "half_end")
CSV_NAME = "ball_outside_summary.csv"


def excess(ball, pitch=PITCH):
    """(ex, ey): metres beyond the goal lines / touchlines per frame (<= 0 inside; NaN if the
    ball is missing)."""
    x, y = ball[:, 0], ball[:, 1]
    ex = np.maximum(-x, x - 1.0) * pitch[0]
    ey = np.maximum(-y, y - 1.0) * pitch[1]
    return ex, ey


def scope(margin, line):
    return "m{:g}:{}".format(margin, line)


def follow_of(run, visible, blocks):
    last = {h: b for h, _, b in blocks}[run["half"]]
    nxt = run["end"] + 1
    if run["end"] >= last:
        return "half_end"
    if nxt >= len(visible) or not visible[nxt]:
        return "missing"
    return "inside"


def analyse_margin(margin, ball, visible, graded, blocks, ex, ey, sp_groups, offsets, w_frames,
                   fps):
    with np.errstate(invalid="ignore"):
        out = visible & graded & (np.maximum(ex, ey) > margin)
        goal = ex >= ey
    runs = common.find_runs(out, blocks)
    for r in runs:
        r["line"] = "goal_line" if goal[r["start"]] else "touchline"
        r["follow"] = follow_of(r, visible, blocks)
    res = {"visible": int(np.sum(visible & graded)),
           "out_frames": {"all": int(out.sum()), "goal_line": int((out & goal).sum()),
                          "touchline": int((out & ~goal).sum())},
           "runs": {}, "follows": {}, "link": None, "crosstab": None}
    by_line = {line: [j for j, r in enumerate(runs) if line == "all" or r["line"] == line]
               for line in LINES}
    for line, idx in by_line.items():
        lengths = {}
        for _, _, label in bg.BINS_S:
            sel = [runs[j]["length"] for j in idx if bg.bin_label(runs[j]["length"], fps) == label]
            lengths[label] = (len(sel), int(sum(sel)))
        res["runs"][line] = lengths
        res["follows"][line] = {f: sum(runs[j]["follow"] == f for j in idx) for f in FOLLOWS}
        res["follows"][line]["n_runs"] = len(idx)
    if sp_groups is None:
        return res
    points = common.link_points(runs)
    res["link"] = {}
    for name, tags in sp_groups.items():
        anchors, _ = tg.graded_anchors(tags, blocks)
        hits = tg.link_hits(points, anchors, blocks, w_frames)
        matrix = tg.link_null_matrix(points, anchors, blocks, offsets, w_frames) \
            if offsets is not None and points else None
        res["link"][name] = {line: tg.null_stats(hits, matrix, idx)
                             for line, idx in by_line.items()}
    sp = sp_groups.get("code:" + tg.SET_PIECE_CODE, [])
    cross = {line: {} for line in LINES}
    for r in runs:
        t = common.nearest_type(r["end"], r["half"], sp, blocks, w_frames)
        for line in ("all", r["line"]):
            cross[line][t] = cross[line].get(t, 0) + 1
    res["crosstab"] = cross
    return res


def analyse(raw_path, blocks, tags_path=None, fps=FPS, pitch=PITCH, margin_m=MARGIN_M,
            margins_also=MARGINS_ALSO, link_window_s=LINK_WINDOW_S, null_n=bg.NULL_N,
            null_seed=bg.NULL_SEED, null_min_shift_s=bg.NULL_MIN_SHIFT_S):
    """All numbers, as a dict. No per-frame or per-run data leaves this function."""
    ball, _ = bg.read_ball(raw_path)
    visible = np.isfinite(ball).all(axis=1)
    half_of = common.graded_half_of(blocks, len(visible))
    graded = half_of > 0
    ex, ey = excess(ball, pitch)
    w_frames = int(round(link_window_s * fps))
    margins = [margin_m] + [m for m in margins_also if m != margin_m]
    res = {"blocks": list(blocks), "fps": fps, "pitch": tuple(pitch), "margins": margins,
           "link_window_s": link_window_s, "null_n": null_n, "null_seed": null_seed,
           "null_min_shift_s": null_min_shift_s, "tags": tags_path is not None,
           "null_skipped": None, "by_margin": {}}
    sp_groups = offsets = None
    if tags_path:
        sp_groups = common.set_piece_groups(tg.read_tags(tags_path, fps))
        offsets, res["null_skipped"] = common.draw_null(blocks, null_n, null_seed,
                                                        null_min_shift_s, fps)
        res["n_tags"] = {k: len(v) for k, v in sp_groups.items()}
    for m in margins:
        res["by_margin"][m] = analyse_margin(m, ball, visible, graded, blocks, ex, ey, sp_groups,
                                             offsets, w_frames, fps)
    return res


def summary_rows(res):
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    add("params", "all", "", "fps", res["fps"], "frames/s")
    add("params", "all", "", "pitch_length", res["pitch"][0], "m")
    add("params", "all", "", "pitch_width", res["pitch"][1], "m")
    for i, m in enumerate(res["margins"]):
        add("params", "all", "main" if i == 0 else "also", "margin", m, "m")
    add("params", "all", "", "link_window", res["link_window_s"], "s")
    add("params", "all", "", "null_n", res["null_n"], "count")
    add("params", "all", "", "null_seed", res["null_seed"], "")
    add("params", "all", "", "null_min_shift", res["null_min_shift_s"], "s")
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")
    for name, n in res.get("n_tags", {}).items():
        add("tags", name, "", "n_tags", n, "count")
    units = {"n_gaps": "count", "real_count": "count", "p_emp": "p", "lift": "ratio"}
    for m, r in res["by_margin"].items():
        for line in LINES:
            sc = scope(m, line)
            add("share", sc, "", "visible_frames", r["visible"], "frames")
            add("share", sc, "", "out_frames", r["out_frames"][line], "frames")
            add("share", sc, "", "share", bg._rate(r["out_frames"][line], r["visible"]),
                "share of frames")
            for label, (n, frames) in r["runs"][line].items():
                add("runs", sc, label, "n_runs", n, "count")
                add("runs", sc, label, "frames", frames, "frames")
            f = r["follows"][line]
            add("follows", sc, "", "n_runs", f["n_runs"], "count")
            for k in FOLLOWS:
                add("follows", sc, k, "n_runs", f[k], "count")
                add("follows", sc, k, "share", bg._rate(f[k], f["n_runs"]), "share of runs")
            if r["link"] is not None:
                for name, by_line in r["link"].items():
                    st = by_line[line]
                    for metric in ("n_gaps", "real_count", "real_share", "null_mean", "null_p5",
                                   "null_p95", "p_emp", "lift"):
                        if metric in st:
                            add("link", sc, name, "n_runs" if metric == "n_gaps" else metric,
                                st[metric], units.get(metric, "share of runs"))
            if r["crosstab"] is not None:
                for t, n in sorted(r["crosstab"][line].items()):
                    add("crosstab", sc, t, "n_runs", n, "count")
    return rows


def _f(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, csv_path):
    for i, (m, r) in enumerate(res["by_margin"].items()):
        f = r["follows"]["all"]
        print("{} margin {:g} m: out on {} of {} visible graded frames ({}; goal line {}, "
              "touchline {}), {} runs; then missing {} / inside {} / half end {}".format(
                  "main" if i == 0 else "also", m, r["out_frames"]["all"], r["visible"],
                  _f(bg._rate(r["out_frames"]["all"], r["visible"]), "{:.3f}"),
                  r["out_frames"]["goal_line"], r["out_frames"]["touchline"], f["n_runs"],
                  f["missing"], f["inside"], f["half_end"]))
    main = res["by_margin"][res["margins"][0]]
    print("  run lengths at {:g} m: ".format(res["margins"][0]) + ", ".join(
        "{} {}".format(label, main["runs"]["all"][label][0]) for _, _, label in bg.BINS_S))
    if not res["tags"]:
        print("no Smart Tagging file: link and crosstab skipped")
    else:
        if res["null_skipped"]:
            print("  null skipped: {}".format(res["null_skipped"]))
        for line in LINES:
            st = main["link"]["code:" + tg.SET_PIECE_CODE][line]
            print("  {:<9} runs ending within {:g} s of a SET PIECES start: {}/{} ({}), null mean "
                  "{}, lift {}, p {}".format(line, res["link_window_s"], st["real_count"],
                                             st["n_gaps"], _f(st["real_share"]),
                                             _f(st.get("null_mean")), _f(st.get("lift"), "{:.1f}"),
                                             _f(st.get("p_emp"), "{:.3f}")))
        for line in ("goal_line", "touchline"):
            cells = sorted(main["crosstab"][line].items(), key=lambda kv: -kv[1])
            print("  {} -> nearest Type: {}".format(
                line, ", ".join("{} {}".format(t, n) for t, n in cells) or "-"))
    print("wrote {}".format(csv_path))


def _margins(text):
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("margins must be comma-separated metres")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (outside the repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--tags", help="Smart Tagging XML; default games.asset_paths()"
                   "['smart_tagging'] when --raw is not given")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--margin-m", type=float, default=MARGIN_M,
                   help="main margin beyond the line, m (default 1)")
    p.add_argument("--margins-also", type=_margins, default=list(MARGINS_ALSO),
                   help="extra margins for sensitivity, comma-separated m (default 0,2)")
    p.add_argument("--link-window", type=float, default=LINK_WINDOW_S,
                   help="a set-piece start within this many s of the run end links (default 5)")
    p.add_argument("--null-n", type=int, default=bg.NULL_N, help="shifts (default 1000; 0 = off)")
    p.add_argument("--null-seed", type=int, default=bg.NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=bg.NULL_MIN_SHIFT_S,
                   help="smallest shift in seconds (default 60)")
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/ball_outside")
    if args.raw:
        if args.frames is None:
            p.error("--frames is required with --raw")
        raw, tags, blocks = args.raw, args.tags, args.frames
    else:
        paths = common.game_paths(args.game, ["atd_raw"], [] if args.tags else ["smart_tagging"])
        raw, tags = paths["atd_raw"], args.tags or paths.get("smart_tagging")
        blocks = args.frames or bg._games().half_video_blocks(args.game)
    res = analyse(raw, blocks, tags, fps=args.fps, pitch=tuple(args.pitch),
                  margin_m=args.margin_m, margins_also=args.margins_also,
                  link_window_s=args.link_window, null_n=args.null_n, null_seed=args.null_seed,
                  null_min_shift_s=args.null_min_shift)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, CSV_NAME)
    bg.write_csv(summary_rows(res), csv_path)
    print_summary(res, csv_path)
    return res


if __name__ == "__main__":
    main()
