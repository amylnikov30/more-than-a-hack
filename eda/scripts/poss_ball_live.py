"""Ball availability during live play: how much must the ball-free signals carry?

On live frames only (graded, not dead): the share with the ball observed, interpolated or
missing; ball gaps (runs of missing ball inside a live run) per length bin, before and after
linear interpolation of short gaps; gap-length quantiles; how many frames other interpolation
caps would fill; the missing share on dead frames for contrast; and the missing share within
--change-window-s of a possession change against the rest of live play (does the ball vanish
exactly where possession changes?).

Usage
    # competition match: paths from games.asset_paths(), dead frames from src/deadball
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_ball_live.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    # reference game (Metrica Sample Game 3), clean and degraded
    python poss_ball_live.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_ball_live_summary.csv (long format: analysis, scope, bin, metric, value,
unit). Labels: true labels with --events, Smart Tagging pseudo-labels otherwise. Definitions
and fixture answers: eda/tests/fixtures/README.md ("Possession fixture"). Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import ball_gaps as bg
import poss_common as pc

CSV_NAME = "poss_ball_live_summary.csv"
EXAMPLE = "~/eda_out/poss_ball_live"
BINS = [label for _, _, label in bg.BINS_S]


def _caps(text):
    return [int(v) for v in text.split(",") if v.strip()]


def bin_table(gaps, fps):
    table = bg.length_table(gaps, fps)
    return {b: table[b] for b in BINS + ["all"]}


def analyse(ctx, interp_also=pc.INTERP_ALSO, change_window_s=pc.CHANGE_WINDOW_S):
    """All numbers, as a dict."""
    live, blocks, fps = ctx["live"], ctx["blocks"], ctx["args"].fps
    cap = ctx["args"].max_interp_frames
    ball, flag = ctx["ball"], ctx["flag"]
    missing = ~np.isfinite(ball).all(axis=1)
    gaps = pc.live_gaps(missing, live, blocks)

    def filled(g, c):
        return g["length"] <= c and g["left"] and g["right"]
    after = [g for g in gaps if not filled(g, cap)]
    halves = {}
    for half, first, last in blocks:
        sl = slice(first, last + 1)
        halves[half] = {k: int(np.sum(flag[sl] == v)) for k, v in
                        (("observed", 0), ("interpolated", 1), ("missing", 2))}
        halves[half]["live"] = int(live[sl].sum())
    total = {k: sum(h[k] for h in halves.values()) for k in
             ("live", "observed", "interpolated", "missing")}
    lengths = np.array([g["length"] for g in gaps], dtype=float) / fps
    caps = sorted(set([cap] + list(interp_also)))
    dead = ctx["graded"] & ~live

    changes = pc.label_changes(ctx["labels"], live, blocks)
    around = pc.window_mask(changes, ctx["n"], int(round(change_window_s * fps)), live)
    rest = live & ~around
    miss_before = live & missing
    miss_after = flag == 2
    return {
        "halves": halves, "total": total,
        "missing_before": int(miss_before.sum()),
        "before": bin_table(gaps, fps), "after": bin_table(after, fps),
        "q": pc.common.percentiles(lengths, [10, 50, 90]),
        "caps": {c: sum(g["length"] for g in gaps if filled(g, c)) for c in caps},
        "dead": int(dead.sum()), "dead_missing": int(np.sum(dead & missing)),
        "n_changes": len(changes), "change_window_s": change_window_s,
        "around": int(around.sum()), "rest": int(rest.sum()),
        "around_before": int(np.sum(around & miss_before)),
        "around_after": int(np.sum(around & miss_after)),
        "rest_before": int(np.sum(rest & miss_before)),
        "rest_after": int(np.sum(rest & miss_after)),
    }


def _share(a, b):
    return a / float(b) if b else math.nan


def summary_rows(ctx, res, interp_also):
    rows = pc.params_rows(ctx, [
        ("params", "all", "", "change_window", res["change_window_s"], "s"),
        ("params", "all", "", "interp_also", ",".join(str(c) for c in interp_also), "frames")])
    rows += pc.label_rows(ctx)
    t = res["total"]
    for scope, cell in [("all", t)] + [("half {}".format(h), v) for h, v in
                                       sorted(res["halves"].items())]:
        rows.append(("coverage", scope, "", "live_frames", cell["live"], "frames"))
        for key in ("observed", "interpolated", "missing"):
            rows += [("coverage", scope, key, "frames", cell[key], "frames"),
                     ("coverage", scope, key, "share", _share(cell[key], cell["live"]),
                      "share of live frames")]
    rows += [("coverage", "all", "missing_before_interp", "frames", res["missing_before"],
              "frames"),
             ("coverage", "all", "missing_before_interp", "share",
              _share(res["missing_before"], t["live"]), "share of live frames")]
    for stage in ("before", "after"):
        table = res[stage]
        missing_total = table["all"][1]
        for b in BINS + ["all"]:
            n_gaps, frames = table[b]
            rows += [("gaps_" + stage, "all", b, "n_gaps", n_gaps, "count"),
                     ("gaps_" + stage, "all", b, "frames", frames, "frames"),
                     ("gaps_" + stage, "all", b, "share", _share(frames, missing_total),
                      "share of missing live frames")]
    for q, v in zip(("p10", "p50", "p90"), res["q"]):
        rows.append(("gap_length", "all", q, "length", v, "s"))
    for c, frames in sorted(res["caps"].items()):
        rows += [("interp_cap", "all", str(c), "filled_frames", frames, "frames"),
                 ("interp_cap", "all", str(c), "share",
                  _share(frames, res["missing_before"]), "share of missing live frames")]
    rows += [("dead", "all", "", "dead_frames", res["dead"], "frames"),
             ("dead", "all", "", "missing_share", _share(res["dead_missing"], res["dead"]),
              "share of dead frames")]
    rows.append(("change", "all", "", "n_changes", res["n_changes"], "count"))
    no_changes = res["n_changes"] == 0
    for scope in ("around", "rest"):
        frames = res[scope]
        rows.append(("change", scope, "", "frames", math.nan if no_changes else frames,
                     "frames"))
        for stage in ("before", "after"):
            value = math.nan if no_changes else _share(res["{}_{}".format(scope, stage)],
                                                       frames)
            rows.append(("change", scope, stage, "missing_share", value,
                         "share of live frames"))
    return rows


def print_summary(ctx, res, path):
    t = res["total"]
    pc.print_labels(ctx)
    print("live frames {}: ball observed {:.3f}, interpolated {:.3f}, missing {:.3f} "
          "(missing before interpolation {:.3f})".format(
              t["live"], _share(t["observed"], t["live"]),
              _share(t["interpolated"], t["live"]), _share(t["missing"], t["live"]),
              _share(res["missing_before"], t["live"])))
    for stage in ("before", "after"):
        table = res[stage]
        print("gaps {:<6}: ".format(stage) + ", ".join(
            "{} {} ({:.3f})".format(b, table[b][0], _share(table[b][1], table["all"][1]))
            for b in BINS))
    print("gap length p10/p50/p90: {:.2f} / {:.2f} / {:.2f} s".format(*res["q"]))
    print("frames filled per cap: " + ", ".join(
        "{} -> {}".format(c, f) for c, f in sorted(res["caps"].items())))
    print("ball missing on dead frames: {:.3f} of {}".format(
        _share(res["dead_missing"], res["dead"]), res["dead"]))
    if res["n_changes"]:
        print("{} changes; missing within {:g} s: {:.3f} (rest {:.3f}); after interp {:.3f} "
              "(rest {:.3f})".format(
                  res["n_changes"], res["change_window_s"],
                  _share(res["around_before"], res["around"]),
                  _share(res["rest_before"], res["rest"]),
                  _share(res["around_after"], res["around"]),
                  _share(res["rest_after"], res["rest"])))
    else:
        print("no labels: change rows are empty")
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--interp-also", type=_caps, default=list(pc.INTERP_ALSO),
                   help="other interpolation caps to report, frames (default 6,25)")
    p.add_argument("--change-window-s", type=float, default=pc.CHANGE_WINDOW_S,
                   help="'around a change' half-width, s (default 2; a norm, not tuned)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p, need_players=False)
    res = analyse(ctx, args.interp_also, args.change_window_s)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res, args.interp_also), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
