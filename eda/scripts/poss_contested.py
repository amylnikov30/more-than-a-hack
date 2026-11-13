"""Contested ball: does the nearest team to the ball flip back and forth, and can that tell a
turnover from a contest that changes nothing?

The nearest team (team of the nearest player to the interpolated ball) is the organisers'
baseline signal; its AUC per --windows is reported for comparison. A flip is a change of nearest
team between consecutive frames of a live run. Per context, the mean flip rate (flips per
second over --flip-window-s) and the share of contested frames (at least two flips in that
window): around label changes (within --context-s), around loose balls that did not change
possession (CHALLENGE or BALL LOST events with no change within --loose-gap-s; reference game
only), and mid-possession (more than --mid-margin-s from all of these and from live-run edges).
The AUC of the flip density for "around a change" vs "mid-possession" and vs "loose ball without
a change" says whether contested play marks turnovers or could suppress false ones.

Usage
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_contested.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    python poss_contested.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_contested_summary.csv (long format). Windows are norms, not tuned: 2 s is a
challenge and its rebound, 1.5 s either side of a turnover holds the duel that decides it, 5 s
keeps mid-possession clear of both. Definitions and fixture answers:
eda/tests/fixtures/README.md. Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import poss_common as pc

CSV_NAME = "poss_contested_summary.csv"
EXAMPLE = "~/eda_out/poss_contested"
FLIP_WINDOW_S = 2.0
CONTEXT_S = 1.5
LOOSE_GAP_S = 3.0
MID_MARGIN_S = 5.0
LOOSE_KINDS = ("CHALLENGE", "BALL LOST")
CONTEXTS = ("all", "around_change", "loose_no_change", "mid")
CHUNK = 20000


def nearest_team(ball, players, team, live):
    """int8 (n,): 0 / 1 team of the nearest player to the ball, -1 where undefined."""
    n = len(ball)
    out = np.full(n, -1, dtype=np.int8)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        pl = players[a:b].astype(float)
        d = np.hypot(pl[..., 0] - ball[a:b, None, 0], pl[..., 1] - ball[a:b, None, 1])
        d = np.where((team[a:b] >= 0) & np.isfinite(d), d, np.inf)
        j = np.argmin(d, axis=1)
        ok = np.isfinite(d[np.arange(b - a), j]) & live[a:b]
        out[a:b] = np.where(ok, team[a:b][np.arange(b - a), j], -1)
    return out


def flip_indicator(nt, live, blocks):
    """float (n,): 1 on flips, 0 on other frames with a nearest team, NaN elsewhere."""
    n = len(nt)
    rid = pc.run_id(live, blocks, n)
    ind = np.where(nt >= 0, 0.0, np.nan)
    flip = np.zeros(n, dtype=bool)
    flip[1:] = (nt[1:] >= 0) & (nt[:-1] >= 0) & (nt[1:] != nt[:-1]) & (rid[1:] == rid[:-1]) & \
        (rid[1:] >= 0)
    ind[flip] = 1.0
    return ind


def contexts(ctx, changes, context_f, loose_gap_f, margin_f):
    """({context: bool mask}, loose anchors)."""
    live, n, fps = ctx["live"], ctx["n"], ctx["args"].fps
    loose = []
    if ctx["events"] is not None:
        for e in ctx["events"]:
            f = e["start"]
            if e["kind"] in LOOSE_KINDS and 0 <= f < n and live[f] and \
                    not np.any(np.abs(changes - f) <= loose_gap_f):
                loose.append(f)
    loose = np.array(sorted(set(loose)), dtype=np.int64)
    masks = {"all": live.copy(),
             "around_change": pc.window_mask(changes, n, context_f, live),
             "loose_no_change": pc.window_mask(loose, n, context_f, live)}
    mid = live.copy()
    for f in list(changes) + list(loose):
        mid[max(f - margin_f, 0):f + margin_f + 1] = False
    for r in pc.live_runs(live, ctx["blocks"]):
        mid[r["start"]:r["start"] + margin_f + 1] = False
        mid[max(r["end"] - margin_f, 0):r["end"] + 1] = False
    masks["mid"] = mid
    return masks, loose


def analyse(ctx, flip_window_s=FLIP_WINDOW_S, context_s=CONTEXT_S, loose_gap_s=LOOSE_GAP_S,
            mid_margin_s=MID_MARGIN_S):
    a = ctx["args"]
    live, labels, blocks, fps, n = ctx["live"], ctx["labels"], ctx["blocks"], a.fps, ctx["n"]
    nt = nearest_team(ctx["ball_interp"], ctx["players"], ctx["team"], live)
    score = np.where(nt == 0, 1.0, np.where(nt == 1, -1.0, np.nan))
    sep = pc.window_separation(score, labels, live, blocks, a.windows, fps)
    ind = flip_indicator(nt, live, blocks)
    h = int(round(flip_window_s * fps / 2.0))
    density = pc.rolling_mean(ind, live, blocks, h) * fps
    count = pc.rolling_sum(ind, live, blocks, h)
    contested = np.where(np.isfinite(density), (count >= 2).astype(float), np.nan)
    changes = pc.label_changes(labels, live, blocks)
    masks, loose = contexts(ctx, changes, int(round(context_s * fps)),
                            int(round(loose_gap_s * fps)), int(round(mid_margin_s * fps)))
    ctx_stats = {}
    for name in CONTEXTS:
        sel = masks[name] & np.isfinite(density)
        m = int(sel.sum())
        ctx_stats[name] = (m, float(density[sel].mean()) if m else math.nan,
                           float(contested[sel].mean()) if m else math.nan)
    pair_auc = {}
    for key, neg in (("contested_change", "mid"), ("contested_loose", "loose_no_change")):
        pair_auc[key] = {}
        for w in [0.0] + [float(x) for x in a.windows]:
            d = pc.rolling_mean(ind, live, blocks, int(round(w * fps / 2.0)))
            pos_m = masks["around_change"] & np.isfinite(d)
            neg_m = masks[neg] & np.isfinite(d) & ~masks["around_change"]
            sel = pos_m | neg_m
            pair_auc[key][w] = pc.auc(d[sel], pos_m[sel]) if pos_m.any() and neg_m.any() \
                else math.nan
    n_live = int(live.sum())
    return {"sep": sep, "coverage": int(np.sum(nt >= 0)) / float(n_live) if n_live else
            math.nan, "n_flips": int(np.nansum(ind)), "minutes": n_live / float(fps) / 60.0,
            "contexts": ctx_stats, "pair_auc": pair_auc, "n_changes": len(changes),
            "n_loose": len(loose), "nt": nt, "density": density, "contested": contested,
            "masks": masks,
            "params": (flip_window_s, context_s, loose_gap_s, mid_margin_s)}


def summary_rows(ctx, res):
    fw, cs, lg, mm = res["params"]
    rows = pc.params_rows(ctx, [("params", "all", "", "flip_window", fw, "s"),
                                ("params", "all", "", "context", cs, "s"),
                                ("params", "all", "", "loose_gap", lg, "s"),
                                ("params", "all", "", "mid_margin", mm, "s")])
    rows += pc.label_rows(ctx)
    rows += [("coverage", "nearest_team", "", "share", res["coverage"], "share of live frames"),
             ("flips", "all", "", "n", res["n_flips"], "count"),
             ("flips", "all", "", "per_min", res["n_flips"] / res["minutes"]
              if res["minutes"] else math.nan, "per live minute"),
             ("contexts", "anchors", "", "n_changes", res["n_changes"], "count"),
             ("contexts", "anchors", "", "n_loose",
              res["n_loose"] if ctx["events"] is not None else math.nan, "count")]
    for name in CONTEXTS:
        m, rate, share = res["contexts"][name]
        rows += [("contexts", name, "", "frames", m, "frames"),
                 ("contexts", name, "", "flip_rate", rate, "flips per s"),
                 ("contexts", name, "", "contested_share", share, "share of frames")]
    rows += pc.window_rows("separation", "nearest_team", res["sep"])
    for key, by_w in res["pair_auc"].items():
        for w, v in sorted(by_w.items()):
            rows.append(("separation", key, "{:g}".format(w), "auc", v,
                         "AUC (around change vs {})".format(
                             "mid" if key == "contested_change" else "loose, no change")))
    return rows


def print_summary(ctx, res, path):
    pc.print_labels(ctx)
    print("nearest team defined on {:.3f} of live frames; {:.1f} flips per live minute".format(
        res["coverage"], res["n_flips"] / max(res["minutes"], 1e-9)))
    print("nearest-team AUC by window: " + ", ".join(
        "{:g}: {:.3f}".format(w, v[0]) for w, v in sorted(res["sep"].items())))
    for name in CONTEXTS:
        m, rate, share = res["contexts"][name]
        print("  {:<16} {:>7} frames: {:.2f} flips/s, contested {:.3f}".format(name, m, rate,
                                                                           share))
    for key, by_w in res["pair_auc"].items():
        print("{} AUC by window: ".format(key) + ", ".join(
            "{:g}: {:.3f}".format(w, v) for w, v in sorted(by_w.items())))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--flip-window-s", type=float, default=FLIP_WINDOW_S,
                   help="window for the flip rate and the contested test, s (default 2)")
    p.add_argument("--context-s", type=float, default=CONTEXT_S,
                   help="frames within this many s of a change or loose ball (default 1.5)")
    p.add_argument("--loose-gap-s", type=float, default=LOOSE_GAP_S,
                   help="a loose ball has no change within this many s (default 3)")
    p.add_argument("--mid-margin-s", type=float, default=MID_MARGIN_S,
                   help="mid-possession: further than this from changes, loose balls and "
                        "live-run edges, s (default 5)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p)
    res = analyse(ctx, args.flip_window_s, args.context_s, args.loose_gap_s, args.mid_margin_s)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
