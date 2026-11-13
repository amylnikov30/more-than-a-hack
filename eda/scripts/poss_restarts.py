"""Restarts: which team restarts play after a dead ball, and which rule says so?

For each restart (first live frame after a dead stretch), five predictions of the restarting
team: geometric (a player beyond a touchline, else the player nearest the ball when play
resumes), last_opposite (the other team than the one nearest the ball when it was last seen
before the stoppage), label_opposite (the other team than the last label before the stoppage),
tag_team (the Smart Tagging SET PIECES team, when not N/A) and label_after (the first label
after the restart). On the reference game, the accuracy of each against the true restart team;
on both, the pairwise agreement matrix. No per-restart rows.

Usage
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_restarts.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    python poss_restarts.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_restarts_summary.csv (long format). Windows are norms, not tuned: 1 s for
the restart touch, 3 s back for the last touch, 5 s for a tag or label to land, 0.5 m beyond the
line for a throw-in taker. Definitions and fixture answers: eda/tests/fixtures/README.md.
Aggregates only.
"""

import argparse
import itertools
import math
import os

import numpy as np

import poss_common as pc
import poss_contested as pco
import tag_gaps as tg

CSV_NAME = "poss_restarts_summary.csv"
EXAMPLE = "~/eda_out/poss_restarts"
GEOM_LOOK_S = 1.0
LAST_LOOK_S = 3.0
TAG_WINDOW_S = 5.0
LOOK_AFTER_S = 5.0
TOUCHLINE_M = 0.5
RULES = ("geometric", "last_opposite", "label_opposite", "tag_team", "label_after")
OTHER = {pc.H: pc.A, pc.A: pc.H}


def restarts(ctx):
    """[(dead_start, restart_frame)]: dead runs followed by a live frame in the same half."""
    dead = ctx["graded"] & ~ctx["live"]
    last = {h: b for h, _, b in ctx["blocks"]}
    out = []
    for r in pc.common.find_runs(dead, ctx["blocks"]):
        f = r["end"] + 1
        if f <= last[r["half"]] and ctx["live"][f]:
            out.append((r["start"], f))
    return out


def tag_starts(ctx, tags_path):
    """[(start, team code)] SET PIECES tags with a team mapped to home / away."""
    if not tags_path:
        return []
    tags = tg.read_tags(tags_path, ctx["args"].fps)
    names = sorted(set(t["team"] for t in tags if t["team"]))
    team_map = pc.tag_team_map(names, [t[1] for t in ctx["teams"]], ctx["args"].tag_teams)
    return [(t["start"], pc.H if team_map[t["team"]] == 0 else pc.A) for t in tags
            if t["code"] == tg.SET_PIECE_CODE.upper() and t["team"] in team_map]


def predict(ctx, tags_path, geom_look_s=GEOM_LOOK_S, last_look_s=LAST_LOOK_S,
            tag_window_s=TAG_WINDOW_S, look_after_s=LOOK_AFTER_S, touchline_m=TOUCHLINE_M):
    """[{rule: team code or 0, 'truth': code or 0}] per restart."""
    a = ctx["args"]
    fps, live, labels, n = a.fps, ctx["live"], ctx["labels"], ctx["n"]
    width = ctx["pitch"][1]
    nt = pco.nearest_team(ctx["ball_interp"], ctx["players"], ctx["team"], live)
    y = ctx["players"][..., 1]
    with np.errstate(invalid="ignore"):
        beyond = ((y < -touchline_m) | (y > width + touchline_m)) & (ctx["team"] >= 0)
    rid = pc.run_id(live, ctx["blocks"], n)
    tags = tag_starts(ctx, tags_path)
    gl, ll = int(round(geom_look_s * fps)), int(round(last_look_s * fps))
    tw, la = int(round(tag_window_s * fps)), int(round(look_after_s * fps))
    truth_known = ctx["label_source"] == "events"
    out = []
    for s, r in restarts(ctx):
        p = {k: 0 for k in RULES}
        for f in range(r, min(r + gl + 1, n)):
            if not live[f] or rid[f] != rid[r]:
                break
            if beyond[f].any():
                p["geometric"] = pc.H if ctx["team"][f][beyond[f]][0] == 0 else pc.A
                break
            if nt[f] >= 0:
                p["geometric"] = pc.H if nt[f] == 0 else pc.A
                break
        prev = s - 1
        if prev >= 0 and live[prev] and ctx["half_of"][prev] == ctx["half_of"][r]:
            for f in range(prev, max(prev - ll, -1), -1):
                if not live[f] or rid[f] != rid[prev]:
                    break
                if nt[f] >= 0:
                    p["last_opposite"] = OTHER[pc.H if nt[f] == 0 else pc.A]
                    break
            seg = np.flatnonzero((rid == rid[prev]) & (labels != pc.NONE) &
                                 (np.arange(n) <= prev))
            if len(seg):
                p["label_opposite"] = OTHER[int(labels[seg[-1]])]
        near = [(abs(t - r), t, code) for t, code in tags if abs(t - r) <= tw]
        if near:
            p["tag_team"] = min(near)[2]
        after = np.flatnonzero(labels[r:min(r + la + 1, n)] != pc.NONE)
        if len(after):
            p["label_after"] = int(labels[r + after[0]])
        p["truth"] = int(labels[r]) if truth_known else 0
        out.append(p)
    return out


def analyse(ctx, tags_path, **kw):
    preds = predict(ctx, tags_path, **kw)
    n = len(preds)
    cov, acc, agree = {}, {}, {}
    for rule in RULES:
        d = [p for p in preds if p[rule]]
        cov[rule] = len(d)
        t = [p for p in d if p["truth"]]
        acc[rule] = (len(t), float(np.mean([p[rule] == p["truth"] for p in t])) if t else
                     math.nan)
    for r1, r2 in itertools.combinations(RULES, 2):
        both = [p for p in preds if p[r1] and p[r2]]
        agree[(r1, r2)] = (len(both), float(np.mean([p[r1] == p[r2] for p in both])) if both
                           else math.nan)
    truth = [p["truth"] for p in preds if p["truth"]]
    return {"n": n, "coverage": cov, "accuracy": acc, "agreement": agree,
            "truth_home_share": float(np.mean([t == pc.H for t in truth])) if truth else
            math.nan, "params": kw}


def summary_rows(ctx, res):
    k = res["params"]
    rows = pc.params_rows(ctx, [
        ("params", "all", "", "geom_look", k.get("geom_look_s", GEOM_LOOK_S), "s"),
        ("params", "all", "", "last_look", k.get("last_look_s", LAST_LOOK_S), "s"),
        ("params", "all", "", "tag_window", k.get("tag_window_s", TAG_WINDOW_S), "s"),
        ("params", "all", "", "look_after", k.get("look_after_s", LOOK_AFTER_S), "s"),
        ("params", "all", "", "touchline", k.get("touchline_m", TOUCHLINE_M), "m")])
    rows += pc.label_rows(ctx)
    rows += [("restarts", "all", "", "n", res["n"], "count"),
             ("restarts", "truth", "", "home_share", res["truth_home_share"], "share")]
    for rule in RULES:
        m, a = res["accuracy"][rule]
        rows += [("rules", rule, "", "defined", res["coverage"][rule], "restarts"),
                 ("rules", rule, "", "defined_share",
                  res["coverage"][rule] / float(res["n"]) if res["n"] else math.nan, "share"),
                 ("rules", rule, "", "n_truth", m, "restarts"),
                 ("rules", rule, "", "accuracy", a, "share")]
    for (r1, r2), (m, a) in res["agreement"].items():
        scope = "{}|{}".format(r1, r2)
        rows += [("agreement_matrix", scope, "", "n", m, "restarts"),
                 ("agreement_matrix", scope, "", "agree_share", a, "share")]
    return rows


def print_summary(ctx, res, path):
    pc.print_labels(ctx)
    print("{} restarts".format(res["n"]))
    for rule in RULES:
        m, a = res["accuracy"][rule]
        print("  {:<15} defined on {:>4}; accuracy {}".format(
            rule, res["coverage"][rule], "{:.3f} ({})".format(a, m) if m else "n/a"))
    print("agreement (n, share): " + "; ".join(
        "{}|{} {} {:.2f}".format(r1, r2, m, a) for (r1, r2), (m, a) in res["agreement"].items()
        if m))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--geom-look-s", type=float, default=GEOM_LOOK_S,
                   help="geometric rule: look this many s from the restart (default 1)")
    p.add_argument("--last-look-s", type=float, default=LAST_LOOK_S,
                   help="last_opposite: look this many s back for the last ball (default 3)")
    p.add_argument("--tag-window-s", type=float, default=TAG_WINDOW_S,
                   help="tag_team: SET PIECES start within this many s (default 5)")
    p.add_argument("--look-after-s", type=float, default=LOOK_AFTER_S,
                   help="label_after: first label within this many s (default 5)")
    p.add_argument("--touchline-m", type=float, default=TOUCHLINE_M,
                   help="a player this far beyond a touchline is a throw-in taker (default 0.5)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p)
    tags_path = pc.resolve_paths(args, p)[2]
    res = analyse(ctx, tags_path, geom_look_s=args.geom_look_s, last_look_s=args.last_look_s,
                  tag_window_s=args.tag_window_s, look_after_s=args.look_after_s,
                  touchline_m=args.touchline_m)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
