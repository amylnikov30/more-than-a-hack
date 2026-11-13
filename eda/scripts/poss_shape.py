"""Team shape: does the possessing team spread out, push up and keep fewer players behind the ball?

Per team and live frame, over its outfield players (the deepest player in the team's attacking
frame is taken as the goalkeeper and dropped): width, length, convex-hull area, centroid advance
toward the opponent goal and its velocity, defensive-line depth (deepest outfield player, metres
from the own goal line), and the share of players goal-side of the ball (of the opponent
centroid when the ball is missing). For each feature: possessing minus defending team, and the
AUC of home-minus-away (signed by the hypothesis) for "home has the ball" at each --windows
rolling mean, over all labelled live frames and split by ball present / ball missing. Also each
team's line depth when defending and when attacking, to test the "low block" assumption about
one team instead of assuming it.

Usage
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_shape.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    python poss_shape.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_shape_summary.csv (long format). --min-players 3 (the fewest outfield
players a hull needs) is a norm, not tuned. Definitions and fixture answers:
eda/tests/fixtures/README.md. Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import poss_common as pc

CSV_NAME = "poss_shape_summary.csv"
EXAMPLE = "~/eda_out/poss_shape"
MIN_PLAYERS = 3
# feature -> (sign of the hypothesis "possessing team has the larger value", unit)
FEATURES = {"width": (1, "m"), "length": (1, "m"), "hull_area": (1, "m2"),
            "centroid_advance": (1, "m"), "centroid_velocity": (1, "m/s"),
            "line_depth": (1, "m"), "goal_side": (-1, "share of outfield players")}
SPLITS = ("all", "ball", "no_ball")


def hull_area(xs, ys):
    """Convex hull area of points (monotone chain, shoelace); 0 for fewer than 3 points."""
    pts = sorted(set(zip(xs, ys)))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(sum(hull[i][0] * hull[i - 1][1] - hull[i - 1][0] * hull[i][1]
                         for i in range(len(hull))))


def direction_by_frame(dirs, half_of, t):
    out = np.zeros(len(half_of), dtype=np.int8)
    for (team, half), d in dirs.items():
        if team == t:
            out[half_of == half] = d
    return out


def team_features(ctx, min_players=MIN_PLAYERS):
    """{team: {feature: float (n,)}} plus 'cx' (outfield mean x, absolute) and 'mask'."""
    players, team, live = ctx["players"], ctx["team"], ctx["live"]
    length = ctx["pitch"][0]
    n = ctx["n"]
    rid = pc.run_id(live, ctx["blocks"], n)
    x = players[..., 0].astype(float)
    y = players[..., 1].astype(float)
    out = {}
    for t in (0, 1):
        d = direction_by_frame(ctx["dirs"], ctx["half_of"], t)
        xg = np.where(d[:, None] > 0, x, length - x)
        m = (team == t) & np.isfinite(x) & np.isfinite(y) & live[:, None] & (d[:, None] != 0)
        deep = np.argmin(np.where(m, xg, np.inf), axis=1)
        m[np.arange(n), deep] = False
        cnt = m.sum(axis=1)
        ok = cnt >= min_players
        with np.errstate(invalid="ignore", divide="ignore"):
            f = {"width": np.where(m, y, -np.inf).max(1) - np.where(m, y, np.inf).min(1),
                 "length": np.where(m, xg, -np.inf).max(1) - np.where(m, xg, np.inf).min(1),
                 "centroid_advance": np.where(m, xg, 0).sum(1) / cnt,
                 "line_depth": np.where(m, xg, np.inf).min(1),
                 "cx": np.where(m, x, 0).sum(1) / cnt}
        for key in f:
            f[key] = np.where(ok, f[key], np.nan)
        area = np.full(n, np.nan)
        for i in np.flatnonzero(ok):
            sel = m[i]
            area[i] = hull_area(x[i, sel].tolist(), y[i, sel].tolist())
        f["hull_area"] = area
        f["centroid_velocity"] = pc.velocity(f["centroid_advance"][:, None], rid, ctx["args"].lag,
                                             ctx["args"].fps)[:, 0]
        f.update({"xg": xg, "mask": m, "dir": d, "ok": ok})
        out[t] = f
    ball = ctx["ball_interp"]
    has_ball = np.isfinite(ball).all(axis=1) & live
    for t in (0, 1):
        f, opp = out[t], out[1 - t]
        d = f["dir"]
        ref_abs = np.where(has_ball, ball[:, 0], opp["cx"])
        ref = np.where(d > 0, ref_abs, length - ref_abs)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = (f["mask"] & (f["xg"] < ref[:, None])).sum(1) / f["mask"].sum(1)
        f["goal_side"] = np.where(f["ok"] & np.isfinite(ref), share, np.nan)
    return out, has_ball


def analyse(ctx, min_players=MIN_PLAYERS):
    a = ctx["args"]
    live, labels, blocks, fps = ctx["live"], ctx["labels"], ctx["blocks"], a.fps
    feats, has_ball = team_features(ctx, min_players)
    flag = ctx["flag"]
    split_mask = {"all": live, "ball": live & (flag >= 0) & (flag <= 1), "no_ball": flag == 2}
    lab = (labels != pc.NONE) & live
    n_live = int(live.sum())
    res = {"features": {}, "n_live": n_live, "min_players": min_players}
    for name, (sign, unit) in FEATURES.items():
        fh, fa = feats[0][name], feats[1][name]
        raw = fh - fa
        defined = np.isfinite(raw) & live
        poss_minus_def = np.where(labels == pc.H, raw, -raw)
        diff = {}
        for split in SPLITS:
            sel = lab & defined & split_mask[split]
            v = poss_minus_def[sel]
            diff[split] = (float(v.mean()) if len(v) else math.nan,
                           float(np.median(v)) if len(v) else math.nan, int(len(v)))
        score = sign * raw
        sep = {split: pc.window_separation(score, labels, live, blocks, a.windows, fps,
                                           subset=split_mask[split]) for split in SPLITS}
        res["features"][name] = {"unit": unit, "sign": sign, "diff": diff, "sep": sep,
                                 "coverage": defined.sum() / float(n_live) if n_live else
                                 math.nan}
    depth = {}
    for t, own in ((0, pc.H), (1, pc.A)):
        other = pc.A if own == pc.H else pc.H
        v = feats[t]["line_depth"]
        for state, code in (("defending", other), ("attacking", own)):
            sel = lab & (labels == code) & np.isfinite(v)
            depth[(pc.LABEL_NAMES[own], state)] = (pc.common.percentiles(v[sel], [25, 50, 75]),
                                                   int(sel.sum()))
    res["depth"] = depth
    return res


def summary_rows(ctx, res):
    rows = pc.params_rows(ctx, [("params", "all", "", "min_players", res["min_players"],
                                 "players")])
    rows += pc.label_rows(ctx)
    for name, f in res["features"].items():
        rows += [("coverage", name, "", "share", f["coverage"], "share of live frames"),
                 ("hypothesis", name, "", "sign", f["sign"], "+1 possessing larger")]
        for split in SPLITS:
            mean, median, m = f["diff"][split]
            rows += [("diff", name, split, "mean", mean, f["unit"]),
                     ("diff", name, split, "median", median, f["unit"]),
                     ("diff", name, split, "n", m, "frames")]
        for split in SPLITS:
            rows += pc.window_rows("separation", "{}:{}".format(name, split), f["sep"][split])
    for (team, state), (q, m) in sorted(res["depth"].items()):
        scope = "{}:{}".format(team, state)
        rows.append(("line_depth", scope, "", "n", m, "frames"))
        rows += [("line_depth", scope, b, "depth", v, "m from own goal line")
                 for b, v in zip(("p25", "p50", "p75"), q)]
    return rows


def print_summary(ctx, res, path):
    pc.print_labels(ctx)
    print("feature: possessing - defending median (all / ball / no ball); AUC at 0 s and best "
          "window (all)")
    for name, f in res["features"].items():
        sep = f["sep"]["all"]
        best = max(sep, key=lambda w: (sep[w][0] if not math.isnan(sep[w][0]) else -1))
        print("  {:<18} {:>8.3g} / {:>8.3g} / {:>8.3g} {:<8} AUC {:.3f}, {:.3f} at {:g} s "
              "(no ball {:.3f})".format(name, f["diff"]["all"][1], f["diff"]["ball"][1],
                                        f["diff"]["no_ball"][1], f["unit"], sep[0.0][0],
                                        sep[best][0], best, f["sep"]["no_ball"][best][0]))
    print("line depth p50 (m from own goal): " + ", ".join(
        "{} {} {:.1f}".format(team, state, q[1]) for (team, state), (q, _) in
        sorted(res["depth"].items())))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--min-players", type=int, default=MIN_PLAYERS,
                   help="fewest outfield players for a team's features (default 3)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p)
    res = analyse(ctx, args.min_players)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
