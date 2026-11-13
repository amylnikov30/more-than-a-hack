"""Ball position vs ball direction vs team drift: which tells who has the ball?

Three scores toward "home has the ball", in home's attacking frame: (a) ball_position, the
ball's x minus half the pitch (is the ball in the away half?); (b) ball_direction, the sign of the
ball's x velocity (is it moving toward the away goal?); (c) team_drift, +1 / -1 when both teams'
outfield players move (median x velocity, as poss_shape.py) toward the away / home goal (no ball needed). AUC of each at each
--windows rolling mean. If (a) underperforms (b) and (c), ball position is confounded (e.g. by
high pressing, which keeps the ball in the pressed team's half while it has it). Also, per team,
the share of ball-present frames it has the ball by the ball's third of the pitch in its attacking
frame.

Usage
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_ball_field.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    python poss_ball_field.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_ball_field_summary.csv (long format). --min-ms 0.5 (below it a direction is
position jitter or a standing team) is a norm, not tuned. Definitions and fixture answers:
eda/tests/fixtures/README.md. Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import poss_common as pc
import poss_shape as ps

CSV_NAME = "poss_ball_field_summary.csv"
EXAMPLE = "~/eda_out/poss_ball_field"
MIN_MS = 0.5
SIGNALS = ("ball_position", "ball_direction", "team_drift")
THIRDS = ("defensive", "middle", "attacking")


def signals(ctx, min_ms=MIN_MS):
    """({signal: float (n,)}, home direction per frame)."""
    a = ctx["args"]
    live, n = ctx["live"], ctx["n"]
    length = ctx["pitch"][0]
    rid = pc.run_id(live, ctx["blocks"], n)
    d_h = ps.direction_by_frame(ctx["dirs"], ctx["half_of"], 0).astype(float)
    d_h[d_h == 0] = np.nan
    ball = ctx["ball_interp"]
    bx = np.where(live, ball[:, 0], np.nan)
    xg = np.where(d_h > 0, bx, length - bx)
    out = {"ball_position": xg - length / 2.0}
    vx = pc.velocity(ball, rid, a.lag, a.fps)[:, 0] * d_h
    with np.errstate(invalid="ignore"):
        out["ball_direction"] = np.where(np.abs(vx) >= min_ms, np.sign(vx), np.nan)
    feats, _ = ps.team_features(ctx, hull=False)
    v = [feats[t]["vx_median"] * d_h for t in (0, 1)]
    with np.errstate(invalid="ignore"):
        up = (v[0] >= min_ms) & (v[1] >= min_ms)
        down = (v[0] <= -min_ms) & (v[1] <= -min_ms)
    out["team_drift"] = np.where(up, 1.0, np.where(down, -1.0, np.nan))
    for k in out:
        out[k] = np.where(live, out[k], np.nan)
    return out, d_h


def thirds(ctx, d_h):
    """{(team, third): (n labelled ball frames, share where the team has the ball)}."""
    length = ctx["pitch"][0]
    live, labels = ctx["live"], ctx["labels"]
    bx = ctx["ball_interp"][:, 0]
    ok = live & np.isfinite(bx) & (labels != pc.NONE) & np.isfinite(d_h)
    out = {}
    for t, code, sign in ((pc.LABEL_NAMES[pc.H], pc.H, 1.0), (pc.LABEL_NAMES[pc.A], pc.A, -1.0)):
        with np.errstate(invalid="ignore"):
            xg = np.where(d_h * sign > 0, bx, length - bx)
            k = np.clip(np.floor(xg / (length / 3.0)), 0, 2)
        for i, name in enumerate(THIRDS):
            sel = ok & (k == i)
            m = int(sel.sum())
            out[(t, name)] = (m, float(np.mean(labels[sel] == code)) if m else math.nan)
    return out


def analyse(ctx, min_ms=MIN_MS):
    a = ctx["args"]
    sig, d_h = signals(ctx, min_ms)
    n_live = int(ctx["live"].sum())
    res = {"sep": {}, "coverage": {}, "min_ms": min_ms}
    for name in SIGNALS:
        res["sep"][name] = pc.window_separation(sig[name], ctx["labels"], ctx["live"],
                                                ctx["blocks"], a.windows, a.fps)
        res["coverage"][name] = int(np.isfinite(sig[name]).sum()) / float(n_live) \
            if n_live else math.nan
    res["thirds"] = thirds(ctx, d_h)
    return res


def summary_rows(ctx, res):
    rows = pc.params_rows(ctx, [("params", "all", "", "min_ms", res["min_ms"], "m/s")])
    rows += pc.label_rows(ctx)
    for name in SIGNALS:
        rows.append(("coverage", name, "", "share", res["coverage"][name],
                     "share of live frames"))
        rows += pc.window_rows("separation", name, res["sep"][name])
    for (team, third), (m, share) in sorted(res["thirds"].items()):
        scope = "{}:{}".format(team, third)
        rows += [("thirds", scope, "", "n", m, "frames"),
                 ("thirds", scope, "", "share_team_has_ball", share, "share of frames")]
    return rows


def print_summary(ctx, res, path):
    pc.print_labels(ctx)
    for name in SIGNALS:
        sep = res["sep"][name]
        print("{:<15} coverage {:.3f}; AUC by window: {}".format(
            name, res["coverage"][name], ", ".join("{:g}: {:.3f} ({:.3f})".format(w, v[0], v[1])
                                                   for w, v in sorted(sep.items()))))
    print("team has the ball, by ball third in its attacking frame: " + "; ".join(
        "{} {} {:.3f} ({})".format(t, third, share, m) for (t, third), (m, share) in
        sorted(res["thirds"].items())))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--min-ms", type=float, default=MIN_MS,
                   help="slowest ball or centroid speed that counts as a direction, m/s "
                        "(default 0.5)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p)
    res = analyse(ctx, args.min_ms)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
