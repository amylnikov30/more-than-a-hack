"""Carries: does the ball move with one player, and is that player's team in possession?

A frame is a carry candidate for a player (of either team) when the ball is within
--carry-dist-m of the player and their velocities differ by less than --carry-dv; the carrier
is the nearest candidate. A carry is a run of frames with the same carrier slot lasting at least
--carry-min-s. Reported: carry durations, the share of live time covered, the precision of
"carry team = possessing team", the AUC of the carry signal (+1 home, -1 away) smoothed over
--windows, how close carry-team switches fall to possession changes, and how often a fast ball
(above --fast-ball-ms) passes within --carry-dist-m of a player (the ball has no height, so this
is a proxy for lofted balls passing over a player) against how often it is called a carry.

Usage
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_carries.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    python poss_carries.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_carries_summary.csv (long format). Defaults are football norms, not tuned:
1.5 m is about a stride, 2 m/s separates a dribbled ball from one passing by, 0.5 s is the
shortest controlled touch-and-go, 15 m/s is a firm pass. Definitions and fixture answers:
eda/tests/fixtures/README.md. Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import poss_common as pc

CSV_NAME = "poss_carries_summary.csv"
EXAMPLE = "~/eda_out/poss_carries"
CARRY_DIST_M = 1.5
CARRY_DV = 2.0
CARRY_MIN_S = 0.5
FAST_BALL_MS = 15.0
DURATION_BINS = [(0.5, 1.0, "0.5-1s"), (1.0, 2.0, "1-2s"), (2.0, 5.0, "2-5s"),
                 (5.0, 10.0, "5-10s"), (10.0, math.inf, ">=10s")]
CHUNK = 20000


def carrier_frames(ball, players, team, rid, lag, fps, dist_m, dv):
    """(carrier slot per frame or -1, distance from the ball to the nearest player of a team,
    ball speed), chunked over frames to bound memory."""
    n, k, _ = players.shape
    carrier = np.full(n, -1, dtype=np.int64)
    near = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    for a in range(0, n, CHUNK):
        lo, b = max(a - lag, 0), min(a + CHUNK, n)
        sl = slice(lo, b)
        vb = pc.velocity(ball[sl], rid[sl], lag, fps)
        vp = pc.velocity(players[sl], rid[sl], lag, fps)
        off = a - lo
        bl, pl, tm = ball[a:b], players[a:b].astype(float), team[a:b]
        vb, vp = vb[off:], vp[off:]
        d = np.hypot(pl[..., 0] - bl[:, None, 0], pl[..., 1] - bl[:, None, 1])
        dvel = np.hypot(vp[..., 0] - vb[:, None, 0], vp[..., 1] - vb[:, None, 1])
        member = tm >= 0
        with np.errstate(invalid="ignore"):
            cand = member & (d < dist_m) & (dvel < dv)
        dd = np.where(cand, d, np.inf)
        best = np.argmin(dd, axis=1) if k else np.zeros(b - a, dtype=np.int64)
        has = np.isfinite(dd[np.arange(b - a), best]) if k else np.zeros(b - a, dtype=bool)
        carrier[a:b] = np.where(has, best, -1)
        dm = np.where(member & np.isfinite(d), d, np.inf)
        m = dm.min(axis=1) if k else np.full(b - a, np.inf)
        near[a:b] = np.where(np.isfinite(m), m, np.nan)
        speed[a:b] = np.hypot(vb[:, 0], vb[:, 1])
    return carrier, near, speed


def find_carries(carrier, team, min_frames):
    """[(start, end, slot, team_code)] runs of one carrier slot of at least min_frames."""
    n = len(carrier)
    change = np.flatnonzero(np.diff(carrier) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [n - 1]])
    out = []
    for s, e in zip(starts, ends):
        slot = int(carrier[s])
        if slot < 0 or e - s + 1 < min_frames:
            continue
        t = int(team[s, slot])
        out.append((int(s), int(e), slot, pc.H if t == 0 else pc.A))
    return out


def switches(carries, rid):
    """(frames, teams): first frames of carries whose team differs from the previous carry in
    the same live run."""
    frames, teams = [], []
    for prev, cur in zip(carries[:-1], carries[1:]):
        if cur[3] != prev[3] and rid[cur[0]] == rid[prev[0]] >= 0:
            frames.append(cur[0])
            teams.append(cur[3])
    return np.array(frames, dtype=np.int64), np.array(teams, dtype=np.int64)


def analyse(ctx, dist_m=CARRY_DIST_M, dv=CARRY_DV, min_s=CARRY_MIN_S, fast_ms=FAST_BALL_MS):
    a = ctx["args"]
    live, labels, blocks, n, fps = ctx["live"], ctx["labels"], ctx["blocks"], ctx["n"], a.fps
    rid = pc.run_id(live, blocks, n)
    carrier, near, speed = carrier_frames(ctx["ball_interp"], ctx["players"], ctx["team"], rid,
                                          a.lag, fps, dist_m, dv)
    carrier[~live] = -1
    min_frames = int(math.ceil(min_s * fps - 1e-9))
    carries = find_carries(carrier, ctx["team"], min_frames)
    code = np.zeros(n, dtype=np.int8)
    for s, e, _, t in carries:
        code[s:e + 1] = t
    in_carry = code != pc.NONE
    lab = (labels != pc.NONE) & in_carry
    lengths = np.array([e - s + 1 for s, e, _, _ in carries], dtype=float) / fps
    bins = {label: int(np.sum((lengths >= lo) & (lengths < hi))) for lo, hi, label in
            DURATION_BINS}
    precision = {}
    for scope, sel in (("all", lab), ("H", lab & (code == pc.H)), ("A", lab & (code == pc.A))):
        m = int(sel.sum())
        precision[scope] = (m, float(np.mean(code[sel] == labels[sel])) if m else math.nan)
    score = np.where(code == pc.H, 1.0, np.where(code == pc.A, -1.0, np.nan))
    sep = pc.window_separation(score, labels, live, blocks, a.windows, fps)
    sw, sw_team = switches(carries, rid)
    changes = pc.label_changes(labels, live, blocks)
    errors, near_sw, n_sw = pc.match_changes(changes, labels[changes], sw, sw_team,
                                             int(round(a.timing_window_s * fps)), fps)
    with np.errstate(invalid="ignore"):
        fast = live & (speed > fast_ms)
        near_fast = fast & (near < dist_m)
    n_fast = int(fast.sum())
    return {"carries": carries, "n_live": int(live.sum()), "carry_frames": int(in_carry.sum()),
            "bins": bins, "q": pc.common.percentiles(lengths, [10, 50, 90]),
            "precision": precision, "sep": sep, "n_changes": len(changes), "errors": errors,
            "near_switches": near_sw, "n_switches": n_sw, "n_fast": n_fast,
            "near_rate": near_fast.sum() / float(n_fast) if n_fast else math.nan,
            "carry_rate": np.sum(fast & in_carry) / float(n_fast) if n_fast else math.nan,
            "params": (dist_m, dv, min_s, fast_ms)}


def summary_rows(ctx, res):
    dist_m, dv, min_s, fast_ms = res["params"]
    rows = pc.params_rows(ctx, [("params", "all", "", "carry_dist", dist_m, "m"),
                                ("params", "all", "", "carry_dv", dv, "m/s"),
                                ("params", "all", "", "carry_min", min_s, "s"),
                                ("params", "all", "", "fast_ball", fast_ms, "m/s")])
    rows += pc.label_rows(ctx)
    rows += [("coverage", "all", "", "live_frames", res["n_live"], "frames"),
             ("coverage", "all", "", "n_carries", len(res["carries"]), "count"),
             ("coverage", "all", "", "carry_frames", res["carry_frames"], "frames"),
             ("coverage", "all", "", "share", res["carry_frames"] / float(res["n_live"])
              if res["n_live"] else math.nan, "share of live frames")]
    for _, _, label in DURATION_BINS:
        rows.append(("durations", "all", label, "n_carries", res["bins"][label], "count"))
    for q, v in zip(("p10", "p50", "p90"), res["q"]):
        rows.append(("durations", "length", q, "length", v, "s"))
    for scope in ("all", "H", "A"):
        m, p = res["precision"][scope]
        rows += [("precision", scope, "", "labelled_frames", m, "frames"),
                 ("precision", scope, "", "precision", p, "share of labelled carry frames")]
    rows += pc.window_rows("separation", "carry", res["sep"])
    rows += pc.timing_rows("timing", "carry_switch", res["errors"], res["n_changes"],
                           res["near_switches"], res["n_switches"])
    rows += [("fast_ball", "all", "", "fast_frames", res["n_fast"], "frames"),
             ("fast_ball", "all", "", "near_rate", res["near_rate"], "share of fast frames"),
             ("fast_ball", "all", "", "carry_rate", res["carry_rate"], "share of fast frames")]
    return rows


def print_summary(ctx, res, path):
    pc.print_labels(ctx)
    print("{} carries cover {:.3f} of {} live frames; length p10/p50/p90 {:.2f} / {:.2f} / "
          "{:.2f} s".format(len(res["carries"]), res["carry_frames"] / max(res["n_live"], 1),
                            res["n_live"], *res["q"]))
    m, p = res["precision"]["all"]
    print("carry team = label on {:.3f} of {} labelled carry frames (H {:.3f}, A {:.3f})".format(
        p, m, res["precision"]["H"][1], res["precision"]["A"][1]))
    print("AUC by window (s): " + ", ".join("{:g}: {:.3f} ({:.3f} defined, coverage {:.3f})"
                                          .format(w, *v) for w, v in sorted(res["sep"].items())))
    if res["errors"]:
        print("switch timing: {} of {} changes matched, median error {:.2f} s; {:.3f} of {} "
              "switches near a change".format(
                  len(res["errors"]), res["n_changes"], float(np.median(res["errors"])),
                  res["near_switches"] / max(res["n_switches"], 1), res["n_switches"]))
    print("fast ball (> {:g} m/s) on {} live frames: near a player {:.3f}, in a carry {:.3f}"
          .format(res["params"][3], res["n_fast"], res["near_rate"], res["carry_rate"]))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--carry-dist-m", type=float, default=CARRY_DIST_M,
                   help="ball within this many m of the carrier (default 1.5)")
    p.add_argument("--carry-dv", type=float, default=CARRY_DV,
                   help="ball and carrier velocities differ by less than this, m/s (default 2)")
    p.add_argument("--carry-min-s", type=float, default=CARRY_MIN_S,
                   help="shortest carry, s (default 0.5)")
    p.add_argument("--fast-ball-ms", type=float, default=FAST_BALL_MS,
                   help="ball speed for the false-carry check, m/s (default 15)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p)
    res = analyse(ctx, args.carry_dist_m, args.carry_dv, args.carry_min_s, args.fast_ball_ms)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
