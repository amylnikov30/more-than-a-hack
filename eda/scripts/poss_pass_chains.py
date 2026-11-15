"""Pass chains: does "a team completes two passes in a row" mean that team has the ball?

CHALLENGE.md suggests "a team completes more than two consecutive passes" as a proxy for
controlled possession. Contacts are runs of frames in which the ball moves with one player (the
carry test of poss_carries.py, at least --touch-min-s); a pass links two consecutive contacts of
different players of one team, in one live run, at most --pass-max-s apart, with the ball faster
than --pass-min-ms in between. A chain is a run of contacts linked by passes. Reported: passes
per team, the chain-length distribution, the share of live time in chains, the precision of
"chain of >= --min-chain passes by T => T has the ball" per chain length, its AUC per smoothing
window, and how close chain-team switches fall to possession changes.

Usage
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_pass_chains.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    python poss_pass_chains.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_pass_chains_summary.csv (long format). Defaults are football norms, not
tuned: 5 m/s is a soft pass, 5 s a long ball's flight with a bounce, 0.2 s the shortest touch
the 0.2 s velocity lag can see. One-touch passes are invisible to the contact test (the ball
never moves with the player), so chains are undercounted. Definitions and fixture answers:
eda/tests/fixtures/README.md. Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import poss_carries as pcar
import poss_common as pc

CSV_NAME = "poss_pass_chains_summary.csv"
EXAMPLE = "~/eda_out/poss_pass_chains"
TOUCH_MIN_S = 0.2
PASS_MIN_MS = 5.0
PASS_MAX_S = 5.0
MIN_CHAIN = 2
LENGTH_BINS = ("1", "2", "3", "4", "5+")


def length_bin(k):
    return "5+" if k >= 5 else str(k)


def contacts(carrier, team, rid, min_frames):
    """[[start, end, slot, team_code]] contact runs, same-slot neighbours in one live run
    merged."""
    out = []
    for s, e, slot, t in pcar.find_carries(carrier, team, min_frames):
        if out and out[-1][2] == slot and rid[out[-1][0]] == rid[s]:
            out[-1][1] = e
        else:
            out.append([s, e, slot, t])
    return out


def is_pass(c1, c2, speed, rid, fps, pass_min_ms, pass_max_s):
    if rid[c1[0]] != rid[c2[0]] or c1[3] != c2[3] or c1[2] == c2[2]:
        return False
    if (c2[0] - c1[1]) / float(fps) > pass_max_s:
        return False
    flight = speed[c1[1] + 1:c2[0] + 1]
    flight = flight[np.isfinite(flight)]
    return bool(len(flight)) and float(flight.max()) > pass_min_ms


def chains(cons, speed, rid, fps, pass_min_ms, pass_max_s):
    """[(start, end, team_code, n_passes)] over contacts; chains of 0 passes included."""
    out = []
    for i, c in enumerate(cons):
        if i and is_pass(cons[i - 1], c, speed, rid, fps, pass_min_ms, pass_max_s):
            s, _, t, k = out[-1]
            out[-1] = (s, c[1], t, k + 1)
        else:
            out.append((c[0], c[1], c[3], 0))
    return out


def analyse(ctx, touch_min_s=TOUCH_MIN_S, pass_min_ms=PASS_MIN_MS, pass_max_s=PASS_MAX_S,
            min_chain=MIN_CHAIN, dist_m=pcar.CARRY_DIST_M, dv=pcar.CARRY_DV):
    a = ctx["args"]
    live, labels, blocks, n, fps = ctx["live"], ctx["labels"], ctx["blocks"], ctx["n"], a.fps
    rid = pc.run_id(live, blocks, n)
    carrier, _, speed = pcar.carrier_frames(ctx["ball_interp"], ctx["players"], ctx["team"],
                                            rid, a.lag, fps, dist_m, dv)
    carrier[~live] = -1
    cons = contacts(carrier, ctx["team"], rid, int(math.ceil(touch_min_s * fps - 1e-9)))
    ch = chains(cons, speed, rid, fps, pass_min_ms, pass_max_s)
    n_pass = {t: sum(k for _, _, tt, k in ch if tt == t) for t in (pc.H, pc.A)}
    hist = {b: 0 for b in LENGTH_BINS}
    frames_by_bin = {b: np.zeros(n, dtype=np.int8) for b in LENGTH_BINS}
    qual = np.zeros(n, dtype=np.int8)
    any_chain = np.zeros(n, dtype=bool)
    for s, e, t, k in ch:
        if k < 1:
            continue
        hist[length_bin(k)] += 1
        frames_by_bin[length_bin(k)][s:e + 1] = t
        any_chain[s:e + 1] = True
        if k >= min_chain:
            qual[s:e + 1] = t
    n_live = int(live.sum())

    def precision(code):
        sel = (code != pc.NONE) & (labels != pc.NONE)
        m = int(sel.sum())
        return m, float(np.mean(code[sel] == labels[sel])) if m else math.nan
    prec = {b: precision(frames_by_bin[b]) for b in LENGTH_BINS}
    prec["min_chain"] = precision(qual)
    score = np.where(qual == pc.H, 1.0, np.where(qual == pc.A, -1.0, np.nan))
    sep = pc.window_separation(score, labels, live, blocks, a.windows, fps)
    q_chains = [(s, e, t) for s, e, t, k in ch if k >= min_chain]
    sw, sw_team = [], []
    for prev, cur in zip(q_chains[:-1], q_chains[1:]):
        if cur[2] != prev[2] and rid[cur[0]] == rid[prev[0]] >= 0:
            sw.append(cur[0])
            sw_team.append(cur[2])
    changes = pc.label_changes(labels, live, blocks)
    errors, near, n_sw = pc.match_changes(changes, labels[changes], np.array(sw, dtype=np.int64),
                                          np.array(sw_team, dtype=np.int64),
                                          int(round(a.timing_window_s * fps)), fps)
    return {"n_contacts": len(cons), "n_pass": n_pass, "hist": hist, "n_live": n_live,
            "chain_frames": int(any_chain.sum()), "qual_frames": int(np.sum(qual != pc.NONE)),
            "precision": prec, "sep": sep, "n_changes": len(changes), "errors": errors,
            "near": near, "n_switches": n_sw,
            "params": (touch_min_s, pass_min_ms, pass_max_s, min_chain, dist_m, dv)}


def summary_rows(ctx, res):
    touch, pmin, pmax, min_chain, dist_m, dv = res["params"]
    rows = pc.params_rows(ctx, [("params", "all", "", "touch_min", touch, "s"),
                                ("params", "all", "", "pass_min", pmin, "m/s"),
                                ("params", "all", "", "pass_max", pmax, "s"),
                                ("params", "all", "", "min_chain", min_chain, "passes"),
                                ("params", "all", "", "carry_dist", dist_m, "m"),
                                ("params", "all", "", "carry_dv", dv, "m/s")])
    rows += pc.label_rows(ctx)
    share = lambda x: x / float(res["n_live"]) if res["n_live"] else math.nan  # noqa: E731
    rows += [("passes", "all", "", "n_contacts", res["n_contacts"], "count"),
             ("passes", "H", "", "n_passes", res["n_pass"][pc.H], "count"),
             ("passes", "A", "", "n_passes", res["n_pass"][pc.A], "count")]
    rows += [("chains", "all", b, "n_chains", res["hist"][b], "count") for b in LENGTH_BINS]
    rows += [("coverage", "all", "", "live_frames", res["n_live"], "frames"),
             ("coverage", "chain", "", "frames", res["chain_frames"], "frames"),
             ("coverage", "chain", "", "share", share(res["chain_frames"]),
              "share of live frames"),
             ("coverage", "min_chain", "", "frames", res["qual_frames"], "frames"),
             ("coverage", "min_chain", "", "share", share(res["qual_frames"]),
              "share of live frames")]
    for b in LENGTH_BINS + ("min_chain",):
        m, p = res["precision"][b]
        rows += [("precision", "length", b, "labelled_frames", m, "frames"),
                 ("precision", "length", b, "precision", p, "share of labelled chain frames")]
    rows += pc.window_rows("separation", "chain", res["sep"])
    rows += pc.timing_rows("timing", "chain_switch", res["errors"], res["n_changes"],
                           res["near"], res["n_switches"])
    return rows


def print_summary(ctx, res, path):
    pc.print_labels(ctx)
    print("{} contacts; passes H {} / A {}; chains by passes: {}".format(
        res["n_contacts"], res["n_pass"][pc.H], res["n_pass"][pc.A],
        ", ".join("{} {}".format(b, res["hist"][b]) for b in LENGTH_BINS)))
    m, p = res["precision"]["min_chain"]
    print("chains of >= {} passes cover {:.3f} of live frames; precision {:.3f} on {} labelled "
          "frames".format(res["params"][3], res["qual_frames"] / max(res["n_live"], 1), p, m))
    print("precision by chain length: " + ", ".join(
        "{} {:.3f} ({})".format(b, res["precision"][b][1], res["precision"][b][0])
        for b in LENGTH_BINS))
    print("AUC by window (s): " + ", ".join("{:g}: {:.3f} (coverage {:.3f})".format(w, v[0], v[2])
                                          for w, v in sorted(res["sep"].items())))
    print("chain switches: {} of {} changes matched{}; {} switches".format(
        len(res["errors"]), res["n_changes"],
        ", median error {:.2f} s".format(float(np.median(res["errors"]))) if res["errors"]
        else "", res["n_switches"]))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--carry-dist-m", type=float, default=pcar.CARRY_DIST_M,
                   help="contact: ball within this many m (default 1.5)")
    p.add_argument("--carry-dv", type=float, default=pcar.CARRY_DV,
                   help="contact: velocity difference below this, m/s (default 2)")
    p.add_argument("--touch-min-s", type=float, default=TOUCH_MIN_S,
                   help="shortest contact, s (default 0.2)")
    p.add_argument("--pass-min-ms", type=float, default=PASS_MIN_MS,
                   help="ball faster than this between contacts, m/s (default 5)")
    p.add_argument("--pass-max-s", type=float, default=PASS_MAX_S,
                   help="longest gap between the two contacts of a pass, s (default 5)")
    p.add_argument("--min-chain", type=int, default=MIN_CHAIN,
                   help="passes in a chain for the proxy (default 2)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p)
    res = analyse(ctx, args.touch_min_s, args.pass_min_ms, args.pass_max_s, args.min_chain,
                  args.carry_dist_m, args.carry_dv)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
