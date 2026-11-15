"""Possession durations on the reference game: the duration prior for a semi-Markov model.

From the true labels of Metrica's open events (poss_reference.py), every live possession (a
maximal run of one team inside a live run) with how it ended: a turnover, a dead ball, or the
end of the half (censored). Reported: length quantiles and bins (all and per team), the share of
possessions ending in a turnover vs a dead ball, a log-normal fit (mean and sd of ln length) and
a discrete hazard per length bin with half-end possessions censored. The match has no true
labels, so this script runs on the reference game only.

Usage
    python poss_durations.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68

Writes OUT_DIR/poss_durations_summary.csv (long format). Definitions and fixture answers:
eda/tests/fixtures/README.md. Aggregates only.
"""

import argparse
import math
import os

import numpy as np

import poss_common as pc

CSV_NAME = "poss_durations_summary.csv"
EXAMPLE = "~/eda_out/poss_durations_sg3"
BINS = [(0.0, 2.0, "0-2s"), (2.0, 5.0, "2-5s"), (5.0, 10.0, "5-10s"), (10.0, 20.0, "10-20s"),
        (20.0, 40.0, "20-40s"), (40.0, math.inf, ">=40s")]
ENDS = ("turnover", "dead", "half_end")
QS = (10, 25, 50, 75, 90)


def possessions_with_ends(ctx):
    """[(length_s, team code, end)] for every live possession."""
    live, labels, fps = ctx["live"], ctx["labels"], ctx["args"].fps
    last = {h: b for h, _, b in ctx["blocks"]}
    out = []
    for s, e, team in pc.possessions(labels, live, ctx["blocks"]):
        half = int(ctx["half_of"][s])
        nxt = e + 1
        if e >= last[half]:
            end = "half_end"
        elif not live[nxt]:
            end = "dead"
        else:
            end = "turnover"
        out.append(((e - s + 1) / float(fps), team, end))
    return out


def stats(lengths):
    q = pc.common.percentiles(lengths, QS)
    return {"n": len(lengths), "q": q,
            "mean": float(np.mean(lengths)) if len(lengths) else math.nan,
            "bins": {label: int(np.sum((np.asarray(lengths) >= lo) & (np.asarray(lengths) < hi)))
                     for lo, hi, label in BINS}}


def analyse(ctx):
    poss = possessions_with_ends(ctx)
    lengths = np.array([p[0] for p in poss])
    res = {"all": stats(lengths),
           "H": stats([p[0] for p in poss if p[1] == pc.H]),
           "A": stats([p[0] for p in poss if p[1] == pc.A])}
    ends = {}
    for end in ENDS:
        sel = [p[0] for p in poss if p[2] == end]
        ends[end] = (len(sel), len(sel) / float(len(poss)) if poss else math.nan,
                     float(np.median(sel)) if sel else math.nan)
    ended = ends["turnover"][0] + ends["dead"][0]
    logs = np.log(lengths) if len(lengths) else np.array([])
    hazard = {}
    for lo, hi, label in BINS:
        at_risk = int(np.sum(lengths >= lo))
        n_end = sum(1 for p in poss if lo <= p[0] < hi and p[2] != "half_end")
        hazard[label] = (at_risk, n_end, n_end / float(at_risk) if at_risk else math.nan)
    res.update({"ends": ends,
                "turnover_share": ends["turnover"][0] / float(ended) if ended else math.nan,
                "log_mean": float(logs.mean()) if len(logs) else math.nan,
                "log_sd": float(logs.std(ddof=1)) if len(logs) > 1 else math.nan,
                "hazard": hazard})
    return res


def summary_rows(ctx, res):
    rows = pc.params_rows(ctx)
    rows += pc.label_rows(ctx)
    for scope in ("all", "H", "A"):
        st = res[scope]
        rows.append(("possessions", scope, "", "n", st["n"], "count"))
        rows += [("possessions", scope, "p{}".format(q), "length", v, "s")
                 for q, v in zip(QS, st["q"])]
        rows.append(("possessions", scope, "mean", "length", st["mean"], "s"))
        rows += [("bins", scope, label, "n", st["bins"][label], "count") for _, _, label in BINS]
    for end in ENDS:
        m, share, med = res["ends"][end]
        rows += [("ends", end, "", "n", m, "count"),
                 ("ends", end, "", "share", share, "share of possessions"),
                 ("ends", end, "p50", "length", med, "s")]
    rows += [("ends", "ended", "", "turnover_share", res["turnover_share"],
              "share of turnover + dead"),
             ("fit", "lognormal", "", "log_mean", res["log_mean"], "ln s"),
             ("fit", "lognormal", "", "log_sd", res["log_sd"], "ln s")]
    for _, _, label in BINS:
        at_risk, n_end, h = res["hazard"][label]
        rows += [("hazard", "all", label, "at_risk", at_risk, "possessions"),
                 ("hazard", "all", label, "ends", n_end, "possessions"),
                 ("hazard", "all", label, "hazard", h, "share of at_risk")]
    return rows


def print_summary(ctx, res, path):
    st = res["all"]
    print("{} possessions; length p10/p50/p90 {:.1f} / {:.1f} / {:.1f} s, mean {:.1f} s".format(
        st["n"], st["q"][0], st["q"][2], st["q"][4], st["mean"]))
    print("home {} (p50 {:.1f} s), away {} (p50 {:.1f} s)".format(
        res["H"]["n"], res["H"]["q"][2], res["A"]["n"], res["A"]["q"][2]))
    print("ends: " + ", ".join("{} {} ({:.3f}, p50 {:.1f} s)".format(e, *res["ends"][e])
                              for e in ENDS) +
          "; turnover share of ended {:.3f}".format(res["turnover_share"]))
    print("log-normal: mean ln(s) {:.3f}, sd {:.3f}".format(res["log_mean"], res["log_sd"]))
    print("hazard per bin: " + ", ".join("{} {:.3f}".format(k, v[2]) for k, v in
                                         res["hazard"].items()))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    if not args.events:
        p.error("--events is required: durations need true labels (reference game only)")
    ctx = pc.load_context(args, p, need_players=False)
    res = analyse(ctx)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
