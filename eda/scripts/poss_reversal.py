"""Joint reversals: do both teams turn around at a turnover, so the change can be placed
without the ball?

Each team's x velocity (median over its outfield players, home's attacking frame) gives a direction state with
hysteresis (--min-ms); a flip is a change of state. A joint reversal is a home flip with an away
flip to the same direction within --reversal-window-s; its direction names the implied new
possessor. Reported: flips and joint reversals per minute against a circular-shift null (are the
teams reversing together more than chance?), timing of reversals against label changes,
change-detection AUC per window, the AUC of the state "the possessor implied by the last
reversal" per window, and each team's centroid velocity in the gaining team's attacking frame in
1 s bins over +-10 s around label changes, against the changes shifted circularly (random live
anchors).

Usage
    PYTHONPATH=/path/to/team-repo python eda/scripts/poss_reversal.py OUT_DIR \\
        --dead ~/eda_out/dead_ball/dead_ball.csv

    python poss_reversal.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_reversal_summary.csv (long format). Defaults are norms, not tuned: 0.5 m/s
separates a drifting team from a standing one, 1 s is how long two teams take to react to the
same turnover. Definitions and fixture answers: eda/tests/fixtures/README.md. Aggregates only.
"""

import argparse
import math
import os
import warnings

import numpy as np

import poss_common as pc
import poss_shape as ps
import tag_gaps as tg

CSV_NAME = "poss_reversal_summary.csv"
EXAMPLE = "~/eda_out/poss_reversal"
MIN_MS = 0.5
REVERSAL_WINDOW_S = 1.0
CHANGE_WINDOW_S = 1.0
PROFILE_WINDOW_S = 10.0
PROFILE_BIN_S = 1.0


def centroid_velocities(ctx):
    """[home, away] team x velocity in home's attacking frame (m/s): the median x velocity of the
    team's outfield players (poss_shape.team_features), NaN where undefined."""
    d_h = ps.direction_by_frame(ctx["dirs"], ctx["half_of"], 0).astype(float)
    d_h[d_h == 0] = np.nan
    feats, _ = ps.team_features(ctx, hull=False)
    return [feats[t]["vx_median"] * d_h for t in (0, 1)]


def flips(v, live, blocks, min_ms):
    """(frames, new directions) where the hysteresis state changes inside a live run."""
    frames, dirs = [], []
    for r in pc.live_runs(live, blocks):
        seg = v[r["start"]:r["end"] + 1]
        with np.errstate(invalid="ignore"):
            mark = np.where(seg >= min_ms, 1, np.where(seg <= -min_ms, -1, 0))
        idx = np.flatnonzero(mark != 0)
        if len(idx) < 2:
            continue
        m = mark[idx]
        ch = np.flatnonzero(m[1:] != m[:-1]) + 1
        frames.extend((r["start"] + idx[ch]).tolist())
        dirs.extend(m[ch].tolist())
    return np.array(frames, dtype=np.int64), np.array(dirs, dtype=np.int64)


def joint(home, away, window):
    """(frames, directions): for each home flip, the nearest away flip to the same direction
    within the window; the reversal is at the later of the two."""
    hf, hd = home
    af, ad = away
    frames, dirs = [], []
    for d in (1, -1):
        a = np.sort(af[ad == d])
        if not len(a):
            continue
        for f in hf[hd == d]:
            i = np.searchsorted(a, f)
            cand = [a[j] for j in (i - 1, i) if 0 <= j < len(a) and abs(a[j] - f) <= window]
            if cand:
                best = min(cand, key=lambda x: (abs(x - f), -x))
                frames.append(max(int(f), int(best)))
                dirs.append(d)
    order = np.argsort(frames, kind="mergesort")
    return np.array(frames, dtype=np.int64)[order], np.array(dirs, dtype=np.int64)[order]


def count_joint(hf, hd, af, ad, window):
    n = 0
    for d in (1, -1):
        a = np.sort(af[ad == d])
        h = hf[hd == d]
        if not len(a) or not len(h):
            continue
        i = np.searchsorted(a, h)
        lo = np.abs(h - a[np.clip(i - 1, 0, len(a) - 1)]) <= window
        hi = np.abs(a[np.clip(i, 0, len(a) - 1)] - h) <= window
        n += int(np.sum(lo | hi))
    return n


def joint_null(home, away, blocks, offsets, window):
    """Joint counts with the away flips shifted circularly within their half, per shift."""
    af, ad = away
    k = len(next(iter(offsets.values())))
    out = np.zeros(k, dtype=np.int64)
    halves = {h: (a, b) for h, a, b in blocks}
    half_of = np.zeros(len(af), dtype=np.int64)
    for h, (a, b) in halves.items():
        half_of[(af >= a) & (af <= b)] = h
    for i in range(k):
        shifted = af.copy()
        for h, (a, b) in halves.items():
            sel = half_of == h
            shifted[sel] = a + (af[sel] - a + offsets[h][i]) % (b - a + 1)
        out[i] = count_joint(home[0], home[1], shifted, ad, window)
    return out


def change_auc(events, changes, live, blocks, windows_s, fps, change_window_f):
    """{window: AUC} of the smoothed reversal indicator for "a change within the window"."""
    n = len(live)
    ind = np.where(live, 0.0, np.nan)
    ind[events] = 1.0
    pos = pc.window_mask(changes, n, change_window_f, live)
    out = {}
    for w in [0.0] + [float(x) for x in windows_s]:
        s = pc.rolling_mean(ind, live, blocks, int(round(w * fps / 2.0)))
        ok = live & np.isfinite(s)
        out[w] = pc.auc(s[ok], pos[ok])
    return out, int(pos.sum())


def profile(parts, blocks, fps, offsets=None, window_s=PROFILE_WINDOW_S,
            bin_s=PROFILE_BIN_S):
    """Pooled mean per bin over several (signal, anchors) parts: (bins_s, value, counted,
    null share array (K, bins) or None). anchors: [(half, frame)]."""
    window_f, bin_f = int(round(window_s * fps)), int(round(bin_s * fps))
    nb = 2 * window_f // bin_f
    total, counted = np.zeros(nb), np.zeros(nb)
    null_t = null_c = None
    for signal, anchors in parts:
        if not anchors:
            continue
        halves = pc.common.signal_halves(signal, blocks)
        t, c = tg.profile_counts(halves, anchors, window_f, bin_f)
        total += np.asarray(t, dtype=float).reshape(-1)[:nb]
        counted += np.asarray(c, dtype=float).reshape(-1)[:nb]
        if offsets is not None:
            tn, cn = tg.profile_counts(halves, anchors, window_f, bin_f, offsets)
            null_t = tn if null_t is None else null_t + tn
            null_c = cn if null_c is None else null_c + cn
    bins = [float((k - nb // 2) * bin_s) for k in range(nb)]
    null = tg._share(null_t, null_c) if null_t is not None else None
    return bins, tg._share(total, counted), counted, null


def implied_state(rev, rdir, live, blocks, n):
    """+1 / -1 from each joint reversal to the next one (or the end of its live run)."""
    state = np.full(n, np.nan)
    rid = pc.run_id(live, blocks, n)
    ends = {i: r["end"] for i, r in enumerate(pc.live_runs(live, blocks))}
    for i, (f, d) in enumerate(zip(rev, rdir)):
        stop = ends[rid[f]] + 1
        if i + 1 < len(rev) and rid[rev[i + 1]] == rid[f]:
            stop = rev[i + 1]
        state[f:stop] = float(d)
    return state


def analyse(ctx, min_ms=MIN_MS, reversal_window_s=REVERSAL_WINDOW_S,
            change_window_s=CHANGE_WINDOW_S):
    a = ctx["args"]
    live, labels, blocks, fps, n = ctx["live"], ctx["labels"], ctx["blocks"], a.fps, ctx["n"]
    vh, va = centroid_velocities(ctx)
    fh, fa = flips(vh, live, blocks, min_ms), flips(va, live, blocks, min_ms)
    window = int(round(reversal_window_s * fps))
    rev, rdir = joint(fh, fa, window)
    rev_team = np.where(rdir > 0, pc.H, pc.A)
    n_live = int(live.sum())
    offsets, null_reason = pc.common.draw_null(blocks, a.null_n, a.null_seed, a.null_min_shift,
                                               fps)
    real = count_joint(fh[0], fh[1], fa[0], fa[1], window)
    null = joint_null(fh, fa, blocks, offsets, window) if offsets is not None else None

    changes = pc.label_changes(labels, live, blocks)
    gain = labels[changes]
    errors, near, n_ev = pc.match_changes(changes, gain, rev, rev_team,
                                          int(round(a.timing_window_s * fps)), fps)
    cauc, n_pos = change_auc(rev, changes, live, blocks, a.windows, fps,
                             int(round(change_window_s * fps)))
    sep = pc.window_separation(implied_state(rev, rdir, live, blocks, n), labels, live, blocks,
                               a.windows, fps)
    anchors = {code: [(tg.half_of(int(f), blocks), int(f)) for f in changes[gain == code]]
               for code in (pc.H, pc.A)}
    # velocities in the gaining team's attacking frame: home's frame for a home gain, minus it
    # for an away gain
    parts = {"gainer": [(vh, anchors[pc.H]), (-va, anchors[pc.A])],
             "loser": [(va, anchors[pc.H]), (-vh, anchors[pc.A])]}
    profiles = {role: profile(p, blocks, fps, offsets) for role, p in parts.items()}
    return {"n_live": n_live, "minutes": n_live / float(fps) / 60.0,
            "flips": (len(fh[0]), len(fa[0])), "rev": rev, "rev_team": rev_team,
            "real": real, "null": null, "null_reason": null_reason, "errors": errors,
            "near": near, "n_events": n_ev, "n_changes": len(changes), "change_auc": cauc,
            "n_pos": n_pos, "sep": sep, "profiles": profiles,
            "n_anchors": (len(anchors[pc.H]), len(anchors[pc.A])),
            "params": (min_ms, reversal_window_s, change_window_s)}


def summary_rows(ctx, res):
    min_ms, rw, cw = res["params"]
    a = ctx["args"]
    rows = pc.params_rows(ctx, [("params", "all", "", "min_ms", min_ms, "m/s"),
                                ("params", "all", "", "reversal_window", rw, "s"),
                                ("params", "all", "", "change_window", cw, "s"),
                                ("params", "all", "", "profile_window", PROFILE_WINDOW_S, "s"),
                                ("params", "all", "", "profile_bin", PROFILE_BIN_S, "s"),
                                ("params", "null", "", "n", a.null_n, "shifts"),
                                ("params", "null", "", "seed", a.null_seed, ""),
                                ("params", "null", "", "min_shift", a.null_min_shift, "s")])
    rows += pc.label_rows(ctx)
    per_min = lambda x: x / res["minutes"] if res["minutes"] else math.nan  # noqa: E731
    nh, na = res["flips"]
    rows += [("flips", "H", "", "n", nh, "count"), ("flips", "H", "", "per_min", per_min(nh),
                                                     "per live minute"),
             ("flips", "A", "", "n", na, "count"), ("flips", "A", "", "per_min", per_min(na),
                                                     "per live minute"),
             ("reversals", "all", "", "n", len(res["rev"]), "count"),
             ("reversals", "all", "", "per_min", per_min(len(res["rev"])), "per live minute"),
             ("reversals", "all", "", "joint_share_of_home_flips",
              res["real"] / float(nh) if nh else math.nan, "share")]
    null = res["null"]
    if null is not None:
        m = float(null.mean())
        p5, p95 = np.percentile(null, [5, 95])
        rows += [("reversals", "null", "", "null_mean", m, "count"),
                 ("reversals", "null", "", "null_p5", float(p5), "count"),
                 ("reversals", "null", "", "null_p95", float(p95), "count"),
                 ("reversals", "null", "", "p_emp",
                  (1 + int(np.sum(null >= res["real"]))) / float(len(null) + 1), "p"),
                 ("reversals", "null", "", "lift", res["real"] / m if m else math.nan, "ratio")]
    else:
        rows += [("reversals", "null", "", k, math.nan, u) for k, u in
                 (("null_mean", "count"), ("null_p5", "count"), ("null_p95", "count"),
                  ("p_emp", "p"), ("lift", "ratio"))]
    rows += pc.timing_rows("timing", "reversal", res["errors"], res["n_changes"], res["near"],
                           res["n_events"])
    rows.append(("separation", "reversal_change", "", "n_positive", res["n_pos"], "frames"))
    for w, v in sorted(res["change_auc"].items()):
        rows.append(("separation", "reversal_change", "{:g}".format(w), "auc", v,
                     "AUC (change within the change window)"))
    rows += pc.window_rows("separation", "reversal_state", res["sep"])
    nh_a, na_a = res["n_anchors"]
    rows += [("profile", "anchors", "", "n_H", nh_a, "count"),
             ("profile", "anchors", "", "n_A", na_a, "count")]
    for role, (bins, value, counted, null) in sorted(res["profiles"].items()):
        if null is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                nm = np.nanmean(null, axis=0)
                p5, p95 = np.nanpercentile(null, [5, 95], axis=0)
        else:
            nm = p5 = p95 = [math.nan] * len(bins)
        for i, b in enumerate(bins):
            bb = "{:g}".format(b)
            rows += [("profile", role, bb, "value", float(value[i]), "m/s, gainer's frame"),
                     ("profile", role, bb, "counted", int(counted[i]), "frames"),
                     ("profile", role, bb, "null_mean", float(nm[i]), "m/s"),
                     ("profile", role, bb, "null_p5", float(p5[i]), "m/s"),
                     ("profile", role, bb, "null_p95", float(p95[i]), "m/s")]
    return rows


def print_summary(ctx, res, path):
    pc.print_labels(ctx)
    nh, na = res["flips"]
    print("direction flips per live minute: home {:.2f}, away {:.2f}; joint reversals {} "
          "({:.2f} per minute)".format(nh / res["minutes"], na / res["minutes"],
                                       len(res["rev"]), len(res["rev"]) / res["minutes"]))
    if res["null"] is not None:
        print("joint count {} vs shifted null mean {:.1f} (p95 {:.1f})".format(
            res["real"], res["null"].mean(), np.percentile(res["null"], 95)))
    else:
        print("null skipped: " + res["null_reason"])
    if res["errors"]:
        print("timing: {} of {} changes matched, median error {:.2f} s; {:.3f} of {} reversals "
              "near a change".format(len(res["errors"]), res["n_changes"],
                                     float(np.median(res["errors"])),
                                     res["near"] / max(res["n_events"], 1), res["n_events"]))
    print("change-detection AUC by window: " + ", ".join(
        "{:g}: {:.3f}".format(w, v) for w, v in sorted(res["change_auc"].items())))
    print("implied-state AUC by window: " + ", ".join(
        "{:g}: {:.3f} (coverage {:.3f})".format(w, v[0], v[2]) for w, v in
        sorted(res["sep"].items())))
    bins, value, _, _ = res["profiles"]["gainer"]
    print("gainer velocity around changes (-3..+3 s): " + ", ".join(
        "{:g}: {:.2f}".format(b, v) for b, v in zip(bins, value) if -3 <= b <= 2))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, EXAMPLE)
    p.add_argument("--min-ms", type=float, default=MIN_MS,
                   help="centroid speed that sets a direction, m/s (default 0.5)")
    p.add_argument("--reversal-window-s", type=float, default=REVERSAL_WINDOW_S,
                   help="both teams flip within this many s (default 1)")
    p.add_argument("--change-window-s", type=float, default=CHANGE_WINDOW_S,
                   help="change-detection target: a change within this many s (default 1)")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, EXAMPLE)
    ctx = pc.load_context(args, p)
    res = analyse(ctx, args.min_ms, args.reversal_window_s, args.change_window_s)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(ctx, res, path)
    return res


if __name__ == "__main__":
    main()
