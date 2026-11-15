"""Check touch detection and the control rule on Metrica Sample Game 3 (open data with events).

Another game, so thresholds may be chosen here; the competition match's public 5 minutes are
not a tuning set (docs/info/RULES.md). Sample Game 3 is in the same EPTS format as the ATD, and
its event data gives the frames to compare against:

  reference touches  PASS start (the kick) and, for a completed pass, PASS end (the reception);
                     SHOT, RECOVERY, BALL LOST and SET PIECE starts. Refs of one team within 2
                     frames of each other count once
  reference labels   per frame, the team of the latest PASS / CARRY / SHOT / RECOVERY / BALL
                     LOST / SET PIECE event; D from a BALL OUT, FAULT RECEIVED or a goal to the
                     next restart (SET PIECE, or a PASS that is a goal kick / kick off ...); X
                     before the kick-off. This is an event-data convention, not the
                     competition's ground truth, so compare rules against each other here, not
                     against the leaderboard

Conditions: the ball as given ("full"), and the ball deleted in bursts until --missing-share of
the graded frames are missing, placed uniformly ("deleted_uniform") or more often where players
crowd the ball ("deleted_crowd", weight (1 + players within 5 m)^2). Burst lengths follow the
competition match's gaps: almost all missing frames sit in gaps of >= 5 s (ball_gaps.py), so a
burst is long (log-normal, median --burst-median-s, clipped to [--burst-min-s, --burst-max-s])
except a share --short-share of short ones (1 to 12 frames).

Per condition:
  detection   precision / recall / F1 of the touches against the reference touches (one-to-one,
              +-MATCH_TOL_S), over a grid of --min-dv-ms and --max-dist-m; recall also on refs
              where the ball is visible; team accuracy, offsets, recall per event kind, and the
              distance to the nearest player for matched (real) and unmatched (noise) touches
  possession  the control rule and the naive rule ("one touch = possession"), with dead ball
              from tracking and from the events: approximate spellF1 (greedy one-to-one IoU
              matching of H / A runs, averaged over IoU 0.3-0.7, X forward-filled), macroF1,
              possession counts, and turnover matching (+-SWITCH_TOL_S, same team)

Usage (open data; any folder outside the repository for OUT_DIR)
    python eda/touch_possession/validate_metrica.py OUT_DIR \\
        --raw .../Sample_Game_3_tracking.txt --meta .../Sample_Game_3_metadata.xml \\
        --events .../Sample_Game_3_events.json

Writes OUT_DIR/touch_validation_summary.csv (long format). Imports touches.py and the EDA
scripts; stdlib + numpy only.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
for _p in (SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ball_gaps as bg  # noqa: E402
import common  # noqa: E402
import touches as tp  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)
FRAMES = "0-69660,69661-143760"
MATCH_TOL_S = 0.4
SWITCH_TOL_S = 1.0
TARGET_MISSING = 0.41
BURST_MEDIAN_S = 15.0
BURST_SIGMA = 0.7
BURST_MIN_S = 5.0
BURST_MAX_S = 90.0
SHORT_SHARE = 0.1
SHORT_MAX_FRAMES = 12
CROWD_RADIUS_M = 5.0
SEED = 2027
CONDITIONS = ("full", "deleted_uniform", "deleted_crowd")
SWEEP_DV = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
SWEEP_DIST = (1.0, 1.5, 2.0, 3.0, math.inf)
SPELL_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
REF_ORDER = ("pass_start", "pass_end", "shot", "recovery", "ball_lost", "set_piece")
POSSESSION_TYPES = ("PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE")
RESTART_SUBTYPES = ("GOAL KICK", "KICK OFF", "THROW IN", "CORNER KICK", "FREE KICK", "PENALTY")
DEAD_START_TYPES = ("BALL OUT", "FAULT RECEIVED")
SUMMARY_NAME = "touch_validation_summary.csv"


# ---------------------------------------------------------------- events

def read_events(path, team_ids, fps=FPS):
    """Metrica JSON events sorted by start: {type, subtypes, team (0 / 1 / -1), start, end
    (video frames = event frame - 1, None if absent), to (a receiver is named), x, y}."""
    with open(path) as handle:
        data = json.load(handle)
    out = []
    for e in data.get("data", []):
        subs = e.get("subtypes")
        names = set()
        if isinstance(subs, dict):
            names.add(subs.get("name"))
        elif isinstance(subs, list):
            names |= {s.get("name") for s in subs if isinstance(s, dict)}
        names.discard(None)
        team = (e.get("team") or {}).get("id")
        start, end = e.get("start") or {}, e.get("end") or {}
        out.append({"type": (e.get("type") or {}).get("name", ""), "subtypes": names,
                    "team": team_ids.index(team) if team in team_ids else -1,
                    "start": start["frame"] - 1 if start.get("frame") is not None else None,
                    "end": end["frame"] - 1 if end.get("frame") is not None else None,
                    "to": e.get("to") is not None, "x": start.get("x"), "y": start.get("y")})
    out = [e for e in out if e["start"] is not None]
    out.sort(key=lambda e: e["start"])
    return out


def reference_touches(events, blocks, dedupe=2):
    """(frames, teams, kinds) arrays of the reference touches on graded frames, sorted. Refs of
    the same team within `dedupe` frames of a kept one are dropped (at equal frames REF_ORDER
    decides which kind is kept)."""
    half_of = {}
    refs = []
    for e in events:
        t, team = e["type"], e["team"]
        if t == "PASS":
            refs.append((e["start"], team, "pass_start"))
            if e["to"] and e["end"] is not None:
                refs.append((e["end"], team, "pass_end"))
        elif t == "SHOT":
            refs.append((e["start"], team, "shot"))
        elif t == "RECOVERY":
            refs.append((e["start"], team, "recovery"))
        elif t == "BALL LOST":
            refs.append((e["start"], team, "ball_lost"))
        elif t == "SET PIECE":
            refs.append((e["start"], team, "set_piece"))
    for h, a, b in blocks:
        half_of[h] = (a, b)
    refs = [r for r in refs if any(a <= r[0] <= b for a, b in half_of.values())]
    refs.sort(key=lambda r: (r[0], REF_ORDER.index(r[2])))
    kept = []
    for f, team, kind in refs:
        if any(abs(f - k[0]) <= dedupe and k[1] == team for k in kept[-6:]):
            continue
        kept.append((f, team, kind))
    return (np.array([k[0] for k in kept], dtype=np.int64),
            np.array([k[1] for k in kept], dtype=np.int64), np.array([k[2] for k in kept]))


def _is_restart(e):
    return e["type"] == "SET PIECE" or bool(e["subtypes"] & set(RESTART_SUBTYPES))


def event_dead(events, blocks):
    """Dead-ball intervals [{half, start, end, length, signals}] from a BALL OUT, FAULT RECEIVED
    or goal to the frame before the next restart of the same half (or the half end)."""
    out = []
    for h, a, b in blocks:
        ev = [e for e in events if a <= e["start"] <= b]
        for i, e in enumerate(ev):
            goal = e["type"] == "SHOT" and "GOAL" in e["subtypes"]
            if e["type"] not in DEAD_START_TYPES and not goal:
                continue
            nxt = next((x["start"] for x in ev[i + 1:] if _is_restart(x) and
                        x["start"] > e["start"]), b + 1)
            s, t = e["start"], nxt - 1
            if out and out[-1]["half"] == h and s <= out[-1]["end"] + 1:
                out[-1]["end"] = max(out[-1]["end"], t)
                out[-1]["length"] = out[-1]["end"] - out[-1]["start"] + 1
            elif t >= s:
                out.append({"half": h, "start": s, "end": t, "length": t - s + 1,
                            "signals": "events"})
    return out


def event_labels(events, n, blocks, dead):
    """Reference labels by frame: the team of the latest possession event (from its start),
    'D' on the event dead-ball intervals, 'X' before a half's first event and off the halves."""
    lab = np.full(n, "X", dtype="<U1")
    for h, a, b in blocks:
        ev = [e for e in events if a <= e["start"] <= b and e["type"] in POSSESSION_TYPES and
              e["team"] >= 0]
        for i, e in enumerate(ev):
            end = ev[i + 1]["start"] if i + 1 < len(ev) else b + 1
            lab[e["start"]:min(end, n)] = tp.LABELS[e["team"]]
    lab[tp.interval_mask(dead, n)] = "D"
    return lab


# ---------------------------------------------------------------- deletion

def crowd_counts(frames, xy, pm, radius, chunk=20000):
    """Players within `radius` m of xy (m) on each frame."""
    out = np.zeros(len(frames), dtype=np.int64)
    for lo in range(0, len(frames), chunk):
        sl = slice(lo, lo + chunk)
        diff = pm[frames[sl]] - xy[sl, None, :]
        d = np.sqrt((diff ** 2).sum(axis=-1))
        out[sl] = (np.where(np.isfinite(d), d, np.inf) <= radius).sum(axis=1)
    return out


def delete_bursts(ball, blocks, target, rng, fps=FPS, median_s=BURST_MEDIAN_S,
                  sigma=BURST_SIGMA, min_s=BURST_MIN_S, max_s=BURST_MAX_S,
                  short_share=SHORT_SHARE, short_max=SHORT_MAX_FRAMES, weights=None):
    """(ball copy with NaN bursts, stats): bursts start on graded ball-visible frames (drawn
    with `weights` if given, uniformly otherwise) and are clipped to their half, until at least
    `target` of the graded frames have no ball."""
    out = ball.copy()
    n = len(out)
    half_of = common.graded_half_of(blocks, n)[:n]
    graded = half_of > 0
    last = {h: b for h, _, b in blocks}
    miss = ~np.isfinite(out).all(axis=1) & graded
    total = int(graded.sum())
    native = int(miss.sum())
    pool = np.flatnonzero(graded & ~miss)
    prob = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)[pool]
        prob = w / w.sum()
    count, lengths = native, []
    while count < target * total and len(pool):
        s = int(rng.choice(pool, p=prob))
        if rng.random() < short_share:
            length = int(rng.integers(1, short_max + 1))
        else:
            sec = float(np.clip(np.exp(rng.normal(math.log(median_s), sigma)), min_s, max_s))
            length = int(round(sec * fps))
        e = min(s + length - 1, last[half_of[s]])
        new = int((~miss[s:e + 1]).sum())
        out[s:e + 1] = np.nan
        miss[s:e + 1] = True
        count += new
        lengths.append(e - s + 1)
    return out, {"native_missing": native / float(total), "missing": count / float(total),
                 "bursts": len(lengths),
                 "burst_median_s": float(np.median(lengths)) / fps if lengths else math.nan}


# ---------------------------------------------------------------- scoring

def match_points(pf, pt, rf, rt, tol, same_team=True):
    """One-to-one pairs {pred index: ref index}, closest first, |pf - rf| <= tol (and same team
    if asked)."""
    order = np.argsort(rf, kind="stable")
    r = np.asarray(rf)[order]
    cands = []
    for i, (f, t) in enumerate(zip(pf, pt)):
        lo, hi = np.searchsorted(r, f - tol, "left"), np.searchsorted(r, f + tol, "right")
        for j in order[lo:hi]:
            if not same_team or rt[j] == t:
                cands.append((abs(int(f) - int(rf[j])), int(f), i, int(j)))
    cands.sort()
    pairs, used = {}, set()
    for _, _, i, j in cands:
        if i not in pairs and j not in used:
            pairs[i] = j
            used.add(j)
    return pairs


def prf(n_match, n_pred, n_ref):
    p = n_match / float(n_pred) if n_pred else math.nan
    r = n_match / float(n_ref) if n_ref else math.nan
    f = 2 * p * r / (p + r) if n_match else (0.0 if n_pred and n_ref else math.nan)
    return p, r, f


def spell_f1(pred, truth, blocks, thresholds=SPELL_THRESHOLDS):
    """Approximate spellF1: H / A runs of the X-filled prediction and of the truth on the scored
    frames (truth in H, A, D), matched one-to-one (same label, greedy by IoU) at each IoU
    threshold; F1 averaged over the thresholds. Returns (score, n_pred, n_true)."""
    n = len(truth)
    scored = np.isin(truth, ["H", "A", "D"]) & (common.graded_half_of(blocks, n)[:n] > 0)
    filled = tp.fill_x(pred[:n], blocks)
    filled[~scored] = "X"
    tr = truth.copy()
    tr[~scored] = "X"
    P, Tr = tp.possession_runs(filled, blocks), tp.possession_runs(tr, blocks)
    pairs = []
    j0 = 0
    for i, (h, s, e, lab) in enumerate(P):
        while j0 < len(Tr) and (Tr[j0][0], Tr[j0][2]) < (h, s):
            j0 += 1
        j = j0
        while j < len(Tr) and Tr[j][0] == h and Tr[j][1] <= e:
            th, ts, te, tl = Tr[j]
            if tl == lab and te >= s:
                inter = min(e, te) - max(s, ts) + 1
                union = max(e, te) - min(s, ts) + 1
                pairs.append((inter / float(union), i, j))
            j += 1
    pairs.sort(reverse=True)
    scores = []
    for thr in thresholds:
        used_p, used_t, tp_ = set(), set(), 0
        for iou, i, j in pairs:
            if iou < thr:
                break
            if i not in used_p and j not in used_t:
                used_p.add(i)
                used_t.add(j)
                tp_ += 1
        scores.append(prf(tp_, len(P), len(Tr))[2])
    return float(np.mean([s if np.isfinite(s) else 0.0 for s in scores])), len(P), len(Tr)


def macro_f1(pred, truth, blocks):
    """Per-frame F1 of H, A, D averaged; an X prediction is a miss."""
    n = len(truth)
    scored = np.isin(truth, ["H", "A", "D"]) & (common.graded_half_of(blocks, n)[:n] > 0)
    pred = pred[:n]
    f1s = []
    for c in "HAD":
        tp_ = int(np.sum((pred == c) & (truth == c) & scored))
        fp = int(np.sum((pred == c) & (truth != c) & scored))
        fn = int(np.sum((pred != c) & (truth == c) & scored))
        f1s.append(prf(tp_, tp_ + fp, tp_ + fn)[2] if tp_ + fn else math.nan)
    return float(np.nanmean(f1s))


def label_turnovers(labels, blocks):
    """(frames, teams): frames where the label goes H -> A or A -> H directly, same half."""
    fr, tm = [], []
    for _, a, b in blocks:
        seg = labels[a:b + 1]
        idx = np.flatnonzero((seg[1:] != seg[:-1]) & np.isin(seg[1:], ["H", "A"]) &
                             np.isin(seg[:-1], ["H", "A"])) + 1
        fr.extend(a + idx)
        tm.extend(tp.LABELS.index(c) for c in seg[idx])
    return np.array(fr, dtype=np.int64), np.array(tm, dtype=np.int64)


# ---------------------------------------------------------------- evaluation

def select(C, thr, dmax, gap_miss, jump_touch):
    dist = np.where(np.isfinite(C["dist"]), C["dist"], np.inf)
    bridged = (C["dv"] >= thr) | (C["miss"] >= gap_miss)
    hit = np.where(C["source"] == 0, C["dv"] >= thr,
                   np.where(C["source"] == 1, bridged, bridged & bool(jump_touch)))
    return hit & ((dist <= dmax) if np.isfinite(dmax) else np.ones(len(dist), dtype=bool))


def detection(res, ref, tol, ball_visible):
    """Detection rows for one condition: the sweep, and details at the default parameters."""
    rows = []
    C, p = res["C"], res["p"]
    rf, rt, rk = ref
    vis_ref = ball_visible[rf]
    for thr in SWEEP_DV:
        for dmax in SWEEP_DIST:
            m = np.flatnonzero(select(C, thr, dmax, p["gap_miss_m"], p["jump_touch"]))
            pairs = match_points(C["frame"][m], np.zeros(len(m)), rf, np.zeros(len(rf)), tol,
                                 same_team=False)
            pr, rc, f1 = prf(len(pairs), len(m), len(rf))
            matched_vis = sum(vis_ref[j] for j in pairs.values())
            key = "dv>={:g}|dist<={:g}".format(thr, dmax)
            rows += [("sweep", key, "", "n_pred", len(m), "touches"),
                     ("sweep", key, "", "precision", pr, "fraction"),
                     ("sweep", key, "", "recall", rc, "fraction"),
                     ("sweep", key, "", "f1", f1, "fraction"),
                     ("sweep", key, "", "recall_ball_visible",
                      matched_vis / float(vis_ref.sum()) if vis_ref.sum() else math.nan,
                      "fraction")]
    # defaults
    m = np.flatnonzero(select(C, p["min_dv_ms"], p["max_dist_m"], p["gap_miss_m"],
                              p["jump_touch"]))
    pairs = match_points(C["frame"][m], np.zeros(len(m)), rf, np.zeros(len(rf)), tol,
                         same_team=False)
    off = [(C["frame"][m[i]] - rf[j]) / float(FPS) for i, j in pairs.items()]
    for q, v in zip((25, 50, 75), common.percentiles(off, (25, 50, 75))):
        rows.append(("default", "offset", "", "p{}".format(q), v, "s"))
    ok = [(i, j) for i, j in pairs.items() if not C["contested"][m[i]]]
    agree = sum(C["team"][m[i]] == rt[j] for i, j in ok)
    rows += [("default", "team", "", "n_uncontested_matched", len(ok), "touches"),
             ("default", "team", "", "agree", agree, "touches"),
             ("default", "team", "", "accuracy", agree / float(len(ok)) if ok else math.nan,
              "fraction"),
             ("default", "contested", "", "share_of_pred",
              float(C["contested"][m].mean()) if len(m) else math.nan, "fraction")]
    hit_ref = np.zeros(len(rf), dtype=bool)
    hit_ref[list(pairs.values())] = True
    for kind in REF_ORDER:
        k = rk == kind
        rows += [("default", "recall_by_kind", kind, "n_ref", int(k.sum()), "refs"),
                 ("default", "recall_by_kind", kind, "recall",
                  float(hit_ref[k].mean()) if k.any() else math.nan, "fraction"),
                 ("default", "recall_by_kind", kind, "recall_ball_visible",
                  float(hit_ref[k & vis_ref].mean()) if (k & vis_ref).any() else math.nan,
                  "fraction")]
    for src, name in ((0, "visible"), (1, "gap"), (2, "jump")):
        s = C["source"][m] == src
        mm = np.zeros(len(m), dtype=bool)
        mm[list(pairs.keys())] = True
        rows.append(("default", "precision_by_source", name, "precision",
                     float(mm[s].mean()) if s.any() else math.nan, "fraction"))
        rows.append(("default", "precision_by_source", name, "n_pred", int(s.sum()), "touches"))
    # distance: every visible peak >= min_dv, matched (real) vs unmatched (noise)
    v = np.flatnonzero((C["source"] == 0) & (C["dv"] >= p["min_dv_ms"]))
    pv = match_points(C["frame"][v], np.zeros(len(v)), rf, np.zeros(len(rf)), tol,
                      same_team=False)
    real = np.zeros(len(v), dtype=bool)
    real[list(pv.keys())] = True
    edges = np.arange(0, tp.DIST_MAX_M + tp.DIST_BIN_M, tp.DIST_BIN_M)
    for name, sel in (("matched", real), ("unmatched", ~real)):
        d = C["dist"][v][sel]
        for a, b, c in zip(edges[:-1], edges[1:], tp.hist(d, edges)):
            label = "[{:g},{:g})".format(a, b) if b < edges[-1] else "[{:g},inf)".format(a)
            rows.append(("distance", name, label, "count", int(c), "touches"))
        for q, x in zip((25, 50, 75, 90), common.percentiles(d[np.isfinite(d)],
                                                              (25, 50, 75, 90))):
            rows.append(("distance", name, "", "p{}".format(q), x, "m"))
        rows.append(("distance", name, "", "n", int(sel.sum()), "touches"))
    return rows


def possession(res, truth, blocks, fps, switch_tol):
    """Possession rows for one pipeline result (control and naive rules)."""
    rows = []
    tf, tt = label_turnovers(truth, blocks)
    for rule, key, lab_key in (("control", "switches", "labels"),
                               ("naive", "switches_naive", "labels_naive")):
        lab = res[lab_key]
        sp, n_p, n_t = spell_f1(lab, truth, blocks)
        rows += [("possession", rule, "", "spellF1_approx", sp, "score"),
                 ("possession", rule, "", "macroF1", macro_f1(lab, truth, blocks), "score"),
                 ("possession", rule, "", "n_pred", n_p, "possessions"),
                 ("possession", rule, "", "n_true", n_t, "possessions")]
        sw = [s for s in res[key] if s["kind"] == "turnover"]
        pf = np.array([s["frame"] for s in sw], dtype=np.int64)
        pt = np.array([s["team"] for s in sw], dtype=np.int64)
        pairs = match_points(pf, pt, tf, tt, int(round(switch_tol * fps)))
        pr, rc, f1 = prf(len(pairs), len(pf), len(tf))
        off = np.array([(pf[i] - tf[j]) for i, j in pairs.items()], dtype=float)
        rows += [("turnovers", rule, "", "n_pred", len(pf), "switches"),
                 ("turnovers", rule, "", "n_true", len(tf), "switches"),
                 ("turnovers", rule, "", "precision", pr, "fraction"),
                 ("turnovers", rule, "", "recall", rc, "fraction"),
                 ("turnovers", rule, "", "f1", f1, "fraction"),
                 ("turnovers", rule, "", "offset_median",
                  float(np.median(off)) / fps if len(off) else math.nan, "s"),
                 ("turnovers", rule, "", "within_3_frames",
                  int((np.abs(off) <= 3).sum()), "switches"),
                 ("turnovers", rule, "", "within_8_frames",
                  int((np.abs(off) <= 8).sum()), "switches")]
    return rows


def alignment(events, ball, pitch, offsets=range(-10, 11)):
    """Median distance (m) between PASS start positions and the tracked ball at start + o."""
    ev = [e for e in events if e["type"] == "PASS" and e["x"] is not None and
          e["y"] is not None]
    f = np.array([e["start"] for e in ev], dtype=np.int64)
    xy = np.array([[e["x"], e["y"]] for e in ev], dtype=float) * np.asarray(pitch)
    out = {}
    for o in offsets:
        g = np.clip(f + o, 0, len(ball) - 1)
        b = ball[g] * np.asarray(pitch)
        d = np.hypot(*(b - xy).T)
        out[o] = float(np.nanmedian(d)) if np.isfinite(d).any() else math.nan
    return out


def run(raw, meta, events_path, blocks, out_dir, fps=FPS, pitch=PITCH, conditions=CONDITIONS,
        target=TARGET_MISSING, seed=SEED, match_tol_s=MATCH_TOL_S, switch_tol_s=SWITCH_TOL_S,
        burst=None, data=None, **params):
    burst = burst or {}
    ball, players, teams = data if data is not None else tp.load_match(raw, meta)
    ids = [tid for tid, _ in tp.team_info(meta)]
    events = read_events(events_path, ids, fps)
    n = len(ball)
    ref = reference_touches(events, blocks)
    dead_ev = event_dead(events, blocks)
    truth = event_labels(events, n, blocks, dead_ev)
    rows = [("params", "all", "", k, "+".join(v) if isinstance(v, tuple) else v, "")
            for k, v in sorted(dict(tp.DEFAULTS, **params).items())]
    rows += [("params", "all", "", "match_tol_s", match_tol_s, "s"),
             ("params", "all", "", "switch_tol_s", switch_tol_s, "s"),
             ("params", "all", "", "target_missing", target, "fraction"),
             ("params", "all", "", "seed", seed, "")]
    rows += [("params", "all", "", k, v, "") for k, v in sorted(burst.items())]
    rows += [("events", "all", "", "n_events", len(events), "events"),
             ("events", "all", "", "n_ref_touches", len(ref[0]), "refs"),
             ("events", "all", "", "n_dead_intervals", len(dead_ev), "intervals"),
             ("events", "all", "", "truth_turnovers", len(label_turnovers(truth, blocks)[0]),
              "switches")]
    for kind in REF_ORDER:
        rows.append(("events", "ref_kind", kind, "n", int((ref[2] == kind).sum()), "refs"))
    for o, d in alignment(events, ball, pitch).items():
        rows.append(("alignment", "pass_start", "{:+d}".format(o), "median_dist", d, "m"))
    graded = common.graded_half_of(blocks, n)[:n] > 0
    for ch in "HADX":
        rows.append(("events", "truth_labels", ch, "share_graded",
                     float(np.mean(truth[graded] == ch)), "fraction"))

    pm = players * np.asarray(pitch, dtype=np.float32)
    for cond in conditions:
        rng = np.random.default_rng(seed)
        b = ball
        stats = {"native_missing": float(np.mean(~np.isfinite(ball[graded]).all(axis=1)))}
        stats["missing"] = stats["native_missing"]
        if cond != "full":
            weights = None
            if cond == "deleted_crowd":
                vis = np.flatnonzero(np.isfinite(ball).all(axis=1))
                cnt = np.zeros(n)
                cnt[vis] = crowd_counts(vis, ball[vis] * np.asarray(pitch), pm, CROWD_RADIUS_M)
                weights = (1.0 + cnt) ** 2
            b, stats = delete_bursts(ball, blocks, target, rng, fps, weights=weights, **burst)
        print("{}: ball missing on {:.1%} of graded frames".format(cond, stats["missing"]))
        for k, v in stats.items():
            rows.append(("condition", cond, "", k, v, ""))
        res = tp.pipeline(b, players, teams, blocks, fps, pitch, **params)
        if cond == "full":
            for w, cells in tp.timescale(res["track"], fps).items():
                for t, v in cells.items():
                    rows.append(("timescale", "w={}".format(w), "dv>={:g}".format(t),
                                 "peaks_per_visible_min", v, "1/min"))
        visible = np.isfinite(b).all(axis=1)
        near_vis = visible.copy()
        for o in (-2, -1, 1, 2):
            near_vis |= np.roll(visible, o)
        rows += [(a, "{}:{}".format(cond, s), bb, m_, v, u) for a, s, bb, m_, v, u in
                 detection(res, ref, int(round(match_tol_s * fps)), near_vis)]
        res_ev = tp.pipeline(b, players, teams, blocks, fps, pitch, dead=dead_ev, **params)
        for name, r in (("tracking_dead", res), ("event_dead", res_ev)):
            pos = possession(r, truth, blocks, fps, switch_tol_s)
            rows += [(a, "{}:{}:{}".format(cond, name, s), bb, m_, v, u)
                     for a, s, bb, m_, v, u in pos]
            sp = {x[1]: x[4] for x in pos if x[0] == "possession" and x[3] == "spellF1_approx"}
            print("  {:13s} spellF1~ control {:.3f}  naive {:.3f}".format(
                name, sp["control"], sp["naive"]))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SUMMARY_NAME)
    bg.write_csv(rows, path)
    print("wrote {}".format(path))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (outside the repo)".format(SUMMARY_NAME))
    p.add_argument("--raw", required=True, help="Sample_Game_3_tracking.txt")
    p.add_argument("--meta", required=True, help="Sample_Game_3_metadata.xml")
    p.add_argument("--events", required=True, help="Sample_Game_3_events.json")
    p.add_argument("--frames", type=bg.parse_frames, default=bg.parse_frames(FRAMES),
                   help="graded video frames per half (default {})".format(FRAMES))
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--conditions", default=",".join(CONDITIONS),
                   help="comma-separated from {}".format(", ".join(CONDITIONS)))
    p.add_argument("--missing-share", type=float, default=TARGET_MISSING)
    p.add_argument("--burst-median-s", type=float, default=BURST_MEDIAN_S)
    p.add_argument("--burst-sigma", type=float, default=BURST_SIGMA)
    p.add_argument("--burst-min-s", type=float, default=BURST_MIN_S)
    p.add_argument("--burst-max-s", type=float, default=BURST_MAX_S)
    p.add_argument("--short-share", type=float, default=SHORT_SHARE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--match-tol-s", type=float, default=MATCH_TOL_S)
    p.add_argument("--switch-tol-s", type=float, default=SWITCH_TOL_S)
    tp.add_param_args(p)
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/touch_validation")
    conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        p.error("unknown condition(s): {}".format(", ".join(bad)))
    burst = {"median_s": args.burst_median_s, "sigma": args.burst_sigma,
             "min_s": args.burst_min_s, "max_s": args.burst_max_s,
             "short_share": args.short_share}
    return run(args.raw, args.meta, args.events, args.frames, args.out_dir, args.fps,
               tuple(args.pitch), conds, args.missing_share, args.seed, args.match_tol_s,
               args.switch_tol_s, burst, **tp.params_from(args))


if __name__ == "__main__":
    main()
