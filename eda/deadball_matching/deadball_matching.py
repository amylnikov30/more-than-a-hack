"""Dead-ball intervals from the EDA signals, matched one-to-one against Smart Tagging set pieces.

Dead ball, per graded frame, is any of the --signals (default all three):

    missing   the ball is missing, in a gap of >= --min-missing-s (ball_gaps.find_gaps)
    outside   the visible ball is beyond a line by more than --margin-m (ball_outside.excess)
    still     the frame is in a ball_stationary still run of >= --min-still-s whose median
              position is one of --still-locations (ball_stationary.classify)

Runs of that mask inside each graded half that are at most --merge-gap-s apart are merged into
one interval, and intervals shorter than --min-dead-s are dropped. Each interval keeps the
signals that flagged at least one of its frames ("missing+outside", ...).

A set piece is a restart, so a correct dead-ball interval ENDS where its SET PIECES tag STARTS;
the two do not overlap. Matching is therefore one-to-one on |interval end - tag start| <=
--link-window-s within the same half, closest pairs first:

    matched       an interval and a tag paired; offset = interval end - tag start (s)
    dead_no_tag   an interval no tag pairs with: a stoppage Smart Tagging did not tag, or a
                  false dead ball
    tag_no_dead   a tag no interval pairs with: a restart the EDA missed, or a false tag

Summary: precision (share of intervals matched) overall, per signal set and per length bin;
recall (share of tags matched) for SET PIECES and per Type; the offset histogram; signal set x
Type; the same one-to-one match against tag starts shifted circularly within each half (the
tag_gaps null). With --gt (a frame,label CSV such as the public ground truth), both the EDA and
Smart Tagging are also scored against the labelled frames: frame-level precision / recall of D,
and restart-level matching against the ends of the labelled D runs.

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/deadball_matching/deadball_matching.py OUT_DIR \\
        [--gt /path/to/BAR-ATH_possession_gt_public.csv]

    # any other game in the raw ATD format
    python deadball_matching.py OUT_DIR --raw RAW.txt --tags TAGS.xml \\
        --frames 0-69660,69661-143760 --pitch 105 68

Writes to OUT_DIR (which must be outside the repository):
    deadball_matching_summary.csv   aggregates only (long format: analysis, scope, bin, metric,
                                    value, unit); safe to bring back
    deadball_matches.csv            FRAME-INDEXED: one row per interval and per unmatched tag.
                                    Stays on the VM; never commit it
    review_sample.csv               FRAME-INDEXED: a seeded sample of each unmatched group with
                                    video seek times and empty category / notes columns to fill
                                    in by watching. Stays on the VM; bring back only
                                    summarise_review.py's counts
stdout carries aggregates only.

Imports the EDA scripts from ../scripts; stdlib + numpy only.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import ball_gaps as bg  # noqa: E402
import ball_outside as bo  # noqa: E402
import ball_stationary as bs  # noqa: E402
import common  # noqa: E402
import sample_long_gaps as slg  # noqa: E402
import tag_gaps as tg  # noqa: E402

FPS = bg.FPS
PITCH = bg.PITCH
SIGNALS = ("missing", "outside", "still")
MIN_MISSING_S = bg.LONG_GAP_S            # 2 s, as ball_gaps (b) and the long-gap work
STILL_LOCATIONS = ("centre_spot", "penalty_spot", "corner_arc", "goal_area", "touchline_band")
MERGE_GAP_S = 1.0
MIN_DEAD_S = 1.0
LINK_WINDOW_S = tg.LINK_WINDOW_S         # 5 s, as tag_gaps, ball_outside, ball_stationary
HIST_BIN_S = 0.5
LENGTH_BINS_S = [(0.0, 2.0, "<2s"), (2.0, 5.0, "2-5s"), (5.0, 10.0, "5-10s"),
                 (10.0, 20.0, "10-20s"), (20.0, 40.0, "20-40s"), (40.0, math.inf, ">=40s")]
GROUPS = ("matched", "dead_no_tag", "tag_no_dead")
NO_TYPE = "(no Type)"
REVIEW_N = 8
REVIEW_SEED = 2027
PAD_S = slg.PAD_S                        # 3 s of context after (and before) an interval
PRE_TAG_S = 10.0                         # build-up shown before an unmatched tag
SUMMARY_NAME = "deadball_matching_summary.csv"
MATCHES_NAME = "deadball_matches.csv"
REVIEW_NAME = "review_sample.csv"
MATCH_COLUMNS = ["group", "half", "start_frame", "end_frame", "length_s", "signals", "edge",
                 "tag_start_frame", "tag_type", "offset_s"]
REVIEW_COLUMNS = ["group", "half", "start_frame", "end_frame", "length_s", "signals",
                  "tag_start_frame", "tag_type", "view_from", "view_to", "category", "notes"]


# ---------------------------------------------------------------- dead-ball intervals

def _frames(seconds, fps):
    return int(math.ceil(seconds * fps - 1e-9))


def signal_masks(ball, blocks, fps=FPS, pitch=PITCH, min_missing_s=MIN_MISSING_S,
                 margin_m=bo.MARGIN_M, lag=bs.LAG, still_ms=bs.STILL_MS,
                 min_still_s=bs.MIN_STILL_S, still_gap=bs.STILL_GAP, spot_tol=bs.SPOT_TOL_M,
                 band=bs.BAND_M, still_locations=STILL_LOCATIONS):
    """{signal: bool array by video frame}, length max(len(ball), last graded frame + 1), True
    only on graded frames."""
    n = max(len(ball), max(b for _, _, b in blocks) + 1)
    if len(ball) < n:
        ball = np.vstack([ball, np.full((n - len(ball), 2), np.nan)])
    visible = np.isfinite(ball).all(axis=1)
    half_of = common.graded_half_of(blocks, n)
    graded = half_of > 0

    missing = np.zeros(n, dtype=bool)
    min_gap = _frames(min_missing_s, fps)
    for g in bg.find_gaps(~visible, blocks):
        if g["length"] >= min_gap:
            missing[g["start"]:g["end"] + 1] = True

    ex, ey = bo.excess(ball, pitch)
    with np.errstate(invalid="ignore"):
        outside = visible & graded & (np.maximum(ex, ey) > margin_m)

    speed = bs.ball_speed(ball, half_of, pitch, fps, lag)
    still_frames = bs.still_mask(speed, visible, lag, still_ms) & graded
    still = np.zeros(n, dtype=bool)
    min_still = _frames(min_still_s, fps)
    for r in bs.merge_runs(common.find_runs(still_frames, blocks), still_gap):
        if r["length"] < min_still:
            continue
        seg = ball[r["start"]:r["end"] + 1]
        seg = seg[np.isfinite(seg).all(axis=1)]
        loc = bs.classify(float(np.median(seg[:, 0])) * pitch[0],
                          float(np.median(seg[:, 1])) * pitch[1], pitch, spot_tol, band)
        if loc in still_locations:
            still[r["start"]:r["end"] + 1] = True
    return {"missing": missing & graded, "outside": outside, "still": still & graded}


def dead_intervals(masks, blocks, merge_gap_frames, min_frames, signals=SIGNALS):
    """[{half, start, end, length, signals, edge}] of the union of `signals`, inclusive frames.

    Runs in the same half at most merge_gap_frames apart are merged (the frames between become
    part of the interval). `signals` names the masks with a True frame in the interval, joined
    by '+' in SIGNALS order; `edge` says the interval touches its half's first or last frame.
    """
    union = np.zeros(len(next(iter(masks.values()))), dtype=bool)
    for s in signals:
        union |= masks[s]
    bounds = {h: (a, b) for h, a, b in blocks}
    out = []
    for r in bs.merge_runs(common.find_runs(union, blocks), merge_gap_frames):
        if r["length"] < min_frames:
            continue
        sl = slice(r["start"], r["end"] + 1)
        first, last = bounds[r["half"]]
        out.append(dict(r, signals="+".join(s for s in SIGNALS if s in signals and
                                            masks[s][sl].any()),
                        edge=r["start"] == first or r["end"] == last))
    return out


def set_piece_tags(tags_path, blocks, fps=FPS):
    """([{half, start, type}], n_dropped): SET PIECES instances (code case-insensitive) whose
    start frame round(s * fps) is graded, in file order; Type '(no Type)' if missing."""
    kept, dropped = [], 0
    for t in tg.read_tags(tags_path, fps):
        if t["code"] != tg.SET_PIECE_CODE:
            continue
        half = tg.half_of(t["start"], blocks)
        if half is None:
            dropped += 1
        else:
            kept.append({"half": half, "start": t["start"], "type": t["type"] or NO_TYPE})
    return kept, dropped


# ---------------------------------------------------------------- one-to-one matching

class Matcher(object):
    """One-to-one pairs between fixed interval ends and (possibly shifted) tag starts."""

    def __init__(self, ends):
        """ends: [(half, frame)], the last frame of each interval."""
        self.by_half = {}
        for i, (h, e) in enumerate(ends):
            self.by_half.setdefault(h, []).append((e, i))
        for h in self.by_half:
            self.by_half[h].sort()
        self.keys = {h: np.array([e for e, _ in v]) for h, v in self.by_half.items()}

    def match(self, starts, w_frames):
        """{interval index: tag index} for starts [(half, frame)].

        Candidates are same-half pairs with |end - start| <= w_frames, taken closest first (ties:
        earlier end, then earlier start); each interval and each tag is used at most once.
        """
        cands = []
        for j, (h, s) in enumerate(starts):
            keys = self.keys.get(h)
            if keys is None:
                continue
            lo = int(np.searchsorted(keys, s - w_frames, "left"))
            hi = int(np.searchsorted(keys, s + w_frames, "right"))
            for e, i in self.by_half[h][lo:hi]:
                cands.append((abs(e - s), e, s, i, j))
        cands.sort()
        pairs, used = {}, set()
        for _, _, _, i, j in cands:
            if i not in pairs and j not in used:
                pairs[i] = j
                used.add(j)
        return pairs


def match(ends, starts, w_frames):
    return Matcher(ends).match(starts, w_frames)


def shift_starts(starts, blocks, offsets, k):
    """Tag starts after circular shift k within each half: first + (s - first + off) % L."""
    bounds = {h: (a, b - a + 1) for h, a, b in blocks}
    out = []
    for h, s in starts:
        first, length = bounds[h]
        out.append((h, first + (s - first + int(offsets[h][k])) % length))
    return out


def null_counts(ends, tags, blocks, offsets, w_frames, types):
    """(total, by_type): matched pairs under each shift (int array (K,)) and matched tags per
    Type ({type: int array (K,)})."""
    k_total = len(next(iter(offsets.values())))
    matcher = Matcher(ends)
    starts = [(t["half"], t["start"]) for t in tags]
    total = np.zeros(k_total, dtype=np.int64)
    by_type = {ty: np.zeros(k_total, dtype=np.int64) for ty in types}
    for k in range(k_total):
        pairs = matcher.match(shift_starts(starts, blocks, offsets, k), w_frames)
        total[k] = len(pairs)
        for j in pairs.values():
            by_type[tags[j]["type"]][k] += 1
    return total, by_type


def null_summary(real, null):
    """Real count against a null of counts: mean, p5, p95, empirical p (null >= real), lift."""
    null = np.asarray(null, dtype=float)
    mean = float(null.mean())
    p5, p95 = np.percentile(null, [5, 95])
    return {"real": float(real), "null_mean": mean, "null_p5": float(p5),
            "null_p95": float(p95),
            "p_emp": (1.0 + float(np.sum(null >= real))) / (1.0 + len(null)),
            "lift": real / mean if mean > 0 else math.nan}


# ---------------------------------------------------------------- ground truth

def read_gt(path):
    """Upper-cased label per video frame from a 'frame,label' CSV (as SUBMISSION.md)."""
    frames, labels = [], []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                frames.append(int(row[0]))
                labels.append(row[1].strip().upper())
    out = np.full(max(frames) + 1 if frames else 0, "X", dtype="<U1")
    out[frames] = labels
    return out


def frame_scores(pred, truth, scored):
    tp = int(np.sum(pred & truth & scored))
    fp = int(np.sum(pred & ~truth & scored))
    fn = int(np.sum(~pred & truth & scored))
    p = tp / float(tp + fp) if tp + fp else math.nan
    r = tp / float(tp + fn) if tp + fn else math.nan
    f1 = 2 * p * r / (p + r) if tp else (0.0 if tp + fp and tp + fn else math.nan)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def restart_scores(pred, anchors, w_frames, fps):
    """One-to-one match of predicted points [(half, frame)] against GT restarts."""
    pairs = match(pred, anchors, w_frames)
    offsets = [(pred[i][1] - anchors[j][1]) / float(fps) for i, j in pairs.items()]
    return {"n_pred": len(pred), "n_gt": len(anchors), "matched": len(pairs),
            "precision": len(pairs) / float(len(pred)) if pred else math.nan,
            "recall": len(pairs) / float(len(anchors)) if anchors else math.nan,
            "median_offset_s": float(np.median(offsets)) if offsets else math.nan}


def gt_analysis(gt, masks, intervals, tags, blocks, w_frames, fps, signals):
    """Scores of the EDA (each signal, their union, the merged intervals) and of Smart Tagging
    against the labelled frames of `gt` (labels H, A, D; X is unscored)."""
    n = max(len(gt), len(next(iter(masks.values()))))
    gt = np.concatenate([gt, np.full(n - len(gt), "X", dtype="<U1")]) if len(gt) < n else gt
    graded = common.graded_half_of(blocks, n) > 0
    scored = np.isin(gt, ["H", "A", "D"]) & graded
    if not scored.any():
        return {"n_scored": 0}
    dead = gt == "D"
    lo, hi = int(np.flatnonzero(scored)[0]), int(np.flatnonzero(scored)[-1])

    def pad(m):
        return np.concatenate([m, np.zeros(n - len(m), dtype=bool)]) if len(m) < n else m[:n]

    span = np.zeros(n, dtype=bool)
    for iv in intervals:
        span[iv["start"]:iv["end"] + 1] = True
    union = np.zeros(n, dtype=bool)
    frames = {}
    for s in signals:
        frames[s] = frame_scores(pad(masks[s]), dead, scored)
        union |= pad(masks[s])
    frames["union"] = frame_scores(union, dead, scored)
    frames["intervals"] = frame_scores(span, dead, scored)

    # GT restarts: the first frame after a maximal labelled D run, when that frame is labelled
    anchors = []
    for r in common.find_runs(dead & scored, blocks):
        nxt = r["end"] + 1
        if nxt < n and scored[nxt]:
            anchors.append((r["half"], nxt))
    eda = [(iv["half"], iv["end"]) for iv in intervals if lo <= iv["end"] <= hi]
    sp = [(t["half"], t["start"]) for t in tags if lo <= t["start"] <= hi]
    return {"n_scored": int(scored.sum()), "n_dead": int((dead & scored).sum()),
            "window": (lo, hi), "frames": frames,
            "restarts": {"eda": restart_scores(eda, anchors, w_frames, fps),
                         "smart_tagging": restart_scores(sp, anchors, w_frames, fps)}}


# ---------------------------------------------------------------- driver

def length_label(frames, fps=FPS):
    s = frames / float(fps)
    for lo, hi, label in LENGTH_BINS_S:
        if lo <= s < hi:
            return label
    raise ValueError(frames)


def hist_edges(link_window_s, bin_s):
    k = int(math.ceil(link_window_s / bin_s - 1e-9))
    return np.arange(-k, k + 1) * bin_s


def analyse(raw_path, blocks, tags_path, gt_path=None, fps=FPS, pitch=PITCH, signals=SIGNALS,
            min_missing_s=MIN_MISSING_S, margin_m=bo.MARGIN_M, lag=bs.LAG,
            still_ms=bs.STILL_MS, min_still_s=bs.MIN_STILL_S, still_gap=bs.STILL_GAP,
            spot_tol=bs.SPOT_TOL_M, band=bs.BAND_M, still_locations=STILL_LOCATIONS,
            merge_gap_s=MERGE_GAP_S, min_dead_s=MIN_DEAD_S, link_window_s=LINK_WINDOW_S,
            hist_bin_s=HIST_BIN_S, null_n=bg.NULL_N, null_seed=bg.NULL_SEED,
            null_min_shift_s=bg.NULL_MIN_SHIFT_S, offsets=None):
    """Intervals, tags, pairs and every summary number, as a dict. `offsets` overrides the drawn
    null (tests). Only main() writes the frame-level parts, and only to OUT_DIR."""
    ball, _ = bg.read_ball(raw_path)
    masks = signal_masks(ball, blocks, fps, pitch, min_missing_s, margin_m, lag, still_ms,
                         min_still_s, still_gap, spot_tol, band, still_locations)
    merge_gap = int(round(merge_gap_s * fps))
    intervals = dead_intervals(masks, blocks, merge_gap, _frames(min_dead_s, fps), signals)
    tags, n_dropped = set_piece_tags(tags_path, blocks, fps)
    w_frames = int(round(link_window_s * fps))
    ends = [(iv["half"], iv["end"]) for iv in intervals]
    pairs = match(ends, [(t["half"], t["start"]) for t in tags], w_frames)
    for i, iv in enumerate(intervals):
        j = pairs.get(i)
        iv["tag"] = j
        iv["offset_s"] = None if j is None else (iv["end"] - tags[j]["start"]) / float(fps)
    matched_tags = set(pairs.values())

    graded_total = sum(b - a + 1 for _, a, b in blocks)
    types = sorted(set(t["type"] for t in tags))
    res = {"blocks": list(blocks), "fps": fps, "signals": list(signals),
           "params": {"min_missing_s": min_missing_s, "margin_m": margin_m,
                      "min_still_s": min_still_s, "still_ms": still_ms,
                      "still_locations": "+".join(still_locations),
                      "merge_gap_s": merge_gap_s, "min_dead_s": min_dead_s,
                      "link_window_s": link_window_s, "null_n": null_n,
                      "null_min_shift_s": null_min_shift_s},
           "graded_frames": graded_total, "intervals": intervals, "tags": tags,
           "n_tags_dropped": n_dropped, "pairs": pairs, "types": types,
           "signal_frames": {s: int(masks[s].sum()) for s in SIGNALS},
           "dead_frames": int(sum(iv["length"] for iv in intervals)),
           "null_skipped": None, "null": None, "gt": None}

    # precision by signal set and length bin
    prec = {"all": [len(intervals), len(pairs)]}
    for iv in intervals:
        for key in ("signals:" + iv["signals"], "length:" + length_label(iv["length"], fps)):
            cell = prec.setdefault(key, [0, 0])
            cell[0] += 1
            cell[1] += iv["tag"] is not None
    res["precision"] = prec
    res["n_edge"] = sum(iv["edge"] for iv in intervals)

    # recall per Type, with offsets
    off_by_type = {ty: [] for ty in types}
    for i, j in pairs.items():
        off_by_type[tags[j]["type"]].append(intervals[i]["offset_s"])
    rec = {"code:" + tg.SET_PIECE_CODE: [len(tags), len(pairs)]}
    for ty in types:
        rec["type:" + ty] = [sum(t["type"] == ty for t in tags), len(off_by_type[ty])]
    res["recall"] = rec
    all_offsets = [o for ty in types for o in off_by_type[ty]]
    res["offsets"] = {"code:" + tg.SET_PIECE_CODE: all_offsets}
    res["offsets"].update({"type:" + ty: off_by_type[ty] for ty in types})
    res["hist_edges"] = hist_edges(link_window_s, hist_bin_s)

    cross = {}
    for i, j in pairs.items():
        key = (intervals[i]["signals"], tags[j]["type"])
        cross[key] = cross.get(key, 0) + 1
    res["crosstab"] = cross

    if offsets is None:
        offsets, res["null_skipped"] = common.draw_null(blocks, null_n, null_seed,
                                                        null_min_shift_s, fps)
    if offsets is not None and tags:
        total, by_type = null_counts(ends, tags, blocks, offsets, w_frames, types)
        res["null"] = {"pairs": null_summary(len(pairs), total)}
        for ty in types:
            res["null"]["type:" + ty] = null_summary(len(off_by_type[ty]), by_type[ty])

    if gt_path:
        res["gt"] = gt_analysis(read_gt(gt_path), masks, intervals, tags, blocks, w_frames, fps,
                                signals)
    res["unmatched_tags"] = [j for j in range(len(tags)) if j not in matched_tags]
    return res


# ---------------------------------------------------------------- frame-level outputs

def match_rows(res):
    """Rows for deadball_matches.csv (FRAME-INDEXED), in frame order."""
    fps, tags, rows = res["fps"], res["tags"], []
    for iv in res["intervals"]:
        j = iv["tag"]
        rows.append(["matched" if j is not None else "dead_no_tag", iv["half"], iv["start"],
                     iv["end"], round(iv["length"] / float(fps), 2), iv["signals"],
                     int(iv["edge"]), "" if j is None else tags[j]["start"],
                     "" if j is None else tags[j]["type"],
                     "" if j is None else round(iv["offset_s"], 2)])
    for j in res["unmatched_tags"]:
        t = tags[j]
        rows.append(["tag_no_dead", t["half"], "", "", "", "", "", t["start"], t["type"], ""])
    rows.sort(key=lambda r: r[2] if r[2] != "" else r[7])
    return rows


def review_sample(res, n=REVIEW_N, seed=REVIEW_SEED, exclude=(), pad_s=PAD_S,
                  pre_tag_s=PRE_TAG_S):
    """Rows for review_sample.csv (FRAME-INDEXED): up to n per unmatched group, spread over the
    halves as evenly as the candidates allow (sample_long_gaps.quotas), seeded. Candidates whose
    span touches an --exclude range are dropped."""
    fps, tags = res["fps"], res["tags"]
    pad, pre = int(round(pad_s * fps)), int(round(pre_tag_s * fps))

    def clear(a, b):
        return not any(lo <= b and a <= hi for lo, hi in exclude)

    cands = {
        "dead_no_tag": [{"half": iv["half"], "start": iv["start"], "end": iv["end"],
                         "row": ["dead_no_tag", iv["half"], iv["start"], iv["end"],
                                 round(iv["length"] / float(fps), 2), iv["signals"], "", "",
                                 slg.ts(max(0, iv["start"] - pad), fps),
                                 slg.ts(iv["end"] + pad, fps), "", ""]}
                        for iv in res["intervals"] if iv["tag"] is None],
        "tag_no_dead": [{"half": tags[j]["half"], "start": tags[j]["start"] - pre,
                         "end": tags[j]["start"] + pad,
                         "row": ["tag_no_dead", tags[j]["half"], "", "", "", "",
                                 tags[j]["start"], tags[j]["type"],
                                 slg.ts(max(0, tags[j]["start"] - pre), fps),
                                 slg.ts(tags[j]["start"] + pad, fps), "", ""]}
                        for j in res["unmatched_tags"]],
    }
    rng = np.random.default_rng(seed)
    rows, counts = [], {}
    for group in ("dead_no_tag", "tag_no_dead"):
        eligible = [c for c in cands[group] if clear(c["start"], c["end"])]
        halves = [h for h, _, _ in res["blocks"]]
        by_half = {h: [c for c in eligible if c["half"] == h] for h in halves}
        q = slg.quotas({h: len(v) for h, v in by_half.items()}, min(n, len(eligible)))
        picked = []
        for h in halves:
            if q[h]:
                idx = rng.choice(len(by_half[h]), size=q[h], replace=False)
                picked.extend(by_half[h][i] for i in sorted(idx))
        picked.sort(key=lambda c: c["start"])
        rows.extend(c["row"] for c in picked)
        counts[group] = (len(cands[group]), len(eligible), len(picked))
    return rows, counts


def write_rows(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


# ---------------------------------------------------------------- aggregate summary

def summary_rows(res):
    fps, rows = res["fps"], []
    for k, v in res["params"].items():
        rows.append(("setup", "params", "", k, v, ""))
    rows.append(("setup", "params", "", "signals", "+".join(res["signals"]), ""))
    g = res["graded_frames"]
    rows.append(("setup", "graded", "", "frames", g, "frames"))
    for s, f in res["signal_frames"].items():
        rows.append(("signals", s, "", "frames", f, "frames"))
        rows.append(("signals", s, "", "share_graded", f / float(g), "fraction"))

    ivs = res["intervals"]
    rows.append(("intervals", "all", "", "n", len(ivs), "intervals"))
    rows.append(("intervals", "all", "", "n_edge", res["n_edge"], "intervals"))
    rows.append(("intervals", "all", "", "frames", res["dead_frames"], "frames"))
    rows.append(("intervals", "all", "", "share_graded", res["dead_frames"] / float(g),
                 "fraction"))
    lengths = [iv["length"] / float(fps) for iv in ivs]
    for q, v in zip((25, 50, 75, 90), common.percentiles(lengths, (25, 50, 75, 90))):
        rows.append(("intervals", "all", "", "length_p{}".format(q), v, "s"))

    for key in sorted(res["precision"], key=lambda k: (k != "all", k)):
        n, m = res["precision"][key]
        rows.append(("precision", key, "", "n_intervals", n, "intervals"))
        rows.append(("precision", key, "", "matched", m, "intervals"))
        rows.append(("precision", key, "", "precision", m / float(n) if n else math.nan,
                     "fraction"))

    rows.append(("recall", "code:" + tg.SET_PIECE_CODE, "", "n_ungraded_dropped",
                 res["n_tags_dropped"], "tags"))
    for key, (n, m) in res["recall"].items():
        rows.append(("recall", key, "", "n_tags", n, "tags"))
        rows.append(("recall", key, "", "matched", m, "tags"))
        rows.append(("recall", key, "", "recall", m / float(n) if n else math.nan, "fraction"))

    edges = res["hist_edges"]
    for key, offs in res["offsets"].items():
        if offs:
            counts, _ = np.histogram(np.clip(offs, edges[0], edges[-1]), bins=edges)
            for a, b, c in zip(edges[:-1], edges[1:], counts):
                rows.append(("offsets", key, "[{:+g},{:+g})".format(a, b), "count", int(c),
                             "pairs"))
        for q, v in zip((25, 50, 75), common.percentiles(offs, (25, 50, 75))):
            rows.append(("offsets", key, "", "p{}".format(q), v, "s"))
        rows.append(("offsets", key, "", "mean", float(np.mean(offs)) if offs else math.nan,
                     "s"))

    for (sig, ty), c in sorted(res["crosstab"].items()):
        rows.append(("crosstab", "signals:" + sig, ty, "matched", c, "pairs"))

    if res["null"]:
        for key, st in res["null"].items():
            for metric, v in st.items():
                rows.append(("null", key, "", metric, v,
                             "" if metric in ("p_emp", "lift") else "pairs"))
    elif res["null_skipped"]:
        rows.append(("null", "skipped", "", res["null_skipped"], "", ""))

    gt = res["gt"]
    if gt:
        rows.append(("gt", "labelled", "", "frames", gt["n_scored"], "frames"))
        if gt["n_scored"]:
            rows.append(("gt", "labelled", "", "dead_frames", gt["n_dead"], "frames"))
            for src, sc in gt["frames"].items():
                for metric, v in sc.items():
                    rows.append(("gt", "frames:" + src, "", metric, v,
                                 "frames" if metric in ("tp", "fp", "fn") else "fraction"))
            for src, sc in gt["restarts"].items():
                for metric, v in sc.items():
                    rows.append(("gt", "restarts:" + src, "", metric, v,
                                 "s" if metric.endswith("_s") else
                                 "fraction" if metric in ("precision", "recall") else "count"))
    return rows


def _f(x, fmt="{:.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, paths, review_counts):
    g, ivs = res["graded_frames"], res["intervals"]
    print("dead-ball signals: {}; graded frames: {}".format("+".join(res["signals"]), g))
    for s, f in res["signal_frames"].items():
        print("  {:8s} {:6.1%} of graded frames".format(s, f / float(g)))
    print("intervals: {} ({:.1%} of graded frames, median {} s, {} touch a half edge)".format(
        len(ivs), res["dead_frames"] / float(g),
        _f(common.percentiles([iv["length"] / float(res["fps"]) for iv in ivs], [50])[0], "{:.1f}"),
        res["n_edge"]))
    n, m = res["precision"]["all"]
    nt, mt = res["recall"]["code:" + tg.SET_PIECE_CODE]
    print("matched pairs: {} (window +-{:g} s)".format(m, res["params"]["link_window_s"]))
    print("  precision {} = {}/{} intervals; recall {} = {}/{} SET PIECES ({} ungraded "
          "dropped)".format(_f(m / float(n) if n else math.nan), m, n,
                            _f(mt / float(nt) if nt else math.nan), mt, nt,
                            res["n_tags_dropped"]))
    for key in sorted(k for k in res["precision"] if k.startswith("signals:")):
        a, b = res["precision"][key]
        print("    precision {:28s} {:>5}/{:<5} {}".format(key, b, a, _f(b / float(a))))
    for key, (a, b) in res["recall"].items():
        if key.startswith("type:"):
            offs = res["offsets"][key]
            print("    recall {:31s} {:>5}/{:<5} {}  median offset {} s".format(
                key, b, a, _f(b / float(a) if a else math.nan),
                _f(float(np.median(offs)) if offs else math.nan)))
    if res["null"]:
        st = res["null"]["pairs"]
        print("  null: {} pairs vs shifted tags {} [{}, {}], lift {}, p {}".format(
            m, _f(st["null_mean"], "{:.1f}"), _f(st["null_p5"], "{:.0f}"),
            _f(st["null_p95"], "{:.0f}"), _f(st["lift"]), _f(st["p_emp"], "{:.3f}")))
    elif res["null_skipped"]:
        print("  null skipped: {}".format(res["null_skipped"]))
    gt = res["gt"]
    if gt and gt["n_scored"]:
        print("ground truth: {} labelled frames, {} dead".format(gt["n_scored"], gt["n_dead"]))
        for src, sc in gt["frames"].items():
            print("  frames {:10s} P {} R {} F1 {}".format(src, _f(sc["precision"]),
                                                       _f(sc["recall"]), _f(sc["f1"])))
        for src, sc in gt["restarts"].items():
            print("  restarts {:14s} {}/{} predicted, {}/{} GT restarts found".format(
                src, sc["matched"], sc["n_pred"], sc["matched"], sc["n_gt"]))
    elif gt:
        print("ground truth: no labelled graded frames")
    for group, (total, eligible, picked) in review_counts.items():
        print("review {}: {} of {} ({} after exclusions)".format(group, picked, total, eligible))
    for p in paths:
        print("wrote {}".format(p))


# ---------------------------------------------------------------- CLI

def _list(allowed):
    def parse(text):
        if text.strip().lower() == "all":
            return tuple(allowed)
        items = tuple(s.strip() for s in text.split(",") if s.strip())
        bad = [s for s in items if s not in allowed]
        if bad or not items:
            raise argparse.ArgumentTypeError("choose from {} (comma-separated) or 'all'".format(
                ", ".join(allowed)))
        return items
    return parse


def _ranges(text):
    if text.strip().lower() in ("", "none"):
        return []
    return slg.parse_ranges(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for the outputs (created if missing; must be outside "
                                   "the repo)")
    p.add_argument("--game", default="BAR-ATH", help="games.py game id (default BAR-ATH)")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--tags", help="Smart Tagging XML; default games.asset_paths()"
                   "['smart_tagging'] (required with --raw)")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--gt", help="optional frame,label CSV (e.g. the public ground truth); "
                                "scores the EDA and Smart Tagging on its H/A/D frames")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--signals", type=_list(SIGNALS), default=SIGNALS,
                   help="dead-ball signals to combine (default missing,outside,still)")
    p.add_argument("--min-missing-s", type=float, default=MIN_MISSING_S,
                   help="missing: shortest gap that counts, s (default 2)")
    p.add_argument("--margin-m", type=float, default=bo.MARGIN_M,
                   help="outside: metres beyond the line (default 1)")
    p.add_argument("--lag", type=int, default=bs.LAG, help="still: speed lag, frames (default 5)")
    p.add_argument("--still-ms", type=float, default=bs.STILL_MS,
                   help="still: speed threshold, m/s (default 0.5)")
    p.add_argument("--min-still-s", type=float, default=bs.MIN_STILL_S,
                   help="still: shortest run, s (default 1)")
    p.add_argument("--still-locations", type=_list(bs.LOCATIONS), default=STILL_LOCATIONS,
                   help="still: run locations that count (default every restart mark, i.e. all "
                        "but 'elsewhere'; 'all' adds free kicks anywhere)")
    p.add_argument("--merge-gap-s", type=float, default=MERGE_GAP_S,
                   help="merge runs at most this far apart, s (default 1)")
    p.add_argument("--min-dead-s", type=float, default=MIN_DEAD_S,
                   help="drop intervals shorter than this, s (default 1)")
    p.add_argument("--link-window", type=float, default=LINK_WINDOW_S,
                   help="match window, s, |interval end - tag start| (default 5)")
    p.add_argument("--hist-bin", type=float, default=HIST_BIN_S,
                   help="offset histogram bin, s (default 0.5)")
    p.add_argument("--null-n", type=int, default=bg.NULL_N, help="shifts (default 1000; 0 = off)")
    p.add_argument("--null-seed", type=int, default=bg.NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=bg.NULL_MIN_SHIFT_S,
                   help="smallest shift, s (default 60)")
    p.add_argument("--review-n", type=int, default=REVIEW_N,
                   help="review sample size per unmatched group (default 8)")
    p.add_argument("--review-seed", type=int, default=REVIEW_SEED)
    p.add_argument("--exclude", type=_ranges, default=None,
                   help="frame ranges kept out of the review sample, 'a-b,c-d' or 'none'; "
                        "default the public GT window and the half-2 pre-kickoff frames for the "
                        "competition match, none with --raw")
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/deadball_matching")

    if args.raw:
        if args.frames is None or not args.tags:
            p.error("--frames and --tags are required with --raw")
        raw, tags, blocks = args.raw, args.tags, args.frames
        exclude = args.exclude or []
    else:
        games = bg._games()
        paths = common.game_paths(args.game, ["atd_raw", "smart_tagging"])
        raw, tags = paths["atd_raw"], args.tags or paths["smart_tagging"]
        blocks = args.frames or games.half_video_blocks(args.game)
        exclude = slg.DEFAULT_EXCLUDE if args.exclude is None else args.exclude

    res = analyse(raw, blocks, tags, args.gt, fps=args.fps, pitch=tuple(args.pitch),
                  signals=args.signals, min_missing_s=args.min_missing_s,
                  margin_m=args.margin_m, lag=args.lag, still_ms=args.still_ms,
                  min_still_s=args.min_still_s, still_locations=args.still_locations,
                  merge_gap_s=args.merge_gap_s, min_dead_s=args.min_dead_s,
                  link_window_s=args.link_window, hist_bin_s=args.hist_bin,
                  null_n=args.null_n, null_seed=args.null_seed,
                  null_min_shift_s=args.null_min_shift)
    review, review_counts = review_sample(res, args.review_n, args.review_seed, exclude)

    os.makedirs(args.out_dir, exist_ok=True)
    out = [os.path.join(args.out_dir, name) for name in (SUMMARY_NAME, MATCHES_NAME,
                                                         REVIEW_NAME)]
    bg.write_csv(summary_rows(res), out[0])
    write_rows(out[1], MATCH_COLUMNS, match_rows(res))
    write_rows(out[2], REVIEW_COLUMNS, review)
    print_summary(res, out, review_counts)
    return res


if __name__ == "__main__":
    main()
