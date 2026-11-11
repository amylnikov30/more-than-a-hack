"""How ball-missing gaps line up with Smart Tagging codes (EDA 2).

Code-agnostic first: for every tag code (and every SET PIECES `Type`), how often do gaps overlap
it, or end (ball reappears) within W seconds of its start, compared with the same tags shifted
circularly within each half? Definitions and hand-derived fixture answers:
tests/fixtures/README.md ("tag_gaps.py"); interpretation notes: EDA_HANDOFF.md.

Views
    A  profile: P(ball missing) in 1 s bins from -30 s to +30 s around each tag's start (and
       end), pooled over the group's instances, with a shift null band
    B  gap x code: overlap and restart-link shares per gap-length stratum, with a shift null
    C  threshold curve: for gap length >= L, the share of gaps that link to a group (proxy
       precision), the share of the group's starts with such a gap (proxy recall), and the
       share of graded frames those gaps cover

Usage
    # competition match: paths from games.asset_paths(), graded frames from
    # games.half_video_blocks(GAME)
    PYTHONPATH=/path/to/team-repo python eda/tag_gaps.py OUT_DIR

    # any other game in the raw ATD format (views A-C need --tags)
    python tag_gaps.py OUT_DIR --raw RAW.txt --frames 0-69660,69661-143760 [--tags TAGS.xml]

Writes OUT_DIR/tag_gaps_summary.csv (long format: analysis, scope, bin, metric, value, unit).

Needs ball_gaps.py next to this file; stdlib + numpy only. Aggregates only: no per-frame or
per-gap rows are written or printed.
"""

import argparse
import math
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np

import ball_gaps as bg

FPS = bg.FPS
WINDOW_S = 30.0                  # view A: seconds either side of the anchor
BIN_S = 1.0                      # view A: bin width, seconds
LINK_WINDOW_S = 5.0              # a gap links to a tag if the ball reappears within this of its start
STRATA_S = [(0.0, 2.0, "0-2s"), (2.0, 5.0, "2-5s"), (5.0, 10.0, "5-10s"),
            (10.0, 20.0, "10-20s"), (20.0, 35.0, "20-35s"), (35.0, math.inf, ">=35s")]
SET_PIECE_CODE = "SET PIECES"
THRESHOLDS_S = [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]
CSV_NAME = "tag_gaps_summary.csv"


# ---------------------------------------------------------------- reading

def read_tags(path, fps=FPS):
    """[{code, type, team, start, end}] in file order. code is stripped and upper-cased;
    type/team are the `Type`/`Team` label texts or None; frames are round(s * fps), end
    exclusive."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        labels = {}
        for label in inst.iter("label"):
            group = (label.findtext("group") or "").strip()
            if group:
                labels[group] = (label.findtext("text") or "").strip()
        out.append({"code": (inst.findtext("code") or "").strip().upper(),
                    "type": labels.get("Type") or None,
                    "team": labels.get("Team") or None,
                    "start": int(round(float(inst.findtext("start")) * fps)),
                    "end": int(round(float(inst.findtext("end")) * fps))})
    return out


def tag_groups(tags):
    """{group: [tag]}: 'code:<CODE>' for every code (sorted), then 'type:<Type>' for SET PIECES
    instances with a Type (sorted)."""
    codes, types = {}, {}
    for t in tags:
        codes.setdefault("code:" + t["code"], []).append(t)
        if t["code"] == SET_PIECE_CODE and t["type"]:
            types.setdefault("type:" + t["type"], []).append(t)
    return dict([(k, codes[k]) for k in sorted(codes)] + [(k, types[k]) for k in sorted(types)])


# ---------------------------------------------------------------- shared helpers

def stratum_label(length_frames, fps=FPS):
    seconds = length_frames / float(fps)
    for lo, hi, label in STRATA_S:
        if lo <= seconds < hi:
            return label
    raise ValueError(length_frames)


def half_of(frame, blocks):
    """Graded half holding `frame`, or None."""
    for half, first, last in blocks:
        if first <= frame <= last:
            return half
    return None


def graded_anchors(tags, blocks, end=False):
    """([(half, frame)], n_dropped): each tag's start (or end) frame, kept only if graded."""
    kept, dropped = [], 0
    for t in tags:
        frame = t["end"] if end else t["start"]
        half = half_of(frame, blocks)
        if half is None:
            dropped += 1
        else:
            kept.append((half, frame))
    return kept, dropped


def linkable(gap, blocks):
    """The ball reappears on a graded frame of the gap's own half."""
    ns = gap["next_seen"]
    return ns is not None and half_of(ns, blocks) == gap["half"]


def link_hits(gaps, anchors, blocks, w_frames):
    """[bool]: linkable gap whose reappearance is within w_frames of an anchor in its half."""
    out = []
    for g in gaps:
        ok = linkable(g, blocks) and any(
            h == g["half"] and abs(g["next_seen"] - a) <= w_frames for h, a in anchors)
        out.append(bool(ok))
    return out


def overlap_hits(gaps, tags, tol_frames=0):
    intervals = [(t["start"], t["end"]) for t in tags]
    return [bg.overlaps(g, intervals, tol_frames) for g in gaps]


def link_null_matrix(gaps, anchors, blocks, offsets, w_frames):
    """(K, len(gaps)) bool: link_hits under each circular shift of the anchors.

    Each linkable gap becomes the point [next_seen, next_seen] and each anchor the tag
    [a, a + 1); with tolerance w the bg.null_overlap_matrix test a - w <= ns < a + 1 + w is
    exactly |ns - a| <= w. Unlinkable gaps stay False.
    """
    k = len(next(iter(offsets.values())))
    out = np.zeros((k, len(gaps)), dtype=bool)
    idx = [j for j, g in enumerate(gaps) if linkable(g, blocks)]
    if not idx or not anchors:
        return out
    points = [{"half": gaps[j]["half"], "start": gaps[j]["next_seen"],
               "end": gaps[j]["next_seen"]} for j in idx]
    out[:, idx] = bg.null_overlap_matrix(points, [(a, a + 1) for _, a in anchors], blocks,
                                         offsets, w_frames)
    return out


def null_stats(real_hits, matrix, idx):
    """Real vs null for the gaps in idx. Shares are fractions (0-1)."""
    n = len(idx)
    real = int(sum(real_hits[j] for j in idx))
    out = {"n_gaps": n, "real_count": real, "real_share": real / float(n) if n else math.nan}
    if matrix is None or n == 0:
        return out
    k = matrix.shape[0]
    null = matrix[:, idx].sum(axis=1)
    share = null / float(n)
    p5, p95 = np.percentile(share, [5, 95])
    mean = float(share.mean())
    out.update({"null_mean": mean, "null_p5": float(p5), "null_p95": float(p95),
                "p_emp": (1.0 + float(np.sum(null >= real))) / (1.0 + k),
                "lift": out["real_share"] / mean if mean > 0 else math.nan})
    return out


# ---------------------------------------------------------------- view B

def view_b(gaps, groups, blocks, fps=FPS, tol_frames=0, w_frames=None, offsets=None):
    """{group: {n_tags, n_dropped, overlap: {stratum: stats}, link: {stratum: stats}}}.

    Strata are STRATA_S labels plus 'all'; empty strata are left out. offsets ({half: (K,)})
    adds the circular-shift null; None gives real counts only.
    """
    if w_frames is None:
        w_frames = int(round(LINK_WINDOW_S * fps))
    strata = {"all": list(range(len(gaps)))}
    for _, _, label in STRATA_S:
        idx = [j for j, g in enumerate(gaps) if stratum_label(g["length"], fps) == label]
        if idx:
            strata[label] = idx
    out = {}
    for name, tags in groups.items():
        anchors, dropped = graded_anchors(tags, blocks)
        over = overlap_hits(gaps, tags, tol_frames)
        link = link_hits(gaps, anchors, blocks, w_frames)
        m_over = m_link = None
        if offsets is not None:
            m_over = bg.null_overlap_matrix(gaps, [(t["start"], t["end"]) for t in tags],
                                            blocks, offsets, tol_frames)
            m_link = link_null_matrix(gaps, anchors, blocks, offsets, w_frames)
        out[name] = {"n_tags": len(tags), "n_dropped": dropped,
                     "overlap": {s: null_stats(over, m_over, idx) for s, idx in strata.items()},
                     "link": {s: null_stats(link, m_link, idx) for s, idx in strata.items()}}
    return out


# ---------------------------------------------------------------- view A

def half_arrays(missing, blocks):
    """{half: (first, L, cumsum)}: cumsum[i] = missing frames among the half's first i frames.
    Frames beyond the end of `missing` count as missing (as in find_gaps)."""
    n = len(missing)
    out = {}
    for half, first, last in blocks:
        length = last - first + 1
        m = np.ones(length, dtype=np.int64)
        lo, hi = max(first, 0), min(last, n - 1)
        if hi >= lo:
            m[lo - first:hi - first + 1] = missing[lo:hi + 1]
        out[half] = (first, length, np.concatenate([[0], np.cumsum(m)]))
    return out


def profile_counts(halves, anchors, window_f, bin_f, offsets=None, chunk=100):
    """Pooled (missing, counted) frames per bin around the anchors.

    Bin k covers [t + k*B, t + (k+1)*B) for k = -window/B .. window/B - 1, clipped to the
    anchor's own half. Without offsets: two int arrays (n_bins,). With offsets
    ({half: (K,)}; each half's anchors move together, circularly within the half, while
    the window is clipped, not wrapped): two arrays (K, n_bins).
    """
    if window_f % bin_f:
        raise ValueError("window must be a multiple of the bin width")
    nb = 2 * window_f // bin_f
    ks = (np.arange(nb) - nb // 2) * bin_f
    k_total = 1 if offsets is None else len(next(iter(offsets.values())))
    miss = np.zeros((k_total, nb), dtype=np.int64)
    counted = np.zeros((k_total, nb), dtype=np.int64)
    for half, (first, length, cs) in halves.items():
        rel = np.array([a - first for h, a in anchors if h == half], dtype=np.int64)
        if len(rel) == 0:
            continue
        for lo_k in range(0, k_total, chunk):
            if offsets is None:
                pos = rel[None, :]
            else:
                off = np.asarray(offsets[half][lo_k:lo_k + chunk], dtype=np.int64)[:, None]
                pos = (rel[None, :] + off) % length                       # (k, m)
            lo = np.clip(pos[:, :, None] + ks, 0, length)                  # (k, m, nb)
            hi = np.clip(pos[:, :, None] + ks + bin_f, 0, length)
            miss[lo_k:lo_k + chunk] += (cs[hi] - cs[lo]).sum(axis=1)
            counted[lo_k:lo_k + chunk] += (hi - lo).sum(axis=1)
    if offsets is None:
        return miss[0], counted[0]
    return miss, counted


def _share(miss, counted):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counted > 0, miss / np.maximum(counted, 1).astype(float), np.nan)


def view_a(missing, groups, blocks, fps=FPS, window_s=WINDOW_S, bin_s=BIN_S, offsets=None,
           end=False):
    """{group: {n, n_dropped, bins_s, missing, counted, share[, null_mean, null_p5, null_p95]}}
    aligned on each tag's start (end=True: its end). Shares are fractions; NaN where no graded
    frame was counted."""
    window_f, bin_f = int(round(window_s * fps)), int(round(bin_s * fps))
    halves = half_arrays(missing, blocks)
    nb = 2 * window_f // bin_f
    bins_s = [float((k - nb // 2) * bin_s) for k in range(nb)]
    out = {}
    for name, tags in groups.items():
        anchors, dropped = graded_anchors(tags, blocks, end=end)
        miss, counted = profile_counts(halves, anchors, window_f, bin_f)
        res = {"n": len(anchors), "n_dropped": dropped, "bins_s": bins_s,
               "missing": miss.tolist(), "counted": counted.tolist(),
               "share": _share(miss, counted).tolist()}
        if offsets is not None and anchors:
            m_null, c_null = profile_counts(halves, anchors, window_f, bin_f, offsets)
            share = _share(m_null, c_null)
            with warnings.catch_warnings():          # all-NaN bins stay NaN, silently
                warnings.simplefilter("ignore", RuntimeWarning)
                res["null_mean"] = np.nanmean(share, axis=0).tolist()
                p5, p95 = np.nanpercentile(share, [5, 95], axis=0)
            res["null_p5"], res["null_p95"] = p5.tolist(), p95.tolist()
        out[name] = res
    return out



# ---------------------------------------------------------------- view C

def view_c(gaps, groups, blocks, graded_total, fps=FPS, w_frames=None, thresholds_s=None):
    """{"n_long": {L: n}, "frame_share": {L: f}, "groups": {group: {L: stats}}}.

    Long gaps: length >= L * fps. stats: anchors (graded), linked (long gaps linked to any
    anchor of the group), precision = linked / n_long, anchors_hit (anchors with at least one
    long gap linked), recall = anchors_hit / anchors. Each gap and anchor counts once.
    """
    if w_frames is None:
        w_frames = int(round(LINK_WINDOW_S * fps))
    thresholds_s = THRESHOLDS_S if thresholds_s is None else thresholds_s
    long_sets = {L: [j for j, g in enumerate(gaps) if g["length"] >= L * fps]
                 for L in thresholds_s}
    out = {"n_long": {L: len(idx) for L, idx in long_sets.items()},
           "frame_share": {L: sum(gaps[j]["length"] for j in idx) / float(graded_total)
                           if graded_total else math.nan for L, idx in long_sets.items()},
           "groups": {}}
    for name, tags in groups.items():
        anchors, _ = graded_anchors(tags, blocks)
        pairs = {}                                   # gap index -> set of anchor indices
        for j, g in enumerate(gaps):
            if linkable(g, blocks):
                hit = {i for i, (h, a) in enumerate(anchors)
                       if h == g["half"] and abs(g["next_seen"] - a) <= w_frames}
                if hit:
                    pairs[j] = hit
        res = {}
        for L, idx in long_sets.items():
            linked = [j for j in idx if j in pairs]
            hit = set().union(*[pairs[j] for j in linked]) if linked else set()
            res[L] = {"n_long": len(idx), "linked": len(linked),
                      "precision": len(linked) / float(len(idx)) if idx else math.nan,
                      "anchors": len(anchors), "anchors_hit": len(hit),
                      "recall": len(hit) / float(len(anchors)) if anchors else math.nan}
        out["groups"][name] = res
    return out


# ---------------------------------------------------------------- driver

def analyse(raw_path, blocks, tags_path=None, fps=FPS, window_s=WINDOW_S, bin_s=BIN_S,
            link_window_s=LINK_WINDOW_S, tag_tolerance_s=0.0, thresholds_s=None,
            null_n=bg.NULL_N, null_seed=bg.NULL_SEED, null_min_shift_s=bg.NULL_MIN_SHIFT_S):
    """All numbers, as a dict. No per-frame or per-gap data leaves this function."""
    ball, _ = bg.read_ball(raw_path)
    missing = ~np.isfinite(ball).all(axis=1)
    gaps = bg.find_gaps(missing, blocks)
    graded_total = sum(b - a + 1 for _, a, b in blocks)
    thresholds_s = THRESHOLDS_S if thresholds_s is None else thresholds_s
    w_frames = int(round(link_window_s * fps))
    tol = int(round(tag_tolerance_s * fps))
    res = {"blocks": list(blocks), "fps": fps, "window_s": window_s, "bin_s": bin_s,
           "link_window_s": link_window_s, "tag_tolerance_s": tag_tolerance_s,
           "thresholds_s": list(thresholds_s), "null_n": null_n, "null_seed": null_seed,
           "null_min_shift_s": null_min_shift_s, "graded_frames": graded_total,
           "missing_frames": int(sum(g["length"] for g in gaps)), "n_gaps": len(gaps),
           "n_unlinkable": sum(not linkable(g, blocks) for g in gaps),
           "tags": None, "null_skipped": None}
    if not tags_path:
        return res
    groups = tag_groups(read_tags(tags_path, fps))
    res["tags"] = {name: {"n_tags": len(t), "n_dropped": graded_anchors(t, blocks)[1]}
                   for name, t in groups.items()}
    offsets = bg.draw_offsets(blocks, null_n, null_seed, int(round(null_min_shift_s * fps))) \
        if null_n > 0 else None
    if offsets is None:
        res["null_skipped"] = ("null_n is 0" if null_n <= 0 else
                               "a half is shorter than 2 x min_shift ({:g} s)".format(
                                   null_min_shift_s))
    res["profile_start"] = view_a(missing, groups, blocks, fps, window_s, bin_s, offsets)
    res["profile_end"] = view_a(missing, groups, blocks, fps, window_s, bin_s, offsets, end=True)
    res["view_b"] = view_b(gaps, groups, blocks, fps, tol, w_frames, offsets)
    res["view_c"] = view_c(gaps, groups, blocks, graded_total, fps, w_frames, thresholds_s)
    return res


# ---------------------------------------------------------------- output

def summary_rows(res):
    """Long-format rows: (analysis, scope, bin, metric, value, unit). Aggregates only."""
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    add("params", "all", "", "fps", res["fps"], "frames/s")
    add("params", "all", "", "window", res["window_s"], "s")
    add("params", "all", "", "bin", res["bin_s"], "s")
    add("params", "all", "", "link_window", res["link_window_s"], "s")
    add("params", "all", "", "tag_tolerance", res["tag_tolerance_s"], "s")
    add("params", "all", "", "null_n", res["null_n"], "count")
    add("params", "all", "", "null_seed", res["null_seed"], "")
    add("params", "all", "", "null_min_shift", res["null_min_shift_s"], "s")
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")
    add("overview", "all", "", "graded_frames", res["graded_frames"], "frames")
    add("overview", "all", "", "missing_frames", res["missing_frames"], "frames")
    add("overview", "all", "", "baseline_missing_share",
        bg._rate(res["missing_frames"], res["graded_frames"]), "fraction")
    add("overview", "all", "", "n_gaps", res["n_gaps"], "count")
    add("overview", "all", "", "n_unlinkable", res["n_unlinkable"], "count")
    if res["tags"] is None:
        return rows
    for name, t in res["tags"].items():
        add("tags", name, "", "n_tags", t["n_tags"], "count")
        add("tags", name, "", "n_dropped", t["n_dropped"], "count")

    for analysis in ("profile_start", "profile_end"):
        for name, a in res[analysis].items():
            add(analysis, name, "", "n_instances", a["n"], "count")
            for k, left in enumerate(a["bins_s"]):
                b = "{:g}".format(left)
                add(analysis, name, b, "share", a["share"][k], "fraction")
                add(analysis, name, b, "missing", a["missing"][k], "frames")
                add(analysis, name, b, "counted", a["counted"][k], "frames")
                for m in ("null_mean", "null_p5", "null_p95"):
                    if m in a:
                        add(analysis, name, b, m, a[m][k], "fraction")

    units = {"n_gaps": "count", "real_count": "count", "p_emp": "p", "lift": "ratio"}
    for name, b in res["view_b"].items():
        for kind in ("overlap", "link"):
            for stratum, st in b[kind].items():
                for m in ("n_gaps", "real_count", "real_share", "null_mean", "null_p5",
                          "null_p95", "p_emp", "lift"):
                    if m in st:
                        add(kind, name, stratum, m, st[m], units.get(m, "fraction"))

    c = res["view_c"]
    for L in res["thresholds_s"]:
        b = "{:g}".format(L)
        add("threshold", "all", b, "n_long", c["n_long"][L], "count")
        add("threshold", "all", b, "frame_share", c["frame_share"][L], "fraction")
        for name, g in c["groups"].items():
            st = g[L]
            for m, unit in (("linked", "count"), ("precision", "fraction"),
                            ("anchors", "count"), ("anchors_hit", "count"),
                            ("recall", "fraction")):
                add("threshold", name, b, m, st[m], unit)
    return rows


def _f(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, csv_path, top=5, pre_s=10.0):
    print("{} graded frames, ball missing on {} ({}), {} gaps ({} unlinkable: ball reappears "
          "off the graded half)".format(res["graded_frames"], res["missing_frames"],
                                        _f(bg._rate(res["missing_frames"], res["graded_frames"])),
                                        res["n_gaps"], res["n_unlinkable"]))
    if res["tags"] is None:
        print("no Smart Tagging file: views A-C skipped")
        print("wrote {}".format(csv_path))
        return
    n_codes = sum(k.startswith("code:") for k in res["tags"])
    print("tags: {} instances, {} codes, {} set-piece types; {} anchors not graded".format(
        sum(t["n_tags"] for k, t in res["tags"].items() if k.startswith("code:")), n_codes,
        len(res["tags"]) - n_codes,
        sum(t["n_dropped"] for k, t in res["tags"].items() if k.startswith("code:"))))
    if res["null_skipped"]:
        print("  null skipped: {}".format(res["null_skipped"]))

    # A: mean P(missing) over the pre_s seconds before each start, vs the null mean
    ranked = []
    for name, a in res["profile_start"].items():
        idx = [k for k, left in enumerate(a["bins_s"]) if -pre_s <= left < 0]
        miss = sum(a["missing"][k] for k in idx)
        cnt = sum(a["counted"][k] for k in idx)
        if not a["n"] or not cnt:
            continue
        obs = miss / float(cnt)
        null = float(np.nanmean([a["null_mean"][k] for k in idx])) if "null_mean" in a \
            else math.nan
        ranked.append((obs - (0 if math.isnan(null) else null), name, a["n"], obs, null))
    ranked.sort(reverse=True)
    print("A  P(missing) in the {:g} s before a tag starts (observed / null mean), top {}:".format(
        pre_s, top))
    print("   " + "; ".join("{} n={} {}/{}".format(name, n, _f(o), _f(nl))
                            for _, name, n, o, nl in ranked[:top]))

    # B: strongest link lifts with enough gaps
    cells = [(st.get("lift", math.nan), name, s, st) for name, b in res["view_b"].items()
             for s, st in b["link"].items() if s != "all" and st["n_gaps"] >= 5
             and st["real_count"] > 0]
    cells = [c for c in cells if not math.isnan(c[0])]
    cells.sort(key=lambda c: -c[0])
    print("B  restart link (reappears within {:g} s of a start), top {} lifts (n >= 5):".format(
        res["link_window_s"], top))
    print("   " + ("; ".join("{} {} {}/{} lift {} p {}".format(
        name, s, st["real_count"], st["n_gaps"], _f(lift, "{:.1f}"), _f(st.get("p_emp"), "{:.3f}"))
        for lift, name, s, st in cells[:top]) or "-"))
    sp = res["view_b"].get("code:" + SET_PIECE_CODE)
    if sp:
        print("   SET PIECES, all gaps: overlap {}/{}, link {}/{} (null mean {})".format(
            sp["overlap"]["all"]["real_count"], sp["overlap"]["all"]["n_gaps"],
            sp["link"]["all"]["real_count"], sp["link"]["all"]["n_gaps"],
            _f(sp["link"]["all"].get("null_mean"))))

    # C: SET PIECES precision / recall at a few thresholds
    g = res["view_c"]["groups"].get("code:" + SET_PIECE_CODE)
    if g:
        show = [L for L in res["thresholds_s"] if L in (2, 5, 10, 20, 30)] or res["thresholds_s"]
        print("C  SET PIECES, gap >= L: " + "; ".join(
            "L={:g}s n={} prec {} rec {} frames {}".format(
                L, g[L]["n_long"], _f(g[L]["precision"]), _f(g[L]["recall"]),
                _f(res["view_c"]["frame_share"][L])) for L in show))
    print("wrote {}".format(csv_path))


# ---------------------------------------------------------------- CLI

def _thresholds(text):
    try:
        out = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated seconds")
    if not out or any(x < 0 for x in out):
        raise argparse.ArgumentTypeError("thresholds must be >= 0")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (created if missing; must be outside the "
                                   "repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH", help="games.py game id (default BAR-ATH)")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--tags", help="Smart Tagging XML; default games.asset_paths()"
                   "['smart_tagging'] when --raw is not given")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--window", type=float, default=WINDOW_S,
                   help="view A: seconds either side of the anchor (default 30)")
    p.add_argument("--bin", type=float, default=BIN_S, help="view A: bin width, s (default 1)")
    p.add_argument("--link-window", type=float, default=LINK_WINDOW_S,
                   help="views B, C: reappearance within this many s of a start (default 5)")
    p.add_argument("--tag-tolerance", type=float, default=0.0,
                   help="view B overlap: widen each tag by this many s on both sides")
    p.add_argument("--thresholds", type=_thresholds, default=THRESHOLDS_S,
                   help="view C: comma-separated gap lengths in s")
    p.add_argument("--null-n", type=int, default=bg.NULL_N, help="shifts (default 1000; 0 = off)")
    p.add_argument("--null-seed", type=int, default=bg.NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=bg.NULL_MIN_SHIFT_S,
                   help="smallest shift in seconds (default 60)")
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/tag_gaps")
    if int(round(args.window * args.fps)) % int(round(args.bin * args.fps)):
        p.error("--window must be a multiple of --bin")

    if args.raw:
        if args.frames is None:
            p.error("--frames is required with --raw")
        raw, tags, blocks = args.raw, args.tags, args.frames
    else:
        games = bg._games()
        try:
            raw = games.require(["atd_raw"], game=args.game)[0]
        except (IOError, OSError, KeyError) as err:
            sys.exit(str(err))
        tags = args.tags
        if not tags:
            try:
                tags = games.require(["smart_tagging"], game=args.game)[0]
            except (IOError, OSError, KeyError):
                tags = None
        blocks = args.frames or games.half_video_blocks(args.game)

    res = analyse(raw, blocks, tags, fps=args.fps, window_s=args.window, bin_s=args.bin,
                  link_window_s=args.link_window, tag_tolerance_s=args.tag_tolerance,
                  thresholds_s=args.thresholds, null_n=args.null_n, null_seed=args.null_seed,
                  null_min_shift_s=args.null_min_shift)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, CSV_NAME)
    bg.write_csv(summary_rows(res), csv_path)
    print_summary(res, csv_path)
    return res


if __name__ == "__main__":
    main()
