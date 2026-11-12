"""Profile the gaps in the ball track (frames where the ball position is NaN), graded frames only.

  a) gap length distribution: number of gaps and share of missing frames in < 0.5 s, 0.5-2 s,
     2-5 s and >= 5 s, overall and per half;
  b) for gaps of 2 s or more: share that overlap a Smart Tagging SET PIECES interval;
  c) crowding: players within 5 m of the last seen ball position at each gap start, compared
     with a seeded random sample of graded frames where the ball is visible;
  d) flips: does the team of the player nearest the ball differ between the last-seen frame and
     the reappearance frame? Compared with ball-visible frame pairs the same distance apart;
  e) set pieces vs chance: the (b) overlap against a null of circularly shifted tags.

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR), graded frames
    # from games.half_video_blocks("BAR-ATH")
    PYTHONPATH=/path/to/team-repo python eda/scripts/ball_gaps.py OUT_DIR

    # any other game in the same raw format, e.g. Metrica Sample Game 3 (no Smart Tagging)
    python ball_gaps.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --frames 0-69660,69661-143760

Prints a few aggregate lines and writes the same numbers to OUT_DIR/ball_gaps_summary.csv.
No per-frame output. Frames are video frames: video_frame = ATD frame_id - 1. OUT_DIR must be
outside the repository that contains this script.
"""

import argparse
import csv
import gzip
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

# Params (defaults; all overridable on the command line)
FPS = 25
PITCH = (105.3, 68.0)            # metres, from games.py
RADIUS_M = 5.0                   # "crowded" = players within this distance of the ball
SAMPLE_SIZE = 2000               # ball-visible frames sampled for the crowding/flip baselines
SEED = 0
LONG_GAP_S = 2.0                 # (b), (e) and the crowding split use gaps >= this
BINS_S = [(0.0, 0.5, "<0.5s"), (0.5, 2.0, "0.5-2s"), (2.0, 5.0, "2-5s"),
          (5.0, math.inf, ">=5s")]   # half-open [lo, hi) in seconds
SET_PIECE_CODE = "set pieces"
NULL_N = 1000                    # (e) number of circular shifts
NULL_SEED = 0
NULL_MIN_SHIFT_S = 60.0          # (e) smallest shift, seconds, in both directions
CSV_NAME = "ball_gaps_summary.csv"
FLIP_REASONS = ("no_last_seen", "no_reappearance", "no_player", "too_far")


# ---------------------------------------------------------------- reading

def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _xy(group):
    """(x, y) floats from 'x,y[,more]'; NaN for anything unreadable."""
    fields = group.split(",")
    try:
        return float(fields[0]), float(fields[1])
    except (ValueError, IndexError):
        return math.nan, math.nan


def read_ball(raw_path):
    """Ball (x, y) normalised, indexed by video frame; NaN where missing or no record.

    Returns (ball, has_record): ball is float array (n, 2), has_record a bool array.
    """
    frames, xs, ys = [], [], []
    with _open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) < 3:
                continue
            x, y = _xy(parts[-1])
            frames.append(int(parts[0]) - 1)
            xs.append(x)
            ys.append(y)
    frames = np.asarray(frames, dtype=np.int64)
    n = int(frames.max()) + 1 if len(frames) else 0
    ball = np.full((n, 2), np.nan)
    has_record = np.zeros(n, dtype=bool)
    ball[frames, 0] = xs
    ball[frames, 1] = ys
    has_record[frames] = True
    return ball, has_record


def read_slots(raw_path, wanted):
    """{video_frame: array (k, 2)} of every slot's normalised position, NaN kept, in line order.

    The position of a row is the slot index used for tie-breaking and for the team lookup.
    """
    wanted = set(int(f) for f in wanted)
    out = {}
    with _open(raw_path) as handle:
        for line in handle:
            head = line[:line.index(":")] if ":" in line else ""
            if not head or int(head) - 1 not in wanted:
                continue
            parts = line.rstrip("\n").split(":")
            out[int(head) - 1] = np.array([_xy(g) for g in parts[1].split(";")],
                                          dtype=float).reshape(-1, 2)
    return out


def read_players(raw_path, wanted):
    """{video_frame: array (k, 2) of present players}; a player is present only if both x and y
    are finite."""
    return {f: pts[np.isfinite(pts).all(axis=1)] for f, pts in read_slots(raw_path, wanted).items()}


def read_slot_teams(meta_path):
    """[(first_frame_id, last_frame_id, teams)] per DataFormatSpecification.

    teams[i] is 0 for the first <Team> in the file (home), 1 for the second (away), and -1 for a
    slot whose player has no team (referee, unknown), in the order the slots appear in a raw line.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [el.get("id") for el in root.iter() if _tag(el) == "Team"]
    team_index = {tid: i for i, tid in enumerate(team_ids[:2])}
    player_team = {el.get("id"): team_index.get(el.get("teamId"), -1)
                   for el in root.iter() if _tag(el) == "Player"}
    channel_player = {el.get("id"): el.get("playerId")
                      for el in root.iter() if _tag(el) == "PlayerChannel"}
    specs = []
    for spec in (el for el in root.iter() if _tag(el) == "DataFormatSpecification"):
        teams = []
        for register in spec:
            if _tag(register) != "SplitRegister" or not any(
                    _tag(el) == "PlayerChannelRef" for el in register.iter()):
                continue
            for group in register:
                refs = [el.get("playerChannelId") for el in group.iter()
                        if _tag(el) == "PlayerChannelRef"]
                pid = channel_player.get(refs[0]) if refs else None
                teams.append(player_team.get(pid, -1))
            break
        # A missing bound means open-ended; defaulting endFrame to 0 would match no frame and
        # silently turn every gap into "no_player".
        end = spec.get("endFrame")
        specs.append((int(spec.get("startFrame") or 0), int(end) if end else math.inf,
                      np.array(teams, dtype=int)))
    return specs


def teams_for(specs, video_frame):
    frame_id = video_frame + 1
    for first, last, teams in specs:
        if first <= frame_id <= last:
            return teams
    return None


def read_set_pieces(path, fps=FPS):
    """[(start_frame, end_frame_exclusive)] of SET PIECES instances, code matched
    case-insensitively, frame = round(seconds * fps) as in starter/load_smart_tagging.py."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        if (inst.findtext("code") or "").strip().lower() != SET_PIECE_CODE:
            continue
        out.append((int(round(float(inst.findtext("start")) * fps)),
                    int(round(float(inst.findtext("end")) * fps))))
    return out


# ---------------------------------------------------------------- (a) gaps

def find_gaps(missing, blocks):
    """Runs of missing ball inside each graded block, clipped to the block.

    missing: bool array by video frame (frames beyond the array count as missing).
    blocks: [(half, first, last)] inclusive video frames.
    Returns [{half, start, end, length, edge, last_seen, next_seen}] where `edge` says the run
    continues past the block boundary (so its true length is longer than counted), `last_seen`
    is the last ball-visible frame before the unclipped run and `next_seen` the first one after
    it (None if there is none; either may be ungraded).
    """
    n = len(missing)

    def is_missing(f):
        return f < 0 or f >= n or bool(missing[f])

    gaps = []
    for half, first, last in blocks:
        f = first
        while f <= last:
            if not is_missing(f):
                f += 1
                continue
            start = f
            while f <= last and is_missing(f):
                f += 1
            end = f - 1
            raw_start = start
            while raw_start - 1 >= 0 and is_missing(raw_start - 1):
                raw_start -= 1
            raw_end = end
            while raw_end + 1 < n and is_missing(raw_end + 1):
                raw_end += 1
            edge = (start == first and (first - 1 < 0 or is_missing(first - 1))) or \
                   (end == last and is_missing(last + 1))
            gaps.append({"half": half, "start": start, "end": end,
                         "length": end - start + 1, "edge": edge,
                         "last_seen": raw_start - 1 if raw_start - 1 >= 0 else None,
                         "next_seen": raw_end + 1 if raw_end + 1 < n else None})
    return gaps


def bin_label(length_frames, fps=FPS):
    seconds = length_frames / float(fps)
    for lo, hi, label in BINS_S:
        if lo <= seconds < hi:
            return label
    raise ValueError(length_frames)


def length_table(gaps, fps=FPS):
    """{label: (n_gaps, missing_frames)} for the bins, plus totals under 'all'."""
    table = {label: [0, 0] for _, _, label in BINS_S}
    for g in gaps:
        cell = table[bin_label(g["length"], fps)]
        cell[0] += 1
        cell[1] += g["length"]
    table["all"] = [len(gaps), sum(g["length"] for g in gaps)]
    return {k: tuple(v) for k, v in table.items()}


# ---------------------------------------------------------------- (b) and (e) set pieces

def overlaps(gap, intervals, tol_frames=0):
    """Does inclusive gap [start, end] overlap any half-open [a - tol, b + tol)?"""
    return any(a - tol_frames <= gap["end"] and b + tol_frames > gap["start"]
               for a, b in intervals)


def clip_to_blocks(intervals, blocks):
    """{half: (starts, ends)} of tags clipped to each block, relative to the block start
    (half-open, 0 <= start < end <= L). A tag crossing blocks is split, never shared."""
    out = {}
    for half, first, last in blocks:
        length = last - first + 1
        rel = [(max(a, first) - first, min(b, last + 1) - first) for a, b in intervals]
        rel = [(s, e) for s, e in rel if e > s and s < length]
        out[half] = (np.array([s for s, _ in rel], dtype=np.int64),
                     np.array([e for _, e in rel], dtype=np.int64))
    return out


def shifted_pieces(intervals, blocks, offsets):
    """Absolute [(half, start, end_exclusive)] of every tag after shifting half h by offsets[h]
    modulo the half length. A wrapped tag becomes two pieces. Reference implementation for tests;
    null_overlap_matrix is the vectorised version."""
    pieces = []
    clipped = clip_to_blocks(intervals, blocks)
    for half, first, last in blocks:
        length = last - first + 1
        for s, e in zip(*clipped[half]):
            s2 = (int(s) + int(offsets[half])) % length
            e2 = s2 + int(e - s)
            pieces.append((half, first + s2, first + min(e2, length)))
            if e2 > length:
                pieces.append((half, first, first + e2 - length))
    return pieces


def null_overlap_matrix(gaps, intervals, blocks, offsets, tol_frames=0, chunk=100):
    """Bool array (K, len(gaps)): does gap j overlap any shifted tag of its own half under shift k?

    offsets: {half: int array (K,)}. Interval arithmetic only; no per-frame masks.
    """
    halves = {h: (a, b) for h, a, b in blocks}
    k_total = len(next(iter(offsets.values()))) if offsets else 0
    out = np.zeros((k_total, len(gaps)), dtype=bool)
    clipped = clip_to_blocks(intervals, blocks)
    for half, (first, last) in halves.items():
        idx = [j for j, g in enumerate(gaps) if g["half"] == half]
        starts, ends = clipped[half]
        if not idx or len(starts) == 0:
            continue
        length = last - first + 1
        gs = np.array([gaps[j]["start"] - first for j in idx])[None, None, :]
        ge = np.array([gaps[j]["end"] - first for j in idx])[None, None, :]
        dur = (ends - starts)[None, :]
        for lo in range(0, k_total, chunk):
            off = np.asarray(offsets[half][lo:lo + chunk], dtype=np.int64)[:, None]
            s1 = (starts[None, :] + off) % length                      # (k, m)
            e1 = s1 + dur
            end1 = np.minimum(e1, length)
            hit = ((s1[:, :, None] - tol_frames <= ge) &
                   (end1[:, :, None] + tol_frames > gs))
            wrap = e1 > length
            if wrap.any():
                hit |= (wrap[:, :, None] & (-tol_frames <= ge) &
                        ((e1 - length)[:, :, None] + tol_frames > gs))
            out[lo:lo + chunk, idx] = hit.any(axis=1)
    return out


def draw_offsets(blocks, k, seed, min_shift_frames):
    """{half: int array (k,)} uniform on [min_shift, L - min_shift], halves drawn in order from
    one default_rng(seed). None if a half is too short for min_shift."""
    rng = np.random.default_rng(seed)
    offsets = {}
    for half, first, last in blocks:
        length = last - first + 1
        if length - min_shift_frames < min_shift_frames:
            return None
        offsets[half] = rng.integers(min_shift_frames, length - min_shift_frames,
                                     size=k, endpoint=True)
    return offsets


# ---------------------------------------------------------------- (c) and (d) players

def crowd_count(players, ball_xy, pitch=PITCH, radius=RADIUS_M):
    """Players within `radius` metres of ball_xy (both normalised), distances in metres."""
    if len(players) == 0:
        return 0
    d = (players - np.asarray(ball_xy)) * np.asarray(pitch)
    return int(np.sum(np.hypot(d[:, 0], d[:, 1]) <= radius))


def nearest_team(slots, teams, ball_xy, pitch=PITCH):
    """(team, distance_m) of the slot nearest the ball, or (-1, nan) if there is none.

    Only slots with finite x and y and a team (>= 0) are considered; ties go to the lowest slot
    index (np.argmin returns the first minimum).
    """
    if slots is None or teams is None or not np.isfinite(ball_xy).all():
        return -1, math.nan
    k = min(len(slots), len(teams))
    pts, tm = slots[:k], teams[:k]
    ok = np.isfinite(pts).all(axis=1) & (tm >= 0)
    if not ok.any():
        return -1, math.nan
    d = (pts - np.asarray(ball_xy)) * np.asarray(pitch)
    dist = np.hypot(d[:, 0], d[:, 1])
    dist[~ok] = np.inf
    i = int(np.argmin(dist))
    return int(tm[i]), float(dist[i])


def flip_status(team_a, dist_a, team_b, dist_b, max_dist=None):
    """'flip', 'same', or a not-evaluable reason ('no_player', 'too_far')."""
    if team_a < 0 or team_b < 0:
        return "no_player"
    if max_dist is not None and (dist_a > max_dist or dist_b > max_dist):
        return "too_far"
    return "flip" if team_a != team_b else "same"


def reference_pairs(sample, delta, ball_ok, half_of, team, dist, max_dist=None):
    """(n_pairs, n_flip) over pairs (f, f + delta) for f in sample.

    A pair is kept only if f + delta exists, both frames are graded in the same half
    (half_of > 0 and equal), the ball is present on both, both have a nearest player and,
    with max_dist, both are within it. team/dist are indexable by frame (array or dict).
    """
    n_pairs = n_flip = 0
    n = len(ball_ok)
    for f in sample:
        f = int(f)
        g = f + int(delta)
        if g >= n or g < 0:
            continue
        if half_of[f] <= 0 or half_of[f] != half_of[g] or not (ball_ok[f] and ball_ok[g]):
            continue
        status = flip_status(team[f], dist[f], team[g], dist[g], max_dist)
        if status in ("flip", "same"):
            n_pairs += 1
            n_flip += status == "flip"
    return n_pairs, n_flip


# ---------------------------------------------------------------- driver

def analyse(raw_path, blocks, tags_path=None, fps=FPS, pitch=PITCH, radius=RADIUS_M,
            sample_size=SAMPLE_SIZE, seed=SEED, tag_tolerance_s=0.0, long_gap_s=LONG_GAP_S,
            meta_path=None, flip_max_dist=None, null_n=NULL_N, null_seed=NULL_SEED,
            null_min_shift_s=NULL_MIN_SHIFT_S):
    """All numbers, as a dict. No per-frame data leaves this function."""
    ball, has_record = read_ball(raw_path)
    missing = ~np.isfinite(ball).all(axis=1)
    n = len(missing)

    graded = [(h, a, b) for h, a, b in blocks]
    graded_frames = {h: b - a + 1 for h, a, b in graded}
    no_record = 0
    for _, a, b in graded:
        inside = has_record[max(a, 0):min(b, n - 1) + 1]
        no_record += int(np.sum(~inside)) + (b - a + 1 - len(inside))

    gaps = find_gaps(missing, graded)
    long_frames = int(math.ceil(long_gap_s * fps - 1e-9))
    tol = int(round(tag_tolerance_s * fps))
    res = {"blocks": graded, "graded_frames": graded_frames, "no_record_frames": no_record,
           "fps": fps, "pitch": pitch, "radius": radius, "seed": seed,
           "sample_size": sample_size, "tag_tolerance_s": tag_tolerance_s,
           "long_gap_s": long_gap_s, "n_gaps": len(gaps),
           "edge_gaps": sum(g["edge"] for g in gaps), "flip_max_dist": flip_max_dist,
           "null_n": null_n, "null_seed": null_seed, "null_min_shift_s": null_min_shift_s}

    # a) lengths
    res["lengths"] = {"overall": length_table(gaps, fps)}
    for h, _, _ in graded:
        res["lengths"]["half {}".format(h)] = length_table([g for g in gaps if g["half"] == h], fps)

    # b) set pieces, and e) the same against circular shifts
    long_gaps = [g for g in gaps if g["length"] >= long_frames]
    res["set_pieces"] = res["set_pieces_null"] = None
    if tags_path:
        intervals = read_set_pieces(tags_path, fps)
        hit = [overlaps(g, intervals, tol) for g in long_gaps]
        by_bin = {}
        for g, h in zip(long_gaps, hit):
            cell = by_bin.setdefault(bin_label(g["length"], fps), [0, 0])
            cell[0] += 1
            cell[1] += int(h)
        res["set_pieces"] = {"n_intervals": len(intervals), "n_long": len(long_gaps),
                             "n_overlap": int(sum(hit)),
                             "by_bin": {k: tuple(v) for k, v in by_bin.items()}}
        res["set_pieces_null"] = set_piece_null(long_gaps, hit, intervals, graded, fps, tol,
                                                null_n, null_seed, null_min_shift_s)

    # c) crowding sample (shared with the d reference)
    in_block = np.zeros(n, dtype=bool)
    half_of = np.zeros(n, dtype=int)
    for h, a, b in graded:
        in_block[max(a, 0):min(b, n - 1) + 1] = True
        half_of[max(a, 0):min(b, n - 1) + 1] = h
    visible = np.flatnonzero(in_block & ~missing)
    if sample_size >= len(visible):
        sample = visible
    else:
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(visible, size=sample_size, replace=False))

    # d) reference offsets: one delta per bin = rounded median clipped gap length of that bin
    deltas = {}
    for _, _, label in BINS_S:
        lengths = [g["length"] for g in gaps if bin_label(g["length"], fps) == label]
        if lengths:
            deltas[label] = int(round(float(np.median(lengths))))

    starts = [g for g in gaps if g["last_seen"] is not None]
    wanted = set(int(g["last_seen"]) for g in starts) | set(sample.tolist())
    if meta_path:
        wanted |= set(int(g["next_seen"]) for g in gaps if g["next_seen"] is not None)
        for d in deltas.values():
            wanted |= set(int(f) + d for f in sample if int(f) + d < n)
    slots = read_slots(raw_path, wanted)
    present = {f: pts[np.isfinite(pts).all(axis=1)] for f, pts in slots.items()}

    def count(f):
        return crowd_count(present.get(int(f), np.empty((0, 2))), ball[f], pitch, radius)

    gap_counts = np.array([count(g["last_seen"]) for g in starts], dtype=float)
    is_long = np.array([g["length"] >= long_frames for g in starts], dtype=bool)
    res["crowding"] = {
        "gap_start_all": _stats(gap_counts),
        "gap_start_long": _stats(gap_counts[is_long]) if len(gap_counts) else _stats([]),
        "gap_start_short": _stats(gap_counts[~is_long]) if len(gap_counts) else _stats([]),
        "gaps_without_last_seen": len(gaps) - len(starts),
        "visible": _stats(np.array([count(f) for f in sample], dtype=float)),
        "visible_population": int(len(visible)),
    }

    # d) flips
    res["flip"] = None
    if meta_path:
        specs = read_slot_teams(meta_path)
        near = {}
        for f in wanted:
            if f < n:
                near[f] = nearest_team(slots.get(f), teams_for(specs, f), ball[f], pitch)
        team = {f: v[0] for f, v in near.items()}
        dist = {f: v[1] for f, v in near.items()}
        res["flip"] = flip_tables(gaps, graded, team, dist, flip_max_dist, fps)
        ball_ok = ~missing
        ref = {}
        for label, d in deltas.items():
            for h, _, _ in graded:
                ref[("half {}".format(h), label)] = reference_pairs(
                    [f for f in sample if half_of[f] == h], d, ball_ok, half_of,
                    team, dist, flip_max_dist)
            ref[("overall", label)] = tuple(
                sum(ref[("half {}".format(h), label)][i] for h, _, _ in graded) for i in (0, 1))
        res["flip"]["reference"] = ref
        res["flip"]["deltas"] = deltas
    return res


def flip_tables(gaps, blocks, team, dist, max_dist, fps):
    """{(scope, bin): {n_gaps, n_evaluable, n_flip, not_eval_<reason>...}}."""
    scopes = ["overall"] + ["half {}".format(h) for h, _, _ in blocks]
    bins = ["all"] + [label for _, _, label in BINS_S]
    tables = {(s, b): dict({"n_gaps": 0, "n_evaluable": 0, "n_flip": 0},
                           **{"not_eval_" + r: 0 for r in FLIP_REASONS})
              for s in scopes for b in bins}
    for g in gaps:
        if g["last_seen"] is None:
            status = "no_last_seen"
        elif g["next_seen"] is None:
            status = "no_reappearance"
        else:
            a, b = g["last_seen"], g["next_seen"]
            status = flip_status(team.get(a, -1), dist.get(a, math.nan),
                                 team.get(b, -1), dist.get(b, math.nan), max_dist)
        for key in [(s, b) for s in ("overall", "half {}".format(g["half"]))
                    for b in ("all", bin_label(g["length"], fps))]:
            cell = tables[key]
            cell["n_gaps"] += 1
            if status in ("flip", "same"):
                cell["n_evaluable"] += 1
                cell["n_flip"] += status == "flip"
            else:
                cell["not_eval_" + status] += 1
    return {"tables": tables}


def set_piece_null(long_gaps, real_hits, intervals, blocks, fps, tol, k, seed, min_shift_s):
    """Real vs circular-shift null for gaps >= long, and for the 2-5 s and >= 5 s bins."""
    min_shift = int(round(min_shift_s * fps))
    offsets = draw_offsets(blocks, k, seed, min_shift)
    if offsets is None:
        return {"skipped": "a half is shorter than 2 x min_shift ({:g} s)".format(min_shift_s)}
    matrix = null_overlap_matrix(long_gaps, intervals, blocks, offsets, tol)
    groups = {">=long": list(range(len(long_gaps)))}
    for label in ("2-5s", ">=5s"):
        groups[label] = [j for j, g in enumerate(long_gaps) if bin_label(g["length"], fps) == label]
    out = {"skipped": None, "K": k, "min_shift_s": min_shift_s, "bins": {}}
    for label, idx in groups.items():
        if not idx:
            continue
        real = int(sum(real_hits[j] for j in idx))
        null = matrix[:, idx].sum(axis=1)
        share = 100.0 * null / len(idx)
        p5, p95 = np.percentile(share, [5, 95])
        out["bins"][label] = {"n_gaps": len(idx), "real_count": real,
                              "real_share": 100.0 * real / len(idx),
                              "null_mean": float(share.mean()), "null_p5": float(p5),
                              "null_p95": float(p95),
                              "p_emp": (1.0 + float(np.sum(null >= real))) / (1.0 + k)}
    return out


def _stats(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan}
    return {"n": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values))}


# ---------------------------------------------------------------- output

def _pct(a, b):
    return 100.0 * a / b if b else math.nan


def _rate(a, b):
    return float(a) / b if b else math.nan


def summary_rows(res):
    """Long-format rows: (analysis, scope, bin, metric, value, unit)."""
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    add("params", "all", "", "fps", res["fps"], "frames/s")
    add("params", "all", "", "pitch_length", res["pitch"][0], "m")
    add("params", "all", "", "pitch_width", res["pitch"][1], "m")
    add("params", "all", "", "radius", res["radius"], "m")
    add("params", "all", "", "long_gap_threshold", res["long_gap_s"], "s")
    add("params", "all", "", "sample_size", res["sample_size"], "frames")
    add("params", "all", "", "seed", res["seed"], "")
    add("params", "all", "", "tag_tolerance", res["tag_tolerance_s"], "s")
    add("params", "all", "", "flip_max_dist",
        math.nan if res["flip_max_dist"] is None else res["flip_max_dist"], "m")
    add("params", "all", "", "null_n", res["null_n"], "count")
    add("params", "all", "", "null_seed", res["null_seed"], "")
    add("params", "all", "", "null_min_shift", res["null_min_shift_s"], "s")
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")

    total_graded = sum(res["graded_frames"].values())
    for scope, table in res["lengths"].items():
        graded = total_graded if scope == "overall" else \
            res["graded_frames"][int(scope.split()[1])]
        n_all, miss_all = table["all"]
        add("gap_length", scope, "all", "graded_frames", graded, "frames")
        add("gap_length", scope, "all", "n_gaps", n_all, "count")
        add("gap_length", scope, "all", "missing_frames", miss_all, "frames")
        add("gap_length", scope, "all", "missing_share_of_graded", _pct(miss_all, graded), "%")
        for _, _, label in BINS_S:
            n, miss = table[label]
            add("gap_length", scope, label, "n_gaps", n, "count")
            add("gap_length", scope, label, "missing_frames", miss, "frames")
            add("gap_length", scope, label, "share_of_missing_frames", _pct(miss, miss_all), "%")
    add("gap_length", "overall", "all", "edge_gaps", res["edge_gaps"], "count")
    add("gap_length", "overall", "all", "no_record_frames", res["no_record_frames"], "frames")

    sp = res["set_pieces"]
    if sp is not None:
        add("set_pieces", "overall", ">=long", "n_set_piece_intervals", sp["n_intervals"], "count")
        add("set_pieces", "overall", ">=long", "n_gaps", sp["n_long"], "count")
        add("set_pieces", "overall", ">=long", "n_overlap", sp["n_overlap"], "count")
        add("set_pieces", "overall", ">=long", "share_overlap", _pct(sp["n_overlap"], sp["n_long"]), "%")
        for label in sorted(sp["by_bin"]):
            n, k = sp["by_bin"][label]
            add("set_pieces", "overall", label, "n_gaps", n, "count")
            add("set_pieces", "overall", label, "n_overlap", k, "count")

    c = res["crowding"]
    for scope in ("gap_start_all", "gap_start_long", "gap_start_short", "visible"):
        for metric in ("n", "mean", "median"):
            unit = "count" if metric == "n" else "players"
            add("crowding", scope, "", metric, c[scope][metric], unit)
    add("crowding", "gap_start_all", "", "gaps_without_last_seen", c["gaps_without_last_seen"], "count")
    add("crowding", "visible", "", "population", c["visible_population"], "frames")

    fl = res["flip"]
    if fl is not None:
        for (scope, label), cell in fl["tables"].items():
            for metric in ("n_gaps", "n_evaluable", "n_flip"):
                add("flip", scope, label, metric, cell[metric], "count")
            add("flip", scope, label, "flip_rate", _rate(cell["n_flip"], cell["n_evaluable"]), "ratio")
            for reason in FLIP_REASONS:
                add("flip", scope, label, "not_eval_" + reason, cell["not_eval_" + reason], "count")
            if label != "all":
                ref_n, ref_flip = fl["reference"].get((scope, label), (0, 0))
                add("flip", scope, label, "ref_delta",
                    fl["deltas"].get(label, math.nan), "frames")
                add("flip", scope, label, "ref_n", ref_n, "count")
                add("flip", scope, label, "ref_flip_rate", _rate(ref_flip, ref_n), "ratio")

    nl = res["set_pieces_null"]
    if nl is not None and not nl["skipped"]:
        for label, b in nl["bins"].items():
            add("set_pieces_null", "overall", label, "n_gaps", b["n_gaps"], "count")
            add("set_pieces_null", "overall", label, "real_share", b["real_share"], "%")
            add("set_pieces_null", "overall", label, "null_mean", b["null_mean"], "%")
            add("set_pieces_null", "overall", label, "null_p5", b["null_p5"], "%")
            add("set_pieces_null", "overall", label, "null_p95", b["null_p95"], "%")
            add("set_pieces_null", "overall", label, "p_emp", b["p_emp"], "ratio")
            add("set_pieces_null", "overall", label, "K", nl["K"], "count")
            add("set_pieces_null", "overall", label, "min_shift", nl["min_shift_s"], "s")
    return rows


def print_summary(res, csv_path):
    total_graded = sum(res["graded_frames"].values())
    overall = res["lengths"]["overall"]
    print("ball gaps on {} graded frames in {} block(s): ball missing on {} ({:.1f}%), "
          "{} gaps, {} touch a half edge".format(
              total_graded, len(res["blocks"]), overall["all"][1],
              _pct(overall["all"][1], total_graded), res["n_gaps"], res["edge_gaps"]))
    if res["no_record_frames"]:
        print("  note: {} graded frames have no tracking record (counted as missing)".format(
            res["no_record_frames"]))
    labels = [label for _, _, label in BINS_S]
    print("  gaps / share of missing frames  " + "".join("{:>15}".format(l) for l in labels))
    for scope, table in res["lengths"].items():
        cells = ["{} / {:.1f}%".format(table[l][0], _pct(table[l][1], table["all"][1]))
                 for l in labels]
        print("  {:<31}".format(scope) + "".join("{:>15}".format(c) for c in cells))

    sp = res["set_pieces"]
    if sp is None:
        print("set pieces: no Smart Tagging file, skipped")
    else:
        parts = ", ".join("{} {}/{}".format(k, v[1], v[0]) for k, v in sorted(sp["by_bin"].items()))
        print("set pieces: {} of {} gaps >= {:g} s overlap a SET PIECES interval ({:.1f}%); {}"
              "  [tolerance {:g} s]".format(sp["n_overlap"], sp["n_long"], res["long_gap_s"],
                                            _pct(sp["n_overlap"], sp["n_long"]), parts or "-",
                                            res["tag_tolerance_s"]))
    c = res["crowding"]
    print("crowding (players <= {:g} m of last-seen ball): gap start mean {:.2f} median {:.1f} "
          "(n={}); >= {:g} s {:.2f} (n={}); < {:g} s {:.2f} (n={})".format(
              res["radius"], c["gap_start_all"]["mean"], c["gap_start_all"]["median"],
              c["gap_start_all"]["n"], res["long_gap_s"], c["gap_start_long"]["mean"],
              c["gap_start_long"]["n"], res["long_gap_s"], c["gap_start_short"]["mean"],
              c["gap_start_short"]["n"]))
    print("  vs ball-visible graded frames: mean {:.2f} median {:.1f} (n={} of {}, seed {})".format(
        c["visible"]["mean"], c["visible"]["median"], c["visible"]["n"],
        c["visible_population"], res["seed"]))

    fl = res["flip"]
    if fl is None:
        print("flip: no ATD metadata (--meta), skipped")
    else:
        t = fl["tables"]
        allc = t[("overall", "all")]
        not_eval = sum(allc["not_eval_" + r] for r in FLIP_REASONS)
        reasons = ", ".join("{} {}".format(r, allc["not_eval_" + r]) for r in FLIP_REASONS)
        per_bin = ", ".join("{} {}/{}".format(l, t[("overall", l)]["n_flip"],
                                              t[("overall", l)]["n_evaluable"]) for l in labels)
        print("flip: overall {:.2f} ({}/{}), not evaluable {} [{}]; per bin {}  [max dist {}]".format(
            _rate(allc["n_flip"], allc["n_evaluable"]), allc["n_flip"], allc["n_evaluable"],
            not_eval, reasons, per_bin,
            "off" if res["flip_max_dist"] is None else "{:g} m".format(res["flip_max_dist"])))
        refs = []
        for l in labels:
            if l in fl["deltas"]:
                n_ref, k_ref = fl["reference"][("overall", l)]
                refs.append("{} {:.2f} (n={}, delta={})".format(
                    l, _rate(k_ref, n_ref), n_ref, fl["deltas"][l]))
            else:
                refs.append("{} -".format(l))
        print("  reference (ball-visible pairs delta frames apart): " + ", ".join(refs))

    nl = res["set_pieces_null"]
    if nl is None:
        print("set pieces vs chance: no Smart Tagging file, skipped")
    elif nl["skipped"]:
        print("set pieces vs chance: skipped, {}".format(nl["skipped"]))
    else:
        cells = []
        for label, b in nl["bins"].items():
            cells.append("{} real {:.1f}%, null mean {:.1f}% [{:.1f}, {:.1f}], p_emp {:.3f}".format(
                ">= {:g} s".format(res["long_gap_s"]) if label == ">=long" else label,
                b["real_share"], b["null_mean"], b["null_p5"], b["null_p95"], b["p_emp"]))
        print("set pieces vs chance: " + "; ".join(cells) +
              "  [tolerance {:g} s, K {}, min_shift {:g} s]".format(
                  res["tag_tolerance_s"], nl["K"], nl["min_shift_s"]))
    print("wrote {}".format(csv_path))


def write_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["analysis", "scope", "bin", "metric", "value", "unit"])
        for row in rows:
            value = row[4]
            if isinstance(value, float):
                value = "" if math.isnan(value) else round(value, 6)
            writer.writerow(list(row[:4]) + [value, row[5]])


# ---------------------------------------------------------------- CLI

def repo_root(start=None):
    """Nearest ancestor of `start` (default: this script's folder) that holds a .git entry;
    the script's own folder if there is none."""
    here = os.path.dirname(os.path.abspath(start or __file__))
    d = here
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return here
        d = parent


def check_out_dir(out_dir, example="~/eda_out/ball_gaps"):
    """Exit if out_dir is inside the repository holding this script: outputs never go in-repo."""
    root = os.path.realpath(repo_root())
    target = os.path.realpath(os.path.abspath(out_dir))
    if target == root or target.startswith(root + os.sep):
        sys.exit("refusing to write into {} (inside the repo {}); pass a folder outside it, "
                 "e.g. {}".format(target, root, example))


def parse_frames(text):
    """'a-b,c-d' -> [(1, a, b), (2, c, d)] inclusive video frames."""
    blocks = []
    for i, part in enumerate(text.split(","), 1):
        a, b = part.strip().split("-")
        a, b = int(a), int(b)
        if b < a:
            raise argparse.ArgumentTypeError("bad range {!r}".format(part))
        blocks.append((i, a, b))
    return blocks


def _games():
    try:
        import games
    except ImportError:
        sys.exit("cannot import games.py -- set PYTHONPATH to the team repo root, "
                 "or pass --raw and --frames")
    return games


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (created if missing; must be outside the "
                                   "repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH", help="games.py game id (default BAR-ATH)")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--meta", help="ATD metadata XML for slot teams (d); default "
                   "games.asset_paths()['atd_meta'] when --raw is not given")
    p.add_argument("--tags", help="Smart Tagging XML; default games.asset_paths()"
                   "['smart_tagging'] when --raw is not given")
    p.add_argument("--frames", type=parse_frames,
                   help="graded video frames, one inclusive range per half, e.g. "
                        "'0-69660,69661-143760'; default games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--radius", type=float, default=RADIUS_M, help="metres (default 5)")
    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--long-gap", type=float, default=LONG_GAP_S, help="seconds (default 2)")
    p.add_argument("--tag-tolerance", type=float, default=0.0,
                   help="widen each SET PIECES interval by this many seconds on both sides")
    p.add_argument("--flip-max-dist", type=float, default=None,
                   help="(d) evaluable only if the nearest player is within this many metres "
                        "on both frames (default off)")
    p.add_argument("--null-n", type=int, default=NULL_N, help="(e) shifts (default 1000)")
    p.add_argument("--null-seed", type=int, default=NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=NULL_MIN_SHIFT_S,
                   help="(e) smallest shift in seconds (default 60)")
    args = p.parse_args(argv)
    check_out_dir(args.out_dir)

    if args.raw:
        raw, tags, meta = args.raw, args.tags, args.meta
        if args.frames is None:
            p.error("--frames is required with --raw")
        blocks = args.frames
    else:
        games = _games()
        try:
            raw, meta = games.require(["atd_raw", "atd_meta"], game=args.game)
        except (IOError, OSError, KeyError) as err:
            sys.exit(str(err))
        meta = args.meta or meta
        tags = args.tags
        if not tags:
            try:
                tags = games.require(["smart_tagging"], game=args.game)[0]
            except (IOError, OSError, KeyError):
                print("note: Smart Tagging file not found; (b) and (e) skipped")
                tags = None
        blocks = args.frames or games.half_video_blocks(args.game)

    res = analyse(raw, blocks, tags, fps=args.fps, pitch=tuple(args.pitch), radius=args.radius,
                  sample_size=args.sample_size, seed=args.seed,
                  tag_tolerance_s=args.tag_tolerance, long_gap_s=args.long_gap,
                  meta_path=meta, flip_max_dist=args.flip_max_dist, null_n=args.null_n,
                  null_seed=args.null_seed, null_min_shift_s=args.null_min_shift)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, CSV_NAME)
    write_csv(summary_rows(res), csv_path)
    print_summary(res, csv_path)
    return res


if __name__ == "__main__":
    main()
