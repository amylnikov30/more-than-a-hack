"""Shared helpers for the dead-ball EDA scripts (stdlib + numpy only).

Builds on ball_gaps.py and tag_gaps.py instead of copying them: readers, null machinery and the
CLI helpers stay there; this module adds what the newer scripts share.

    stream_frames   one pass over the raw ATD text, one frame at a time (players and ball with
                    their declared speed); nothing is kept between frames
    graded_half_of  int array by video frame: graded half number, 0 if not graded
    is_step         (previous, current) are consecutive frames graded in the same half
    find_runs       maximal runs of a per-frame condition inside each graded half
    link_points     run ends as point "gaps" for the tag_gaps link test and its null
    set_piece_groups  SET PIECES and each set-piece Type, from tag_gaps.read_tags
    nearest_type    Type of the SET PIECES start nearest to a frame, within a window
    draw_null       circular-shift offsets for the link null, or the reason it is skipped
    player_signals  per frame: share of players below walking pace, number of tracked slots
    read_reprojection  per frame homography reprojection_error (NaN where not valid)
    signal_halves   cumulative sums of any per-frame signal per half, for tag_gaps.profile_counts
    game_paths      asset paths from games.py, with optional ones allowed to be missing
"""

import collections
import csv
import math
import sys

import numpy as np

import ball_gaps as bg
import tag_gaps as tg


def _parse_body(body, n_fields):
    """'x,y[,s];x,y[,s];...' -> float array (k, 3); a missing speed field becomes NaN."""
    values = np.array(body.replace(";", ",").split(","), dtype=float)
    if n_fields >= 3:
        out = values.reshape(-1, n_fields)[:, :3]
    else:
        xy = values.reshape(-1, 2)
        out = np.column_stack([xy, np.full(len(xy), np.nan)])
    return out


def _fields(group):
    return group.count(",") + 1


def stream_frames(raw_path):
    """Yield (video_frame, players, ball) per line of the raw ATD file, in file order.

    players: float array (k, 3) of x, y (normalised) and declared speed per slot, NaN where
    missing; ball: float array (3,). A line that cannot be parsed is skipped.
    """
    with bg._open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) < 3:
                continue
            try:
                frame = int(parts[0]) - 1
                first = parts[1].split(";", 1)[0]
                players = _parse_body(parts[1], _fields(first)) if parts[1] else \
                    np.empty((0, 3))
                ball = _parse_body(parts[-1], _fields(parts[-1]))[0]
            except ValueError:
                continue
            yield frame, players, ball


def graded_half_of(blocks, n=None):
    """Int array by video frame (length n, default last graded frame + 1): the half number of
    each graded frame, 0 elsewhere."""
    if n is None:
        n = max(b for _, _, b in blocks) + 1
    out = np.zeros(n, dtype=int)
    for half, a, b in blocks:
        lo, hi = max(a, 0), min(b, n - 1)
        if hi >= lo:
            out[lo:hi + 1] = half
    return out


def half_at(half_of, frame):
    return int(half_of[frame]) if 0 <= frame < len(half_of) else 0


def is_step(prev_frame, frame, half_of):
    """(prev_frame, frame) are consecutive video frames graded in the same half."""
    return (prev_frame is not None and frame == prev_frame + 1 and
            half_at(half_of, frame) > 0 and half_at(half_of, frame) == half_at(half_of, prev_frame))


def find_runs(mask, blocks):
    """[{half, start, end, length}] maximal runs of True in `mask` inside each graded block,
    clipped to it (inclusive frames). Frames beyond the end of `mask` count as False."""
    mask = np.asarray(mask, dtype=bool)
    out = []
    for half, first, last in blocks:
        seg = np.zeros(last - first + 1, dtype=bool)
        lo, hi = max(first, 0), min(last, len(mask) - 1)
        if hi >= lo:
            seg[lo - first:hi - first + 1] = mask[lo:hi + 1]
        edges = np.diff(np.concatenate([[0], seg.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
            out.append({"half": half, "start": first + int(s), "end": first + int(e),
                        "length": int(e - s + 1)})
    return out


def link_points(runs):
    """Runs as tag_gaps "gaps" whose reappearance is the run's last frame, so tag_gaps.link_hits
    and tag_gaps.link_null_matrix test |end - anchor| <= W."""
    return [{"half": r["half"], "start": r["end"], "end": r["end"], "next_seen": r["end"]}
            for r in runs]


def set_piece_groups(tags):
    """{group: [tag]}: 'code:SET PIECES' first, then 'type:<Type>' (sorted), as tag_gaps names
    them. Other codes are dropped."""
    groups = tg.tag_groups(tags)
    return {k: v for k, v in groups.items()
            if k == "code:" + tg.SET_PIECE_CODE or k.startswith("type:")}


def nearest_type(frame, half, set_pieces, blocks, w_frames):
    """Type of the graded SET PIECES start nearest to `frame` in the same half with
    |start - frame| <= w_frames (ties: the earlier start); '(no Type)' if that tag has none,
    'none' if there is no such start."""
    best = None
    for t in set_pieces:
        if tg.half_of(t["start"], blocks) != half:
            continue
        d = abs(t["start"] - frame)
        if d <= w_frames and (best is None or (d, t["start"]) < best[:2]):
            best = (d, t["start"], t["type"] or "(no Type)")
    return "none" if best is None else best[2]


def draw_null(blocks, null_n, null_seed, null_min_shift_s, fps):
    """(offsets, None) or (None, reason)."""
    if null_n <= 0:
        return None, "null_n is 0"
    offsets = bg.draw_offsets(blocks, null_n, null_seed, int(round(null_min_shift_s * fps)))
    if offsets is None:
        return None, "a half is shorter than 2 x min_shift ({:g} s)".format(null_min_shift_s)
    return offsets, None


def player_signals(raw_path, half_of, pitch, fps, lag, walk_ms):
    """(slow_share, n_tracked), float arrays by video frame (length len(half_of)), NaN off the
    graded frames.

    slow_share[f]: among slots present on f and f - lag (graded in the same half, records for
    both, same slot count), the share whose speed over the lag is below walk_ms; NaN if none.
    n_tracked[f]: slots with finite x and y on f.
    """
    n = len(half_of)
    slow = np.full(n, np.nan)
    tracked = np.full(n, np.nan)
    recent = collections.deque(maxlen=lag + 1)
    for frame, players, _ in stream_frames(raw_path):
        if not 0 <= frame < n or half_of[frame] == 0:
            continue
        xy = players[:, :2]
        tracked[frame] = float(np.isfinite(xy).all(axis=1).sum())
        recent.append((frame, xy))
        prev = recent[0]
        if prev[0] == frame - lag and half_of[prev[0]] == half_of[frame] and \
                len(prev[1]) == len(xy):
            v = step_speed(prev[1], xy, pitch, fps) / lag
            ok = np.isfinite(v)
            if ok.any():
                slow[frame] = float(np.mean(v[ok] < walk_ms))
    return slow, tracked


def read_reprojection(path, n):
    """Float array (n,): reprojection_error per video frame where valid == 1, else NaN."""
    out = np.full(n, np.nan)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame = int(row["frame"])
                if 0 <= frame < n and int(float(row.get("valid", 1))) == 1:
                    out[frame] = float(row["reprojection_error"])
            except (KeyError, ValueError):
                continue
    return out


def signal_halves(values, blocks):
    """{half: (first, L, cs_sum, cs_count)} for tag_gaps.profile_counts. NaN values (and frames
    beyond the array) are undefined: they add to neither sum."""
    values = np.asarray(values, dtype=float)
    out = {}
    for half, first, last in blocks:
        length = last - first + 1
        seg = np.full(length, np.nan)
        lo, hi = max(first, 0), min(last, len(values) - 1)
        if hi >= lo:
            seg[lo - first:hi - first + 1] = values[lo:hi + 1]
        ok = np.isfinite(seg)
        out[half] = (first, length, np.concatenate([[0.0], np.cumsum(np.where(ok, seg, 0.0))]),
                     np.concatenate([[0], np.cumsum(ok)]).astype(np.int64))
    return out


def step_speed(prev_xy, xy, pitch, fps):
    """Derived speed in m/s per row from two (k, 2) normalised position arrays; NaN where either
    position is missing."""
    d = (np.asarray(xy, dtype=float) - np.asarray(prev_xy, dtype=float)) * np.asarray(pitch)
    with np.errstate(invalid="ignore"):
        return np.hypot(d[..., 0], d[..., 1]) * fps


def game_paths(game, required, optional=()):
    """{name: path} from games.require; required names exit on failure, optional ones map to
    None (with a stdout note) when missing."""
    games = bg._games()
    out = {}
    try:
        for name, path in zip(required, games.require(list(required), game=game)):
            out[name] = path
    except (IOError, OSError, KeyError) as err:
        sys.exit(str(err))
    for name in optional:
        try:
            out[name] = games.require([name], game=game)[0]
        except (IOError, OSError, KeyError):
            print("note: {} not found; the parts that need it are skipped".format(name))
            out[name] = None
    return out


def percentiles(values, qs):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [math.nan] * len(qs)
    return [float(v) for v in np.percentile(values, qs)]
