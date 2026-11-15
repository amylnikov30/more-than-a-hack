"""Touches and possession from the ball track: split the match at touches, decide possession on top.

Between two touches the ball moves freely, so each piece of the track is one action by one
player. A ball in flight still belongs to the team that played it, so a change of possession goes
on the new team's FIRST touch. One touch is not control, though: H passes, an A defender clips the
ball, it still reaches an H team-mate, and that is all H. So the touches are first a list of
events, and possession is decided in a separate step on top of them.

  1 touches     per graded half, on each stretch where the ball is visible, in metres: light
                Savitzky-Golay smoothing, then dv(t) = |mean velocity over the next w frames -
                mean velocity over the previous w frames|. A touch is a peak of dv of at least
                --min-dv-ms, peaks at least --min-sep-s apart. Velocity, not heading: heading is
                undefined for a slow ball and misses a kick from rest, a first touch that kills
                the ball and a touch onward in the same direction (the angle is still recorded).
                A step faster than --ball-cap-ms is the tracker switching objects: the stretch is
                split there and, by default, the jump is never a touch (--jump-touch bridges it
                like a gap; the summary reports how far jumps land from a player either way).
                Gaps of at most --short-gap-s are bridged with straight lines; if the ball
                leaves the gap on another line than it entered (velocity change >= --min-dv-ms,
                or a miss >= --gap-miss-m), a touch is placed inside the gap where the two lines
                come closest
  2 attribution closest approach of each team over +-ATTR_HALF frames. The nearest team owns
                the touch if within --max-dist-m; otherwise it is no touch (bounce, post,
                tracker jump, lofted ball). Contested: the other team is also within the limit
                and less than --contest-margin-m further away (inf = the plan's literal "both
                teams inside the limit", which marks ~1 in 4 touches on Sample Game 3)
  3 segments    each stretch from a touch to the next: carry (a player of one team within
                --carry-radius-m on most visible frames), flight, still, out, missing; carries
                of one team in a row are merged and the touches inside are dribbles
  4 control     one value, the last team in control. It switches to B only when B shows control:
                two B touches in a row; a B touch followed by a B carry of >= --min-carry-s; or a
                B touch that slows the ball to <= --receive-ms with the B player within the carry
                radius. The switch goes at B's FIRST touch of that run, never at the moment
                control was confirmed. A contested touch counts only if a carry of >=
                --min-carry-s follows it (for the carrier). A long ball gap clears a pending
                candidate but not the team in control
  5 dead ball   D from --dead-signals (default: ball outside the pitch, ball still on a restart
                mark; "missing" is available but flags ~40% of graded frames). The first touch
                after D, or after a half start, is a restart and sets the team without
                confirmation

Measurements (aggregates only):
  M1 distance from each touch to the nearest player, against random ball-visible frames: two
     clusters (real touches close, noise far) would set --max-dist-m from the dip between them
  M2 how often consecutive touches switch team, and the share of switches that switch back
     after one touch or within --switch-back-s: how much work the control rule has to do
  M3 the share of team switches whose interval (last touch of the old team -> first touch of the
     new one) contains a ball gap, by gap length: the ceiling of a touch-only approach
  M4 switches against ATTACKING / DEFENSIVE TRANSITION tag starts, with the circular-shift null
     of tag_gaps.py, and whether the tag's team is the switch's new team

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/touch_possession/touches.py OUT_DIR

    # any other game in the raw ATD format (with its metadata for the slot teams)
    python touches.py OUT_DIR --raw RAW.txt --meta META.xml --frames 0-69660,69661-143760 \\
        --pitch 105 68

Writes to OUT_DIR (which must be outside the repository):
    touch_possession_summary.csv  aggregates only (long format: analysis, scope, bin, metric,
                                  value, unit); safe to bring back
    touches.csv                   FRAME-INDEXED: one row per touch. Stays on the VM
    touch_labels.csv              FRAME-INDEXED: frame,label from the control rule. Stays on the VM
stdout carries aggregates only. --no-tables skips the two frame-indexed files.

Imports the EDA scripts from ../scripts; stdlib + numpy only.
"""

import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import ball_gaps as bg  # noqa: E402
import ball_outside as bo  # noqa: E402
import ball_stationary as bs  # noqa: E402
import common  # noqa: E402
import tag_gaps as tg  # noqa: E402

FPS = bg.FPS
PITCH = bg.PITCH

# 1 touches
BALL_CAP_MS = 40.0        # as speed_units.py: a faster step is a jump to another object
SG_WINDOW = 7             # frames, odd: light smoothing (0.28 s)
SG_ORDER = 2
DV_HALF = 12              # frames each side for the mean velocity before / after (0.48 s). The
                          # plan said 3-5, but the ATD ball is already low-pass filtered: a kick
                          # spreads over ~0.5 s, and the touch rate only levels off at w ~ 12
                          # (the "timescale" rows); Sample Game 3 is flat from w = 1
TIMESCALE_W = (2, 4, 8, 12, 25)
TIMESCALE_DV = (3.0, 5.0, 8.0, 12.0)
GLITCH_MAX = 2            # a run of <= this many frames next to a jump is a tracker glitch
MIN_DV_MS = 3.0           # touch threshold on |v_after - v_before|
PEAK_FLOOR_MS = 1.0       # weaker peaks are dropped; the ones above feed the dv histogram
MIN_SEP_S = 0.2
ANGLE_MIN_MS = 1.0        # heading is only defined above this speed
SHORT_GAP_S = 0.5         # gaps up to this are bridged with straight lines (hmm_writeup: ~0.5 s)
GAP_MISS_M = 1.0
JUMP_TOUCH = False        # jumps are the tracker switching objects, not touches (see README)
# 2 attribution
ATTR_HALF = 3
MAX_DIST_M = 2.0
CONTEST_MARGIN_M = 0.5    # contested only if the other team is this close to being nearest too
# 3 segments
CARRY_RADIUS_M = 1.5
CARRY_SHARE = 0.6         # share of a segment's visible frames with one team within the radius
CARRY_MAX_MS = 10.0       # a ball faster than this (median) is passed, not carried
MISSING_SHARE = 0.5
OUT_SHARE = 0.5
# 4 control
MIN_CARRY_S = 1.0
RECEIVE_MS = 4.0
# 5 dead ball
DEAD_SIGNALS = ("outside", "still")
ALL_DEAD_SIGNALS = ("missing", "outside", "still")
MERGE_GAP_S = 1.0
MIN_DEAD_S = 1.0
# measurements
SWITCH_BACK_S = 2.0
TAG_CODES = ("ATTACKING TRANSITION", "DEFENSIVE TRANSITION")
LINK_WINDOWS_S = (2.0, 5.0)
HOME_TAG_TEAM = "Barcelona"
SAMPLE_SIZE = 5000
SEED = 0
DIST_BIN_M = 0.25
DIST_MAX_M = 10.0
DV_BIN_MS = 0.5
DV_MAX_MS = 30.0
OFFSET_BIN_S = 0.5
SWEEP_DV = (2.0, 3.0, 4.0, 6.0)
DV_BANDS = ((1.0, 3.0, "1-3"), (3.0, 5.0, "3-5"), (5.0, 10.0, "5-10"), (10.0, math.inf, "10+"))
GAP_BINS_S = ((0.0, 0.5, "<0.5s"), (0.5, 2.0, "0.5-2s"), (2.0, 5.0, "2-5s"),
              (5.0, math.inf, ">=5s"))
SEG_TYPES = ("carry", "flight", "still", "out", "missing")
LABELS = "HA"             # team 0 (first <Team> of the metadata) -> H, team 1 -> A
SUMMARY_NAME = "touch_possession_summary.csv"
TOUCHES_NAME = "touches.csv"
LABELS_NAME = "touch_labels.csv"

DEFAULTS = dict(
    ball_cap_ms=BALL_CAP_MS, sg_window=SG_WINDOW, sg_order=SG_ORDER, dv_half=DV_HALF,
    min_dv_ms=MIN_DV_MS, peak_floor_ms=PEAK_FLOOR_MS, min_sep_s=MIN_SEP_S,
    short_gap_s=SHORT_GAP_S, gap_miss_m=GAP_MISS_M, jump_touch=JUMP_TOUCH, attr_half=ATTR_HALF,
    max_dist_m=MAX_DIST_M, contest_margin_m=CONTEST_MARGIN_M,
    carry_radius_m=CARRY_RADIUS_M, carry_share=CARRY_SHARE, carry_max_ms=CARRY_MAX_MS,
    min_carry_s=MIN_CARRY_S, receive_ms=RECEIVE_MS, dead_signals=DEAD_SIGNALS,
    margin_m=bo.MARGIN_M, still_ms=bs.STILL_MS, min_still_s=bs.MIN_STILL_S, lag=bs.LAG,
    still_gap=bs.STILL_GAP, spot_tol_m=bs.SPOT_TOL_M, band_m=bs.BAND_M,
    min_missing_s=bg.LONG_GAP_S, merge_gap_s=MERGE_GAP_S, min_dead_s=MIN_DEAD_S)


# ---------------------------------------------------------------- reading

def team_info(meta_path):
    """[(team id, name)] for the first two <Team> elements of an EPTS metadata XML, in file order
    (index 0 is the home team, as in ball_gaps.read_slot_teams)."""
    out = []
    for el in ET.parse(meta_path).getroot().iter():
        if bg._tag(el) == "Team":
            name = next((c.text for c in el.iter() if bg._tag(c) == "Name"), None)
            out.append((el.get("id"), (name or "").strip()))
    return out[:2]


def load_match(raw_path, meta_path=None):
    """(ball, players, teams) indexed by video frame (ATD frame_id - 1).

    ball (n, 2) and players (n, K, 2, float32) are normalised positions, NaN where missing or
    where a frame has no record; teams (n, K) int8 is 0 / 1 for the first / second <Team>, -1 for
    a slot without a team (all -1 without metadata).
    """
    specs = bg.read_slot_teams(meta_path) if meta_path else []
    k = max([len(t) for _, _, t in specs] + [1])
    cap = 1 << 16
    ball = np.full((cap, 2), np.nan)
    players = np.full((cap, k, 2), np.nan, dtype=np.float32)
    n = 0
    for frame, pl, b in common.stream_frames(raw_path):
        if frame < 0:
            continue
        if frame >= len(ball):
            new = len(ball)
            while new <= frame:
                new *= 2
            ball = np.vstack([ball, np.full((new - len(ball), 2), np.nan)])
            players = np.concatenate(
                [players, np.full((new - len(players), players.shape[1], 2), np.nan,
                                  dtype=np.float32)])
        if len(pl) > players.shape[1]:
            players = np.concatenate(
                [players, np.full((len(players), len(pl) - players.shape[1], 2), np.nan,
                                  dtype=np.float32)], axis=1)
        if len(pl):
            players[frame, :len(pl)] = pl[:, :2]
        ball[frame] = b[:2]
        n = max(n, frame + 1)
    ball, players = ball[:n], players[:n]
    teams = np.full((n, players.shape[1]), -1, dtype=np.int8)
    for first, last, t in specs:
        lo = max(int(first) - 1, 0)
        hi = n - 1 if last == math.inf else min(int(last) - 1, n - 1)
        if hi >= lo and len(t):
            m = min(len(t), teams.shape[1])
            teams[lo:hi + 1, :m] = t[:m]
    return ball, players, teams


def read_gt(path):
    """Upper-cased label per video frame from a 'frame,label' CSV ('X' where absent)."""
    frames, labels = [], []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                frames.append(int(row[0]))
                labels.append(row[1].strip().upper()[:1] or "X")
    out = np.full(max(frames) + 1 if frames else 0, "X", dtype="<U1")
    out[frames] = labels
    return out


# ---------------------------------------------------------------- 1 touches

def visible_runs(bm, half_of, fps, cap_ms, glitch_max=GLITCH_MAX):
    """([(half, start, end)], n_jumps, glitch): maximal runs of graded frames with the ball
    visible, inclusive, split wherever one frame-to-frame step is faster than cap_ms.

    A run of at most glitch_max frames with a jump at either end is a tracker glitch (the
    tracker on another object for a frame or two): its frames are dropped (`glitch`, a bool mask)
    and the runs recomputed, so the track either side meets across a short gap. n_jumps counts
    the jumps left after that."""
    n = len(bm)
    hv = half_of[:n]
    vis = np.isfinite(bm).all(axis=1) & (hv > 0)
    glitch = np.zeros(n, dtype=bool)
    d = np.diff(bm, axis=0)
    step = np.full(n, np.nan)
    if n > 1:
        step[1:] = np.hypot(d[:, 0], d[:, 1]) * fps
    for _ in range(5):
        link = np.zeros(n, dtype=bool)
        if n > 1:
            link[1:] = vis[1:] & vis[:-1] & (hv[1:] == hv[:-1])
        with np.errstate(invalid="ignore"):
            jump = link & (step > cap_ms)
        cont = link & ~jump
        nxt = np.zeros(n, dtype=bool)
        nxt[:-1] = cont[1:]
        starts = np.flatnonzero(vis & ~cont)
        ends = np.flatnonzero(vis & ~nxt)
        nxt_jump = np.zeros(n, dtype=bool)
        nxt_jump[:-1] = jump[1:]
        bad = (ends - starts + 1 <= glitch_max) & (jump[starts] | nxt_jump[ends])
        if not bad.any():
            break
        for a, b in zip(starts[bad], ends[bad]):
            vis[a:b + 1] = False
            glitch[a:b + 1] = True
    runs = [(int(hv[a]), int(a), int(b)) for a, b in zip(starts, ends)]
    return runs, int(jump.sum()), glitch


def savgol_matrix(window, order):
    """(window, window): row j is the least-squares polynomial of degree `order` through
    `window` samples, evaluated at sample j. The middle row is the Savitzky-Golay filter; the
    others smooth the edges of a run (scipy's mode='interp')."""
    i = np.arange(window, dtype=float)
    return np.array([np.linalg.pinv(np.vander(i - j, order + 1, increasing=True))[0]
                     for j in range(window)])


def smooth(xy, window=SG_WINDOW, order=SG_ORDER):
    """Savitzky-Golay smoothing of an (L, 2) run; a run shorter than the window gets the
    largest odd window that fits (and is returned unchanged below 3 samples)."""
    xy = np.asarray(xy, dtype=float)
    length = len(xy)
    w = min(window, length if length % 2 else length - 1)
    if w < 3:
        return xy.copy()
    o = min(order, w - 1)
    m = savgol_matrix(w, o)
    h = w // 2
    out = np.empty_like(xy)
    for d in range(xy.shape[1]):
        out[h:length - h, d] = np.convolve(xy[:, d], m[h][::-1], mode="valid")
    out[:h] = m[:h] @ xy[:w]
    out[length - h:] = m[w - h:] @ xy[length - w:]
    return out


def touch_signal(xy, w, fps):
    """(dv, v_before, v_after) per sample of a run: mean velocity over the w frames before and
    after (m/s), dv = |v_after - v_before|; NaN where either window leaves the run."""
    length = len(xy)
    vb = np.full((length, 2), np.nan)
    va = np.full((length, 2), np.nan)
    if length > w:
        mean = (xy[w:] - xy[:-w]) * fps / float(w)
        vb[w:] = mean
        va[:-w] = mean
    dv = np.hypot(va[:, 0] - vb[:, 0], va[:, 1] - vb[:, 1])
    return dv, vb, va


def pick_peaks(dv, floor, min_sep):
    """Indices (sorted) of local maxima of dv >= floor, taken highest first, each at least
    min_sep samples from every peak already taken. NaN never peaks. Because the order is by
    height, the peaks >= any higher threshold are exactly those this returns above it."""
    x = np.where(np.isfinite(dv), dv, -np.inf)
    if len(x) == 0:
        return np.array([], dtype=np.int64)
    left = np.concatenate([[-np.inf], x[:-1]])
    right = np.concatenate([x[1:], [-np.inf]])
    cand = np.flatnonzero((x >= floor) & (x > left) & (x >= right))
    order = cand[np.argsort(-x[cand], kind="stable")]
    blocked = np.zeros(len(x), dtype=bool)
    keep = []
    for i in order:
        if blocked[i]:
            continue
        keep.append(i)
        blocked[max(0, i - min_sep + 1):i + min_sep] = True
    return np.array(sorted(keep), dtype=np.int64)


def angle_deg(vb, va, min_speed=ANGLE_MIN_MS):
    """Heading change in degrees between two (k, 2) velocities; NaN if either is slower than
    min_speed (heading undefined)."""
    vb, va = np.atleast_2d(vb), np.atleast_2d(va)
    sb, sa = np.hypot(vb[:, 0], vb[:, 1]), np.hypot(va[:, 0], va[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vb * va).sum(axis=1) / (sb * sa)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    ang[~((sb >= min_speed) & (sa >= min_speed))] = np.nan
    return ang


def build_track(bm, half_of, fps, p):
    """Visible runs, smoothed positions (m) and speed (m/s) per frame, run index per frame."""
    n = len(bm)
    runs, n_jumps, _ = visible_runs(bm, half_of, fps, p["ball_cap_ms"])
    sm = np.full((n, 2), np.nan)
    speed = np.full(n, np.nan)
    run_of = np.full(n, -1, dtype=np.int64)
    for k, (_, a, b) in enumerate(runs):
        xy = smooth(bm[a:b + 1], p["sg_window"], p["sg_order"])
        sm[a:b + 1] = xy
        run_of[a:b + 1] = k
        if b > a:
            v = np.gradient(xy, axis=0) * fps
            speed[a:b + 1] = np.hypot(v[:, 0], v[:, 1])
    return {"runs": runs, "n_jumps": n_jumps, "sm": sm, "speed": speed, "run_of": run_of,
            "run_start": np.array([a for _, a, _ in runs], dtype=np.int64),
            "run_end": np.array([b for _, _, b in runs], dtype=np.int64)}


CAND_KEYS = ("half", "frame", "source", "dv", "sp_b", "sp_a", "angle", "x", "y", "miss",
             "gap_len")


def _cols(rows=None):
    return {k: [] for k in CAND_KEYS} if rows is None else rows


def _arrays(cols):
    out = {}
    for k in CAND_KEYS:
        dtype = np.int64 if k in ("half", "frame", "source", "gap_len") else float
        out[k] = np.array(cols[k], dtype=dtype)
    return out


def detect_peaks(track, fps, p):
    """Visible-run peaks of dv >= peak_floor (source 0) as candidate columns."""
    w = int(p["dv_half"])
    sep = max(1, int(round(p["min_sep_s"] * fps)))
    cols = _cols()
    for half, a, b in track["runs"]:
        xy = track["sm"][a:b + 1]
        dv, vb, va = touch_signal(xy, w, fps)
        idx = pick_peaks(dv, p["peak_floor_ms"], sep)
        if not len(idx):
            continue
        ang = angle_deg(vb[idx], va[idx])
        for j, i in enumerate(idx):
            cols["half"].append(half)
            cols["frame"].append(a + int(i))
            cols["source"].append(0)
            cols["dv"].append(float(dv[i]))
            cols["sp_b"].append(float(np.hypot(*vb[i])))
            cols["sp_a"].append(float(np.hypot(*va[i])))
            cols["angle"].append(float(ang[j]))
            cols["x"].append(float(xy[i, 0]))
            cols["y"].append(float(xy[i, 1]))
            cols["miss"].append(math.nan)
            cols["gap_len"].append(0)
    return _arrays(cols)


def gap_candidates(track, fps, p):
    """Straight-line bridges between consecutive visible runs of a half, and their lines for
    attribution: over a short gap (source 1: 1 to short_gap_s of missing ball) or over a jump
    (source 2: a step faster than ball_cap_ms split the runs, nothing missing).

    Each side needs at least one more frame in its run: v_in from the last min(w, available)
    frames before, v_out from the first ones after. dv = |v_out - v_in|, miss = distance
    between where the entry line puts the ball at the exit frame and where it is (for a jump:
    the jump). A gap's candidate frame is where the two lines come closest (the middle of the
    gap when they are parallel), clipped into the gap; a jump's is the first frame after it.
    Returns (columns, lines, counts)."""
    sm, runs = track["sm"], track["runs"]
    short = int(round(p["short_gap_s"] * fps))
    w = int(p["dv_half"])
    cols, lines = _cols(), []
    counts = {"short": 0, "bridged": 0, "jumps": 0, "jumps_bridged": 0}
    for (h1, s1, a), (h2, b, e2) in zip(runs[:-1], runs[1:]):
        g = b - a - 1
        if h1 != h2 or g > short:
            continue
        jump = g == 0
        counts["jumps" if jump else "short"] += 1
        k1, k2 = min(w, a - s1), min(w, e2 - b)
        if k1 < 1 or k2 < 1:
            continue
        counts["jumps_bridged" if jump else "bridged"] += 1
        p1, p2 = sm[a], sm[b]
        v_in = (sm[a] - sm[a - k1]) * fps / float(k1)
        v_out = (sm[b + k2] - sm[b]) * fps / float(k2)
        dt = (b - a) / float(fps)
        d = v_in - v_out
        if jump:
            f, pos = b, p2
        else:
            c = p1 - p2 + v_out * dt
            dd = float(d @ d)
            s = -float(c @ d) / dd if dd > 1e-9 else dt / 2.0
            f = int(min(max(a + int(round(s * fps)), a + 1), b - 1))
            s = (f - a) / float(fps)
            pos = 0.5 * (p1 + v_in * s + p2 + v_out * (s - dt))
        cols["half"].append(h1)
        cols["frame"].append(f)
        cols["source"].append(2 if jump else 1)
        cols["dv"].append(float(np.hypot(*d)))
        cols["sp_b"].append(float(np.hypot(*v_in)))
        cols["sp_a"].append(float(np.hypot(*v_out)))
        cols["angle"].append(float(angle_deg(v_in, v_out)[0]))
        cols["x"].append(float(pos[0]))
        cols["y"].append(float(pos[1]))
        cols["miss"].append(float(np.hypot(*(p1 + v_in * dt - p2))))
        cols["gap_len"].append(g)
        lines.append((a, b, p1, v_in, p2, v_out))
    return _arrays(cols), lines, counts


def windows_visible(frames, track, h):
    """(T, 2h + 1, 2) smoothed ball positions around each frame, clipped to its run."""
    offs = np.arange(-h, h + 1)
    run = track["run_of"][frames]
    idx = np.clip(frames[:, None] + offs, track["run_start"][run][:, None],
                  track["run_end"][run][:, None])
    return track["sm"][idx]


def windows_gap(frames, lines, h, fps):
    """(T, 2h + 1, 2) ball positions on the bridge lines: the entry line up to the candidate
    frame (and never at or after the first frame after the gap), the exit line after it."""
    out = np.empty((len(frames), 2 * h + 1, 2))
    for t, (f, (a, b, p1, v_in, p2, v_out)) in enumerate(zip(frames, lines)):
        for j, o in enumerate(range(-h, h + 1)):
            g = f + o
            s = (g - a) / float(fps)
            out[t, j] = p1 + v_in * s if (g <= f and g < b) else \
                p2 + v_out * (s - (b - a) / float(fps))
    return out


# ---------------------------------------------------------------- 2 attribution

def closest_approach(frames, ball_win, pm, teams, first, last, chunk=2000):
    """(T, 2): per team, the smallest player-ball distance (m) over the window frames; inf if
    the team has no player there. ball_win (T, W, 2) is the ball on frames + (-h..h), clipped to
    [first, last] per row (the half)."""
    frames = np.asarray(frames, dtype=np.int64)
    t_total, width = ball_win.shape[:2]
    h = width // 2
    offs = np.arange(-h, h + 1)
    out = np.full((t_total, 2), np.inf)
    for lo in range(0, t_total, chunk):
        sl = slice(lo, lo + chunk)
        idx = np.clip(frames[sl, None] + offs, first[sl, None], last[sl, None])
        diff = pm[idx] - ball_win[sl, :, None, :]
        d = np.sqrt((diff ** 2).sum(axis=-1))
        d = np.where(np.isfinite(d), d, np.inf)
        tm = teams[idx]
        for k in (0, 1):
            out[sl, k] = np.where(tm == k, d, np.inf).min(axis=(1, 2)) if d.size else np.inf
    return out


def attribution(d_team, max_dist, margin=math.inf):
    """(team, dist, d_other, contested): nearest team within max_dist (else -1), its distance
    (NaN if no player at all), the other team's distance; contested when the other team is also
    within max_dist and less than `margin` further than the nearest (margin inf: both teams
    within the limit is enough, the plan's literal rule)."""
    dist = d_team.min(axis=1)
    near = d_team.argmin(axis=1)
    other = d_team.max(axis=1)
    team = np.where(dist <= max_dist, near, -1)
    with np.errstate(invalid="ignore"):
        contested = (dist <= max_dist) & (other <= max_dist) & (other - dist < margin)
    return (team.astype(np.int64), np.where(np.isfinite(dist), dist, np.nan),
            np.where(np.isfinite(other), other, np.nan), contested)


def frame_nearest(frames, xy, pm, teams, chunk=20000):
    """(len(frames), 2): per team, the nearest player's distance (m) to xy on that frame."""
    frames = np.asarray(frames, dtype=np.int64)
    out = np.full((len(frames), 2), np.inf)
    for lo in range(0, len(frames), chunk):
        sl = slice(lo, lo + chunk)
        diff = pm[frames[sl]] - xy[sl, None, :]
        d = np.sqrt((diff ** 2).sum(axis=-1))
        d = np.where(np.isfinite(d), d, np.inf)
        tm = teams[frames[sl]]
        for k in (0, 1):
            out[sl, k] = np.where(tm == k, d, np.inf).min(axis=1) if d.size else np.inf
    return out


# ---------------------------------------------------------------- 5 dead ball

def dead_intervals(ball, blocks, fps, pitch, p):
    """([{half, start, end, length, signals}], {signal: frames}) of the dead-ball mask.

    outside: visible ball more than margin_m beyond a line (ball_outside.excess); still: a still
    run (ball_stationary) of >= min_still_s whose median position is a restart mark (anything
    but 'elsewhere'); missing: a ball gap of >= min_missing_s. Runs of the union at most
    merge_gap_s apart are merged; intervals under min_dead_s are dropped.
    """
    n = len(ball)
    half_of = common.graded_half_of(blocks, n)[:n]
    graded = half_of > 0
    visible = np.isfinite(ball).all(axis=1)
    masks = {}
    if "outside" in p["dead_signals"]:
        ex, ey = bo.excess(ball, pitch)
        with np.errstate(invalid="ignore"):
            masks["outside"] = visible & graded & (np.maximum(ex, ey) > p["margin_m"])
    if "still" in p["dead_signals"]:
        speed = bs.ball_speed(ball, half_of, pitch, fps, p["lag"])
        frames = bs.still_mask(speed, visible, p["lag"], p["still_ms"]) & graded
        still = np.zeros(n, dtype=bool)
        min_still = int(math.ceil(p["min_still_s"] * fps - 1e-9))
        for r in bs.merge_runs(common.find_runs(frames, blocks), p["still_gap"]):
            if r["length"] < min_still:
                continue
            seg = ball[r["start"]:r["end"] + 1]
            seg = seg[np.isfinite(seg).all(axis=1)]
            if not len(seg):
                continue
            loc = bs.classify(float(np.median(seg[:, 0])) * pitch[0],
                              float(np.median(seg[:, 1])) * pitch[1], pitch, p["spot_tol_m"],
                              p["band_m"])
            if loc != "elsewhere":
                still[r["start"]:r["end"] + 1] = True
        masks["still"] = still
    if "missing" in p["dead_signals"]:
        miss = np.zeros(n, dtype=bool)
        min_gap = int(math.ceil(p["min_missing_s"] * fps - 1e-9))
        for g in bg.find_gaps(~visible, blocks):
            if g["length"] >= min_gap:
                miss[g["start"]:g["end"] + 1] = True
        masks["missing"] = miss & graded
    union = np.zeros(n, dtype=bool)
    for m in masks.values():
        union |= m
    merge = int(round(p["merge_gap_s"] * fps))
    min_len = int(math.ceil(p["min_dead_s"] * fps - 1e-9))
    out = []
    for r in bs.merge_runs(common.find_runs(union, blocks), merge):
        if r["length"] >= min_len:
            sl = slice(r["start"], r["end"] + 1)
            out.append(dict(r, signals="+".join(s for s in ALL_DEAD_SIGNALS
                                                if s in masks and masks[s][sl].any())))
    return out, {s: int(m.sum()) for s, m in masks.items()}


def interval_mask(intervals, n):
    m = np.zeros(n, dtype=bool)
    for iv in intervals:
        m[iv["start"]:iv["end"] + 1] = True
    return m


# ---------------------------------------------------------------- 3 segments

def segments(T, blocks, dead, track, close, outside, gaps, fps, p):
    """Per touch (rows of T, sorted by frame, outside dead ball), the stretch [frame, end) to
    the next touch of the half, the next dead-ball start or the half end: type, length,
    carry team; then consecutive carries of one team are merged (carry_s) and the touches
    inside them flagged as dribbles. Adds the columns to T in place."""
    n = len(track["sm"])
    k = len(T["frame"])
    last_of = {h: b for h, _, b in blocks}
    dead_starts = np.array(sorted(iv["start"] for iv in dead), dtype=np.int64)
    vis = track["run_of"] >= 0
    cs = {name: np.concatenate([[0], np.cumsum(arr)]) for name, arr in
          (("vis", vis), ("out", vis & outside), ("c0", vis & (close == 0)),
           ("c1", vis & (close == 1)))}
    g_start = np.array([g["start"] for g in gaps], dtype=np.int64)
    g_end = np.array([g["end"] for g in gaps], dtype=np.int64)
    short = int(round(p["short_gap_s"] * fps))

    seg_end = np.zeros(k, dtype=np.int64)
    seg_type = np.empty(k, dtype=object)
    carry_team = np.full(k, -1, dtype=np.int64)
    longest_gap = np.zeros(k, dtype=np.int64)
    for i in range(k):
        f, h = int(T["frame"][i]), int(T["half"][i])
        end = last_of[h] + 1
        if i + 1 < k and T["half"][i + 1] == h:
            end = min(end, int(T["frame"][i + 1]))
        j = np.searchsorted(dead_starts, f, side="right")
        if j < len(dead_starts):
            end = min(end, int(dead_starts[j]))
        end = max(end, f + 1)
        seg_end[i] = end
        a, b = f, min(end, n)
        n_f = b - a
        n_vis = cs["vis"][b] - cs["vis"][a]
        lo = np.searchsorted(g_end, a, side="left")
        hi = np.searchsorted(g_start, b, side="left")
        longest = 0
        for gi in range(lo, hi):
            longest = max(longest, min(g_end[gi], b - 1) - max(g_start[gi], a) + 1)
        longest_gap[i] = longest
        if n_f <= 0 or n_vis < MISSING_SHARE * n_f or n_vis == 0:
            seg_type[i] = "missing"
            continue
        if (cs["out"][b] - cs["out"][a]) >= OUT_SHARE * n_vis:
            seg_type[i] = "out"
            continue
        med = float(np.nanmedian(track["speed"][a:b])) if np.isfinite(
            track["speed"][a:b]).any() else math.nan
        if np.isfinite(med) and med < p["still_ms"]:
            seg_type[i] = "still"
            continue
        c = [cs["c0"][b] - cs["c0"][a], cs["c1"][b] - cs["c1"][a]]
        t = int(np.argmax(c))
        if c[t] >= p["carry_share"] * n_vis and (not np.isfinite(med) or
                                                 med <= p["carry_max_ms"]):
            seg_type[i] = "carry"
            carry_team[i] = t
            continue
        seg_type[i] = "flight"
    seg_s = (seg_end - T["frame"]) / float(fps)

    carry_s = np.where(seg_type == "carry", seg_s, 0.0)
    dribble = np.zeros(k, dtype=bool)
    for i in range(k - 2, -1, -1):
        j = i + 1
        if (seg_type[i] == "carry" and seg_type[j] == "carry" and
                carry_team[j] == carry_team[i] and T["half"][j] == T["half"][i] and
                seg_end[i] == T["frame"][j] and
                (T["team"][j] == carry_team[i] or T["contested"][j])):
            carry_s[i] = seg_s[i] + carry_s[j]
            dribble[j] = True
    T.update(seg_type=seg_type, seg_end=seg_end, seg_s=seg_s, carry_team=carry_team,
             carry_s=carry_s, dribble=dribble, seg_longest_gap=longest_gap,
             seg_long_gap=longest_gap > short)
    return T


# ---------------------------------------------------------------- 4 control

def control_rule(T, dead, long_gaps, blocks, p, naive=False):
    """Possession switches from the touch list (T sorted by frame).

    Returns [{half, frame, team, kind, confirm, first, confirm_touch, prev, prev_team}]: kind
    'restart' (first touch after a half start or dead ball) or 'turnover'; `first` is the touch
    the switch is placed at, `confirm_touch` the one that confirmed it, `prev` the last touch of
    the team that had the ball (None for a restart). naive=True switches on every attributed,
    uncontested touch ("one touch = possession")."""
    items = [(int(f), 1, "touch", i) for i, f in enumerate(T["frame"])
             if T["team"][i] >= 0 and not T["in_dead"][i]]
    items += [(iv["start"], 0, "reset", None) for iv in dead]
    items += [(g["start"], 0, "gap", None) for g in long_gaps]
    items += [(first, -1, "reset", None) for _, first, _ in blocks]
    items.sort(key=lambda it: (it[0], it[1]))
    out = []
    ctrl, fresh, cand, last_ctrl = None, True, None, None
    for _, _, kind, i in items:
        if kind == "reset":
            ctrl, fresh, cand, last_ctrl = None, True, None, None
            continue
        if kind == "gap":
            cand = None
            continue
        team, contested = int(T["team"][i]), bool(T["contested"][i])
        carry = int(T["carry_team"][i]) if T["carry_s"][i] >= p["min_carry_s"] else -1
        if contested:
            if naive or carry < 0:
                continue
            eff = carry
        else:
            eff = team
        if fresh:
            out.append(dict(half=int(T["half"][i]), frame=int(T["frame"][i]), team=eff,
                            kind="restart", confirm="restart", first=i, confirm_touch=i,
                            prev=None, prev_team=None))
            ctrl, fresh, cand, last_ctrl = eff, False, None, i
            continue
        if eff == ctrl:
            cand, last_ctrl = None, i
            continue
        if naive:
            confirm, first = "touch", i
        elif cand is not None and cand["team"] == eff:
            confirm, first = "two_touches", cand["first"]
        else:
            cand, first = {"team": eff, "first": i}, i
            if carry == eff:
                confirm = "carry"
            elif (not contested and T["sp_a"][i] <= p["receive_ms"] and
                  T["sp_b"][i] > T["sp_a"][i] and T["dist"][i] <= p["carry_radius_m"]):
                confirm = "receive"
            else:
                confirm = None
        if confirm:
            out.append(dict(half=int(T["half"][first]), frame=int(T["frame"][first]), team=eff,
                            kind="turnover", confirm=confirm, first=first, confirm_touch=i,
                            prev=last_ctrl, prev_team=ctrl))
            ctrl, cand, last_ctrl = eff, None, i
    return out


def team_frames(n, blocks, switches, dead):
    """Int array by frame: team in possession from each switch to the next switch, the next
    dead-ball start or the half end; -1 elsewhere."""
    out = np.full(n, -1, dtype=np.int64)
    dead_starts = np.array(sorted(iv["start"] for iv in dead), dtype=np.int64)
    last_of = {h: b for h, _, b in blocks}
    sw = sorted(switches, key=lambda s: s["frame"])
    for j, s in enumerate(sw):
        end = last_of[s["half"]] + 1
        if j + 1 < len(sw) and sw[j + 1]["half"] == s["half"]:
            end = min(end, sw[j + 1]["frame"])
        k = np.searchsorted(dead_starts, s["frame"], side="right")
        if k < len(dead_starts):
            end = min(end, int(dead_starts[k]))
        out[s["frame"]:min(end, n)] = s["team"]
    return out


def spanned_gaps(long_gaps, switches, T):
    """Indices of the long gaps a turnover spans: its last touch of the old team is before the
    gap and the new team's first touch after it (the switch happened somewhere inside)."""
    turn = [(int(T["frame"][s["prev"]]), s["frame"]) for s in switches
            if s["kind"] == "turnover" and s["prev"] is not None]
    return {i for i, g in enumerate(long_gaps)
            if any(a < g["start"] and b > g["end"] for a, b in turn)}


def make_labels(n, blocks, switches, dead, long_gaps, differs=()):
    """(labels, gap_counts): 'H'/'A' from team_frames, 'D' on dead-ball frames, 'X' elsewhere.
    A long gap in `differs` (spanned by a turnover) keeps the old team up to the new team's
    first touch, which is what the scorer's fill would do anyway; the plan leaves where the
    switch goes inside it to the ball-free features or the HMM. Any other long gap with the
    same team on both sides becomes 'X' (free on spellF1 inside a possession, a miss on
    macroF1)."""
    team = team_frames(n, blocks, switches, dead)
    labels = np.full(n, "X", dtype="<U1")
    for t, ch in enumerate(LABELS):
        labels[team == t] = ch
    labels[interval_mask(dead, n)] = "D"
    counts = {"same": 0, "differs": 0, "unknown": 0}
    for i, g in enumerate(long_gaps):
        if i in differs:
            counts["differs"] += 1
            continue
        before = team[g["start"] - 1] if g["start"] > 0 else -1
        after = team[g["end"] + 1] if g["end"] + 1 < n else -1
        if before < 0 or after < 0:
            counts["unknown"] += 1
        elif before == after:
            counts["same"] += 1
            sl = slice(g["start"], g["end"] + 1)
            labels[sl] = np.where(labels[sl] == "D", "D", "X")
        else:
            counts["differs"] += 1
    return labels, counts


def fill_x(labels, blocks):
    """The scorer's view: each X forward-filled with the last non-X label of its half
    (back-filled at the head); a half with no label at all stays X."""
    out = labels.copy()
    for _, a, b in blocks:
        seg = out[a:b + 1]
        idx = np.where(seg != "X", np.arange(len(seg)), -1)
        np.maximum.accumulate(idx, out=idx)
        ok = idx >= 0
        if not ok.any():
            continue
        seg[ok] = seg[idx[ok]]
        seg[~ok] = seg[int(np.flatnonzero(ok)[0])]
    return out


def possession_runs(labels, blocks):
    """[(half, start, end, label)] maximal runs of H or A (inclusive) inside each half."""
    out = []
    for h, a, b in blocks:
        seg = labels[a:b + 1]
        if not len(seg):
            continue
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change - 1, [len(seg) - 1]])
        for s, e in zip(starts, ends):
            if seg[s] in ("H", "A"):
                out.append((h, a + int(s), a + int(e), str(seg[s])))
    return out


# ---------------------------------------------------------------- the whole pipeline

def pipeline(ball, players, teams, blocks, fps=FPS, pitch=PITCH, dead=None, **kw):
    """Steps 1-5 on arrays (load_match). `dead` overrides the dead-ball intervals. Returns a
    dict with the track, every candidate (C), the touches (T, sorted, with attribution and
    segments), the dead intervals, long gaps, switches (control and naive) and labels."""
    p = dict(DEFAULTS)
    p.update(kw)
    n = len(ball)
    half_of = common.graded_half_of(blocks, n)[:n]
    _, _, glitch = visible_runs(ball * np.asarray(pitch), half_of, fps, p["ball_cap_ms"])
    if glitch.any():
        ball = ball.copy()
        ball[glitch] = np.nan
    bm = ball * np.asarray(pitch)
    pm = players * np.asarray(pitch, dtype=np.float32)
    track = build_track(bm, half_of, fps, p)
    track["n_glitch"] = int(glitch.sum())
    missing = ~np.isfinite(bm).all(axis=1)
    gaps = bg.find_gaps(missing, blocks)

    peaks = detect_peaks(track, fps, p)
    gapc, lines, bridge = gap_candidates(track, fps, p)
    C = {k: np.concatenate([peaks[k], gapc[k]]) for k in CAND_KEYS}
    h = int(p["attr_half"])
    win = np.concatenate([windows_visible(peaks["frame"], track, h).reshape(-1, 2 * h + 1, 2),
                          windows_gap(gapc["frame"], lines, h, fps).reshape(-1, 2 * h + 1, 2)])
    bounds = {hh: (a, b) for hh, a, b in blocks}
    first = np.array([bounds[x][0] for x in C["half"]], dtype=np.int64)
    last = np.array([min(bounds[x][1], n - 1) for x in C["half"]], dtype=np.int64)
    d_team = closest_approach(C["frame"], win, pm, teams, first, last)
    C["d_home"], C["d_away"] = d_team[:, 0], d_team[:, 1]
    C["team"], C["dist"], C["d_other"], C["contested"] = attribution(
        d_team, p["max_dist_m"], p["contest_margin_m"])
    bridged = (C["dv"] >= p["min_dv_ms"]) | (C["miss"] >= p["gap_miss_m"])
    C["is_touch"] = np.where(C["source"] == 0, C["dv"] >= p["min_dv_ms"],
                             np.where(C["source"] == 1, bridged, bridged & bool(p["jump_touch"])))

    if dead is None:
        dead, dead_frames = dead_intervals(ball, blocks, fps, pitch, p)
    else:
        dead_frames = {}
    dmask = interval_mask(dead, n)

    keep = np.flatnonzero(C["is_touch"] & (C["team"] >= 0))
    keep = keep[np.argsort(C["frame"][keep], kind="stable")]
    T = {k: v[keep] for k, v in C.items()}
    T["cand"] = keep
    # the restart kick sits on the last frames of a still run, so the last `lag` frames of a
    # dead interval do not swallow touches
    core = interval_mask([dict(iv, end=iv["end"] - int(p["lag"])) for iv in dead], n)
    T["in_dead"] = core[T["frame"]]
    n_in_dead = int(T["in_dead"].sum())
    live = np.flatnonzero(~T["in_dead"])
    T = {k: v[live] for k, v in T.items()}
    T["in_dead"] = np.zeros(len(live), dtype=bool)

    vis_frames = np.flatnonzero(track["run_of"] >= 0)
    near = frame_nearest(vis_frames, track["sm"][vis_frames], pm, teams)
    close = np.full(n, -1, dtype=np.int64)
    r = p["carry_radius_m"]
    c0, c1 = near[:, 0] <= r, near[:, 1] <= r
    close[vis_frames[c0 & ~c1]] = 0
    close[vis_frames[c1 & ~c0]] = 1
    close[vis_frames[c0 & c1]] = 2
    ex, ey = bo.excess(ball, pitch)
    with np.errstate(invalid="ignore"):
        outside = np.maximum(ex, ey) > p["margin_m"]
    segments(T, blocks, dead, track, close, outside, gaps, fps, p)

    short = int(round(p["short_gap_s"] * fps))
    long_gaps = [g for g in gaps if g["length"] > short and not dmask[g["start"]:g["end"] + 1].all()]
    sw = control_rule(T, dead, long_gaps, blocks, p)
    sw_naive = control_rule(T, dead, long_gaps, blocks, p, naive=True)
    labels, gap_counts = make_labels(n, blocks, sw, dead, long_gaps,
                                     spanned_gaps(long_gaps, sw, T))
    labels_naive, _ = make_labels(n, blocks, sw_naive, dead, long_gaps,
                                  spanned_gaps(long_gaps, sw_naive, T))
    return {"p": p, "n": n, "fps": fps, "pitch": pitch, "blocks": list(blocks),
            "half_of": half_of, "track": track, "missing": missing, "gaps": gaps,
            "bridge": bridge, "C": C, "T": T, "n_in_dead": n_in_dead, "dead": dead,
            "dead_frames": dead_frames,
            "long_gaps": long_gaps, "switches": sw, "switches_naive": sw_naive,
            "labels": labels, "labels_naive": labels_naive, "gap_counts": gap_counts,
            "pm": pm, "teams": teams, "close": close}


# ---------------------------------------------------------------- measurements

def timescale(track, fps, ws=TIMESCALE_W, dvs=TIMESCALE_DV, min_sep_s=MIN_SEP_S):
    """{w: {dv: peaks per visible minute}}: dv peaks (min_sep_s apart) at each window
    half-width w. On a raw track the rate is flat in w; on a low-pass filtered one it grows
    until w spans the filter, which is where the touch detector's window belongs."""
    sep = max(1, int(round(min_sep_s * fps)))
    vis_min = sum(b - a + 1 for _, a, b in track["runs"]) / float(fps) / 60.0
    out = {}
    for w in ws:
        cnt = dict.fromkeys(dvs, 0)
        for _, a, b in track["runs"]:
            dv, _, _ = touch_signal(track["sm"][a:b + 1], w, fps)
            pk = pick_peaks(dv, min(dvs), sep)
            for t in dvs:
                cnt[t] += int((dv[pk] >= t).sum())
        out[w] = {t: c / vis_min if vis_min else math.nan for t, c in cnt.items()}
    return out


def hist(values, edges):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), bins=edges)
    return counts


def dip(counts, edges, first_mode_max=2.5, min_sep=1.0):
    """(mode1, dip, mode2, bimodal) in metres from a distance histogram: mode1 = peak of the
    3-bin moving average below first_mode_max, mode2 = its peak at least min_sep further out,
    dip = the minimum between them; bimodal if the dip is at most 75 % of the lower mode."""
    c = np.convolve(np.asarray(counts, dtype=float), np.ones(3) / 3.0, mode="same")
    mid = 0.5 * (edges[:-1] + edges[1:])
    lo = np.flatnonzero(mid < first_mode_max)
    if not len(lo) or c.sum() == 0:
        return math.nan, math.nan, math.nan, False
    i1 = int(lo[np.argmax(c[lo])])
    far = np.flatnonzero(mid >= mid[i1] + min_sep)
    far = far[far < len(c) - 1]           # the last bin holds the overflow
    if not len(far):
        return float(mid[i1]), math.nan, math.nan, False
    i2 = int(far[np.argmax(c[far])])
    if i2 <= i1 + 1:
        return float(mid[i1]), math.nan, math.nan, False
    j = i1 + int(np.argmin(c[i1:i2 + 1]))
    bimodal = c[j] <= 0.75 * min(c[i1], c[i2]) and j not in (i1, i2)
    return float(mid[i1]), float(mid[j]), float(mid[i2]), bool(bimodal)


def gap_between(a, b, gaps):
    """(missing frames, longest gap) strictly between frames a and b (a < b): overlap of the
    ball gaps with [a + 1, b - 1]."""
    total = longest = 0
    for g in gaps:
        lo, hi = max(g["start"], a + 1), min(g["end"], b - 1)
        if hi >= lo:
            total += hi - lo + 1
            longest = max(longest, hi - lo + 1)
    return total, longest


def gap_class(longest, short):
    return "no_gap" if longest == 0 else ("short_gap" if longest <= short else "long_gap")


def gap_bin(frames, fps):
    s = frames / float(fps)
    for lo, hi, label in GAP_BINS_S:
        if lo <= s < hi:
            return label
    return GAP_BINS_S[-1][2]


class GapIndex(object):
    """Ball gaps sorted by start, for gap_between on many intervals."""

    def __init__(self, gaps):
        self.gaps = sorted(gaps, key=lambda g: g["start"])
        self.ends = np.array([g["end"] for g in self.gaps], dtype=np.int64)
        self.starts = np.array([g["start"] for g in self.gaps], dtype=np.int64)

    def between(self, a, b):
        lo = int(np.searchsorted(self.ends, a + 1, side="left"))
        hi = int(np.searchsorted(self.starts, b - 1, side="right"))
        return gap_between(a, b, self.gaps[lo:hi])


def consecutive(T, dead, long_gaps, blocks, fps, switch_back_s=SWITCH_BACK_S):
    """M2 and the raw half of M3: consecutive uncontested touches of the same block (no half
    start, dead-ball start or long gap between them). Returns {n_touches, n_pairs, n_switch,
    back_1, back_2s, n_back_eval, switch_pairs [(frame_a, frame_b)]}."""
    use = np.flatnonzero((T["team"] >= 0) & ~T["contested"] & ~T["in_dead"])
    frames, team = T["frame"][use], T["team"][use]
    breaks = np.array(sorted([iv["start"] for iv in dead] + [g["start"] for g in long_gaps] +
                             [a for _, a, _ in blocks]), dtype=np.int64)
    block = np.searchsorted(breaks, frames, side="right")
    win = int(round(switch_back_s * fps))
    out = {"n_touches": int(len(use)), "n_pairs": 0, "n_switch": 0, "back_1": 0,
           "back_2s": 0, "n_back_eval": 0, "switch_pairs": []}
    for i in range(len(use) - 1):
        if block[i + 1] != block[i]:
            continue
        out["n_pairs"] += 1
        if team[i + 1] == team[i]:
            continue
        out["n_switch"] += 1
        out["switch_pairs"].append((int(frames[i]), int(frames[i + 1])))
        j = i + 2
        while j < len(use) and block[j] == block[i] and team[j] == team[i + 1]:
            j += 1
        if j < len(use) and block[j] == block[i]:
            out["n_back_eval"] += 1
            out["back_1"] += j == i + 2
            out["back_2s"] += frames[j] - frames[i + 1] <= win
    return out


def gap_spans(pairs, gidx, fps, short):
    """M3 table for [(frame_a, frame_b)]: counts per class and per longest-gap bin."""
    out = {"n": len(pairs), "no_gap": 0, "short_gap": 0, "long_gap": 0}
    out.update({"bin:" + lab: 0 for _, _, lab in GAP_BINS_S})
    for a, b in pairs:
        _, longest = gidx.between(a, b)
        out[gap_class(longest, short)] += 1
        if longest:
            out["bin:" + gap_bin(longest, fps)] += 1
    return out


def tag_anchors(tags_path, blocks, fps, home_tag_team=HOME_TAG_TEAM, codes=TAG_CODES):
    """([{half, frame, code, expected}], n_dropped): starts of the transition tags on graded
    frames. expected is the team that should have the ball after the switch: the tag's Team for
    ATTACKING TRANSITION, the other team for DEFENSIVE TRANSITION, None for a missing / N/A team.
    """
    out, dropped = [], 0
    for t in tg.read_tags(tags_path, fps):
        if t["code"] not in codes:
            continue
        half = tg.half_of(t["start"], blocks)
        if half is None:
            dropped += 1
            continue
        team = (t["team"] or "").strip()
        idx = None
        if team and team.upper() != "N/A":
            idx = 0 if home_tag_team.lower() in team.lower() else 1
        if idx is not None and t["code"] == "DEFENSIVE TRANSITION":
            idx = 1 - idx
        out.append({"half": half, "frame": t["start"], "code": t["code"], "expected": idx})
    return out, dropped


def nearest(points, refs):
    """(distance, index into refs) of each point's nearest ref; inf / -1 if no refs."""
    points = np.asarray(points, dtype=np.int64)
    refs = np.asarray(refs, dtype=np.int64)
    if not len(refs):
        return np.full(len(points), np.inf), np.full(len(points), -1)
    order = np.argsort(refs, kind="stable")
    r = refs[order]
    j = np.searchsorted(r, points)
    jl, jr = np.clip(j - 1, 0, len(r) - 1), np.clip(j, 0, len(r) - 1)
    dl, dr = np.abs(points - r[jl]), np.abs(points - r[jr])
    pick = np.where(dl <= dr, jl, jr)
    return np.minimum(dl, dr).astype(float), order[pick]


def link_shares(sw_half, sw_frames, an_half, an_frames, halves, w):
    """(n switches with an anchor within w, n anchors with a switch within w), same half."""
    hit_s = hit_a = 0
    for h in halves:
        s, a = sw_frames[sw_half == h], an_frames[an_half == h]
        if len(s) and len(a):
            hit_s += int((nearest(s, a)[0] <= w).sum())
            hit_a += int((nearest(a, s)[0] <= w).sum())
    return hit_s, hit_a


def tag_alignment(switches, anchors, blocks, fps, windows_s, offsets, hist_bin_s=OFFSET_BIN_S):
    """M4 for one switch list: per anchor group (all, per code) and window W, the share of
    switches with an anchor within W (precision-like), the share of anchors with a switch
    within W (recall-like), both against circularly shifted anchors, team agreement, and the
    offset histogram (nearest switch - anchor) within the largest window."""
    sw = [s for s in switches if s["kind"] == "turnover"]
    sw_h = np.array([s["half"] for s in sw], dtype=np.int64)
    sw_f = np.array([s["frame"] for s in sw], dtype=np.int64)
    sw_t = np.array([s["team"] for s in sw], dtype=np.int64)
    halves = [h for h, _, _ in blocks]
    bounds = {h: (a, b - a + 1) for h, a, b in blocks}
    groups = {"all": anchors}
    for code in TAG_CODES:
        groups[code] = [a for a in anchors if a["code"] == code]
    res = {"n_switches": len(sw), "groups": {}}
    wmax = int(round(max(windows_s) * fps))
    for name, an in groups.items():
        an_h = np.array([a["half"] for a in an], dtype=np.int64)
        an_f = np.array([a["frame"] for a in an], dtype=np.int64)
        g = {"n_anchors": len(an), "windows": {}}
        for ws in windows_s:
            w = int(round(ws * fps))
            hs, ha = link_shares(sw_h, sw_f, an_h, an_f, halves, w)
            cell = {"switch_hits": hs, "anchor_hits": ha,
                    "precision": hs / float(len(sw)) if sw else math.nan,
                    "recall": ha / float(len(an)) if an else math.nan}
            if offsets is not None and len(an) and sw:
                k_total = len(next(iter(offsets.values())))
                ps, ra = np.zeros(k_total), np.zeros(k_total)
                for k in range(k_total):
                    shifted = an_f.copy()
                    for h in halves:
                        m = an_h == h
                        first, length = bounds[h]
                        shifted[m] = first + (an_f[m] - first + int(offsets[h][k])) % length
                    a1, a2 = link_shares(sw_h, sw_f, an_h, shifted, halves, w)
                    ps[k], ra[k] = a1 / float(len(sw)), a2 / float(len(an))
                for key, real, null in (("precision", cell["precision"], ps),
                                        ("recall", cell["recall"], ra)):
                    cell[key + "_null_mean"] = float(null.mean())
                    cell[key + "_null_p5"], cell[key + "_null_p95"] = (
                        float(v) for v in np.percentile(null, [5, 95]))
                    cell[key + "_p_emp"] = (1.0 + float(np.sum(null >= real - 1e-12))) / \
                        (1.0 + k_total)
                    cell[key + "_lift"] = real / null.mean() if null.mean() > 0 else math.nan
            agree = disagree = 0
            for a in an:
                if a["expected"] is None:
                    continue
                m = sw_h == a["half"]
                if not m.any():
                    continue
                d, j = nearest([a["frame"]], sw_f[m])
                if d[0] <= w:
                    if sw_t[m][j[0]] == a["expected"]:
                        agree += 1
                    else:
                        disagree += 1
            cell["team_agree"], cell["team_disagree"] = agree, disagree
            g["windows"][ws] = cell
        offs = []
        for a in an:
            m = sw_h == a["half"]
            if m.any():
                d, j = nearest([a["frame"]], sw_f[m])
                if d[0] <= wmax:
                    offs.append((sw_f[m][j[0]] - a["frame"]) / float(fps))
        g["offsets"] = offs
        res["groups"][name] = g
    k = int(math.ceil(max(windows_s) / hist_bin_s - 1e-9))
    res["hist_edges"] = np.arange(-k, k + 1) * hist_bin_s
    return res


def gt_convention(gt, blocks, T, switches, fps, window_s=2.0):
    """Where the ground truth switches, relative to the touches (convention check only).

    GT turnovers are frames where the label goes H -> A or A -> H directly, GT restarts where it
    goes D -> H / A. For each: offset (s) to the nearest attributed touch of the new team within
    +-window_s, and to the nearest predicted switch to that team; plus possession counts in the
    labelled window, truth against the control rule (after the scorer's fill)."""
    n = len(gt)
    w = int(round(window_s * fps))
    graded = common.graded_half_of(blocks, n)[:n] > 0
    lab = np.isin(gt, ["H", "A", "D"]) & graded
    if not lab.any():
        return {"n_labelled": 0}
    out = {"n_labelled": int(lab.sum()), "kinds": {}}
    tf, tt = T["frame"], T["team"]
    ok = tt >= 0
    sw_f = np.array([s["frame"] for s in switches], dtype=np.int64)
    sw_t = np.array([s["team"] for s in switches], dtype=np.int64)
    for kind in ("turnover", "restart"):
        rows = []
        for f in np.flatnonzero(lab):
            if f == 0 or not lab[f - 1] or gt[f] not in ("H", "A"):
                continue
            prev = gt[f - 1]
            if (kind == "turnover" and prev in ("H", "A") and prev != gt[f]) or \
                    (kind == "restart" and prev == "D"):
                team = LABELS.index(gt[f])
                m = ok & (tt == team)
                d, j = nearest([f], tf[m])
                touch = (tf[m][j[0]] - f) / float(fps) if d[0] <= w else math.nan
                ms = sw_t == team
                d2, j2 = nearest([f], sw_f[ms])
                sw = (sw_f[ms][j2[0]] - f) / float(fps) if d2[0] <= w else math.nan
                rows.append((touch, sw))
        touch = np.array([r[0] for r in rows], dtype=float)
        sw = np.array([r[1] for r in rows], dtype=float)
        out["kinds"][kind] = {"n": len(rows), "touch": touch, "switch": sw}
    return out


# ---------------------------------------------------------------- driver

def analyse(raw_path, blocks, meta_path, tags_path=None, gt_path=None, fps=FPS, pitch=PITCH,
            sample_size=SAMPLE_SIZE, seed=SEED, switch_back_s=SWITCH_BACK_S,
            windows_s=LINK_WINDOWS_S, home_tag_team=HOME_TAG_TEAM, null_n=bg.NULL_N,
            null_seed=bg.NULL_SEED, null_min_shift_s=bg.NULL_MIN_SHIFT_S, offsets=None,
            data=None, **kw):
    """Pipeline plus every measurement. `data` = (ball, players, teams) skips reading (tests);
    `offsets` overrides the drawn null."""
    ball, players, teams = data if data is not None else load_match(raw_path, meta_path)
    res = pipeline(ball, players, teams, blocks, fps, pitch, **kw)
    res.update(team_names=[nm for _, nm in team_info(meta_path)] if meta_path else [],
               sample_size=sample_size, seed=seed, switch_back_s=switch_back_s,
               windows_s=list(windows_s), null_n=null_n, null_min_shift_s=null_min_shift_s,
               home_tag_team=home_tag_team)
    p, C, T, n = res["p"], res["C"], res["T"], res["n"]
    short = int(round(p["short_gap_s"] * fps))
    graded = int(sum(min(b, n - 1) - a + 1 for _, a, b in blocks))
    res["graded_frames"] = graded
    res["visible_frames"] = int(((res["half_of"] > 0) & ~res["missing"]).sum())
    res["timescale"] = timescale(res["track"], fps)

    # M1: distance at touches (visible candidates above the threshold) against random frames
    vis_t = (C["source"] == 0) & (C["dv"] >= p["min_dv_ms"])
    res["m1_touch"] = C["dist"][vis_t]
    res["m1_touch_dv"] = C["dv"][vis_t]
    tr = res["track"]
    pool = np.flatnonzero(tr["run_of"] >= 0)
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(pool, size=min(sample_size, len(pool)), replace=False)) \
        if len(pool) else pool
    h = int(p["attr_half"])
    bounds = {hh: (a, b) for hh, a, b in blocks}
    first = np.array([bounds[x][0] for x in res["half_of"][sample]], dtype=np.int64)
    last = np.array([min(bounds[x][1], n - 1) for x in res["half_of"][sample]], dtype=np.int64)
    d_base = closest_approach(sample, windows_visible(sample, tr, h), res["pm"], res["teams"],
                              first, last)
    base = d_base.min(axis=1)
    res["m1_base"] = np.where(np.isfinite(base), base, np.nan)

    # M2 and M3
    gidx = GapIndex(res["gaps"])
    cons = consecutive(T, res["dead"], res["long_gaps"], blocks, fps, switch_back_s)
    res["m2"] = cons
    turn = [s for s in res["switches"] if s["kind"] == "turnover"]
    res["m3_control"] = gap_spans([(int(T["frame"][s["prev"]]), s["frame"]) for s in turn
                                   if s["prev"] is not None], gidx, fps, short)
    # raw: consecutive touches broken only by half starts and dead ball, so a team change
    # across a ball gap counts here (M2's pairs stop at long gaps)
    res["m3_raw"] = gap_spans(consecutive(T, res["dead"], [], blocks, fps,
                                          switch_back_s)["switch_pairs"], gidx, fps, short)

    # M4
    res["m4"] = None
    res["null_skipped"] = None
    if tags_path:
        anchors, dropped = tag_anchors(tags_path, blocks, fps, home_tag_team)
        if offsets is None:
            offsets, res["null_skipped"] = common.draw_null(blocks, null_n, null_seed,
                                                            null_min_shift_s, fps)
        res["m4"] = {"n_dropped": dropped,
                     "control": tag_alignment(res["switches"], anchors, blocks, fps, windows_s,
                                              offsets),
                     "naive": tag_alignment(res["switches_naive"], anchors, blocks, fps,
                                            windows_s, offsets)}

    res["gt"] = None
    if gt_path:
        res["gt"] = gt_convention(read_gt(gt_path), blocks, T, res["switches"], fps)
        gt = read_gt(gt_path)
        if res["gt"].get("n_labelled"):
            m = np.isin(gt, ["H", "A", "D"])
            lo, hi = int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1])
            sub = [(hh, max(a, lo), min(b, hi)) for hh, a, b in blocks if max(a, lo) <= min(b, hi)]
            res["gt"]["poss_truth"] = len(possession_runs(gt, sub))
            res["gt"]["poss_control"] = len(possession_runs(fill_x(res["labels"], sub), sub))
            res["gt"]["poss_naive"] = len(possession_runs(fill_x(res["labels_naive"], sub), sub))
    return res


# ---------------------------------------------------------------- outputs

def _q(values, qs=(25, 50, 75)):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return common.percentiles(v, qs)


def _share(a, b):
    return a / float(b) if b else math.nan


def possession_stats(labels, blocks, fps):
    runs = possession_runs(fill_x(labels, blocks), blocks)
    lengths = np.array([(e - s + 1) / float(fps) for _, s, e, _ in runs])
    return {"n": len(runs), "mean_s": float(lengths.mean()) if len(runs) else math.nan,
            "median_s": float(np.median(lengths)) if len(runs) else math.nan,
            "share_under_2s": float(np.mean(lengths < 2.0)) if len(runs) else math.nan}


def summary_rows(res):
    """Long-format rows (analysis, scope, bin, metric, value, unit). Aggregates only."""
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    p, fps, C, T = res["p"], res["fps"], res["C"], res["T"]
    for k in sorted(p):
        v = p[k]
        add("params", "all", "", k, "+".join(v) if isinstance(v, (tuple, list)) else v, "")
    for k in ("sample_size", "seed", "switch_back_s", "null_n", "null_min_shift_s",
              "home_tag_team"):
        add("params", "all", "", k, res[k], "")
    add("params", "all", "", "windows_s", "+".join("{:g}".format(w) for w in res["windows_s"]),
        "s")
    for i, nm in enumerate(res["team_names"]):
        add("params", "team {}".format(i), "", "name", nm, "label {}".format(LABELS[i]))
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")

    # funnel
    g = res["graded_frames"]
    tr = res["track"]
    add("funnel", "all", "", "graded_frames", g, "frames")
    add("funnel", "all", "", "visible_frames", res["visible_frames"], "frames")
    add("funnel", "all", "", "visible_runs", len(tr["runs"]), "runs")
    add("funnel", "all", "", "glitch_frames_dropped", tr.get("n_glitch", 0), "frames")
    add("funnel", "all", "", "jumps_over_cap", tr["n_jumps"], "steps")
    add("funnel", "all", "", "peaks_over_floor", int((C["source"] == 0).sum()), "peaks")
    vis = C["source"] == 0
    det = vis & (C["dv"] >= p["min_dv_ms"])
    add("funnel", "all", "", "visible_touches", int(det.sum()), "touches")
    add("funnel", "all", "", "short_gaps", res["bridge"]["short"], "gaps")
    add("funnel", "all", "", "short_gaps_bridged", res["bridge"]["bridged"], "gaps")
    add("funnel", "all", "", "jumps_between_runs", res["bridge"]["jumps"], "jumps")
    add("funnel", "all", "", "jumps_bridged", res["bridge"]["jumps_bridged"], "jumps")
    gap_t = (C["source"] == 1) & C["is_touch"]
    jump_t = (C["source"] == 2) & C["is_touch"]
    add("funnel", "all", "", "gap_touches", int(gap_t.sum()), "touches")
    add("funnel", "all", "", "jump_touches", int(jump_t.sum()), "touches")
    for name, m in (("visible", det), ("gap", gap_t), ("jump", jump_t)):
        att = m & (C["team"] >= 0)
        add("funnel", name, "", "attributed", int(att.sum()), "touches")
        add("funnel", name, "", "contested", int((att & C["contested"]).sum()), "touches")
        add("funnel", name, "", "no_player", int((m & (C["team"] < 0)).sum()), "touches")
    add("funnel", "all", "", "touches_in_dead_ball", res["n_in_dead"], "touches")
    add("funnel", "all", "", "touches_live", len(T["frame"]), "touches")
    add("funnel", "all", "", "dribbles", int(T["dribble"].sum()), "touches")
    # unattributed touches that look like a bending ball in flight (lofted, projected)
    far = det & (C["team"] < 0)
    lofted = far & (C["sp_b"] >= 8) & (C["sp_a"] >= 8) & (C["angle"] < 45)
    add("funnel", "visible", "", "no_player_fast_bend", int(lofted.sum()), "touches")
    add("funnel", "visible", "", "no_player_fast_bend_share", _share(lofted.sum(), far.sum()),
        "fraction")
    for w, cells in res["timescale"].items():
        for t, v in cells.items():
            add("timescale", "w={}".format(w), "dv>={:g}".format(t), "peaks_per_visible_min", v,
                "1/min")
    for thr in SWEEP_DV:
        m = vis & (C["dv"] >= thr)
        add("sweep", "visible", "dv>={:g}".format(thr), "touches", int(m.sum()), "touches")
        add("sweep", "visible", "dv>={:g}".format(thr), "attributed_share",
            _share((m & (C["team"] >= 0)).sum(), m.sum()), "fraction")
        add("sweep", "visible", "dv>={:g}".format(thr), "per_min",
            m.sum() / (g / float(fps) / 60.0), "touches/min")
    dv_edges = np.arange(0, DV_MAX_MS + DV_BIN_MS, DV_BIN_MS)
    for a, b, c in zip(dv_edges[:-1], dv_edges[1:], hist(C["dv"][vis], dv_edges)):
        add("dv_hist", "visible_peaks", "[{:g},{:g})".format(a, b), "count", int(c), "peaks")
    for a, b, c in zip(dv_edges[:-1], dv_edges[1:], hist(C["dv"][C["source"] == 1], dv_edges)):
        add("dv_hist", "gap_candidates", "[{:g},{:g})".format(a, b), "count", int(c), "gaps")
    # where do jumps land? (whatever --jump-touch says): nearest player at the landing frame
    jm = C["source"] == 2
    for q, v in zip((25, 50, 75), _q(C["dist"][jm])):
        add("jumps", "landing", "", "nearest_player_p{}".format(q), v, "m")
    add("jumps", "landing", "", "share_within_max_dist",
        _share((C["dist"][jm] <= p["max_dist_m"]).sum(), np.isfinite(C["dist"][jm]).sum()),
        "fraction")
    for q, v in zip((25, 50, 75), _q(C["miss"][jm])):
        add("jumps", "length", "", "p{}".format(q), v, "m")

    # M1
    edges = np.arange(0, DIST_MAX_M + DIST_BIN_M, DIST_BIN_M)
    t_hist = hist(res["m1_touch"], edges)
    b_hist = hist(res["m1_base"], edges)
    for a, b, c1, c2 in zip(edges[:-1], edges[1:], t_hist, b_hist):
        label = "[{:g},{:g})".format(a, b) if b < edges[-1] else "[{:g},inf)".format(a)
        add("m1_distance", "touches", label, "count", int(c1), "touches")
        add("m1_distance", "random_frames", label, "count", int(c2), "frames")
    for scope, vals in (("touches", res["m1_touch"]), ("random_frames", res["m1_base"])):
        v = np.asarray(vals, dtype=float)
        add("m1_distance", scope, "", "n", int(np.isfinite(v).sum()), "count")
        add("m1_distance", scope, "", "n_no_player", int((~np.isfinite(v)).sum()), "count")
        for q, x in zip((10, 25, 50, 75, 90), _q(v, (10, 25, 50, 75, 90))):
            add("m1_distance", scope, "", "p{}".format(q), x, "m")
        for r in (1.0, 1.5, 2.0, 3.0):
            add("m1_distance", scope, "", "share_within_{:g}m".format(r),
                _share((v <= r).sum(), np.isfinite(v).sum()), "fraction")
    m1, dp, m2, bim = dip(t_hist, edges)
    add("m1_distance", "touches", "", "mode1", m1, "m")
    add("m1_distance", "touches", "", "dip", dp, "m")
    add("m1_distance", "touches", "", "mode2", m2, "m")
    add("m1_distance", "touches", "", "bimodal", int(bim), "flag")
    for lo, hi, lab in DV_BANDS:
        m = (res["m1_touch_dv"] >= lo) & (res["m1_touch_dv"] < hi)
        v = res["m1_touch"][m]
        add("m1_by_dv", "touches", lab, "n", int(m.sum()), "count")
        add("m1_by_dv", "touches", lab, "median", _q(v, (50,))[0], "m")
        add("m1_by_dv", "touches", lab, "share_within_2m",
            _share((v <= 2.0).sum(), np.isfinite(v).sum()), "fraction")

    # segments
    st = T["seg_type"]
    for s in SEG_TYPES:
        m = st == s
        add("segments", s, "", "n", int(m.sum()), "segments")
        add("segments", s, "", "seconds", float(T["seg_s"][m].sum()), "s")
        add("segments", s, "", "median_s", _q(T["seg_s"][m], (50,))[0], "s")
    add("segments", "carry", "", "merged_carries_over_min",
        int(((T["carry_s"] >= p["min_carry_s"]) & ~T["dribble"]).sum()), "carries")

    # M2
    m2 = res["m2"]
    add("m2_consecutive", "all", "", "n_touches", m2["n_touches"], "touches")
    add("m2_consecutive", "all", "", "n_pairs", m2["n_pairs"], "pairs")
    add("m2_consecutive", "all", "", "n_switch", m2["n_switch"], "pairs")
    add("m2_consecutive", "all", "", "switch_share", _share(m2["n_switch"], m2["n_pairs"]),
        "fraction")
    add("m2_consecutive", "all", "", "n_back_evaluable", m2["n_back_eval"], "switches")
    add("m2_consecutive", "all", "", "back_after_1_touch", m2["back_1"], "switches")
    add("m2_consecutive", "all", "", "back_after_1_touch_share",
        _share(m2["back_1"], m2["n_back_eval"]), "fraction")
    add("m2_consecutive", "all", "", "back_within_s", m2["back_2s"], "switches")
    add("m2_consecutive", "all", "", "back_within_s_share",
        _share(m2["back_2s"], m2["n_back_eval"]), "fraction")

    # M3
    for scope in ("control", "raw"):
        t = res["m3_" + scope]
        add("m3_gap_span", scope, "", "n_switches", t["n"], "switches")
        for k in ("no_gap", "short_gap", "long_gap"):
            add("m3_gap_span", scope, k, "count", t[k], "switches")
            add("m3_gap_span", scope, k, "share", _share(t[k], t["n"]), "fraction")
        for _, _, lab in GAP_BINS_S:
            add("m3_gap_span", scope, "longest:" + lab, "count", t["bin:" + lab], "switches")

    # control rule and possessions
    for name in ("control", "naive"):
        sw = res["switches" if name == "control" else "switches_naive"]
        add("switches", name, "", "restarts", sum(s["kind"] == "restart" for s in sw), "count")
        add("switches", name, "", "turnovers", sum(s["kind"] == "turnover" for s in sw), "count")
        for c in ("two_touches", "carry", "receive"):
            add("switches", name, c, "turnovers",
                sum(s["kind"] == "turnover" and s["confirm"] == c for s in sw), "count")
        lag = [(T["frame"][s["confirm_touch"]] - s["frame"]) / float(fps) for s in sw
               if s["kind"] == "turnover"]
        for q, v in zip((50, 90), _q(lag, (50, 90))):
            add("switches", name, "", "confirm_delay_p{}".format(q), v, "s")
        ps = possession_stats(res["labels" if name == "control" else "labels_naive"],
                              res["blocks"], fps)
        add("possessions", name, "", "n", ps["n"], "possessions")
        add("possessions", name, "", "per_min", ps["n"] / (g / float(fps) / 60.0), "1/min")
        add("possessions", name, "", "mean_s", ps["mean_s"], "s")
        add("possessions", name, "", "median_s", ps["median_s"], "s")
        add("possessions", name, "", "share_under_2s", ps["share_under_2s"], "fraction")
        lab = res["labels" if name == "control" else "labels_naive"]
        gm = res["half_of"] > 0
        for ch in "HADX":
            add("labels", name, ch, "share_graded", float(np.mean(lab[gm] == ch)), "fraction")
    add("dead", "all", "", "intervals", len(res["dead"]), "intervals")
    add("dead", "all", "", "frames", int(sum(iv["length"] for iv in res["dead"])), "frames")
    for s, f in sorted(res["dead_frames"].items()):
        add("dead", s, "", "frames", f, "frames")
    add("gaps", "long", "", "n", len(res["long_gaps"]), "gaps")
    for k, v in res["gap_counts"].items():
        add("gaps", "long", k, "count", v, "gaps")

    # M4
    m4 = res["m4"]
    if m4:
        add("m4_tags", "all", "", "n_ungraded_dropped", m4["n_dropped"], "tags")
        for rule in ("control", "naive"):
            r = m4[rule]
            add("m4_tags", rule, "", "n_switches", r["n_switches"], "switches")
            for grp, gg in r["groups"].items():
                scope = "{}:{}".format(rule, grp)
                add("m4_tags", scope, "", "n_anchors", gg["n_anchors"], "tags")
                for ws, cell in gg["windows"].items():
                    for k, v in cell.items():
                        unit = "count" if k.endswith(("hits", "agree", "disagree")) else \
                            "ratio" if k.endswith(("p_emp", "lift")) else "fraction"
                        add("m4_tags", scope, "w{:g}s".format(ws), k, v, unit)
                offs = gg["offsets"]
                e = r["hist_edges"]
                for a, b, c in zip(e[:-1], e[1:], hist(offs, e) if offs else
                                   np.zeros(len(e) - 1, dtype=int)):
                    add("m4_offsets", scope, "[{:+g},{:+g})".format(a, b), "count", int(c),
                        "tags")
                for q, v in zip((25, 50, 75), _q(offs)):
                    add("m4_offsets", scope, "", "p{}".format(q), v, "s")
    elif res.get("null_skipped"):
        add("m4_tags", "skipped", "", res["null_skipped"], "", "")

    gt = res["gt"]
    if gt and gt.get("n_labelled"):
        add("gt_convention", "all", "", "labelled_frames", gt["n_labelled"], "frames")
        for kind, c in gt["kinds"].items():
            add("gt_convention", kind, "", "n", c["n"], "switches")
            for src in ("touch", "switch"):
                v = c[src]
                add("gt_convention", kind, src, "found", int(np.isfinite(v).sum()), "switches")
                for q, x in zip((25, 50, 75), _q(v)):
                    add("gt_convention", kind, src, "offset_p{}".format(q), x, "s")
                add("gt_convention", kind, src, "within_3_frames",
                    int((np.abs(v) <= 3.0 / fps + 1e-9).sum()), "switches")
        for k in ("poss_truth", "poss_control", "poss_naive"):
            add("gt_convention", "window", "", k, gt[k], "possessions")
    return rows


def touch_rows(res):
    """Rows for touches.csv (FRAME-INDEXED)."""
    T, fps = res["T"], res["fps"]
    at = {}
    for s in res["switches"]:
        at[s["first"]] = "{}:{}".format(s["kind"], s["confirm"])
    cols = ["half", "frame", "source", "dv_ms", "speed_before_ms", "speed_after_ms",
            "angle_deg", "x_m", "y_m", "miss_m", "dist_m", "team", "contested", "d_other_m",
            "seg_type", "seg_s", "carry_team", "carry_s", "dribble", "switch"]
    rows = []
    for i in range(len(T["frame"])):
        rows.append([int(T["half"][i]), int(T["frame"][i]),
                     ("visible", "gap", "jump")[int(T["source"][i])]] +
                    [_r(T[k][i]) for k in ("dv", "sp_b", "sp_a", "angle", "x", "y", "miss",
                                           "dist")] +
                    [int(T["team"][i]), int(T["contested"][i]), _r(T["d_other"][i]),
                     T["seg_type"][i], _r(T["seg_s"][i]), int(T["carry_team"][i]),
                     _r(T["carry_s"][i]), int(T["dribble"][i]), at.get(i, "")])
    return cols, rows


def _r(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(
        float(x), nd)


def write_rows(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def write_labels(path, labels, n_frames):
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        for f in range(n_frames):
            handle.write("{},{}\n".format(f, labels[f] if f < len(labels) else "X"))


def _f(x, fmt="{:.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, paths):
    p, C, T, fps = res["p"], res["C"], res["T"], res["fps"]
    det = (C["source"] == 0) & (C["dv"] >= p["min_dv_ms"])
    print("graded frames {}, ball visible {:.1%}; {} visible runs, {} glitch frames dropped, "
          "{} jumps over {:g} m/s left".format(
              res["graded_frames"], _share(res["visible_frames"], res["graded_frames"]),
              len(res["track"]["runs"]), res["track"].get("n_glitch", 0),
              res["track"]["n_jumps"], p["ball_cap_ms"]))
    print("touches (dv >= {:g} m/s): {} visible + {} in short gaps; attributed {} "
          "(contested {}), live {}".format(
              p["min_dv_ms"], int(det.sum()), int(((C["source"] == 1) & C["is_touch"]).sum()),
              int((C["is_touch"] & (C["team"] >= 0)).sum()),
              int((C["is_touch"] & C["contested"]).sum()), len(T["frame"])))
    v = res["m1_touch"]
    print("M1 nearest player at touches: median {} m, within {:g} m {}; random frames median "
          "{} m".format(_f(_q(v, (50,))[0]), p["max_dist_m"],
                        _f(_share((v <= p["max_dist_m"]).sum(), np.isfinite(v).sum())),
                        _f(_q(res["m1_base"], (50,))[0])))
    m2 = res["m2"]
    print("M2 consecutive touches: {} pairs, {} switch team ({}); back after 1 touch {}, "
          "within {:g} s {}".format(m2["n_pairs"], m2["n_switch"],
                                    _f(_share(m2["n_switch"], m2["n_pairs"])),
                                    _f(_share(m2["back_1"], m2["n_back_eval"])),
                                    res["switch_back_s"],
                                    _f(_share(m2["back_2s"], m2["n_back_eval"]))))
    for scope in ("control", "raw"):
        t = res["m3_" + scope]
        print("M3 {} switches spanning a ball gap: {} of {} (long gap {})".format(
            scope, t["short_gap"] + t["long_gap"], t["n"], t["long_gap"]))
    for name in ("control", "naive"):
        ps = possession_stats(res["labels" if name == "control" else "labels_naive"],
                              res["blocks"], fps)
        print("possessions, {} rule: {} (mean {} s, {} under 2 s)".format(
            name, ps["n"], _f(ps["mean_s"], "{:.1f}"), _f(ps["share_under_2s"])))
    if res["m4"]:
        for rule in ("control", "naive"):
            cell = res["m4"][rule]["groups"]["all"]["windows"][max(res["windows_s"])]
            print("M4 {} switches vs transition tags (+-{:g} s): precision {} (null {}), "
                  "recall {} (null {}), team agree {}/{}".format(
                      rule, max(res["windows_s"]), _f(cell["precision"]),
                      _f(cell.get("precision_null_mean", math.nan)), _f(cell["recall"]),
                      _f(cell.get("recall_null_mean", math.nan)), cell["team_agree"],
                      cell["team_agree"] + cell["team_disagree"]))
    for pth in paths:
        print("wrote {}".format(pth))


# ---------------------------------------------------------------- CLI

def _signals(text):
    items = tuple(s.strip() for s in text.split(",") if s.strip())
    bad = [s for s in items if s not in ALL_DEAD_SIGNALS]
    if bad:
        raise argparse.ArgumentTypeError("choose from {}".format(", ".join(ALL_DEAD_SIGNALS)))
    return items


def _odd(text):
    v = int(text)
    if v < 1 or v % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd number")
    return v


def add_param_args(p):
    """The pipeline parameters as CLI options (shared with validate_metrica.py)."""
    p.add_argument("--ball-cap-ms", type=float, default=BALL_CAP_MS,
                   help="faster ball steps are tracker jumps (default 40)")
    p.add_argument("--sg-window", type=_odd, default=SG_WINDOW,
                   help="Savitzky-Golay window, odd frames (default 7; 1 = off)")
    p.add_argument("--sg-order", type=int, default=SG_ORDER)
    p.add_argument("--dv-half", type=int, default=DV_HALF,
                   help="frames each side for the mean velocities (default 4)")
    p.add_argument("--min-dv-ms", type=float, default=MIN_DV_MS,
                   help="touch threshold on the velocity change, m/s (default 3)")
    p.add_argument("--peak-floor-ms", type=float, default=PEAK_FLOOR_MS)
    p.add_argument("--min-sep-s", type=float, default=MIN_SEP_S)
    p.add_argument("--short-gap-s", type=float, default=SHORT_GAP_S,
                   help="gaps up to this are bridged (default 0.5)")
    p.add_argument("--gap-miss-m", type=float, default=GAP_MISS_M)
    p.add_argument("--jump-touch", action="store_true",
                   help="a jump between two runs (step > --ball-cap-ms) may be a touch, bridged "
                        "like a short gap (default off: jumps are the tracker switching objects)")
    p.add_argument("--attr-half", type=int, default=ATTR_HALF,
                   help="closest approach over +- this many frames (default 3)")
    p.add_argument("--max-dist-m", type=float, default=MAX_DIST_M,
                   help="attribution limit, m (default 2)")
    p.add_argument("--contest-margin-m", type=float, default=CONTEST_MARGIN_M,
                   help="contested only if the other team is also within --max-dist-m and less "
                        "than this much further away (default 0.5; inf = both within the limit)")
    p.add_argument("--carry-radius-m", type=float, default=CARRY_RADIUS_M)
    p.add_argument("--carry-share", type=float, default=CARRY_SHARE)
    p.add_argument("--carry-max-ms", type=float, default=CARRY_MAX_MS)
    p.add_argument("--min-carry-s", type=float, default=MIN_CARRY_S)
    p.add_argument("--receive-ms", type=float, default=RECEIVE_MS)
    p.add_argument("--dead-signals", type=_signals, default=DEAD_SIGNALS,
                   help="comma-separated from missing, outside, still (default outside,still)")


def params_from(args):
    return {k: getattr(args, k) for k in ("ball_cap_ms", "sg_window", "sg_order", "dv_half",
                                          "min_dv_ms", "peak_floor_ms", "min_sep_s",
                                          "short_gap_s", "gap_miss_m", "jump_touch",
                                          "attr_half", "max_dist_m", "contest_margin_m",
                                          "carry_radius_m", "carry_share",
                                          "carry_max_ms", "min_carry_s", "receive_ms",
                                          "dead_signals")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for the outputs (created if missing; must be outside "
                                   "the repo)")
    p.add_argument("--game", default="BAR-ATH", help="games.py game id (default BAR-ATH)")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--meta", help="ATD metadata XML (slot teams); required with --raw")
    p.add_argument("--tags", help="Smart Tagging XML for M4; default games.asset_paths()"
                                  "['smart_tagging'] without --raw")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--gt", help="optional frame,label CSV (the public ground truth): where it "
                                "switches relative to the touches. A convention check, never a "
                                "tuning target")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    add_param_args(p)
    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                   help="M1 random ball-visible frames (default 5000)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--switch-back-s", type=float, default=SWITCH_BACK_S)
    p.add_argument("--home-tag-team", default=HOME_TAG_TEAM,
                   help="Smart Tagging Team text of the home team (default Barcelona)")
    p.add_argument("--null-n", type=int, default=bg.NULL_N, help="shifts (default 1000; 0 = off)")
    p.add_argument("--null-seed", type=int, default=bg.NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=bg.NULL_MIN_SHIFT_S)
    p.add_argument("--no-tables", action="store_true",
                   help="write only the summary (no frame-indexed touches / labels)")
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/touch_possession")

    n_frames = None
    if args.raw:
        if args.frames is None or not args.meta:
            p.error("--frames and --meta are required with --raw")
        raw, meta, tags, blocks = args.raw, args.meta, args.tags, args.frames
    else:
        games = bg._games()
        paths = common.game_paths(args.game, ["atd_raw", "atd_meta"], ["smart_tagging"])
        raw, meta = paths["atd_raw"], args.meta or paths["atd_meta"]
        tags = args.tags or paths["smart_tagging"]
        blocks = args.frames or games.half_video_blocks(args.game)
        n_frames = games.GAMES.get(args.game, {}).get("n_video_frames")

    res = analyse(raw, blocks, meta, tags, args.gt, fps=args.fps, pitch=tuple(args.pitch),
                  sample_size=args.sample_size, seed=args.seed,
                  switch_back_s=args.switch_back_s, home_tag_team=args.home_tag_team,
                  null_n=args.null_n, null_seed=args.null_seed,
                  null_min_shift_s=args.null_min_shift, **params_from(args))
    os.makedirs(args.out_dir, exist_ok=True)
    out = [os.path.join(args.out_dir, SUMMARY_NAME)]
    bg.write_csv(summary_rows(res), out[0])
    if not args.no_tables:
        out.append(os.path.join(args.out_dir, TOUCHES_NAME))
        write_rows(out[-1], *touch_rows(res))
        out.append(os.path.join(args.out_dir, LABELS_NAME))
        write_labels(out[-1], res["labels"], n_frames or res["n"])
    print_summary(res, out)
    return res


if __name__ == "__main__":
    main()
