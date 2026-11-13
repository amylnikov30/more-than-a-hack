"""Shared preparation for the possession-signal EDA scripts (poss_*.py; stdlib + numpy only).

Kept apart from common.py, which serves the dead-ball scripts: this module holds what every
poss_*.py script needs before its own analysis.

    load_tracking      one streamed pass over the raw ATD: ball and players in metres, the team
                       of every slot on every frame (from the metadata, per spec)
    read_teams         team ids and names from the metadata, first <Team> = home
    attack_directions  attacking direction per team and half: from the metadata's
                       attack_direction_first_half, else inferred from the team centroids
    toward_goal        x measured toward the opponent goal for a team in a half
    read_dead_csv      the per-frame output of src/deadball/run.py (dead_ball.csv)
    live_runs          maximal runs of live frames
    interpolate_ball   linear fill of short ball gaps inside live runs, with a flag
    pseudo_labels      Smart Tagging phases -> H / A per live frame, agreement per code
    label_changes      first frames of the other team inside a live run
    degrade            the Sample Game 3 "looks like our ATD" copy: ball bursts, broken tracks
    add_args / load_context   the CLI and loading shared by every poss_*.py script
    write_csv          the long-format summary, floats to 3 significant figures

Label codes on frames: NONE 0, H 1 (home), A 2 (away). Dead frames are those not in `live`.
Team codes on slots: 0 home, 1 away, -1 anything else (referee, unknown).
"""

import csv
import math
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

import ball_gaps as bg
import common
import tag_gaps as tg

FPS = bg.FPS
PITCH = bg.PITCH
NONE, H, A = 0, 1, 2
LABEL_NAMES = {H: "H", A: "A"}

MAX_INTERP_FRAMES = 12      # 0.48 s: a ball can be lost and refound along its line in that time
INTERP_ALSO = (6, 25)       # sensitivity caps, frames
CHANGE_WINDOW_S = 2.0       # "around a change" = within this many seconds
DIRECTION_FIRST_S = 60.0    # centroid window for inferring the attacking direction
# --degrade defaults. Bursts are log-normal; median 1 s and sigma 1.1 put about a quarter of
# bursts under 0.5 s and a tenth over 4 s. A guess to compare with poss_ball_live.py on the
# match, not fitted to it.
MISSING_SHARE = 0.41        # DATA.md: the ball is missing on 41% of graded frames
GAP_MEDIAN_S, GAP_SIGMA, GAP_CAP_S = 1.0, 1.1, 30.0
FRAG_MEAN_S = 20.0          # mean tracklet length before a break; a guess, not tuned
FRAG_GAP_FRAMES = 5         # frames a broken track stays lost before it is re-slotted
FRAG_POOL = 46              # slots available, as in the match ATD

HAS, DEFENDS, IGNORE = "has", "defends", "ignore"
# Smart Tagging code -> whose ball it is during the phase. Codes flagged uncertain in
# docs/analyses.md: LONG BALL (the receiver may be the other team), ATTACKING / DEFENSIVE
# TRANSITION (they straddle the turnover that starts them).
PHASE_TO_POSSESSION = {
    "BUILD UP": HAS, "PROGRESSION": HAS, "ATTACKING TRANSITION": HAS,
    "CREATING CHANCES": HAS, "BALL IN FINAL THIRD": HAS, "LONG BALL": HAS,
    "DEFENDING IN DEFENSIVE THIRD": DEFENDS, "DEFENDING IN MIDDLE THIRD": DEFENDS,
    "DEFENDING IN ATTACKING THIRD": DEFENDS, "DEFENSIVE TRANSITION": DEFENDS,
    "BALL IN THE BOX": IGNORE, "PLAYERS IN THE BOX": IGNORE, "SET PIECES": IGNORE,
    "GOALS": IGNORE,
}


# ---------------------------------------------------------------- metadata and tracking

def read_teams(meta_path):
    """[(team_id, name)] for the first two <Team> elements (home first)."""
    root = ET.parse(meta_path).getroot()
    out = []
    for el in root.iter():
        if bg._tag(el) == "Team":
            name = next((c.text or "" for c in el if bg._tag(c) == "Name"), "")
            out.append((el.get("id"), name.strip()))
    return out[:2]


def read_meta_directions(meta_path):
    """{team: +1 / -1} attacking direction in the first half, from the teams'
    attack_direction_first_half parameter; None unless both teams declare one."""
    root = ET.parse(meta_path).getroot()
    teams = [el for el in root.iter() if bg._tag(el) == "Team"][:2]
    out = {}
    for t, el in enumerate(teams):
        for par in el.iter():
            if bg._tag(par) != "ProviderParameter":
                continue
            name = value = ""
            for c in par:
                if bg._tag(c) == "Name":
                    name = (c.text or "").strip().lower()
                elif bg._tag(c) == "Value":
                    value = (c.text or "").strip().lower()
            if name == "attack_direction_first_half":
                if value.startswith("left"):
                    out[t] = 1
                elif value.startswith("right"):
                    out[t] = -1
    return out if len(out) == 2 else None


def load_tracking(raw_path, meta_path, n, pitch=PITCH):
    """One pass over the raw ATD. Returns a dict:

    ball      float (n, 2), metres, NaN where missing or no record
    players   float32 (n, k, 2), metres, NaN where missing
    team      int8 (n, k): 0 home, 1 away, -1 other or no slot
    record    bool (n,): the frame has a line in the file
    Frames at or beyond n are skipped. k is the largest slot count of any line.
    """
    length, width = pitch
    ball = np.full((n, 2), np.nan)
    record = np.zeros(n, dtype=bool)
    k = 0
    players = np.full((n, 0, 2), np.nan, dtype=np.float32)
    for frame, pl, b in common.stream_frames(raw_path):
        if not 0 <= frame < n:
            continue
        if len(pl) > k:
            grown = np.full((n, len(pl), 2), np.nan, dtype=np.float32)
            grown[:, :k] = players
            players, k = grown, len(pl)
        players[frame, :len(pl)] = pl[:, :2]
        ball[frame] = b[:2]
        record[frame] = True
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    team = np.full((n, k), -1, dtype=np.int8)
    for first_id, last_id, teams in bg.read_slot_teams(meta_path):
        lo = max(int(first_id) - 1, 0)
        hi = n if last_id == math.inf else min(int(last_id), n)
        m = min(len(teams), k)
        if hi > lo and m:
            team[lo:hi, :m] = teams[:m]
    return {"ball": ball, "players": players, "team": team, "record": record}


def team_mean_x(players, team, t, frames):
    """Mean x of team t's present players over the given frames (NaN if none)."""
    xs = players[frames, :, 0]
    ok = (team[frames] == t) & np.isfinite(xs)
    return float(xs[ok].mean()) if ok.any() else math.nan


def infer_directions(players, team, blocks, live, fps=FPS, first_s=DIRECTION_FIRST_S):
    """{(team, half): +1 / -1} from the first `first_s` seconds of live frames of each half:
    the team whose players sit at lower mean x attacks toward +x."""
    out = {}
    for half, first, last in blocks:
        idx = np.flatnonzero(live[first:last + 1])[:int(round(first_s * fps))] + first
        h, a = team_mean_x(players, team, 0, idx), team_mean_x(players, team, 1, idx)
        if math.isnan(h) or math.isnan(a):
            continue
        out[(0, half)] = 1 if h < a else -1
        out[(1, half)] = -out[(0, half)]
    return out


def attack_directions(meta_path, players, team, blocks, live, fps=FPS):
    """({(team, half): +1 / -1}, source, agreement). Source 'metadata' when the metadata
    declares the first-half direction (it flips each half), else 'centroids'. agreement is the
    share of (team, half) pairs where the centroid inference matches the metadata (NaN without
    metadata)."""
    inferred = infer_directions(players, team, blocks, live, fps)
    declared = read_meta_directions(meta_path) if meta_path else None
    if declared is None:
        return inferred, "centroids", math.nan
    dirs = {}
    for half, _, _ in blocks:
        for t in (0, 1):
            dirs[(t, half)] = declared[t] * (1 if half % 2 == 1 else -1)
    keys = [key for key in dirs if key in inferred]
    agree = float(np.mean([dirs[key] == inferred[key] for key in keys])) if keys else math.nan
    return dirs, "metadata", agree


def toward_goal(x, direction, length):
    """x in metres measured toward the opponent goal: x when attacking +x, length - x
    otherwise."""
    return x if direction > 0 else length - x


# ---------------------------------------------------------------- live mask and labels

def read_dead_csv(path, n):
    """int8 (n,): the `dead` column of dead_ball.csv (1 dead, 0 live, -1 not decoded); -1 for
    frames the file does not list."""
    out = np.full(n, -1, dtype=np.int8)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                f, d = int(row["frame"]), int(row["dead"])
            except (KeyError, ValueError):
                continue
            if 0 <= f < n:
                out[f] = d
    return out


def live_runs(live, blocks):
    """[{half, start, end, length}] maximal runs of live frames inside each graded block."""
    return common.find_runs(live, blocks)


def run_id(live, blocks, n):
    """int (n,): index of the live run holding each frame, -1 off live frames."""
    out = np.full(n, -1, dtype=np.int64)
    for i, r in enumerate(live_runs(live, blocks)):
        out[r["start"]:r["end"] + 1] = i
    return out


def tag_team_map(tag_names, meta_names, override=None):
    """{tag team name: 0 / 1}. `override` 'HOME,AWAY' gives the tag names directly. Otherwise a
    tag name maps to a team if they match case-insensitively, or if it shares a word with that
    team's metadata name and with no word of the other (words shared by both names are
    ignored). Names that match neither (N/A) are left out."""
    if override:
        home, away = [s.strip().lower() for s in override.split(",")]
        return {name: (0 if name.strip().lower() == home else 1) for name in tag_names
                if name and name.strip().lower() in (home, away)}

    def words(s):
        return set(re.findall(r"[a-z0-9]+", s.lower()))
    meta_words = [words(m) for m in meta_names]
    common_words = meta_words[0] & meta_words[1] if len(meta_words) == 2 else set()
    out = {}
    for name in tag_names:
        if not name:
            continue
        exact = [i for i, m in enumerate(meta_names) if m.lower() == name.strip().lower()]
        if exact:
            out[name] = exact[0]
            continue
        hits = [i for i, w in enumerate(meta_words) if (words(name) & w) - common_words]
        if len(hits) == 1:
            out[name] = hits[0]
    return out


def pseudo_labels(tags, n, live, team_map):
    """Smart Tagging -> (labels int8 (n,), stats). Only live frames are labelled.

    stats: labelled, dropped (frames where mapped tags disagree), agreement {code: (n, share)}
    for every code in PHASE_TO_POSSESSION and every mapped code found in the tags, unmapped
    {code: instances} for codes the table does not know.
    """
    per_code = {}
    unmapped = {}
    for t in tags:
        rule = PHASE_TO_POSSESSION.get(t["code"])
        if rule is None:
            unmapped[t["code"]] = unmapped.get(t["code"], 0) + 1
            continue
        if rule == IGNORE or t["team"] not in team_map:
            continue
        tagged = team_map[t["team"]]
        lab = (H if tagged == 0 else A) if rule == HAS else (A if tagged == 0 else H)
        arr = per_code.setdefault(t["code"], np.zeros(n, dtype=np.int8))
        lo, hi = max(t["start"], 0), min(t["end"], n)
        if hi > lo:
            arr[lo:hi] = lab
    count_h = np.zeros(n, dtype=np.int32)
    count_a = np.zeros(n, dtype=np.int32)
    for arr in per_code.values():
        arr[~live] = NONE
        count_h += arr == H
        count_a += arr == A
    labels = np.zeros(n, dtype=np.int8)
    labels[(count_h > 0) & (count_a == 0)] = H
    labels[(count_a > 0) & (count_h == 0)] = A
    dropped = int(np.sum((count_h > 0) & (count_a > 0)))
    agreement = {}
    for code in sorted(set(c for c, r in PHASE_TO_POSSESSION.items() if r != IGNORE)
                       | set(per_code)):
        arr = per_code.get(code)
        if arr is None:
            agreement[code] = (0, math.nan)
            continue
        own_h, own_a = arr == H, arr == A
        others = count_h + count_a - own_h - own_a
        both = (arr != NONE) & (others > 0)
        agree = (own_h & (count_a == 0)) | (own_a & (count_h == 0))
        m = int(both.sum())
        agreement[code] = (m, float(np.mean(agree[both])) if m else math.nan)
    return labels, {"labelled": int(np.sum(labels != NONE)), "dropped": dropped,
                    "agreement": agreement, "unmapped": unmapped}


def label_changes(labels, live, blocks):
    """Sorted frames where the label switches team: the first frame of the new team, the
    previous labelled frame being of the other team and in the same live run."""
    n = len(labels)
    rid = run_id(live, blocks, n)
    idx = np.flatnonzero((labels != NONE) & (rid >= 0))
    if len(idx) < 2:
        return np.zeros(0, dtype=np.int64)
    lab, r = labels[idx], rid[idx]
    switch = (lab[1:] != lab[:-1]) & (r[1:] == r[:-1])
    return idx[1:][switch]


def possessions(labels, live, blocks):
    """[(start, end, label)] maximal runs of one label inside live runs (inclusive)."""
    out = []
    for r in live_runs(live, blocks):
        seg = labels[r["start"]:r["end"] + 1]
        edges = np.flatnonzero(np.diff(seg)) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges - 1, [len(seg) - 1]])
        for s, e in zip(starts, ends):
            if seg[s] != NONE:
                out.append((r["start"] + int(s), r["start"] + int(e), int(seg[s])))
    return out


def window_mask(frames, n, half_width, live):
    """bool (n,): live frames within half_width frames of any of `frames`."""
    out = np.zeros(n, dtype=bool)
    for f in frames:
        out[max(f - half_width, 0):min(f + half_width + 1, n)] = True
    return out & live


# ---------------------------------------------------------------- ball gaps

def live_gaps(missing, live, blocks):
    """[{half, start, end, length, left, right}] runs of missing ball inside each live run;
    left / right: the gap has an observed neighbour inside the same run on that side."""
    out = []
    for r in live_runs(live, blocks):
        seg = missing[r["start"]:r["end"] + 1]
        edges = np.diff(np.concatenate([[0], seg.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
            out.append({"half": r["half"], "start": r["start"] + int(s),
                        "end": r["start"] + int(e), "length": int(e - s + 1),
                        "left": s > 0, "right": e < len(seg) - 1})
    return out


def interpolate_ball(ball, live, blocks, max_frames=MAX_INTERP_FRAMES):
    """(ball_filled, flag): live gaps of at most max_frames with an observed neighbour on both
    sides inside the same live run are filled on a straight line. flag int8 (n,): 0 observed,
    1 interpolated, 2 missing (on live frames; -1 elsewhere)."""
    ball = np.array(ball, dtype=float)
    missing = ~np.isfinite(ball).all(axis=1)
    flag = np.full(len(ball), -1, dtype=np.int8)
    flag[live] = np.where(missing[live], 2, 0)
    for g in live_gaps(missing, live, blocks):
        if g["length"] > max_frames or not (g["left"] and g["right"]):
            continue
        a, b = g["start"] - 1, g["end"] + 1
        t = (np.arange(g["start"], g["end"] + 1) - a) / float(b - a)
        ball[g["start"]:g["end"] + 1] = ball[a] + (ball[b] - ball[a]) * t[:, None]
        flag[g["start"]:g["end"] + 1] = 1
    return ball, flag


# ---------------------------------------------------------------- signals and separation

WINDOWS_S = (0.5, 1.0, 2.0, 5.0, 10.0)   # rolling-mean windows for every AUC; 0 = the frame


def velocity(xy, rid, lag, fps=FPS):
    """Backward-difference velocity in m/s, (xy[f] - xy[f - lag]) * fps / lag, for (n, 2) or
    (n, k, 2) positions; NaN unless f and f - lag are in the same live run (rid >= 0) and both
    positions are present."""
    xy = np.asarray(xy, dtype=float)
    out = np.full(xy.shape, np.nan)
    if lag <= 0 or len(xy) <= lag:
        return out
    same = (rid[lag:] >= 0) & (rid[lag:] == rid[:-lag])
    d = (xy[lag:] - xy[:-lag]) * (fps / float(lag))
    shape = (-1,) + (1,) * (xy.ndim - 1)
    out[lag:] = np.where(same.reshape(shape), d, np.nan)
    return out


def rolling_mean(values, live, blocks, half_width):
    """Centred mean of `values` over the defined (finite) frames of [f - h, f + h] inside f's
    live run; NaN where none is defined or off live frames. half_width 0 returns the values on
    live frames."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    for r in live_runs(live, blocks):
        seg = values[r["start"]:r["end"] + 1]
        ok = np.isfinite(seg)
        cs = np.concatenate([[0.0], np.cumsum(np.where(ok, seg, 0.0))])
        cc = np.concatenate([[0], np.cumsum(ok)])
        i = np.arange(len(seg))
        lo = np.maximum(i - half_width, 0)
        hi = np.minimum(i + half_width, len(seg) - 1) + 1
        count = cc[hi] - cc[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r["start"]:r["end"] + 1] = np.where(count > 0, (cs[hi] - cs[lo]) / count,
                                                    np.nan)
    return out


def rankdata(x):
    """Ranks 1..n with ties given their average rank."""
    x = np.asarray(x, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv.ravel()]


def auc(score, positive):
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney); NaN if
    either class is empty. Frames with a NaN score must be removed by the caller."""
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if not n1 or not n0:
        return math.nan
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def window_separation(score, labels, live, blocks, windows_s, fps=FPS, subset=None):
    """{window_s: (auc, auc_defined, coverage)} for "home has the ball" over labelled live
    frames. score: per frame, NaN where the signal is absent. auc treats absent as 0 (no
    evidence), auc_defined uses only frames with a defined smoothed score, coverage is the
    share of live frames with one. Window 0 is the frame itself. `subset` (bool per frame)
    restricts the frames scored and counted, not the frames the window averages over."""
    out = {}
    sub = live if subset is None else (live & subset)
    lab = (labels != NONE) & sub
    n_live = int(sub.sum())
    for w in [0.0] + [float(v) for v in windows_s]:
        s = rolling_mean(score, live, blocks, int(round(w * fps / 2.0)))
        defined = np.isfinite(s) & sub
        pos = labels == H
        a = auc(np.where(defined, s, 0.0)[lab], pos[lab])
        m = lab & defined
        out[w] = (a, auc(s[m], pos[m]), defined.sum() / float(n_live) if n_live else math.nan)
    return out


def window_rows(analysis, scope, sep):
    rows = []
    for w, (a, ad, cov) in sorted(sep.items()):
        b = "{:g}".format(w)
        rows += [(analysis, scope, b, "auc", a, "AUC"),
                 (analysis, scope, b, "auc_defined", ad, "AUC"),
                 (analysis, scope, b, "coverage", cov, "share of live frames")]
    return rows


def match_changes(changes, change_team, events, event_team, window_frames, fps=FPS):
    """Timing of a signal that fires at changes. changes / events: sorted frames; *_team their
    new team. Each change takes the nearest event of the same team within the window (events
    may be shared). Returns (errors in s for matched changes, n_events_near, n_events), where
    n_events_near counts events with a same-team change within the window."""
    changes, events = np.asarray(changes), np.asarray(events)
    change_team, event_team = np.asarray(change_team), np.asarray(event_team)
    errors = []
    for c, t in zip(changes, change_team):
        cand = events[(event_team == t) & (np.abs(events - c) <= window_frames)]
        if len(cand):
            best = cand[np.argmin(np.abs(cand - c) + 1e-3 * (cand < c))]
            errors.append((best - c) / float(fps))
    near = sum(1 for e, t in zip(events, event_team)
               if np.any((change_team == t) & (np.abs(changes - e) <= window_frames)))
    return errors, near, len(events)


def timing_rows(analysis, scope, errors, n_changes, near, n_events):
    q = common.percentiles(errors, [25, 50, 75])
    rows = [(analysis, scope, "", "n_changes", n_changes, "count"),
            (analysis, scope, "", "matched", len(errors), "count"),
            (analysis, scope, "", "matched_share", len(errors) / float(n_changes)
             if n_changes else math.nan, "share of changes"),
            (analysis, scope, "", "n_events", n_events, "count"),
            (analysis, scope, "", "event_precision", near / float(n_events)
             if n_events else math.nan, "share of events")]
    rows += [(analysis, scope, b, "error", v, "s") for b, v in zip(("p25", "p50", "p75"), q)]
    rows.append((analysis, scope, "", "median_abs_error",
                 float(np.median(np.abs(errors))) if errors else math.nan, "s"))
    return rows


# ---------------------------------------------------------------- degraded copy

def degrade_ball(ball, live, rng, missing_share=MISSING_SHARE, median_s=GAP_MEDIAN_S,
                 sigma=GAP_SIGMA, cap_s=GAP_CAP_S, fps=FPS):
    """(ball copy, n_bursts): bursts of missing ball, log-normal lengths (median, sigma, capped,
    at least one frame), placed at uniform live start frames and clipped to live frames, until
    the missing share of live frames reaches missing_share. Non-live frames are untouched."""
    ball = np.array(ball, dtype=float)
    live_idx = np.flatnonzero(live)
    if not len(live_idx):
        return ball, 0
    missing = ~np.isfinite(ball).all(axis=1) & live
    target = int(math.ceil(missing_share * len(live_idx)))
    have = int(missing.sum())
    bursts = 0
    cap = max(1, int(round(cap_s * fps)))
    while have < target:
        length = int(round(rng.lognormal(math.log(median_s * fps), sigma)))
        length = min(max(length, 1), cap)
        start = int(live_idx[rng.integers(len(live_idx))])
        seg = slice(start, min(start + length, len(ball)))
        new = live[seg] & ~missing[seg]
        have += int(new.sum())
        missing[seg] |= live[seg]
        bursts += 1
    ball[missing] = np.nan
    return ball, bursts


def fragment_tracks(players, team, rng, frag_mean_s=FRAG_MEAN_S, gap_frames=FRAG_GAP_FRAMES,
                    pool=FRAG_POOL, fps=FPS):
    """(players, team, n_pieces, n_dropped): each slot's presence runs are cut into pieces of
    exponential length (mean frag_mean_s), losing gap_frames after each cut; pieces are
    re-slotted in time order into a random slot of `pool` that is free, and dropped if none is.
    Teams travel with the pieces."""
    n, k, _ = players.shape
    pool = max(pool, k)
    out_p = np.full((n, pool, 2), np.nan, dtype=np.float32)
    out_t = np.full((n, pool), -1, dtype=np.int8)
    present = np.isfinite(players).all(axis=2)
    mean_frames = max(frag_mean_s * fps, 1.0)
    pieces = []
    for j in range(k):
        edges = np.diff(np.concatenate([[0], present[:, j].astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
            t = int(s)
            while t <= e:
                length = max(1, int(round(rng.exponential(mean_frames))))
                end = min(t + length - 1, int(e))
                pieces.append((t, end, j))
                t = end + 1 + gap_frames
    pieces.sort()
    busy = np.full(pool, -1, dtype=np.int64)
    dropped = 0
    for a, b, j in pieces:
        free = np.flatnonzero(busy < a)
        if not len(free):
            dropped += 1
            continue
        s = int(free[rng.integers(len(free))])
        out_p[a:b + 1, s] = players[a:b + 1, j]
        out_t[a:b + 1, s] = team[a:b + 1, j]
        busy[s] = b
    return out_p, out_t, len(pieces), dropped


# ---------------------------------------------------------------- CLI and loading

def _floats(text):
    return [float(v) for v in text.split(",") if v.strip()]


def add_args(p, csv_name, example):
    """The options every poss_*.py script shares."""
    p.add_argument("out_dir", help="folder for {} (outside the repo, e.g. {})".format(
        csv_name, example))
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--meta", help="ATD metadata XML; default games.asset_paths()['atd_meta'] "
                   "(required with --raw)")
    p.add_argument("--tags", help="Smart Tagging XML for pseudo-labels; default "
                   "games.asset_paths()['smart_tagging'] when --raw is not given")
    p.add_argument("--events", help="Metrica events JSON (reference game only): true labels "
                   "and dead frames; never pass event data about the competition match")
    p.add_argument("--dead", help="dead_ball.csv from `python -m deadball.run` "
                   "(required on the match, e.g. ~/eda_out/dead_ball/dead_ball.csv); "
                   "overrides the dead frames from --events")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--lag", type=int, default=5,
                   help="frames for derived speeds (default 5, as the dead-ball scripts)")
    p.add_argument("--max-interp-frames", type=int, default=MAX_INTERP_FRAMES,
                   help="fill ball gaps up to this many frames (default 12 = 0.48 s; a norm, "
                        "not tuned)")
    p.add_argument("--tag-teams", help="'HOME,AWAY' Smart Tagging team names, when the "
                   "automatic match to the metadata team names fails")
    p.add_argument("--degrade", action="store_true",
                   help="reference game: drop the ball in bursts and break player tracks")
    p.add_argument("--seed", type=int, default=0, help="--degrade seed (default 0)")
    p.add_argument("--missing-share", type=float, default=MISSING_SHARE,
                   help="--degrade: missing share of live frames to reach (default 0.41)")
    p.add_argument("--gap-lengths", type=_floats,
                   default=[GAP_MEDIAN_S, GAP_SIGMA, GAP_CAP_S],
                   help="--degrade burst lengths 'MEDIAN_S,SIGMA,CAP_S' (log-normal; default "
                        "1,1.1,30)")
    p.add_argument("--frag-mean-s", type=float, default=FRAG_MEAN_S,
                   help="--degrade: mean tracklet length, s (default 20)")
    p.add_argument("--frag-gap-frames", type=int, default=FRAG_GAP_FRAMES,
                   help="--degrade: frames lost at each track break (default 5)")
    p.add_argument("--windows", type=_floats, default=list(WINDOWS_S),
                   help="rolling-mean windows for the AUCs, s (default 0.5,1,2,5,10; the "
                        "frame itself, 0, is always added)")
    p.add_argument("--timing-window-s", type=float, default=5.0,
                   help="a signal event within this many s of a change can match it "
                        "(default 5; a norm, not tuned)")
    p.add_argument("--null-n", type=int, default=bg.NULL_N, help="shifts (default 1000; 0 = off)")
    p.add_argument("--null-seed", type=int, default=bg.NULL_SEED)
    p.add_argument("--null-min-shift", type=float, default=bg.NULL_MIN_SHIFT_S,
                   help="smallest shift in seconds (default 60)")


def resolve_paths(args, parser):
    """(raw, meta, tags, events, dead, blocks) from the arguments, or games.py defaults."""
    if args.raw:
        if args.frames is None or not args.meta:
            parser.error("--frames and --meta are required with --raw")
        tags = args.tags
        blocks = args.frames
        raw, meta = args.raw, args.meta
    else:
        paths = common.game_paths(args.game, ["atd_raw", "atd_meta"],
                                  [] if args.tags else ["smart_tagging"])
        raw, meta = paths["atd_raw"], args.meta or paths["atd_meta"]
        tags = args.tags or paths.get("smart_tagging")
        blocks = args.frames or bg._games().half_video_blocks(args.game)
    if not args.dead and not args.events:
        parser.error("pass --dead DEAD_BALL_CSV (from `python -m deadball.run OUT_DIR`), or "
                     "--events on the reference game")
    return raw, meta, tags, args.events, args.dead, blocks


def load_context(args, parser, need_players=True):
    """Everything shared, as a dict: blocks, n, half_of, live, labels, label_source,
    live_source, pseudo stats, tracking arrays (degraded with --degrade), interpolated ball and
    flag, directions, team names, params rows."""
    import poss_reference as pr
    raw, meta, tags, events, dead_path, blocks = resolve_paths(args, parser)
    n = max(b for _, _, b in blocks) + 1
    pitch = tuple(args.pitch)
    half_of = common.graded_half_of(blocks, n)
    graded = half_of > 0
    teams = read_teams(meta)
    team_ids = [t[0] for t in teams]

    labels = np.zeros(n, dtype=np.int8)
    ref_dead = None
    label_source = "none"
    if events:
        evs = pr.read_events(events, args.fps)
        labels, ref_dead = pr.reference_labels(evs, team_ids, blocks, n)
        label_source = "events"
    if dead_path:
        dead = read_dead_csv(dead_path, n)
        live = graded & (dead == 0)
        live_source = "dead_csv"
    else:
        live = graded & ~ref_dead
        live_source = "events"
    labels = np.where(live, labels, NONE).astype(np.int8)

    pseudo = None
    team_map = {}
    if not events and tags:
        tag_list = tg.read_tags(tags, args.fps)
        names = sorted(set(t["team"] for t in tag_list if t["team"]))
        team_map = tag_team_map(names, [t[1] for t in teams], args.tag_teams)
        labels, pseudo = pseudo_labels(tag_list, n, live, team_map)
        label_source = "smart_tagging"

    trk = load_tracking(raw, meta, n, pitch) if need_players else None
    if trk is None:
        ball, _ = bg.read_ball(raw)
        full = np.full((n, 2), np.nan)
        m = min(len(ball), n)
        full[:m] = ball[:m] * np.asarray(pitch)
        trk = {"ball": full, "players": None, "team": None}
    degraded = {}
    if args.degrade:
        rng = np.random.default_rng(args.seed)
        median_s, sigma, cap_s = args.gap_lengths
        trk["ball"], bursts = degrade_ball(trk["ball"], live, rng, args.missing_share,
                                           median_s, sigma, cap_s, args.fps)
        degraded = {"bursts": bursts}
        if trk["players"] is not None:
            trk["players"], trk["team"], pieces, dropped = fragment_tracks(
                trk["players"], trk["team"], rng, args.frag_mean_s, args.frag_gap_frames,
                fps=args.fps)
            degraded.update({"pieces": pieces, "pieces_dropped": dropped})
    ball_i, flag = interpolate_ball(trk["ball"], live, blocks, args.max_interp_frames)

    dirs, dir_source, dir_agree = ({}, "none", math.nan)
    if trk["players"] is not None:
        dirs, dir_source, dir_agree = attack_directions(meta, trk["players"], trk["team"],
                                                        blocks, live, args.fps)
    return {"blocks": blocks, "n": n, "pitch": pitch, "half_of": half_of, "graded": graded,
            "live": live, "labels": labels, "label_source": label_source,
            "live_source": live_source, "pseudo": pseudo, "team_map": team_map,
            "teams": teams, "ball": trk["ball"], "players": trk["players"],
            "team": trk["team"], "ball_interp": ball_i, "flag": flag, "dirs": dirs,
            "dir_source": dir_source, "dir_agree": dir_agree, "degraded": degraded,
            "args": args}


def params_rows(ctx, extra=()):
    """The params rows every poss_*.py summary starts with."""
    a = ctx["args"]
    rows = [("params", "all", "", "fps", a.fps, "frames/s"),
            ("params", "all", "", "pitch_length", ctx["pitch"][0], "m"),
            ("params", "all", "", "pitch_width", ctx["pitch"][1], "m"),
            ("params", "all", "", "lag", a.lag, "frames"),
            ("params", "all", "", "max_interp_frames", a.max_interp_frames, "frames"),
            ("params", "all", "", "degrade", int(bool(a.degrade)), "flag"),
            ("params", "all", "", "windows", ",".join("{:g}".format(w) for w in a.windows),
             "s"),
            ("params", "all", "", "timing_window", a.timing_window_s, "s")]
    if a.degrade:
        rows += [("params", "degrade", "", "seed", a.seed, ""),
                 ("params", "degrade", "", "missing_share", a.missing_share, "share"),
                 ("params", "degrade", "", "gap_median", a.gap_lengths[0], "s"),
                 ("params", "degrade", "", "gap_sigma", a.gap_lengths[1], "log"),
                 ("params", "degrade", "", "gap_cap", a.gap_lengths[2], "s"),
                 ("params", "degrade", "", "frag_mean", a.frag_mean_s, "s"),
                 ("params", "degrade", "", "frag_gap", a.frag_gap_frames, "frames")]
        for key, value in sorted(ctx["degraded"].items()):
            rows.append(("params", "degrade", "", key, value, "count"))
    rows += list(extra)
    rows += [("params", "labels", "", "source", ctx["label_source"], ""),
             ("params", "live", "", "source", ctx["live_source"], ""),
             ("params", "direction", "", "source", ctx["dir_source"], ""),
             ("params", "direction", "", "centroid_agreement", ctx["dir_agree"], "share")]
    for half, first, last in ctx["blocks"]:
        rows += [("params", "half {}".format(half), "", "first_frame", first, "video frame"),
                 ("params", "half {}".format(half), "", "last_frame", last, "video frame")]
    return rows


def label_rows(ctx):
    """Label counts and, for pseudo-labels, agreement per code (NaN rows otherwise, so match
    and reference CSVs have the same rows)."""
    labels, live = ctx["labels"], ctx["live"]
    changes = label_changes(labels, live, ctx["blocks"])
    rows = [("labels", "all", "", "live_frames", int(live.sum()), "frames"),
            ("labels", "all", "", "labelled_frames", int(np.sum(labels != NONE)), "frames"),
            ("labels", "H", "", "frames", int(np.sum(labels == H)), "frames"),
            ("labels", "A", "", "frames", int(np.sum(labels == A)), "frames"),
            ("labels", "all", "", "n_changes", len(changes), "count")]
    pseudo = ctx["pseudo"]
    rows.append(("labels", "all", "", "dropped_frames", pseudo["dropped"] if pseudo else
                 math.nan, "frames"))
    codes = sorted(c for c, r in PHASE_TO_POSSESSION.items() if r != IGNORE)
    for code in codes:
        m, share = pseudo["agreement"].get(code, (0, math.nan)) if pseudo else (math.nan,
                                                                                 math.nan)
        rows += [("agreement", "code:" + code, "", "n_frames", m, "frames"),
                 ("agreement", "code:" + code, "", "agree_share", share, "share")]
    return rows


def sig3(value):
    """Round a float to 3 significant figures (ints and text unchanged)."""
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v) or v == 0 or math.isinf(v):
            return v
        return float("{:.3g}".format(v))
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(rows, path):
    """The long format of bg.write_csv, floats rounded to 3 significant figures, NaN empty."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["analysis", "scope", "bin", "metric", "value", "unit"])
        for row in rows:
            value = sig3(row[4])
            if isinstance(value, float) and math.isnan(value):
                value = ""
            writer.writerow(list(row[:4]) + [value, row[5]])


def check_out_dir(out_dir, example):
    bg.check_out_dir(out_dir, example=example)


def print_labels(ctx):
    """One line on where labels and the live mask come from; for pseudo-labels, the tag team
    map and any codes PHASE_TO_POSSESSION does not know (aggregate text only)."""
    print("labels: {}; live from {}".format(ctx["label_source"], ctx["live_source"]))
    if ctx["pseudo"] is not None:
        names = {0: "home", 1: "away"}
        print("tag teams: " + (", ".join("{} -> {}".format(k, names[v]) for k, v in
                                         sorted(ctx["team_map"].items())) or
                               "none matched; pass --tag-teams HOME,AWAY"))
        if ctx["pseudo"]["unmapped"]:
            print("codes not in PHASE_TO_POSSESSION (ignored): " + ", ".join(
                sorted(ctx["pseudo"]["unmapped"])))
    sys.stdout.flush()
