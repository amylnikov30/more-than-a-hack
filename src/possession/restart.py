"""Restart prior (§4.1): which team starts each live spell, as +1 (H) / -1 (A) / 0 (none).

1. Smart Tagging: a SET PIECES instance with a known Team, linked to the spell it restarts. A
   tag links to a spell when it starts inside the stoppage before it or at most TAG_LINK_S into
   it, i.e. in [end of the previous live spell of the half - TAG_LINK_S, spell start +
   TAG_LINK_S] (from the half's first graded frame for the half's first spell); the latest such
   tag wins. A Type that starts with "Defending" names the defending team, so it gives the
   other one.
   (Deviation from the spec, which linked only tags starting within TAG_LINK_S of the spell
   start: on the match, set-piece tags start when the stoppage starts, often far more than
   3 s before the ball is played again, so that rule linked 3 spells of 112.)
2. Otherwise, the first frame in the spell's first ONSET_S seconds where the ball is faster than
   FAST_BALL_MS: the team of the nearest player over the PRE_KICK_FRAMES frames before it
   (majority; ties and no player give nothing).
3. Otherwise no prior.

The decoder adds beta for the prior team on the first frame; evidence can overrule it.
"""

import numpy as np

import poss_common as pc
import tag_gaps as tg

from possession import config

TAG, BALL, NONE = "tag", "fast_ball", "none"


def tag_restarts(tags_path, team_names, override=None, fps=config.FPS):
    """([(frame, +1 / -1)] for SET PIECES tags whose team maps to the metadata teams, stats).

    stats: set_pieces (all SET PIECES tags), mapped (with a team), team_map {tag name: home /
    away}, unmapped (tag team names that matched neither team, N/A included)."""
    stats = {"set_pieces": 0, "mapped": 0, "team_map": {}, "unmapped": []}
    if not tags_path:
        return [], stats
    tags = tg.read_tags(tags_path, fps)
    names = sorted(set(t["team"] for t in tags if t["team"]))
    team_map = pc.tag_team_map(names, team_names, override)
    stats["team_map"] = {k: ("home" if v == 0 else "away") for k, v in sorted(team_map.items())}
    out, unmapped = [], set()
    for t in tags:
        if t["code"] != "SET PIECES":
            continue
        stats["set_pieces"] += 1
        if t["team"] not in team_map:
            unmapped.add(t["team"] or "(none)")
            continue
        sign = 1 if team_map[t["team"]] == 0 else -1
        if (t["type"] or "").strip().lower().startswith("defending"):
            sign = -sign
        out.append((t["start"], sign))
    stats["mapped"] = len(out)
    stats["unmapped"] = sorted(unmapped)
    return sorted(out), stats


def fast_ball_team(start, stop, signals, fps=config.FPS):
    """+1 / -1 / 0 from the first fast-ball frame in the spell's opening ONSET_S."""
    speed, nearest = signals["speed"], signals["nearest"]
    end = min(stop, start + int(round(config.ONSET_S * fps)))
    with np.errstate(invalid="ignore"):
        fast = np.flatnonzero(speed[start:end] > config.FAST_BALL_MS)
    if not len(fast):
        return 0
    f = start + int(fast[0])
    before = nearest[max(start, f - config.PRE_KICK_FRAMES):f + 1]
    home, away = int(np.sum(before == 0)), int(np.sum(before == 1))
    return 1 if home > away else (-1 if away > home else 0)


def link_windows(spells, blocks=None, fps=config.FPS):
    """{spell start: (lo, hi)} inclusive frame windows a tag start must fall in."""
    link = int(round(config.TAG_LINK_S * fps))
    first = {half: a for half, a, _ in (blocks or [])}
    out, prev = {}, {}
    for half, a, b in spells:
        lo = prev.get(half, first.get(half, a))
        out[a] = (lo - link, a + link)
        prev[half] = b
    return out


def priors(spells, signals, tag_list=(), blocks=None, fps=config.FPS):
    """({spell start: +1 / -1 / 0}, {source: count}, [s from linked tag to spell start])."""
    windows = link_windows(spells, blocks, fps)
    tag_frames = np.array([f for f, _ in tag_list], dtype=np.int64)
    out, counts, offsets = {}, {TAG: 0, BALL: 0, NONE: 0}, []
    for _, a, b in spells:
        sign = 0
        if len(tag_frames):
            lo, hi = windows[a]
            inside = np.flatnonzero((tag_frames >= lo) & (tag_frames <= hi))
            if len(inside):
                j = int(inside[-1])                      # latest tag: the restart itself
                sign = tag_list[j][1]
                offsets.append((a - int(tag_frames[j])) / float(fps))
        if sign:
            counts[TAG] += 1
        else:
            sign = fast_ball_team(a, b, signals, fps)
            counts[BALL if sign else NONE] += 1
        out[a] = sign
    return out, counts, offsets
