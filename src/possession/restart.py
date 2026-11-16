"""Restart prior (§4.1): which team starts each live spell, as +1 (H) / -1 (A) / 0 (none).

1. Smart Tagging: a SET PIECES instance starting within TAG_LINK_S of the spell start whose Team
   is a known team gives that team. A Type that starts with "Defending" names the defending
   team, so it gives the other one.
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


def tag_restarts(tags_path, team_names, fps=config.FPS):
    """[(frame, +1 / -1)] for SET PIECES tags with a team that maps to the metadata teams."""
    if not tags_path:
        return []
    tags = tg.read_tags(tags_path, fps)
    names = sorted(set(t["team"] for t in tags if t["team"]))
    team_map = pc.tag_team_map(names, team_names)
    out = []
    for t in tags:
        if t["code"] != "SET PIECES" or t["team"] not in team_map:
            continue
        sign = 1 if team_map[t["team"]] == 0 else -1
        if (t["type"] or "").strip().lower().startswith("defending"):
            sign = -sign
        out.append((t["start"], sign))
    return out


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


def priors(spells, signals, tag_list=(), fps=config.FPS):
    """({spell start: +1 / -1 / 0}, {source: count})."""
    link = int(round(config.TAG_LINK_S * fps))
    tag_frames = np.array([f for f, _ in tag_list], dtype=np.int64)
    out, counts = {}, {TAG: 0, BALL: 0, NONE: 0}
    for _, a, b in spells:
        sign = 0
        if len(tag_frames):
            near = np.flatnonzero(np.abs(tag_frames - a) <= link)
            if len(near):
                best = near[np.argmin(np.abs(tag_frames[near] - a))]
                sign = tag_list[int(best)][1]
        if sign:
            counts[TAG] += 1
        else:
            sign = fast_ball_team(a, b, signals, fps)
            counts[BALL if sign else NONE] += 1
        out[a] = sign
    return out, counts
