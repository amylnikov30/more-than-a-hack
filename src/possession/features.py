"""Per-frame features for the evidence model (§4.2): home minus away, centred windows only.

Built from the EDA signal code, unchanged:

    shape      poss_shape.team_features: convex-hull area, width, length, share of outfield
               players goal-side of the ball (of the opponent centroid when it is missing),
               each in the team's own attacking direction for the half
    carry      poss_carries: team of the player the ball moves with (+1 H, -1 A)
    pass chain poss_pass_chains: team of the current chain of >= MIN_CHAIN passes
    nearest    poss_contested.nearest_team: team of the nearest player to the visible ball
    missing    ball missing after the short-gap fill, and its products with the ball features

Every windowed feature is a centred mean over 2, 5 and 10 s inside the frame's live spell
(poss_common.rolling_mean), so no window crosses a dead spell and none lags a change. Ball
features average over the frames where they are defined and are 0 where none is. Excluded on
purpose (§4.2): raw ball position, team velocity, centroid advance, line depth, flip density.
"""

import math
import types

import numpy as np

import common
import poss_carries as pcar
import poss_common as pc
import poss_contested as pcon
import poss_pass_chains as ppc
import poss_shape as pshape

from possession import config

SHAPE = ("hull_area", "width", "length", "goal_side")
BALL = ("carry", "chain", "nearest")


def feature_names(windows=config.WINDOWS_S):
    names = ["{}_{:g}s".format(f, w) for f in SHAPE + BALL for w in windows]
    names += ["missing"] + ["missing_x_{}_{:g}s".format(f, windows[0]) for f in BALL]
    return names


def _sign(code):
    """+1 / -1 / NaN from pc.H / pc.A / NONE codes."""
    return np.where(code == pc.H, 1.0, np.where(code == pc.A, -1.0, np.nan))


def ball_signals(ball, players, team, live, blocks, fps=config.FPS):
    """Frame-level ball signals shared by features, restart and snap. Returns a dict:

    ball      interpolated ball (n, 2), m        flag     0 seen, 1 filled, 2 missing (-1 off live)
    speed     ball speed, m/s (NaN undefined)    near     distance to the nearest team player, m
    nearest   0 home / 1 away / -1 undefined     carry, chain   +1 / -1 / NaN per frame
    """
    n = len(ball)
    ball_i, flag = pc.interpolate_ball(ball, live, blocks, config.MAX_INTERP_FRAMES)
    rid = pc.run_id(live, blocks, n)
    carrier, near, speed = pcar.carrier_frames(ball_i, players, team, rid, config.LAG, fps,
                                               config.CARRY_DIST_M, config.CARRY_DV)
    carrier[~live] = -1
    code = np.zeros(n, dtype=np.int8)
    for s, e, _, t in pcar.find_carries(carrier, team,
                                        int(math.ceil(config.CARRY_MIN_S * fps - 1e-9))):
        code[s:e + 1] = t
    cons = ppc.contacts(carrier, team, rid, int(math.ceil(config.TOUCH_MIN_S * fps - 1e-9)))
    chain = np.zeros(n, dtype=np.int8)
    for s, e, t, k in ppc.chains(cons, speed, rid, fps, config.PASS_MIN_MS, config.PASS_MAX_S):
        if k >= config.MIN_CHAIN:
            chain[s:e + 1] = t
    nearest = pcon.nearest_team(ball_i, players, team, live)
    return {"ball": ball_i, "flag": flag, "speed": speed, "near": near, "nearest": nearest,
            "carry": _sign(code), "chain": _sign(chain),
            "nearest_sign": np.where(nearest == 0, 1.0, np.where(nearest == 1, -1.0, np.nan))}


def _shape_ctx(players, team, live, blocks, dirs, pitch, ball, fps=config.FPS):
    """The context dict poss_shape.team_features reads."""
    n = len(live)
    return {"players": players, "team": team, "live": live, "pitch": tuple(pitch), "n": n,
            "blocks": blocks, "dirs": dirs, "half_of": common.graded_half_of(blocks, n),
            "ball_interp": ball, "args": types.SimpleNamespace(lag=config.LAG, fps=fps)}


def build(track, blocks, live, meta_path, pitch, fps=config.FPS, windows=config.WINDOWS_S):
    """(X float (n, F) with NaN off live frames, names, info).

    track: {ball (n, 2) m, players (n, k, 2) m, team (n, k)}. info carries the ball signals
    (for restart and snap) and the attacking directions with their source and the agreement of
    the centroid inference with the metadata (the §6 direction check)."""
    ball, players, team = track["ball"], track["players"], track["team"]
    n = len(live)
    dirs, dir_source, dir_agree = pc.attack_directions(meta_path, players, team, blocks, live,
                                                       fps)
    sig = ball_signals(ball, players, team, live, blocks, fps)
    ctx = _shape_ctx(players, team, live, blocks, dirs, pitch, sig["ball"], fps)
    shape, _ = pshape.team_features(ctx, config.MIN_PLAYERS)
    diff = {f: shape[0][f] - shape[1][f] for f in SHAPE}
    frame = {"carry": sig["carry"], "chain": sig["chain"], "nearest": sig["nearest_sign"]}

    def window(values, w):
        return pc.rolling_mean(values, live, blocks, int(round(w * fps / 2.0)))

    cols = []
    for f in SHAPE:
        cols += [window(diff[f], w) for w in windows]
    first = {}
    for f in BALL:
        for i, w in enumerate(windows):
            col = np.where(live, np.nan_to_num(window(frame[f], w)), np.nan)
            cols.append(col)
            if i == 0:
                first[f] = col
    missing = np.where(live, (sig["flag"] == 2).astype(float), np.nan)
    cols.append(missing)
    cols += [missing * first[f] for f in BALL]
    X = np.column_stack(cols)
    X[~live] = np.nan
    info = {"signals": sig, "dirs": dirs, "dir_source": dir_source, "dir_agree": dir_agree}
    return X, feature_names(windows), info
