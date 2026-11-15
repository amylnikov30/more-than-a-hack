"""Per-frame signals from tracking arrays. numpy only; nothing here reads a ground truth.

Arrays are indexed by video frame: players (n, k, 2) and ball (n, 2), normalised 0-1, NaN where
missing. The same functions run on the competition ATD and on the Metrica sample games, which is
what lets a model fitted on the sample games be applied to the match.
"""

import warnings

import numpy as np

FPS = 25
LAG = 5                   # frames; speeds are displacement over 0.2 s, robust to per-frame jitter
SPEED_CAP = 12.0          # m/s; faster "players" are slot swaps
WALK_MS = 2.0             # below this a player is walking or standing
RUN_MS = 4.0              # above this a player is running
WINDOWS = (12, 37, 75, 150)   # half-widths in frames: about 1 s, 3 s, 6 s and 12 s centred
SIDE = 37                 # half-width of the one-sided (past / future) 3 s windows
OUT_MARGIN_M = 1.0        # ball this far beyond a line counts as outside
BALL_SPEED_EDGES = (1.0, 4.0, 12.0)   # m/s
BALL_DIST_EDGES = (2.0, 5.0)          # m to the nearest tracked player

MISSING, OUTSIDE = 0, 1
BALL_CATEGORIES = ["missing", "outside"] + [
    "speed {} m/s, nearest player {} m".format(s, d)
    for s in ("<1", "1-4", "4-12", ">12") for d in ("<2", "2-5", ">5")]


def load_atd(raw_path, n_frames):
    """(players, ball) from a raw ATD text file: '<frame>:<x,y,s;...>:<x,y,z>' per line."""
    players, ball = None, np.full((n_frames, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # frame ids count from 1, video frames from 0
            groups = parts[1].split(";")
            if players is None:
                players = np.full((n_frames, len(groups), 2), np.nan)
            n_fields = groups[0].count(",") + 1
            values = np.array(parts[1].replace(";", ",").split(","), dtype=float)
            players[row, :len(groups)] = values.reshape(-1, n_fields)[:, :2]
            ball[row] = [float(v) for v in parts[2].split(",")[:2]]
    return players, ball


def rolling_mean(values, half_width):
    """Centred NaN-aware mean over 2 * half_width + 1 frames."""
    seen = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total[high] - total[low]) / (count[high] - count[low])


def player_speeds(players, pitch):
    metres = players * np.asarray(pitch, dtype=float)
    speed = np.full(players.shape[:2], np.nan)
    step = metres[LAG:] - metres[:-LAG]
    speed[LAG:] = np.hypot(step[:, :, 0], step[:, :, 1]) * FPS / LAG
    speed[speed > SPEED_CAP] = np.nan
    return speed


def player_features(players, pitch):
    """(n, 18) ball-free features: how many players walk or run, and how fast they move on average.

    Each of the three per-frame quantities is averaged over four centred windows and over the 3 s
    before and the 3 s after the frame, so the model sees both the level and whether it is changing.
    """
    speed = player_speeds(players, pitch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tracked = np.sum(~np.isnan(speed), axis=1).astype(float)
        tracked[tracked == 0] = np.nan
        per_frame = (np.nansum(speed < WALK_MS, axis=1) / tracked,
                     np.nanmean(speed, axis=1),
                     np.nansum(speed > RUN_MS, axis=1) / tracked)
    columns = []
    for values in per_frame:
        columns += [rolling_mean(values, w) for w in WINDOWS]
        centred = rolling_mean(values, SIDE)
        pad = np.full(SIDE + 1, np.nan)
        columns += [np.concatenate((pad, centred[:-(SIDE + 1)])),
                    np.concatenate((centred[SIDE + 1:], pad))]
    return np.column_stack(columns)


def ball_categories(players, ball, pitch):
    """One of len(BALL_CATEGORIES) discrete states of the ball track per frame."""
    pitch = np.asarray(pitch, dtype=float)
    xy = ball * pitch
    visible = ~np.isnan(ball[:, 0])
    inside_by = np.minimum.reduce([xy[:, 0], pitch[0] - xy[:, 0], xy[:, 1], pitch[1] - xy[:, 1]])
    speed = np.full(len(ball), np.nan)
    step = xy[LAG:] - xy[:-LAG]
    speed[LAG:] = np.hypot(step[:, 0], step[:, 1]) * FPS / LAG
    metres = players * pitch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nearest = np.sqrt(np.nanmin((metres[:, :, 0] - xy[:, None, 0]) ** 2
                                    + (metres[:, :, 1] - xy[:, None, 1]) ** 2, axis=1))
    # A visible ball with no speed yet (first frames after a gap) is binned as moving, and one
    # with no tracked player as close: the two least informative bins.
    speed_bin = np.digitize(np.where(np.isnan(speed), 5.0, speed), BALL_SPEED_EDGES)
    dist_bin = np.digitize(np.where(np.isnan(nearest), 1.0, nearest), BALL_DIST_EDGES)
    category = np.full(len(ball), MISSING, dtype=int)
    outside = visible & (inside_by < -OUT_MARGIN_M)
    category[outside] = OUTSIDE
    inside = visible & ~outside
    category[inside] = 2 + speed_bin[inside] * (len(BALL_DIST_EDGES) + 1) + dist_bin[inside]
    return category
