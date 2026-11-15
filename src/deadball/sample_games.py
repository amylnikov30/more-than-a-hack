"""Metrica's open sample games as training data: tracking arrays plus dead-ball labels.

Source: https://github.com/metrica-sports/sample-data (three anonymised games, corrected tracking
and events). In that data the ball position is blank while the ball is out of play: it is blank
on 97-98% of frames just before a SET PIECE event and from the end of every BALL OUT event. So
"ball blank for at least a second" is a dead-ball label nobody on this team produced, on games
that are not the competition match.
"""

import os

import numpy as np
import pandas as pd

from deadball.signals import FPS

PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
GAME3_FIRST_HALF_LAST_FRAME = 69661
MIN_DEAD_S = 1.0  # shorter blanks are tracking drop-outs, left out of training


def _folder(root, game):
    return os.path.join(root, "data", "Sample_Game_{}".format(game))


def _load_csv_game(root, game):
    stem = os.path.join(_folder(root, game), "Sample_Game_{}_RawTrackingData_".format(game))
    home = pd.read_csv(stem + "Home_Team.csv", skiprows=2)
    away = pd.read_csv(stem + "Away_Team.csv", skiprows=2)
    players = np.concatenate([home.iloc[:, 3:-2].values.reshape(len(home), -1, 2),
                              away.iloc[:, 3:-2].values.reshape(len(away), -1, 2)], axis=1)
    return players, home.iloc[:, -2:].values, home["Period"].values


def _load_epts_game(root, game):
    path = os.path.join(_folder(root, game), "Sample_Game_{}_tracking.txt".format(game))
    frames, players, ball = [], [], []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            frames.append(int(parts[0]))
            players.append(np.array(parts[1].replace(";", ",").split(","), dtype=float))
            ball.append(np.array(parts[2].split(","), dtype=float)[:2])
    frames = np.array(frames)
    period = np.where(frames <= GAME3_FIRST_HALF_LAST_FRAME, 1, 2)
    return np.array(players).reshape(len(frames), -1, 2), np.array(ball), period


def load(root, game):
    """(players, ball, period) for one sample game."""
    return _load_epts_game(root, game) if game == 3 else _load_csv_game(root, game)


def runs(mask):
    """[(start, stop)] half-open runs where mask is true."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(int), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def dead_labels(ball):
    """1 dead, 0 alive, -1 left out (blank for under MIN_DEAD_S)."""
    labels = np.zeros(len(ball), dtype=int)
    for start, stop in runs(np.isnan(ball[:, 0])):
        labels[start:stop] = 1 if stop - start >= MIN_DEAD_S * FPS else -1
    return labels
