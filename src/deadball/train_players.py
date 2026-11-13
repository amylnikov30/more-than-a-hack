"""Fit the player-movement stoppage model on Metrica's open sample games.

    cd src && PYTHONPATH=.. python -m deadball.train_players /path/to/sample-data

Writes src/deadball/player_model.json: a logistic model of "is the ball dead?" from ball-free
player features, and the two decoder constants (tempering and switch cost) chosen by
leave-one-game-out on the sample games. Nothing here touches the competition match, so every
number it produces is safe to tune on. Deterministic: no random numbers anywhere.
"""

import argparse
import json
import os

import numpy as np

from deadball import hmm, sample_games, signals

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_model.json")
L2 = 1e-3
NEWTON_STEPS = 50
TEMPERS = (10.0, 25.0)
SWITCH_COSTS = (2.0, 4.0, 6.0, 8.0, 12.0)


def fit_logistic(x, y):
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    weights = np.zeros(z.shape[1])
    penalty = L2 * np.eye(len(weights))
    penalty[0, 0] = 0.0
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-z @ weights))
        gradient = z.T @ (p - y) / len(y) + penalty @ weights
        hessian = (z * (p * (1 - p))[:, None]).T @ z / len(y) + penalty + 1e-9 * np.eye(len(weights))
        weights -= np.linalg.solve(hessian, gradient)
    return {"mean": mean, "scale": scale, "weights": weights, "prior": float(y.mean())}


def log_odds_ratio(model, x):
    """log P(x | dead) - log P(x | alive): the model's log-odds with the training prior removed."""
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    x = np.where(np.isfinite(x), x, mean)  # no players tracked: fall back to the average frame
    logit = np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ np.asarray(model["weights"])
    return logit - np.log(model["prior"] / (1 - model["prior"]))


def auc(y, score):
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n_pos = y.sum()
    return float((rank[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos)))


def decode(llr, period, temper, switch_cost):
    log_a = hmm.sticky_log_transitions(switch_cost)
    path = np.zeros(len(llr), dtype=np.int8)
    for half in np.unique(period):
        index = np.flatnonzero(period == half)
        path[index] = hmm.viterbi(np.column_stack([-llr[index] / 2, llr[index] / 2]) / temper, log_a)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample_data", help="a clone of github.com/metrica-sports/sample-data")
    args = parser.parse_args()

    games = {}
    for game in sample_games.GAMES:
        players, ball, period = sample_games.load(args.sample_data, game)
        labels = sample_games.dead_labels(ball)
        features = signals.player_features(players, sample_games.PITCH)
        games[game] = (features, labels, period)
        print("game {}: {} frames, dead share {:.3f}".format(game, len(labels), np.mean(labels == 1)))

    def training_rows(names):
        keep = [(games[g][1] >= 0) & np.isfinite(games[g][0]).all(axis=1) for g in names]
        return (np.vstack([games[g][0][k] for g, k in zip(names, keep)]),
                np.concatenate([games[g][1][k] for g, k in zip(names, keep)]))

    # Leave one game out: fit on two, score on the third, for the frame model and each decoder.
    grid = {(t, s): [] for t in TEMPERS for s in SWITCH_COSTS}
    held_out_auc = {}
    for held_out in sample_games.GAMES:
        model = fit_logistic(*training_rows([g for g in sample_games.GAMES if g != held_out]))
        features, labels, period = games[held_out]
        llr = log_odds_ratio(model, features)
        held_out_auc[held_out] = auc(labels[labels >= 0], llr[labels >= 0])
        for temper, switch_cost in grid:
            path = decode(llr, period, temper, switch_cost)
            grid[(temper, switch_cost)].append(float(np.mean(
                [hmm.alive_spell_f1(labels[period == h] != 1, path[period == h] == hmm.ALIVE)
                 for h in np.unique(period)])))
    print("held-out AUC per game:", {g: round(a, 3) for g, a in held_out_auc.items()})
    print("temper  switch  alive-spell F1 per held-out game   mean")
    for (temper, switch_cost), scores in sorted(grid.items()):
        print("{:>6} {:>7}  {}   {:.3f}".format(temper, switch_cost,
                                               "  ".join("{:.3f}".format(s) for s in scores),
                                               np.mean(scores)))
    temper, switch_cost = max(sorted(grid), key=lambda key: np.mean(grid[key]))
    print("chosen: temper {} switch cost {}".format(temper, switch_cost))

    model = fit_logistic(*training_rows(sample_games.GAMES))
    out = {
        "trained_on": "metrica-sports/sample-data, Sample Games 1-3; label = ball blank >= 1 s",
        "features": "signals.player_features (18 columns)",
        "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
        "weights": model["weights"].tolist(), "prior": model["prior"],
        "temper": temper, "switch_cost": switch_cost,
        "held_out_auc": held_out_auc,
        "held_out_alive_spell_f1": grid[(temper, switch_cost)],
    }
    with open(MODEL_PATH, "w") as handle:
        json.dump(out, handle, indent=1)
    print("wrote", MODEL_PATH)


if __name__ == '__main__':
    main()
