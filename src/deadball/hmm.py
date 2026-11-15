"""Two-state (alive / dead) hidden Markov model: forward-backward, Viterbi, spell matching."""

import math

import numpy as np

ALIVE, DEAD = 0, 1
SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)  # the thresholds the official spellF1 averages over


def sticky_log_transitions(switch_cost):
    """Symmetric 2x2 log transition matrix where staying beats switching by switch_cost nats."""
    p_switch = 1.0 / (1.0 + math.exp(switch_cost))
    return np.log(np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]]))


def forward_backward(log_emission, log_a):
    """Posterior P(state | all frames), shape (T, 2), from (T, 2) log emissions. Flat start."""
    n = len(log_emission)
    alpha = np.zeros((n, 2))
    beta = np.zeros((n, 2))
    alpha[0] = math.log(0.5) + log_emission[0]
    for t in range(1, n):
        prev = alpha[t - 1]
        alpha[t, 0] = log_emission[t, 0] + np.logaddexp(prev[0] + log_a[0, 0], prev[1] + log_a[1, 0])
        alpha[t, 1] = log_emission[t, 1] + np.logaddexp(prev[0] + log_a[0, 1], prev[1] + log_a[1, 1])
    for t in range(n - 2, -1, -1):
        nxt = log_emission[t + 1] + beta[t + 1]
        beta[t, 0] = np.logaddexp(log_a[0, 0] + nxt[0], log_a[0, 1] + nxt[1])
        beta[t, 1] = np.logaddexp(log_a[1, 0] + nxt[0], log_a[1, 1] + nxt[1])
    return np.exp(alpha + beta - np.logaddexp(alpha[-1, 0], alpha[-1, 1]))


def viterbi(log_emission, log_a):
    """Most probable state path, (T,) of ALIVE / DEAD. Flat start; ties stay alive."""
    n = len(log_emission)
    back = np.zeros((n, 2), dtype=np.int8)
    score = [math.log(0.5) + log_emission[0, 0], math.log(0.5) + log_emission[0, 1]]
    a00, a01, a10, a11 = log_a[0, 0], log_a[0, 1], log_a[1, 0], log_a[1, 1]
    emission = log_emission.tolist()
    for t in range(1, n):
        to_alive = (score[0] + a00, score[1] + a10)
        to_dead = (score[0] + a01, score[1] + a11)
        back[t, 0] = to_alive[1] > to_alive[0]
        back[t, 1] = to_dead[1] > to_dead[0]
        score = [max(to_alive) + emission[t][0], max(to_dead) + emission[t][1]]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score[1] > score[0]
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def spells(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(int), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def alive_spell_f1(true_alive, pred_alive):
    """The official spellF1 matching, applied to ball-in-play spells with the team ignored.

    Used on the sample games only, to choose the decoder constants: it rewards exactly what the
    ranking metric rewards about dead ball, which is cutting play into the right spells.
    """
    truth, pred = spells(true_alive), spells(pred_alive)
    overlaps = []
    for i, (a, b) in enumerate(truth):
        for j, (c, d) in enumerate(pred):
            if d > a and b > c:
                overlaps.append(((min(b, d) - max(a, c)) / float(max(b, d) - min(a, c)), i, j))
    overlaps.sort(key=lambda item: (-item[0], item[1], item[2]))
    scores = []
    for threshold in SPELL_IOUS:
        matched_truth, matched_pred = set(), set()
        for iou, i, j in overlaps:
            if iou < threshold:
                break
            if i not in matched_truth and j not in matched_pred:
                matched_truth.add(i)
                matched_pred.add(j)
        total = len(truth) + len(pred)
        scores.append(2.0 * len(matched_truth) / total if total else 0.0)
    return float(np.mean(scores))
