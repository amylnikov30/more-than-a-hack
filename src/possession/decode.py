"""Per-spell decoder (§4.3): two states, a switch cost and a minimum segment length.

For one live spell with per-frame log-odds l_t (positive = H), choose s_t in {+1 H, -1 A} to

    maximise   sum_t l_t * s_t  -  lam * (#switches)  +  beta * prior * s_0

with every segment (first and last included) at least m frames long. prior is +1 / -1 for the
restart team, 0 for none.

Exact dynamic programme over segment ends, O(T): F[t, k] is the best score of frames [0, t)
whose last segment has team k and ends at t. With S_k the cumulative sum of k * l,

    F[t, k] = S_k[t] + max( beta * prior * k                     (one segment from 0),
                            max_{m <= s <= t-m} F[s, -k] - lam - S_k[s] )

and the running maximum over s grows by one candidate per step. Ties keep the earlier switch
and, for the first team, H; the result is deterministic.
"""

import numpy as np

H, A = 1, -1


def decode_spell(llr, lam, m, beta=0.0, prior=0):
    """int8 (T,) of +1 / -1 for one spell. m >= 1 frames; spells shorter than 2 m cannot switch."""
    llr = np.asarray(llr, dtype=float)
    n = len(llr)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    m = max(int(m), 1)
    if n < m:
        return np.full(n, single_team(llr, beta, prior), dtype=np.int8)
    cum = np.concatenate(([0.0], np.cumsum(llr))).tolist()   # S_H; S_A = -S_H
    bonus = {H: beta * prior, A: -beta * prior}
    neg_inf = float("-inf")
    # F[k][t] and the start of the segment that ends at t (-1 = no switch before it)
    F = {H: [neg_inf] * (n + 1), A: [neg_inf] * (n + 1)}
    start = {H: [0] * (n + 1), A: [0] * (n + 1)}
    best = {H: neg_inf, A: neg_inf}      # max over s of F[s, -k] - lam - S_k[s]
    best_at = {H: -1, A: -1}
    for t in range(m, n + 1):
        s = t - m                         # new switch candidate: previous segment ends at s
        if s >= m:
            for k in (H, A):
                cand = F[-k][s] - lam - k * cum[s]
                if cand > best[k]:
                    best[k], best_at[k] = cand, s
        for k in (H, A):
            first = bonus[k]
            if best[k] > first:
                F[k][t], start[k][t] = k * cum[t] + best[k], best_at[k]
            else:
                F[k][t], start[k][t] = k * cum[t] + first, 0
    out = np.zeros(n, dtype=np.int8)
    k = H if F[H][n] >= F[A][n] else A
    t = n
    while t > 0:
        s = start[k][t]
        out[s:t] = k
        t, k = s, -k
    return out


def single_team(llr, beta=0.0, prior=0):
    """The one-team answer for a short spell: sign of the summed evidence plus the prior."""
    score = float(np.sum(llr)) + beta * prior
    return H if score >= 0 else A


def decode_all(llr, spells, lam, m, beta, priors, min_spell):
    """int8 (n,) team per frame (+1 / -1 on live frames, 0 elsewhere).

    spells: [(half, start, stop)]; priors: {start: +1 / -1 / 0}; min_spell: frames."""
    team = np.zeros(len(llr), dtype=np.int8)
    for _, a, b in spells:
        prior = priors.get(a, 0)
        if b - a < min_spell:
            team[a:b] = single_team(llr[a:b], beta, prior)
        else:
            team[a:b] = decode_spell(llr[a:b], lam, m, beta, prior)
    return team


def switches(team, spells):
    """Sorted first frames of a new team inside a spell."""
    out = []
    for _, a, b in spells:
        seg = team[a:b]
        out.extend((a + np.flatnonzero(seg[1:] != seg[:-1]) + 1).tolist())
    return out
