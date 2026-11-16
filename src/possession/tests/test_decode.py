"""Decoder (§4.3). Expected answers are worked out by hand in each docstring."""

import itertools

import numpy as np

from possession import decode


def test_one_clear_change():
    """l = +1 on frames 0-99, -1 on 100-199; lam 5, m 25. One switch at 100 scores 200 - 5 =
    195; no switch scores 0; moving the switch by d frames loses 2d. So exactly one switch, at
    frame 100, H before and A after."""
    llr = np.r_[np.ones(100), -np.ones(100)]
    out = decode.decode_spell(llr, lam=5.0, m=25)
    assert out[:100].tolist() == [1] * 100
    assert out[100:].tolist() == [-1] * 100
    assert decode.switches(out, [(1, 0, 200)]) == [100]


def test_noise_below_lambda():
    """l = +1 on 200 frames except -1 on frames 100-102; lam 10, m 1. Flipping the blip to A
    gains 2 * 3 = 6 but costs two switches, 20. So no switch: all H."""
    llr = np.ones(200)
    llr[100:103] = -1
    out = decode.decode_spell(llr, lam=10.0, m=1)
    assert (out == 1).all()


def test_segments_never_shorter_than_m():
    """Random evidence, lam 0, m 10: every segment, the first and last included, has >= 10
    frames."""
    llr = np.random.default_rng(1).normal(size=500) * 3
    out = decode.decode_spell(llr, lam=0.0, m=10)
    edges = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1, len(out)]
    assert len(edges) > 3
    assert np.diff(edges).min() >= 10


def _brute(llr, lam, m, beta, prior):
    best, best_path = -np.inf, None
    for path in itertools.product((1, -1), repeat=len(llr)):
        path = np.array(path)
        edges = np.r_[0, np.flatnonzero(path[1:] != path[:-1]) + 1, len(path)]
        if np.diff(edges).min() < m:
            continue
        score = float(llr @ path) - lam * (len(edges) - 2) + beta * prior * path[0]
        if score > best + 1e-12:
            best, best_path = score, path
    return best


def test_matches_brute_force():
    """The O(T) programme reaches the exhaustive optimum on 12-frame sequences."""
    rng = np.random.default_rng(7)
    for trial in range(30):
        llr = rng.normal(size=12) * 2
        lam, m, beta, prior = (float(rng.choice([0, 1, 3])), int(rng.integers(1, 4)),
                               float(rng.choice([0, 2])), int(rng.choice([-1, 0, 1])))
        out = decode.decode_spell(llr, lam, m, beta, prior)
        edges = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1, len(out)]
        score = float(llr @ out) - lam * (len(edges) - 2) + beta * prior * out[0]
        assert np.diff(edges).min() >= m
        assert abs(score - _brute(llr, lam, m, beta, prior)) < 1e-9


def test_prior_and_short_spells():
    """Evidence -0.1 on 10 frames (sum -1), lam 10, beta 5, prior H, m 1. All H: -1 + 5 = 4.
    All A: 1 - 5 = -4. H for one frame then A: -0.1 + 5 - 10 + 0.9 = -4.2. So all H.
    A spell shorter than min_spell gets one team even when its evidence changes sign."""
    assert (decode.decode_spell(np.full(10, -0.1), 10.0, 1, beta=5.0, prior=1) == 1).all()
    llr = np.r_[np.full(20, 2.0), np.full(20, -3.0)]
    team = decode.decode_all(llr, [(1, 0, 40)], 0.0, 1, 0.0, {}, min_spell=50)
    assert (team == -1).all()          # sum 40 - 60 = -20 -> A, no switch
