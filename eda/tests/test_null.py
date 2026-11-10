"""Analysis (e): set-piece overlap against circularly shifted tags. Expected values are derived
in tests/fixtures/README.md ("Analysis (e)")."""

import os

import numpy as np
import pytest

import ball_gaps as bg
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_atd.txt.gz")
TAGS = os.path.join(FIXTURES, "fixture_smart_tagging.xml")
BLOCKS = [(1, 100, 1499), (2, 1600, 2899)]


def long_gaps_and_tags(tol_frames=0):
    ball, _ = bg.read_ball(RAW)
    gaps = [g for g in bg.find_gaps(~np.isfinite(ball).all(axis=1), BLOCKS) if g["length"] >= 50]
    tags = bg.read_set_pieces(TAGS)
    hits = [bg.overlaps(g, tags, tol_frames) for g in gaps]
    return gaps, tags, hits


@pytest.mark.parametrize("tol, expected", [(0, 3), (25, 5)])
def test_offset_zero_reproduces_real(tol, expected):
    gaps, tags, hits = long_gaps_and_tags(tol)
    assert sum(hits) == expected
    zero = {1: np.array([0]), 2: np.array([0])}
    matrix = bg.null_overlap_matrix(gaps, tags, BLOCKS, zero, tol)
    assert matrix[0].tolist() == hits


def test_vectorised_matches_reference_pieces():
    gaps, tags, _ = long_gaps_and_tags()
    rng = np.random.default_rng(7)
    for _ in range(25):
        offsets = {1: np.array([rng.integers(0, 1400)]), 2: np.array([rng.integers(0, 1300)])}
        for tol in (0, 25):
            pieces = bg.shifted_pieces(tags, BLOCKS, {h: int(o[0]) for h, o in offsets.items()})
            brute = [any(h == g["half"] and a - tol <= g["end"] and b + tol > g["start"]
                         for h, a, b in pieces) for g in gaps]
            assert bg.null_overlap_matrix(gaps, tags, BLOCKS, offsets, tol)[0].tolist() == brute


def test_wrap_around_and_no_crossing():
    blocks = [(1, 0, 99), (2, 200, 299)]
    assert bg.shifted_pieces([(280, 295)], blocks, {1: 0, 2: 10}) == [(2, 290, 300), (2, 200, 205)]
    pieces = bg.shifted_pieces([(90, 210)], blocks, {1: 0, 2: 0})
    assert pieces == [(1, 90, 100), (2, 200, 210)]
    for o1 in range(0, 100, 7):
        for o2 in range(0, 100, 11):
            for h, a, b in bg.shifted_pieces([(90, 210)], blocks, {1: o1, 2: o2}):
                lo, hi = (0, 100) if h == 1 else (200, 300)
                assert lo <= a < b <= hi


def test_same_seed_identical(tmp_path):
    kw = dict(sample_size=10, null_n=200, null_min_shift_s=5.0)
    a = bg.analyse(RAW, BLOCKS, TAGS, null_seed=3, **kw)["set_pieces_null"]
    b = bg.analyse(RAW, BLOCKS, TAGS, null_seed=3, **kw)["set_pieces_null"]
    assert a == b and a["skipped"] is None
    assert a["bins"][">=long"]["real_share"] == pytest.approx(60.0)
    assert a["bins"]["2-5s"]["real_share"] == pytest.approx(100 / 3)
    assert a["bins"][">=5s"]["real_share"] == pytest.approx(100.0)
    c = bg.analyse(RAW, BLOCKS, TAGS, null_seed=4, **kw)["set_pieces_null"]
    assert c != a


def test_default_min_shift_too_long_for_fixture():
    res = bg.analyse(RAW, BLOCKS, TAGS, sample_size=10)
    assert res["set_pieces_null"]["skipped"]


GAP_STARTS = [300, 1100, 1750, 3020, 4400]
ONE_HALF = [(1, 0, 4999)]


def synthetic_null(tags, k=1000, seed=0, min_shift=1500):
    gaps = [{"half": 1, "start": s, "end": s + 49, "length": 50} for s in GAP_STARTS]
    real = [bg.overlaps(g, tags) for g in gaps]
    offsets = bg.draw_offsets(ONE_HALF, k, seed, min_shift)
    null = bg.null_overlap_matrix(gaps, tags, ONE_HALF, offsets).sum(axis=1)
    p_emp = (1 + np.sum(null >= sum(real))) / (1 + k)
    return sum(real), null, p_emp


def test_tags_on_gaps_give_minimal_p():
    real, null, p_emp = synthetic_null([(s, s + 50) for s in GAP_STARTS])
    assert real == 5
    assert p_emp < 0.005
    assert p_emp >= 1 / 1001


def test_independent_tags_give_large_p():
    starts = [0, 500, 1050, 1500, 2000, 2500, 3200, 3700, 4200, 4700]
    real, null, p_emp = synthetic_null([(s, s + 100) for s in starts])
    assert real == 1
    assert p_emp > 0.5


def test_null_csv_and_print(tmp_path, capsys):
    bg.main([str(tmp_path), "--raw", RAW, "--tags", TAGS, "--frames", "100-1499,1600-2899",
             "--sample-size", "50", "--null-n", "100", "--null-min-shift", "5"])
    out = capsys.readouterr().out
    assert "set pieces vs chance:" in out
    import csv
    rows = list(csv.DictReader(open(os.path.join(str(tmp_path), bg.CSV_NAME))))
    get = {(r["analysis"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert float(get[("set_pieces_null", ">=long", "real_share")]) == pytest.approx(60.0)
    assert get[("set_pieces_null", ">=long", "K")] == "100"
    p = float(get[("set_pieces_null", ">=long", "p_emp")])
    assert 1 / 101 <= p <= 1.0
