"""Analysis (d): nearest-team flips across gaps. Expected values are derived by hand in
tests/fixtures/README.md ("Flip fixture")."""

import csv
import math
import os

import numpy as np
import pytest

import ball_gaps as bg
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_flip_atd.txt")
META = os.path.join(FIXTURES, "fixture_flip_atd.xml")
BLOCKS = [(1, 10, 99), (2, 110, 189)]
BIG = 10 ** 6   # sample size >= population: use every ball-visible graded frame


def cell(res, scope="overall", label="all"):
    c = res["flip"]["tables"][(scope, label)]
    return (c["n_gaps"], c["n_evaluable"], c["n_flip"])


def reasons(res, scope="overall", label="all"):
    c = res["flip"]["tables"][(scope, label)]
    return tuple(c["not_eval_" + r] for r in bg.FLIP_REASONS)


@pytest.fixture(scope="module")
def flip():
    return bg.analyse(RAW, BLOCKS, None, sample_size=BIG, meta_path=META)


@pytest.fixture(scope="module")
def flip_capped():
    return bg.analyse(RAW, BLOCKS, None, sample_size=BIG, meta_path=META, flip_max_dist=5.0)


def test_slot_teams_and_referee():
    (first, last, teams), = bg.read_slot_teams(META)
    assert (first, last) == (1, 200)
    assert teams.tolist() == [0, 1, 0, 1, -1, 0]       # REF has no <Team>


def test_nearest_skips_referee_and_breaks_ties_low():
    (_, _, teams), = bg.read_slot_teams(META)
    ball, _ = bg.read_ball(RAW)
    slots = bg.read_slots(RAW, [19, 30, 59, 79, 114, 175])
    assert bg.nearest_team(slots[19], teams, ball[19]) == (0, pytest.approx(2.0, abs=1e-3))
    assert bg.nearest_team(slots[30], teams, ball[30]) == (1, pytest.approx(3.0, abs=1e-3))
    # referee at 0.5 m is skipped -> home at 2 m
    assert bg.nearest_team(slots[59], teams, ball[59]) == (0, pytest.approx(2.0, abs=1e-3))
    # exact tie at 26.325 m between slot 2 (away) and slot 3 (home): lowest slot wins
    team, dist = bg.nearest_team(slots[79], teams, ball[79])
    assert team == 1 and dist == pytest.approx(0.25 * 105.3)
    # only the referee is present -> no player
    team, dist = bg.nearest_team(slots[114], teams, ball[114])
    assert team == -1 and math.isnan(dist)
    assert bg.nearest_team(slots[175], teams, ball[175]) == (1, pytest.approx(7.0, abs=1e-3))


def test_gap_endpoints():
    ball, _ = bg.read_ball(RAW)
    gaps = bg.find_gaps(~np.isfinite(ball).all(axis=1), BLOCKS)
    got = [(g["start"], g["end"], g["last_seen"], g["next_seen"]) for g in gaps]
    assert got == [(10, 12, None, 13), (20, 29, 19, 30), (40, 44, 39, 45), (60, 74, 59, 75),
                   (80, 84, 79, 85), (115, 119, 114, 120), (125, 174, 124, 175),
                   (180, 189, 179, None)]


def test_flip_overall_halves_bins(flip):
    assert cell(flip) == (8, 5, 2)
    assert reasons(flip) == (1, 1, 1, 0)   # no_last_seen, no_reappearance, no_player, too_far
    assert cell(flip, "half 1") == (5, 4, 1)
    assert cell(flip, "half 2") == (3, 1, 1)
    assert cell(flip, label="<0.5s") == (6, 3, 1)
    assert cell(flip, label="0.5-2s") == (1, 1, 0)
    assert cell(flip, label="2-5s") == (1, 1, 1)
    assert cell(flip, label=">=5s") == (0, 0, 0)


def test_flip_max_dist(flip_capped):
    assert cell(flip_capped) == (8, 3, 1)
    assert reasons(flip_capped) == (1, 1, 1, 2)
    assert cell(flip_capped, label="<0.5s") == (6, 2, 1)
    assert cell(flip_capped, label="2-5s") == (1, 0, 0)


def test_reference_on_fixture(flip, flip_capped):
    assert flip["flip"]["deltas"] == {"<0.5s": 5, "0.5-2s": 15, "2-5s": 50}
    ref = flip["flip"]["reference"]
    assert ref[("overall", "<0.5s")] == (27, 2)
    assert ref[("overall", "0.5-2s")] == (20, 3)
    assert ref[("overall", "2-5s")] == (10, 1)
    assert ref[("half 2", "<0.5s")] == (0, 0)
    capped = flip_capped["flip"]["reference"]
    assert capped[("overall", "<0.5s")] == (27, 2)
    assert capped[("overall", "0.5-2s")] == (19, 2)
    assert capped[("overall", "2-5s")] == (10, 1)
    assert flip["crowding"]["visible_population"] == 67


def test_reference_pairs_hand_case():
    n = 20
    half_of = np.array([1] * 10 + [2] * 10)
    ball_ok = np.ones(n, dtype=bool)
    ball_ok[4] = False
    team = np.zeros(n, dtype=int)
    team[6] = 1
    team[12] = -1
    dist = np.ones(n)
    dist[5] = 9.0
    sample = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 15, 17]
    assert bg.reference_pairs(sample, 3, ball_ok, half_of, team, dist) == (7, 2)
    assert bg.reference_pairs(sample, 3, ball_ok, half_of, team, dist, max_dist=5.0) == (5, 2)


def test_main_fixture_never_flips():
    raw = os.path.join(FIXTURES, "fixture_atd.txt.gz")
    meta = os.path.join(FIXTURES, "fixture_atd.xml")
    res = bg.analyse(raw, [(1, 100, 1499), (2, 1600, 2899)], None, sample_size=BIG,
                     meta_path=meta)
    assert cell(res) == (11, 11, 0)
    for label in ("<0.5s", "0.5-2s", "2-5s", ">=5s"):
        n_ref, k_ref = res["flip"]["reference"][("overall", label)]
        assert n_ref > 0 and k_ref == 0


def test_flip_skipped_without_meta():
    assert bg.analyse(RAW, BLOCKS, None, sample_size=10)["flip"] is None


def test_flip_csv_rows(tmp_path):
    bg.main([str(tmp_path), "--raw", RAW, "--meta", META, "--frames", "10-99,110-189",
             "--sample-size", str(BIG), "--flip-max-dist", "5"])
    rows = list(csv.DictReader(open(os.path.join(str(tmp_path), bg.CSV_NAME))))
    get = {(r["analysis"], r["scope"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert get[("flip", "overall", "all", "n_evaluable")] == "3"
    assert get[("flip", "overall", "all", "not_eval_too_far")] == "2"
    assert float(get[("flip", "overall", "all", "flip_rate")]) == pytest.approx(1 / 3, abs=1e-6)
    assert get[("flip", "overall", "0.5-2s", "ref_n")] == "19"
    assert float(get[("flip", "overall", "0.5-2s", "ref_flip_rate")]) == \
        pytest.approx(2 / 19, abs=1e-6)
    assert get[("params", "all", "", "flip_max_dist")] == "5.0"


def test_out_dir_inside_repo_is_refused():
    inside = os.path.join(os.path.dirname(FIXTURES), "should_not_exist_out")
    with pytest.raises(SystemExit):
        bg.main([inside, "--raw", RAW, "--frames", "10-99,110-189"])
    assert not os.path.exists(inside)
