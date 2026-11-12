"""ball_stationary.py on fixture_still_*. Expected values: tests/fixtures/README.md
("ball_stationary.py")."""

import csv
import os

import numpy as np
import pytest

import ball_gaps as bg
import ball_stationary as bs
import common
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_still_atd.txt")
META = os.path.join(FIXTURES, "fixture_still_atd.xml")
TAGS = os.path.join(FIXTURES, "fixture_still_tags.xml")
BLOCKS = [(1, 0, 999), (2, 1000, 1999)]
SP = "code:SET PIECES"
LOCS = ("centre_spot", "corner_arc", "goal_area", "penalty_spot", "touchline_band", "elsewhere")


@pytest.fixture(scope="module")
def res():
    return bs.analyse(RAW, BLOCKS, TAGS, META, null_n=50, null_min_shift_s=5)


def _locs(r):
    return tuple((r["locations"][l]["n_runs"], r["locations"][l]["frames"]) for l in LOCS)


@pytest.mark.parametrize("m, runs, locs, frames, linked", [
    (1.0, 9, ((3, 150), (1, 63), (1, 37), (1, 75), (1, 31), (2, 60)), 416, 6),
    (0.5, 10, ((3, 150), (1, 63), (1, 37), (1, 75), (1, 31), (3, 80)), 436, 7),
    (2.0, 5, ((3, 150), (1, 63), (0, 0), (1, 75), (0, 0), (0, 0)), 288, 4),
])
def test_runs_by_min(res, m, runs, locs, frames, linked):
    r = res["by_min"][m]
    assert r["locations"]["all"]["n_runs"] == runs
    assert _locs(r) == locs
    assert r["locations"]["all"]["frames"] == frames
    assert r["link"][SP]["all"]["real_count"] == linked


def test_shares_and_lengths(res):
    assert res["mins"] == [1.0, 0.5, 2.0]
    rows = {(a, s, b, m): v for a, s, b, m, v, _ in bs.summary_rows(res)}
    assert rows[("location", "s1:all", "", "share")] == pytest.approx(0.208)
    assert res["by_min"][1.0]["locations"]["all"]["lengths"] == {
        "<1s": 0, "1-2s": 4, "2-5s": 5, "5-10s": 0, ">=10s": 0}


def test_jitter_and_missing_frame_stay_one_run():
    ball, _ = bg.read_ball(RAW)
    visible = np.isfinite(ball).all(axis=1)
    half = common.graded_half_of(BLOCKS, len(ball))
    still = bs.still_mask(bs.ball_speed(ball, half), visible) & (half > 0)
    raw_runs = [(r["start"], r["end"]) for r in common.find_runs(still, BLOCKS)
                if 1490 <= r["start"] <= 1560]
    assert raw_runs == [(1500, 1519), (1521, 1549)]
    merged = [(r["start"], r["end"]) for r in bs.merge_runs(common.find_runs(still, BLOCKS), 3)]
    assert (1500, 1549) in merged and (1700, 1729) in merged and (1740, 1769) in merged


def test_links_by_location_and_type(res):
    link = res["by_min"][1.0]["link"]
    sp = link[SP]
    assert [(sp[l]["real_count"], sp[l]["n_gaps"]) for l in LOCS] == \
        [(3, 3), (1, 1), (1, 1), (0, 1), (1, 1), (0, 2)]
    assert {k: v["all"]["real_count"] for k, v in link.items() if k != SP} == {
        "type:Kick Off": 3, "type:Corner Kick": 1, "type:Goal Kick": 1, "type:Throw-in": 1,
        "type:Free Kick": 0}
    assert res["by_min"][0.5]["link"]["type:Free Kick"]["all"]["real_count"] == 1


def test_crosstab(res):
    c = res["by_min"][1.0]["crosstab"]
    assert c["centre_spot"] == {"Kick Off": 3}
    assert c["corner_arc"] == {"Corner Kick": 1}
    assert c["goal_area"] == {"Goal Kick": 1}
    assert c["penalty_spot"] == {"none": 1}
    assert c["touchline_band"] == {"Throw-in": 1}
    assert c["elsewhere"] == {"none": 2}


def test_kickoff(res):
    k = res["by_min"][1.0]["kickoff"]
    assert k == {1: {"centre_runs": 1, "kickoff_runs": 1}, 2: {"centre_runs": 2, "kickoff_runs": 1}}
    assert bs.analyse(RAW, BLOCKS)["by_min"][1.0]["kickoff"] is None


def test_classify_symmetric():
    L, W = bs.PITCH
    for x, y in ((0.5, 0.5), (11.0, 34.0), (0.5, 0.6), (3.0, 37.0), (40.0, 67.5), (40.0, 20.0)):
        assert bs.classify(x, y) == bs.classify(L - x, W - y)
    assert bs.classify(11.0, 34.0) == "penalty_spot"
    assert bs.classify(L / 2, W / 2) == "centre_spot"
    assert bs.classify(40.0, -1.5) == "touchline_band"


def test_main_writes_csv(tmp_path, capsys):
    out = tmp_path / "out"
    bs.main([str(out), "--raw", RAW, "--meta", META, "--tags", TAGS, "--frames",
             "0-999,1000-1999", "--null-n", "20", "--null-min-shift", "5"])
    with open(out / bs.CSV_NAME) as handle:
        rows = list(csv.DictReader(handle))
    v = {(r["analysis"], r["scope"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert v[("kickoff", "s1:centre_spot", "half 2", "kickoff_runs")] == "1"
    assert v[("crosstab", "s1:corner_arc", "Corner Kick", "n_runs")] == "1"
    assert len(capsys.readouterr().out.strip().splitlines()) <= 10
