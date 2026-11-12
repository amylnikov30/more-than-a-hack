"""signal_profiles.py and gap_stoppage.py on fixture_signal_*, plus the reproduction of
tag_gaps view A. Expected values: tests/fixtures/README.md ("signal_profiles.py and
gap_stoppage.py")."""

import csv
import math
import os

import numpy as np
import pytest

import ball_gaps as bg
import common
import gap_stoppage as gs
import signal_profiles as spf
import tag_gaps as tg
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_signal_atd.txt")
HOM = os.path.join(FIXTURES, "fixture_signal_homography.csv")
TAGS = os.path.join(FIXTURES, "fixture_signal_tags.xml")
BLOCKS = [(1, 0, 199), (2, 200, 399)]
SP = "code:SET PIECES"
A = pytest.approx


@pytest.fixture(scope="module")
def signals():
    return spf.compute_signals(RAW, BLOCKS, HOM, lag=1)


def test_per_frame_signals(signals):
    slow, tracked = signals["slow_players"], signals["n_tracked"]
    assert math.isnan(slow[0]) and math.isnan(slow[200])
    assert slow[1] == A(0.25) and slow[199] == A(0.25) and slow[399] == A(0.25)
    assert slow[100] == A(1 / 3) and slow[150] == A(1 / 3)
    assert np.allclose(slow[101:150], 2 / 3)
    assert slow[300] == A(0.25) and np.allclose(slow[301:350], 0.5) and slow[350] == A(0.25)
    assert np.all(tracked[100:150] == 3) and tracked[99] == 4 and tracked[150] == 4
    rep = signals["reproj_error"]
    assert np.all(np.isnan(rep[300:310])) and rep[310] == A(2.0) and rep[120] == A(3.0)
    assert np.nansum(signals["ball_outside"]) == 0 and np.nansum(signals["ball_still"]) == 0
    assert np.isnan(signals["ball_outside"][20]) and signals["ball_missing"][20] == 1


@pytest.fixture(scope="module")
def res():
    return spf.analyse(RAW, BLOCKS, TAGS, HOM, lag=1, window_s=2, bin_s=1, null_n=0)


def test_profiles(res):
    p = res["profiles"]
    assert p[("start", "slow_players")]["type:Throw-in"]["value"] == \
        [A(16.333333 / 25), A(2 / 3), A(6.333333 / 25), A(0.25)]
    assert p[("start", "slow_players")]["type:Corner Kick"]["value"] == \
        [A(0.49), A(0.5), A(0.25), A(0.25)]
    assert p[("start", "slow_players")][SP]["value"] == \
        [A(28.583333 / 50), A(29.166667 / 50), A(12.583333 / 50), A(0.25)]
    assert p[("start", "n_tracked")]["type:Throw-in"]["value"] == [3, 3, 4, 4]
    rep = p[("start", "reproj_error")]["type:Corner Kick"]
    assert rep["value"] == [A(2.0), A(2.0), A(0.5), A(0.5)]
    assert rep["counted"] == [15, 25, 25, 25]
    assert p[("start", "ball_missing")]["type:Throw-in"]["value"] == [1.0, 1.0, 0.0, 0.0]
    for s in ("ball_outside", "ball_still"):   # undefined while the ball is missing
        a = p[("start", s)][SP]
        assert a["counted"][:2] == [0, 0] and all(math.isnan(x) for x in a["value"][:2])
        assert a["value"][2:] == [0, 0]


def test_ball_missing_reproduces_tag_gaps_view_a():
    raw = os.path.join(FIXTURES, "fixture_atd.txt.gz")
    tags = os.path.join(FIXTURES, "fixture_tags.xml")
    blocks = [(1, 100, 1499), (2, 1600, 2899)]
    offsets = bg.draw_offsets(blocks, 30, 0, 125)
    ball, _ = bg.read_ball(raw)
    missing = ~np.isfinite(ball).all(axis=1)
    groups = common.set_piece_groups(tg.read_tags(tags))
    values = spf.compute_signals(raw, blocks)["ball_missing"]
    for end in (False, True):
        want = tg.view_a(missing, groups, blocks, offsets=offsets, end=end)
        got = spf.profile(values, groups, blocks, offsets=offsets, end=end)
        for name in groups:
            w, g = want[name], got[name]
            assert g["n"] == w["n"] and g["counted"] == w["counted"]
            assert g["sum"] == w["missing"]
            np.testing.assert_allclose(g["value"], w["share"], equal_nan=True)
            for m in ("null_mean", "null_p5", "null_p95"):
                if m in w:
                    np.testing.assert_allclose(g[m], w[m], equal_nan=True)


@pytest.fixture(scope="module")
def gap_res():
    return gs.analyse(RAW, BLOCKS, HOM, lag=1, n_clusters=2)


def test_gap_bins(gap_res):
    b = gap_res["bins"]
    assert gap_res["n_gaps"] == 4
    assert b["0-2s"]["n_gaps"] == 2 and b["2-5s"]["n_gaps"] == 2
    assert b["0-2s"]["slow_players"]["q"] == [A(0.25)] * 3
    assert b["0-2s"]["reproj_error"]["q"] == [A(0.5)] * 3
    assert b["0-2s"]["n_tracked"]["q"] == [A(4)] * 3
    assert b["2-5s"]["slow_players"]["q"] == [A(0.5775), A(0.53625), A(0.61875)]
    assert b["2-5s"]["reproj_error"]["q"] == [A(2.5), A(2.25), A(2.75)]
    assert b["2-5s"]["n_tracked"]["q"] == [A(3.5), A(3.25), A(3.75)]
    for label in ("5-10s", "10-20s", "20-35s", "35-60s", ">=60s"):
        assert b[label]["n_gaps"] == 0 and math.isnan(b[label]["slow_players"]["q"][0])


def test_clusters(gap_res):
    c = gap_res["clusters"]["clusters"]
    assert [x["size"] for x in c] == [2, 2]
    assert (c[0]["length_s"], c[0]["slow_players"], c[0]["reproj_error"]) == \
        (A(0.4), A(0.25), A(0.5))
    assert (c[1]["length_s"], c[1]["slow_players"], c[1]["reproj_error"]) == \
        (A(2.0), A(0.5775), A(2.5))


def test_without_homography_or_tags():
    r = gs.analyse(RAW, BLOCKS, lag=1, n_clusters=2)
    assert r["clusters"] == {"skipped": "no homography"}
    assert r["bins"]["2-5s"]["reproj_error"]["n"] == 0
    s = spf.analyse(RAW, BLOCKS, lag=1, null_n=0)
    assert "reproj_error" not in s["signals"] and s["profiles"] == {}


def test_mains_write_csv(tmp_path, capsys):
    out = tmp_path / "sp"
    spf.main([str(out), "--raw", RAW, "--tags", TAGS, "--homography", HOM, "--frames",
              "0-199,200-399", "--lag", "1", "--window", "2", "--null-n", "0", "--ends"])
    with open(out / spf.CSV_NAME) as handle:
        rows = list(csv.DictReader(handle))
    v = {(r["analysis"], r["scope"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert float(v[("profile_start", "n_tracked:type:Throw-in", "-2", "value")]) == 3
    assert ("profile_end", "slow_players:" + SP, "0", "value") in v
    assert len(capsys.readouterr().out.strip().splitlines()) <= 10
    out = tmp_path / "gs"
    gs.main([str(out), "--raw", RAW, "--homography", HOM, "--frames", "0-199,200-399",
             "--lag", "1", "--clusters", "2"])
    with open(out / gs.CSV_NAME) as handle:
        rows = list(csv.DictReader(handle))
    v = {(r["analysis"], r["scope"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert float(v[("gap_bin", "slow_players", "2-5s", "median")]) == A(0.5775)
    assert v[("cluster", "cluster 1", "", "size")] == "2"
    assert len(capsys.readouterr().out.strip().splitlines()) <= 10
