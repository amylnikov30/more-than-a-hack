"""poss_carries.py on fixture_poss_*: expected values in fixtures/README.md."""

import math
import os

import numpy as np
import pytest

import poss_carries as pcar
import poss_common as pc
from conftest import FIXTURES
from test_poss_ball_live import BASE, DEAD, EVENTS, TAGS, value

CARRIES = [(55, 89, 4, pc.H), (105, 174, 2, pc.H), (195, 209, 3, pc.H), (250, 279, 4, pc.H),
           (305, 339, 8, pc.A), (360, 399, 9, pc.A), (455, 479, 10, pc.A),
           (561, 699, 3, pc.H), (705, 729, 7, pc.A), (745, 999, 5, pc.H),
           (1055, 1099, 9, pc.A), (1118, 1199, 9, pc.A), (1216, 1299, 10, pc.A),
           (1430, 1989, 2, pc.H)]


def run(tmp_path, *extra):
    import csv
    res = pcar.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), pcar.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def test_carries(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    # slots in the CSV-free result are 0-based
    assert [(s, e, slot + 1, t) for s, e, slot, t in res["carries"]] == CARRIES
    assert res["carry_frames"] == 1440 and value(rows, "coverage", "all", "", "share") == 0.778
    assert res["bins"] == {"0.5-1s": 1, "1-2s": 7, "2-5s": 3, "5-10s": 1, ">=10s": 2}
    assert res["q"] == pytest.approx([1.0, 1.7, 8.808])


def test_precision_and_auc_events(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["precision"]["all"] == (1440, pytest.approx(1415 / 1440))
    assert res["precision"]["H"] == (1104, 1.0)
    assert res["precision"]["A"] == (336, pytest.approx(311 / 336))
    a, ad, cov = res["sep"][0.0]
    assert a == pytest.approx(684703 / 715000) and ad == pytest.approx(1116.5 / 1129)
    assert cov == pytest.approx(1440 / 1850)
    assert value(rows, "separation", "carry", "0", "auc") == 0.958
    assert sorted(res["sep"]) == [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def test_timing_events(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["errors"] == pytest.approx([0.2, 1.2])
    assert (res["near_switches"], res["n_switches"]) == (2, 4)
    assert value(rows, "timing", "carry_switch", "p50", "error") == 0.7
    assert value(rows, "timing", "carry_switch", "p25", "error") == 0.45
    assert value(rows, "timing", "carry_switch", "", "median_abs_error") == 0.7
    assert value(rows, "timing", "carry_switch", "", "event_precision") == 0.5


def test_tags(tmp_path):
    res, rows = run(tmp_path, "--dead", DEAD, "--tags", TAGS)
    assert res["precision"]["all"] == (556, 1.0)
    assert res["precision"]["H"][0] == 315 and res["precision"]["A"][0] == 241
    assert res["errors"] == pytest.approx([-0.2, 0.8])
    assert value(rows, "timing", "carry_switch", "p50", "error") == 0.3


def test_fast_ball(tmp_path):
    res, _ = run(tmp_path, "--events", EVENTS)
    assert res["carry_rate"] == 0.0
    ctx_args = BASE + ["--events", EVENTS]
    import argparse
    p = argparse.ArgumentParser()
    pc.add_args(p, "x", "x")
    ctx = pc.load_context(p.parse_args([str(tmp_path)] + ctx_args), p)
    ball, players, team, live = ctx["ball_interp"], ctx["players"], ctx["team"], ctx["live"]
    rid = pc.run_id(live, ctx["blocks"], ctx["n"])
    fast = near = 0
    for f in range(5, ctx["n"]):           # the slow, obvious loop
        if not live[f] or rid[f] != rid[f - 5]:
            continue
        b0, b1 = ball[f - 5], ball[f]
        if not (np.isfinite(b0).all() and np.isfinite(b1).all()):
            continue
        if math.hypot(*(b1 - b0)) * 5 <= 15.0:
            continue
        fast += 1
        d = [math.hypot(players[f, j, 0] - b1[0], players[f, j, 1] - b1[1])
             for j in range(players.shape[1]) if team[f, j] >= 0]
        near += min(d) < 1.5
    assert res["n_fast"] == fast and fast > 0
    assert res["near_rate"] == pytest.approx(near / float(fast))


def test_degrade_runs(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, "--degrade", "--frag-mean-s", "5")
    assert value(rows, "params", "all", "", "degrade") == 1
    assert res["carry_frames"] < 1440


def test_helpers():
    assert pc.rankdata([3, 1, 3, 2]).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert pc.auc([1, 0, 0], [True, False, True]) == 0.75
    assert math.isnan(pc.auc([1, 2], [True, True]))
    live = np.ones(6, dtype=bool)
    rm = pc.rolling_mean([1, np.nan, 3, np.nan, np.nan, np.nan], live, [(1, 0, 5)], 1)
    assert rm[:4].tolist() == [1.0, 2.0, 3.0, 3.0] and np.isnan(rm[4:]).all()
    rid = np.array([0, 0, 0, -1, 1, 1])
    v = pc.velocity(np.arange(12, dtype=float).reshape(6, 2), rid, 1, 25)
    assert v[1].tolist() == [50.0, 50.0] and np.isnan(v[[0, 3, 4]]).all()
    errors, near, n = pc.match_changes([100], [1], [90, 110, 300], [1, 1, 1], 50)
    assert errors == [0.4] and (near, n) == (2, 3)
