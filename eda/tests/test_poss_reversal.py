"""poss_reversal.py on fixture_poss_*: expected values in fixtures/README.md."""

import csv
import math
import os

import numpy as np
import pytest

import poss_common as pc
import poss_reversal as pr
from test_poss_ball_live import BASE, DEAD, EVENTS, TAGS, value

NULL = ["--null-min-shift", "5", "--null-n", "50"]


def run(tmp_path, *extra):
    res = pr.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), pr.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def test_reversals_and_null(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, *NULL)
    assert res["flips"] == (2, 2)
    assert res["rev"].tolist() == [304, 1404] and res["rev_team"].tolist() == [pc.A, pc.H]
    assert value(rows, "reversals", "all", "", "per_min") == 1.62
    assert res["real"] == 2 and res["null"].tolist() == [0] * 50
    assert value(rows, "reversals", "null", "", "p_emp") == pytest.approx(1 / 51.0, abs=1e-3)


def test_null_skipped_by_default(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["null"] is None and "shorter" in res["null_reason"]
    assert math.isnan(value(rows, "reversals", "null", "", "null_mean"))


def test_timing(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, *NULL)
    assert res["errors"] == pytest.approx([0.16, 0.16])
    assert value(rows, "timing", "reversal", "", "event_precision") == 1.0
    res, _ = run(tmp_path / "t", "--dead", DEAD, "--tags", TAGS, *NULL)
    assert res["errors"] == pytest.approx([-0.24, -0.24])


def test_change_auc_and_state(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, *NULL)
    assert res["n_pos"] == 102
    assert res["change_auc"][0.0] == pytest.approx(52 / 102.0)
    assert res["change_auc"][1.0] == pytest.approx(76 / 102.0)
    a, ad, cov = res["sep"][0.0]
    assert a == pytest.approx(590392 / 715000.0) and ad == 1.0
    assert cov == pytest.approx(792 / 1850.0)


def test_profile(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, *NULL)
    for role in ("gainer", "loser"):
        bins, val, counted, null = res["profiles"][role]
        assert bins[0] == -10 and bins[-1] == 9
        assert np.allclose(val[:10], -1, atol=0.01)
        assert val[10] == pytest.approx(0.76, abs=0.01)
        assert np.allclose(val[11:], 1, atol=0.01)
        assert counted[0] == 45 and counted[18] == 25 and counted[19] == 25 and counted[5] == 50
        assert null.shape == (50, 20)
    assert value(rows, "profile", "gainer", "0", "value") == 0.76


def test_flips_hysteresis():
    live = np.ones(10, dtype=bool)
    v = np.array([np.nan, 1, 0.3, -0.3, 0.2, -0.6, -1, 0.4, 0.6, np.nan])
    f, d = pr.flips(v, live, [(1, 0, 9)], 0.5)
    assert f.tolist() == [5, 8] and d.tolist() == [-1, 1]
    rev, rd = pr.joint((np.array([10, 50]), np.array([1, -1])),
                       (np.array([30, 52, 40]), np.array([1, -1, 1])), 25)
    assert rev.tolist() == [30, 52] and rd.tolist() == [1, -1]


def test_degrade_runs(tmp_path):
    res, _ = run(tmp_path, "--events", EVENTS, "--degrade", *NULL)
    assert res["n_live"] == 1850
