"""poss_pass_chains.py on fixture_poss_*: expected values in fixtures/README.md."""

import csv
import math
import os

import pytest

import poss_common as pc
import poss_pass_chains as ppc
from test_poss_ball_live import BASE, DEAD, EVENTS, TAGS, value


def run(tmp_path, *extra):
    res = ppc.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), ppc.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def test_passes_and_chains(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["n_contacts"] == 14
    assert res["n_pass"] == {pc.H: 4, pc.A: 2}
    assert res["hist"] == {"1": 2, "2": 0, "3": 0, "4": 1, "5+": 0}
    assert res["qual_frames"] == 225 and res["chain_frames"] == 565
    assert value(rows, "coverage", "min_chain", "", "share") == 0.122
    assert value(rows, "coverage", "chain", "", "share") == 0.305


def test_precision_auc_events(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["precision"]["min_chain"] == (225, 1.0)
    assert res["precision"]["1"] == (340, 1.0) and res["precision"]["4"] == (225, 1.0)
    assert res["precision"]["2"][0] == 0 and math.isnan(res["precision"]["2"][1])
    a, ad, cov = res["sep"][0.0]
    assert a == pytest.approx(762.5 / 1300) and math.isnan(ad)
    assert value(rows, "separation", "chain", "0", "auc") == 0.587
    assert res["errors"] == [] and res["n_switches"] == 0
    assert value(rows, "timing", "chain_switch", "", "matched_share") == 0.0


def test_min_chain_one(tmp_path):
    res, _ = run(tmp_path, "--events", EVENTS, "--min-chain", "1")
    assert res["qual_frames"] == 565
    assert res["errors"] == pytest.approx([0.2]) and (res["near"], res["n_switches"]) == (1, 1)


def test_tags(tmp_path):
    res, _ = run(tmp_path, "--dead", DEAD, "--tags", TAGS)
    assert res["precision"]["min_chain"] == (220, 1.0)


def test_no_speed_is_no_pass():
    import numpy as np
    speed = np.full(20, np.nan)
    rid = np.zeros(20, dtype=int)
    c1, c2 = [0, 4, 1, pc.H], [10, 15, 2, pc.H]
    assert not ppc.is_pass(c1, c2, speed, rid, 25, 5.0, 5.0)
    speed[10] = 6.0
    assert ppc.is_pass(c1, c2, speed, rid, 25, 5.0, 5.0)
    assert not ppc.is_pass(c1, [10, 15, 1, pc.H], speed, rid, 25, 5.0, 5.0)   # same slot
    assert not ppc.is_pass(c1, [10, 15, 2, pc.A], speed, rid, 25, 5.0, 5.0)   # other team
    assert not ppc.is_pass(c1, c2, speed, rid, 25, 5.0, 0.2)                  # too long


def test_degrade_runs(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, "--degrade")
    assert value(rows, "params", "all", "", "degrade") == 1
