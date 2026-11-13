"""poss_restarts.py on fixture_poss_*: expected values in fixtures/README.md."""

import csv
import math
import os

import numpy as np
import pytest

import poss_common as pc
import poss_restarts as prs
from test_poss_ball_live import BASE, DEAD, EVENTS, TAGS, value

H, A = pc.H, pc.A


def run(tmp_path, *extra):
    res = prs.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), prs.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def test_events_with_tags(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, "--tags", TAGS)
    assert res["n"] == 3
    assert res["coverage"] == {"geometric": 3, "last_opposite": 1, "label_opposite": 1,
                               "tag_team": 2, "label_after": 3}
    assert res["accuracy"]["geometric"] == (3, 1.0) and res["accuracy"]["tag_team"] == (2, 1.0)
    assert res["accuracy"]["last_opposite"] == (1, 1.0)
    ag = res["agreement"]
    assert ag[("geometric", "last_opposite")] == (1, 1.0)
    assert ag[("geometric", "tag_team")] == (2, 1.0)
    assert ag[("geometric", "label_after")] == (3, 1.0)
    assert ag[("last_opposite", "tag_team")][0] == 0
    assert math.isnan(ag[("last_opposite", "tag_team")][1])
    assert value(rows, "agreement_matrix", "geometric|tag_team", "", "agree_share") == 1.0
    assert value(rows, "restarts", "truth", "", "home_share") == pytest.approx(0.667, abs=1e-3)


def test_predictions(tmp_path):
    import argparse
    p = argparse.ArgumentParser()
    pc.add_args(p, "x", "x")
    ctx = pc.load_context(p.parse_args([str(tmp_path)] + BASE + ["--events", EVENTS]), p)
    assert prs.restarts(ctx) == [(0, 50), (500, 550), (1000, 1050)]
    preds = prs.predict(ctx, TAGS)
    assert [q["geometric"] for q in preds] == [H, H, A]
    assert [q["last_opposite"] for q in preds] == [0, H, 0]
    assert [q["label_opposite"] for q in preds] == [0, H, 0]
    assert [q["tag_team"] for q in preds] == [H, 0, A]
    assert [q["label_after"] for q in preds] == [H, H, A]
    assert [q["truth"] for q in preds] == [H, H, A]


def test_tags_run(tmp_path):
    res, rows = run(tmp_path, "--dead", DEAD, "--tags", TAGS)
    assert res["coverage"]["label_after"] == 2
    assert math.isnan(res["accuracy"]["geometric"][1])
    assert res["agreement"][("geometric", "label_after")] == (2, 1.0)
    assert res["coverage"]["label_opposite"] == 1


def test_throw_in_taker(tmp_path):
    import argparse
    p = argparse.ArgumentParser()
    pc.add_args(p, "x", "x")
    ctx = pc.load_context(p.parse_args([str(tmp_path)] + BASE + ["--events", EVENTS]), p)
    ctx["players"] = ctx["players"].copy()
    ctx["players"][1050, 1] = (50.0, 69.0)        # home 2 beyond the far touchline
    assert prs.predict(ctx, None)[2]["geometric"] == H


def test_degrade_runs(tmp_path):
    res, _ = run(tmp_path, "--events", EVENTS, "--degrade")
    assert res["n"] == 3
