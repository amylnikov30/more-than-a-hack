"""poss_shape.py on fixture_poss_*: expected values in fixtures/README.md."""

import argparse
import csv
import os

import pytest

import poss_common as pc
import poss_shape as ps
from test_poss_ball_live import BASE, DEAD, EVENTS, TAGS, value


def run(tmp_path, *extra):
    res = ps.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), ps.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def context(tmp_path, *extra):
    p = argparse.ArgumentParser()
    pc.add_args(p, "x", "x")
    return pc.load_context(p.parse_args([str(tmp_path)] + BASE + list(extra)), p)


@pytest.mark.parametrize("frame,home,away", [
    (200, {"width": 30, "length": 12, "hull_area": 276, "centroid_advance": 46,
           "centroid_velocity": 1, "line_depth": 40, "goal_side": 0.5},
     {"width": 18, "length": 12, "hull_area": 165.6, "centroid_advance": 49,
      "centroid_velocity": -1, "line_depth": 43, "goal_side": 1.0}),
    (420, {"goal_side": 1.0, "centroid_advance": 45.2}, {"goal_side": 1.0}),
])
def test_features_on_one_frame(tmp_path, frame, home, away):
    feats, _ = ps.team_features(context(tmp_path, "--events", EVENTS))
    for t, expected in ((0, home), (1, away)):
        for name, v in expected.items():
            tol = 0.05 if name == "hull_area" else 0.01
            assert feats[t][name][frame] == pytest.approx(v, abs=tol), (t, name)


def test_hull_area():
    assert ps.hull_area([0, 1, 1, 0, 0.5], [0, 0, 1, 1, 0.5]) == pytest.approx(1.0)
    assert ps.hull_area([0, 1], [0, 1]) == 0.0


def test_diff_and_auc(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    f = res["features"]
    assert all(f[k]["coverage"] == 1.0 for k in f if k != "centroid_velocity")
    assert f["centroid_velocity"]["coverage"] == pytest.approx(1835 / 1850)
    assert f["width"]["diff"]["all"][0] == pytest.approx(11.494, abs=0.01)
    assert f["width"]["diff"]["all"][1] == pytest.approx(12, abs=0.01)
    assert f["width"]["diff"]["all"][2] == 1850
    assert f["hull_area"]["diff"]["all"][0] == pytest.approx(105.75, abs=0.05)
    assert f["hull_area"]["diff"]["all"][1] == pytest.approx(110.4, abs=0.05)
    assert f["length"]["diff"]["all"][1] == pytest.approx(0, abs=0.01)
    assert f["width"]["sep"]["all"][0.0][0] == pytest.approx(0.9974, abs=0.0017)
    assert f["width"]["diff"]["no_ball"][2] == 224
    assert f["width"]["diff"]["ball"][2] == 1626
    assert value(rows, "hypothesis", "goal_side", "", "sign") == -1


def test_line_depth(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    d = res["depth"]
    expected = {("H", "defending"): ([30.53, 36.02, 38.76], 550),
                ("A", "defending"): ([38.03, 42.68, 47.01], 1300),
                ("H", "attacking"): ([35.99, 40.32, 44.97], 1300),
                ("A", "attacking"): ([44.24, 46.98, 52.47], 550)}
    for key, (q, m) in expected.items():
        assert d[key][1] == m
        assert d[key][0] == pytest.approx(q, abs=0.01), key
    assert value(rows, "line_depth", "H:defending", "p50", "depth") == 36.0


def test_tags_and_degrade_run(tmp_path):
    res, _ = run(tmp_path / "t", "--dead", DEAD, "--tags", TAGS)
    assert res["features"]["width"]["diff"]["all"][2] == 850
    res, _ = run(tmp_path / "d", "--events", EVENTS, "--degrade")
    assert res["features"]["width"]["coverage"] < 1.0
