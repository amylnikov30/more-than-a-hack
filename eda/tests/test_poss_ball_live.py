"""poss_ball_live.py on fixture_poss_*: expected values in fixtures/README.md."""

import csv
import math
import os

import pytest

import poss_ball_live as bl
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_poss_atd.txt")
META = os.path.join(FIXTURES, "fixture_poss_atd.xml")
EVENTS = os.path.join(FIXTURES, "fixture_poss_events.json")
TAGS = os.path.join(FIXTURES, "fixture_poss_tags.xml")
DEAD = os.path.join(FIXTURES, "fixture_poss_dead.csv")
BASE = ["--raw", RAW, "--meta", META, "--frames", "0-999,1000-1999", "--pitch", "105", "68"]


def run(tmp_path, *extra):
    res = bl.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), bl.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def value(rows, analysis, scope, bin_, metric):
    hits = [r["value"] for r in rows if (r["analysis"], r["scope"], r["bin"], r["metric"]) ==
            (analysis, scope, bin_, metric)]
    assert len(hits) == 1, (analysis, scope, bin_, metric)
    return float(hits[0]) if hits[0] != "" else math.nan


def test_coverage(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["total"] == {"live": 1850, "observed": 1611, "interpolated": 15, "missing": 224}
    assert res["halves"][1] == {"live": 900, "observed": 809, "interpolated": 15, "missing": 76}
    assert res["halves"][2] == {"live": 950, "observed": 802, "interpolated": 0, "missing": 148}
    assert res["missing_before"] == 239
    assert value(rows, "coverage", "all", "observed", "share") == 0.871
    assert value(rows, "coverage", "all", "interpolated", "share") == 0.00811
    assert value(rows, "coverage", "all", "missing", "share") == 0.121
    assert value(rows, "coverage", "all", "missing_before_interp", "share") == 0.129


def test_gap_bins(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["before"] == {"<0.5s": (4, 31), "0.5-2s": (2, 33), "2-5s": (1, 50),
                             ">=5s": (1, 125), "all": (8, 239)}
    assert res["after"] == {"<0.5s": (2, 16), "0.5-2s": (2, 33), "2-5s": (1, 50),
                            ">=5s": (1, 125), "all": (6, 224)}
    assert [value(rows, "gaps_before", "all", b, "share") for b in bl.BINS] == \
        [0.13, 0.138, 0.209, 0.523]
    assert [value(rows, "gaps_after", "all", b, "share") for b in bl.BINS] == \
        [0.0714, 0.147, 0.223, 0.558]
    assert res["q"] == pytest.approx([0.204, 0.5, 2.9])


def test_caps_and_dead(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["caps"] == {6: 3, 12: 15, 25: 28}
    assert res["dead"] == 150 and res["dead_missing"] == 120
    assert value(rows, "dead", "all", "", "missing_share") == 0.8


def test_changes_events(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["n_changes"] == 2 and res["around"] == 202 and res["rest"] == 1648
    assert (res["around_before"], res["around_after"]) == (87, 75)
    assert (res["rest_before"], res["rest_after"]) == (152, 149)
    assert value(rows, "change", "around", "before", "missing_share") == 0.431
    assert value(rows, "change", "rest", "after", "missing_share") == 0.0904


def test_changes_tags(tmp_path):
    res, rows = run(tmp_path, "--dead", DEAD, "--tags", TAGS)
    assert res["n_changes"] == 2
    assert (res["around_before"], res["around_after"]) == (67, 65)
    assert (res["rest_before"], res["rest_after"]) == (172, 159)
    assert value(rows, "agreement", "code:BUILD UP", "", "agree_share") == 1.0
    assert value(rows, "agreement", "code:LONG BALL", "", "n_frames") == 40
    assert value(rows, "labels", "all", "", "dropped_frames") == 40


def test_same_rows_for_both_label_sources(tmp_path):
    _, a = run(tmp_path / "a", "--events", EVENTS)
    _, b = run(tmp_path / "b", "--dead", DEAD, "--tags", TAGS)
    key = lambda rows: [(r["analysis"], r["scope"], r["bin"], r["metric"]) for r in rows]  # noqa
    assert key(a) == key(b)


def test_no_labels(tmp_path):
    res, rows = run(tmp_path, "--dead", DEAD)
    assert res["n_changes"] == 0
    assert math.isnan(value(rows, "change", "around", "before", "missing_share"))
    assert res["total"]["live"] == 1850


def test_degrade(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS, "--degrade")
    t = res["total"]
    assert res["missing_before"] / t["live"] >= 0.41
    assert value(rows, "params", "all", "", "degrade") == 1
    assert res["dead_missing"] == 120          # dead frames untouched


def test_refuses_repo_out_dir():
    with pytest.raises(SystemExit):
        bl.main([os.path.join(FIXTURES, "out")] + BASE + ["--events", EVENTS])


def test_needs_dead_or_events(tmp_path):
    with pytest.raises(SystemExit):
        bl.main([str(tmp_path)] + BASE + ["--tags", TAGS])
