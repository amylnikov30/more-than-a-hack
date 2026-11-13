"""poss_durations.py on fixture_poss_*: expected values in fixtures/README.md."""

import csv
import math
import os

import pytest

import poss_durations as pd_
from test_poss_ball_live import BASE, DEAD, EVENTS, TAGS, value


def run(tmp_path, *extra):
    res = pd_.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), pd_.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def test_durations(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    a = res["all"]
    assert a["n"] == 5 and a["q"] == pytest.approx([8.8, 10, 14, 18, 21.6])
    assert a["mean"] == pytest.approx(14.8)
    assert a["bins"] == {"0-2s": 0, "2-5s": 0, "5-10s": 1, "10-20s": 3, "20-40s": 1,
                         ">=40s": 0}
    assert res["H"]["n"] == 3 and res["H"]["q"][2] == 18
    assert res["A"]["n"] == 2 and res["A"]["q"][2] == 11
    assert res["ends"]["turnover"] == (2, 0.4, 12.0)
    assert res["ends"]["dead"] == (1, 0.2, 8.0)
    assert res["ends"]["half_end"] == (2, 0.4, 21.0)
    assert res["turnover_share"] == pytest.approx(2 / 3.0)
    assert res["log_mean"] == pytest.approx(2.618, abs=1e-3)
    assert res["log_sd"] == pytest.approx(0.441, abs=1e-3)
    h = res["hazard"]
    assert h["5-10s"] == (5, 1, 0.2) and h["10-20s"] == (4, 2, 0.5)
    assert h["20-40s"] == (1, 0, 0.0) and h[">=40s"][0] == 0 and math.isnan(h[">=40s"][2])
    assert value(rows, "ends", "ended", "", "turnover_share") == 0.667


def test_needs_events(tmp_path):
    with pytest.raises(SystemExit):
        pd_.main([str(tmp_path)] + BASE + ["--dead", DEAD, "--tags", TAGS])
