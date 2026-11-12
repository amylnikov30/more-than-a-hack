"""ball_outside.py on fixture_outside_*. Expected values: tests/fixtures/README.md
("ball_outside.py")."""

import csv
import os

import numpy as np
import pytest

import ball_gaps as bg
import ball_outside as bo
import common
import tag_gaps as tg
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_outside_atd.txt")
TAGS = os.path.join(FIXTURES, "fixture_outside_tags.xml")
BLOCKS = [(1, 0, 999), (2, 1000, 1999)]
SP = "code:SET PIECES"


@pytest.fixture(scope="module")
def res():
    return bo.analyse(RAW, BLOCKS, TAGS, null_n=50, null_min_shift_s=5)


@pytest.mark.parametrize("margin, runs, frames, goal, touch, follows, linked", [
    (0.0, 9, 323, 58, 265, (4, 4, 1), 6),
    (1.0, 8, 273, 58, 215, (3, 4, 1), 6),
    (2.0, 6, 253, 48, 205, (3, 3, 0), 4),
])
def test_margins(res, margin, runs, frames, goal, touch, follows, linked):
    r = res["by_margin"][margin]
    assert r["visible"] == 1824
    assert r["out_frames"] == {"all": frames, "goal_line": goal, "touchline": touch}
    f = r["follows"]["all"]
    assert f["n_runs"] == runs
    assert (f["missing"], f["inside"], f["half_end"]) == follows
    assert r["link"][SP]["all"]["real_count"] == linked
    assert r["link"][SP]["all"]["n_gaps"] == runs


def test_order_of_margins(res):
    assert res["margins"] == [1.0, 0.0, 2.0]


def test_run_lengths_and_lines(res):
    r = res["by_margin"][1.0]
    assert r["runs"]["all"] == {"<0.5s": (6, 48), "0.5-2s": (1, 25), "2-5s": (0, 0),
                                ">=5s": (1, 200)}
    assert r["follows"]["goal_line"]["n_runs"] == 5
    assert r["follows"]["touchline"]["n_runs"] == 3


def test_links_by_type_and_line(res):
    link = res["by_margin"][1.0]["link"]
    expect = {SP: (6, 4, 2), "type:Corner Kick": (2, 2, 0), "type:Free Kick": (2, 2, 0),
              "type:Goal Kick": (2, 1, 1), "type:Throw-in": (2, 0, 2)}
    assert set(link) == set(expect)
    for name, counts in expect.items():
        got = tuple(link[name][line]["real_count"] for line in bo.LINES)
        assert got == counts, name


def test_crosstab(res):
    c = res["by_margin"][1.0]["crosstab"]
    assert c["goal_line"] == {"Goal Kick": 1, "Corner Kick": 2, "Free Kick": 1, "none": 1}
    assert c["touchline"] == {"Throw-in": 2, "none": 1}


def test_null_offset_zero_reproduces_observed():
    offsets = {1: np.zeros(3, dtype=np.int64), 2: np.zeros(3, dtype=np.int64)}
    ball, _ = bg.read_ball(RAW)
    visible = np.isfinite(ball).all(axis=1)
    ex, ey = bo.excess(ball)
    graded = common.graded_half_of(BLOCKS, len(visible)) > 0
    groups = common.set_piece_groups(tg.read_tags(TAGS))
    r = bo.analyse_margin(1.0, ball, visible, graded, BLOCKS, ex, ey, groups, offsets, 125, 25)
    st = r["link"][SP]["all"]
    assert st["null_mean"] == pytest.approx(6 / 8) and st["p_emp"] == pytest.approx(1.0)


def test_same_seed_same_null(res):
    again = bo.analyse(RAW, BLOCKS, TAGS, null_n=50, null_min_shift_s=5)
    a = res["by_margin"][1.0]["link"][SP]["all"]
    b = again["by_margin"][1.0]["link"][SP]["all"]
    assert a == b


def test_without_tags():
    r = bo.analyse(RAW, BLOCKS)
    assert r["by_margin"][1.0]["link"] is None and r["by_margin"][1.0]["crosstab"] is None
    assert r["by_margin"][1.0]["out_frames"]["all"] == 273


def test_find_runs_matches_find_gaps():
    ball, _ = bg.read_ball(os.path.join(FIXTURES, "fixture_atd.txt.gz"))
    missing = ~np.isfinite(ball).all(axis=1)
    blocks = [(1, 100, 1499), (2, 1600, 2899)]
    gaps = [(g["half"], g["start"], g["end"]) for g in bg.find_gaps(missing, blocks)]
    runs = [(r["half"], r["start"], r["end"]) for r in common.find_runs(missing, blocks)]
    assert runs == gaps


def test_main_writes_csv(tmp_path, capsys):
    out = tmp_path / "out"
    bo.main([str(out), "--raw", RAW, "--tags", TAGS, "--frames", "0-999,1000-1999",
             "--null-n", "20", "--null-min-shift", "5"])
    with open(out / bo.CSV_NAME) as handle:
        rows = list(csv.DictReader(handle))
    v = {(r["analysis"], r["scope"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert float(v[("share", "m1:all", "", "share")]) == pytest.approx(273 / 1824)
    assert v[("link", "m1:goal_line", SP, "real_count")] == "4"
    assert v[("crosstab", "m1:touchline", "Throw-in", "n_runs")] == "2"
    assert len(capsys.readouterr().out.strip().splitlines()) <= 10
