"""speed_units.py on fixture_speed_atd. Expected values: tests/fixtures/README.md
("speed_units.py")."""

import csv
import os

import numpy as np
import pytest

import common
import speed_units as su
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_speed_atd.txt")
META = os.path.join(FIXTURES, "fixture_speed_atd.xml")
BLOCKS = [(1, 0, 5), (2, 6, 11)]
A = pytest.approx


@pytest.fixture(scope="module")
def res():
    return su.analyse(RAW, BLOCKS, META)


def test_players(res):
    r = res["classes"]["players"]
    assert (r["n_pairs"], r["n_over_cap"], r["n_ratio"]) == (28, 1, 17)
    assert (r["ratio_median"], r["ratio_p25"], r["ratio_p75"]) == (A(3.6), A(3.6), A(4.0))
    assert r["derived_q"] == [A(2.125), A(128.35), A(169.66)]
    assert r["n_declared"] == 35
    assert r["declared_q"] == [A(7.65), A(17.0), A(17.0)]
    assert r["verdict"] == "km/h"


def test_ball(res):
    r = res["classes"]["ball"]
    assert (r["n_pairs"], r["n_over_cap"], r["n_ratio"]) == (7, 0, 7)
    assert (r["ratio_median"], r["ratio_p25"], r["ratio_p75"]) == (A(3.6), A(3.6), A(3.6))
    assert r["derived_q"] == [A(8.5)] * 3
    assert r["n_declared"] == 10 and r["declared_q"] == [A(30.6)] * 3
    assert r["verdict"] == "km/h"


def test_overall_and_unit(res):
    assert res["verdict"] == "km/h"
    assert res["declared_unit"] == "meters per second"
    assert res["skipped_steps"] == 0


def test_one_block_adds_the_halftime_step():
    r = su.analyse(RAW, [(1, 0, 11)])["classes"]
    # eleven ratios of 3.6 (slot 1) and eight of 4.0 (slot 3 gains 5->6)
    assert (r["players"]["n_pairs"], r["players"]["n_ratio"]) == (31, 19)
    assert (r["players"]["ratio_median"], r["players"]["ratio_p75"]) == (A(3.6), A(4.0))
    assert r["ball"]["n_pairs"] == 7


def _rewrite(tmp_path, fn):
    out = tmp_path / "raw.txt"
    with open(RAW) as src:
        lines = [fn(line) for line in src]
    out.write_text("".join(l for l in lines if l is not None))
    return str(out)


def test_no_speed_field(tmp_path):
    def strip(line):
        head, body, ball = line.rstrip("\n").split(":")
        cut = lambda part: ";".join(",".join(g.split(",")[:2]) for g in part.split(";"))  # noqa
        return "{}:{}:{}\n".format(head, cut(body), cut(ball))
    res = su.analyse(_rewrite(tmp_path, strip), BLOCKS)
    assert res["verdict"] == "no declared speed"
    assert res["classes"]["players"]["n_ratio"] == 0
    assert res["classes"]["players"]["derived_q"][0] == A(2.125)
    assert res["declared_unit"] is None


def test_missing_line_breaks_steps(tmp_path):
    raw = _rewrite(tmp_path, lambda l: None if l.startswith("4:") else l)
    r = su.analyse(raw, BLOCKS)["classes"]
    # steps 2->3 and 3->4 drop for every object: slot 1 and 2 lose 2 each, slot 3 already had
    # them missing, the ball loses 2
    assert r["players"]["n_pairs"] == 24 and r["ball"]["n_pairs"] == 5


def test_stream_and_halves():
    frames = [f for f, _, _ in common.stream_frames(RAW)]
    assert frames == list(range(12))
    half = common.graded_half_of(BLOCKS)
    assert common.is_step(4, 5, half) and not common.is_step(5, 6, half)
    assert not common.is_step(None, 0, half) and not common.is_step(11, 12, half)


def test_verdict_bands():
    assert su.verdict(1.08) == "m/s" and su.verdict(3.25) == "km/h"
    assert su.verdict(2.0) == "unclear" and su.verdict(float("nan")) is None
    assert su.overall_verdict(["km/h", None]) == "km/h"
    assert su.overall_verdict(["km/h", "m/s"]) == "conflict"


def test_main_writes_csv(tmp_path, capsys):
    out = tmp_path / "out"
    su.main([str(out), "--raw", RAW, "--meta", META, "--frames", "0-5,6-11"])
    with open(out / su.CSV_NAME) as handle:
        rows = list(csv.DictReader(handle))
    v = {(r["analysis"], r["scope"], r["metric"]): r["value"] for r in rows}
    assert v[("verdict", "overall", "verdict")] == "km/h"
    assert len(capsys.readouterr().out.strip().splitlines()) <= 10
    with pytest.raises(SystemExit):
        su.main([os.path.join(FIXTURES, "x"), "--raw", RAW, "--frames", "0-11"])
    assert np.isfinite(float(v[("ratio", "players", "n_pairs")]))
