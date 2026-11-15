"""poss_ball_field.py on fixture_poss_*: expected values in fixtures/README.md.

Ball position and direction are continuous in the fixture, so `reference()` recomputes them
from the fixture text with plain loops, sharing no code with the script."""

import bisect
import csv
import math
import os

import pytest

import poss_ball_field as pbf
from test_poss_ball_live import BASE, DEAD, EVENTS, RAW, TAGS, value

RUNS = [(50, 499), (550, 999), (1050, 1999)]
LABEL_SPANS = [(50, 299, "H"), (300, 499, "A"), (550, 999, "H"), (1050, 1399, "A"),
               (1400, 1999, "H")]
FILLED = [(150, 152), (250, 261)]


def run(tmp_path, *extra):
    res = pbf.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), pbf.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def label(f):
    for a, b, t in LABEL_SPANS:
        if a <= f <= b:
            return t
    return None


def run_of(f):
    for i, (a, b) in enumerate(RUNS):
        if a <= f <= b:
            return i
    return None


def reference():
    ball = {}
    with open(RAW) as handle:
        for line in handle:
            fid, _, b = line.strip().split(":")
            x, y = b.split(",")
            if x != "NaN":
                ball[int(fid) - 1] = (float(x) * 105.0, float(y) * 68.0)
    for a, b in FILLED:
        x0, x1 = ball[a - 1][0], ball[b + 1][0]
        for f in range(a, b + 1):
            ball[f] = (x0 + (x1 - x0) * (f - a + 1) / (b - a + 2), None)
    pos, dirn, third = {}, {}, {}
    for f in range(2000):
        if run_of(f) is None or f not in ball:
            continue
        d = 1 if f < 1000 else -1
        xg = ball[f][0] if d > 0 else 105.0 - ball[f][0]
        pos[f] = xg - 52.5
        g = f - 5
        if run_of(g) == run_of(f) and g in ball:
            vx = (ball[f][0] - ball[g][0]) * 5.0 * d
            if abs(vx) >= 0.5:
                dirn[f] = 1.0 if vx > 0 else -1.0
        if label(f):
            third[("H", min(max(int(xg // 35), 0), 2))] = third.get(("H", min(max(
                int(xg // 35), 0), 2)), [0, 0])
            cell = third[("H", min(max(int(xg // 35), 0), 2))]
            cell[0] += 1
            cell[1] += label(f) == "H"
            xa = 105.0 - xg
            key = ("A", min(max(int(xa // 35), 0), 2))
            cell = third.setdefault(key, [0, 0])
            cell[0] += 1
            cell[1] += label(f) == "A"
    return pos, dirn, third


def pair_auc(scores):
    """AUC over labelled live frames, absent scores as 0, by sorting and bisecting."""
    h, a = [], []
    for f in range(2000):
        if run_of(f) is None:
            continue
        (h if label(f) == "H" else a).append(scores.get(f, 0.0))
    a.sort()
    total = 0.0
    for s in h:
        lo, hi = bisect.bisect_left(a, s), bisect.bisect_right(a, s)
        total += lo + 0.5 * (hi - lo)
    return total / (len(h) * len(a))


def test_team_drift(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    a, ad, cov = res["sep"]["team_drift"][0.0]
    assert a == pytest.approx(713089 / 715000)
    assert ad == pytest.approx(697553 / 699384)
    assert res["coverage"]["team_drift"] == pytest.approx(1831 / 1850)


def test_spot_values(tmp_path):
    import argparse
    import poss_common as pc
    p = argparse.ArgumentParser()
    pc.add_args(p, "x", "x")
    ctx = pc.load_context(p.parse_args([str(tmp_path)] + BASE + ["--events", EVENTS]), p)
    sig, _ = pbf.signals(ctx)
    assert sig["ball_position"][200] == pytest.approx(-12, abs=0.01)
    assert sig["ball_direction"][200] == 1 and sig["team_drift"][200] == 1
    assert math.isnan(sig["ball_position"][420]) and math.isnan(sig["ball_direction"][420])
    assert sig["team_drift"][420] == -1
    assert sig["ball_position"][1500] == pytest.approx(-12, abs=0.01)
    assert sig["ball_direction"][1500] == 1


def test_against_reference(tmp_path):
    res, _ = run(tmp_path, "--events", EVENTS)
    pos, dirn, third = reference()
    assert res["coverage"]["ball_position"] == pytest.approx(len(pos) / 1850.0)
    assert res["coverage"]["ball_direction"] == pytest.approx(len(dirn) / 1850.0)
    assert res["sep"]["ball_position"][0.0][0] == pytest.approx(pair_auc(pos), abs=1e-9)
    assert res["sep"]["ball_direction"][0.0][0] == pytest.approx(pair_auc(dirn), abs=1e-9)
    for (team, k), (m, hits) in third.items():
        got = res["thirds"][(team, pbf.THIRDS[k])]
        assert got[0] == m and got[1] == pytest.approx(hits / float(m))


def test_tags_and_degrade(tmp_path):
    res, rows = run(tmp_path / "t", "--dead", DEAD, "--tags", TAGS)
    assert [r["value"] for r in rows if r["metric"] == "source" and r["scope"] == "labels"] \
        == ["smart_tagging"]
    assert value(rows, "labels", "all", "", "labelled_frames") == 850
    res, _ = run(tmp_path / "d", "--events", EVENTS, "--degrade")
    assert res["coverage"]["ball_position"] < 0.8
