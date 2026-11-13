"""poss_contested.py on fixture_poss_*: hand facts in fixtures/README.md, the rest against
`reference()`, a plain per-frame recomputation from the fixture text sharing no code with the
script."""

import csv
import math
import os

import numpy as np
import pytest

import poss_contested as pco
from test_poss_ball_field import RUNS, FILLED, label, pair_auc, run_of
from test_poss_ball_live import BASE, DEAD, EVENTS, RAW, TAGS, value

HOME, AWAY = range(0, 5), range(5, 10)        # 0-based slots; slot index 10 is the referee


def run(tmp_path, *extra):
    res = pco.main([str(tmp_path)] + BASE + list(extra))
    with open(os.path.join(str(tmp_path), pco.CSV_NAME)) as handle:
        rows = list(csv.DictReader(handle))
    return res, rows


def reference():
    players, ball = {}, {}
    with open(RAW) as handle:
        for line in handle:
            fid, pl, b = line.strip().split(":")
            f = int(fid) - 1
            players[f] = [(float(g.split(",")[0]) * 105.0, float(g.split(",")[1]) * 68.0)
                          for g in pl.split(";")]
            if not b.startswith("NaN"):
                ball[f] = (float(b.split(",")[0]) * 105.0, float(b.split(",")[1]) * 68.0)
    for a, b in FILLED:
        (x0, y0), (x1, y1) = ball[a - 1], ball[b + 1]
        for f in range(a, b + 1):
            t = (f - a + 1) / float(b - a + 2)
            ball[f] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    nt = {}
    for f in range(2000):
        if run_of(f) is None or f not in ball:
            continue
        best = min((math.hypot(players[f][j][0] - ball[f][0], players[f][j][1] - ball[f][1]), j)
                   for j in list(HOME) + list(AWAY))
        nt[f] = 0 if best[1] in HOME else 1
    flips = {f for f in nt if f - 1 in nt and run_of(f - 1) == run_of(f) and nt[f] != nt[f - 1]}
    dens, cont = {}, {}
    for f in range(2000):
        r = run_of(f)
        if r is None:
            continue
        win = [g for g in range(f - 25, f + 26) if run_of(g) == r and g in nt]
        if win:
            dens[f] = 25.0 * sum(g in flips for g in win) / len(win)
            cont[f] = 1.0 if sum(g in flips for g in win) >= 2 else 0.0
    return nt, flips, dens, cont


def ctx_sets():
    around = {f for c in (300, 1400) for f in range(c - 38, c + 39)}
    loose = {f for c in (700, 1200) for f in range(c - 38, c + 39)}
    mid = set()
    for f in range(2000):
        r = run_of(f)
        if r is None:
            continue
        a, b = RUNS[r]
        if f - a <= 125 or b - f <= 125:
            continue
        if any(abs(f - c) <= 125 for c in (300, 1400, 700, 1200)):
            continue
        mid.add(f)
    return {"around_change": around, "loose_no_change": loose, "mid": mid,
            "all": {f for f in range(2000) if run_of(f) is not None}}


def test_against_reference(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    nt, flips, dens, cont = reference()
    assert res["coverage"] == pytest.approx(len(nt) / 1850.0)
    assert res["n_flips"] == len(flips)
    got = {f: int(v) for f, v in enumerate(res["nt"]) if v >= 0}
    assert got == nt
    for name, frames in ctx_sets().items():
        sel = [f for f in frames if f in dens]
        m, rate, share = res["contexts"][name]
        assert m == len(sel), name
        assert rate == pytest.approx(np.mean([dens[f] for f in sel])), name
        assert share == pytest.approx(np.mean([cont[f] for f in sel])), name
    score = {f: (1.0 if t == 0 else -1.0) for f, t in nt.items()}
    assert res["sep"][0.0][0] == pytest.approx(pair_auc(score), abs=1e-9)


def test_hand_facts(tmp_path):
    res, rows = run(tmp_path, "--events", EVENTS)
    assert res["n_changes"] == 2 and res["n_loose"] == 2
    m = res["masks"]
    assert np.flatnonzero(m["loose_no_change"]).tolist() == list(range(662, 739)) + \
        list(range(1162, 1239))
    assert np.flatnonzero(m["around_change"]).tolist() == list(range(262, 339)) + \
        list(range(1362, 1439))
    assert value(rows, "contexts", "anchors", "", "n_loose") == 2


def test_tags(tmp_path):
    res, rows = run(tmp_path, "--dead", DEAD, "--tags", TAGS)
    assert res["contexts"]["loose_no_change"][0] == 0
    assert math.isnan(value(rows, "contexts", "anchors", "", "n_loose"))
    assert np.flatnonzero(res["masks"]["around_change"]).tolist() == \
        list(range(272, 349)) + list(range(1372, 1449))


def test_degrade_runs(tmp_path):
    res, _ = run(tmp_path, "--events", EVENTS, "--degrade")
    assert res["coverage"] < 0.8
