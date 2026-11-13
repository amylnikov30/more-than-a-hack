"""Shared possession preparation on fixture_poss_*: expected values in fixtures/README.md
("Possession fixture")."""

import argparse
import math
import os

import numpy as np
import pytest

import poss_common as pc
import poss_reference as pr
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_poss_atd.txt")
META = os.path.join(FIXTURES, "fixture_poss_atd.xml")
EVENTS = os.path.join(FIXTURES, "fixture_poss_events.json")
TAGS = os.path.join(FIXTURES, "fixture_poss_tags.xml")
DEAD = os.path.join(FIXTURES, "fixture_poss_dead.csv")
BLOCKS = [(1, 0, 999), (2, 1000, 1999)]
N = 2000


def parse(*extra, need_players=True):
    p = argparse.ArgumentParser()
    pc.add_args(p, "x.csv", "~/eda_out/x")
    args = p.parse_args(["/tmp/unused", "--raw", RAW, "--meta", META, "--frames",
                         "0-999,1000-1999", "--pitch", "105", "68"] + list(extra))
    return pc.load_context(args, p, need_players=need_players)


def runs_of(mask):
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def test_reference_labels():
    evs = pr.read_events(EVENTS)
    labels, dead = pr.reference_labels(evs, ["T1", "T2"], BLOCKS, N)
    assert runs_of(dead) == [(0, 49), (500, 549), (1000, 1049)]
    live = ~dead
    lab = np.where(live, labels, 0)
    assert int(np.sum(lab == pc.H)) == 1300 and int(np.sum(lab == pc.A)) == 550
    assert pc.possessions(lab, live, BLOCKS) == [(50, 299, 1), (300, 499, 2), (550, 999, 1),
                                                 (1050, 1399, 2), (1400, 1999, 1)]
    assert pc.label_changes(lab, live, BLOCKS).tolist() == [300, 1400]


def test_goal_kick_is_a_restart_and_card_is_skipped():
    evs = [{"kind": "SET PIECE", "subtypes": {"KICK OFF"}, "team": "T1", "start": 10,
            "end": 10, "period": 1},
           {"kind": "SHOT", "subtypes": {"OFF TARGET"}, "team": "T1", "start": 30, "end": 40,
            "period": 1},
           {"kind": "CARD", "subtypes": {"YELLOW"}, "team": "T2", "start": 45, "end": 60,
            "period": 1},
           {"kind": "RECOVERY", "subtypes": set(), "team": "T2", "start": 70, "end": 70,
            "period": 1},
           {"kind": "PASS", "subtypes": {"GOAL KICK"}, "team": "T2", "start": 70, "end": 80,
            "period": 1},
           {"kind": "BALL LOST", "subtypes": {"END HALF"}, "team": "T2", "start": 90,
            "end": 92, "period": 1}]
    labels, dead = pr.reference_labels(evs, ["T1", "T2"], [(1, 0, 99)], 100)
    assert runs_of(dead) == [(0, 9), (40, 69), (92, 99)]
    assert labels[69] == pc.H and labels[70] == pc.A and labels[5] == 0


def test_ball_out_without_restart():
    evs = [{"kind": "SET PIECE", "subtypes": {"KICK OFF"}, "team": "T1", "start": 0, "end": 0,
            "period": 1},
           {"kind": "SHOT", "subtypes": {"OFF TARGET", "OUT"}, "team": "T1", "start": 20,
            "end": 30, "period": 1},
           {"kind": "CARRY", "subtypes": set(), "team": "T1", "start": 20, "end": 21,
            "period": 1},
           {"kind": "BALL LOST", "subtypes": {"FORCED"}, "team": "T2", "start": 60, "end": 70,
            "period": 1}]
    _, dead = pr.reference_labels(evs, ["T1", "T2"], [(1, 0, 99)], 100)
    assert runs_of(dead) == [(30, 59)]


def test_context_events_and_dead_csv_agree():
    ev = parse("--events", EVENTS)
    dc = parse("--dead", DEAD, "--tags", TAGS)
    assert int(ev["live"].sum()) == 1850 and np.array_equal(ev["live"], dc["live"])
    assert ev["label_source"] == "events" and ev["live_source"] == "events"
    assert dc["label_source"] == "smart_tagging" and dc["live_source"] == "dead_csv"
    assert int(np.sum(ev["graded"] & ~ev["live"])) == 150


def test_dead_csv_minus_one_is_not_live(tmp_path):
    path = tmp_path / "dead.csv"
    path.write_text("frame,dead,p_dead,llr_dead\n0,0,,\n1,1,,\n2,-1,,\n")
    assert pc.read_dead_csv(str(path), 5).tolist() == [0, 1, -1, -1, -1]


def test_slot_teams_and_directions():
    ctx = parse("--events", EVENTS)
    team = ctx["team"][0]
    assert (team == 0).sum() == 5 and (team == 1).sum() == 5 and (team == -1).sum() == 1
    assert ctx["dir_source"] == "metadata" and ctx["dir_agree"] == 1.0
    assert ctx["dirs"] == {(0, 1): 1, (1, 1): -1, (0, 2): -1, (1, 2): 1}
    inferred = pc.infer_directions(ctx["players"], ctx["team"], BLOCKS, ctx["live"])
    assert inferred == ctx["dirs"]
    assert pc.toward_goal(30.0, -1, 105.0) == 75.0 and pc.toward_goal(30.0, 1, 105.0) == 30.0


def test_directions_without_metadata(tmp_path):
    text = open(META).read().replace("attack_direction_first_half", "something_else")
    meta = tmp_path / "nodir.xml"
    meta.write_text(text)
    ctx = parse("--events", EVENTS)
    dirs, source, agree = pc.attack_directions(str(meta), ctx["players"], ctx["team"], BLOCKS,
                                               ctx["live"])
    assert source == "centroids" and math.isnan(agree) and dirs == ctx["dirs"]


def test_tracking_in_metres():
    ctx = parse("--events", EVENTS)
    # slot 1 is the home goalkeeper at (5, 34) m in half 1
    assert ctx["players"][100, 0].tolist() == pytest.approx([5.0, 34.0], abs=1e-3)
    assert np.isnan(ctx["ball"][150]).all() and np.isfinite(ctx["ball"][149]).all()


def test_tag_team_map():
    m = pc.tag_team_map(["Alpha", "Beta Club", "N/A"], ["Alpha FC", "Beta Club"])
    assert m == {"Alpha": 0, "Beta Club": 1}
    m = pc.tag_team_map(["Barcelona", "Atletic Club", "N/A"], ["FC Barcelona", "Athletic Club"])
    assert m == {"Barcelona": 0, "Atletic Club": 1}
    assert pc.tag_team_map(["X", "Y"], ["Home", "Away"], "x,y") == {"X": 0, "Y": 1}


def test_pseudo_labels():
    ctx = parse("--dead", DEAD, "--tags", TAGS)
    labels, stats = ctx["labels"], ctx["pseudo"]
    assert stats["labelled"] == 850 and stats["dropped"] == 40
    assert int(np.sum(labels == pc.H)) == 230 + 190
    assert int(np.sum(labels == pc.A)) == 140 + 290
    assert pc.label_changes(labels, ctx["live"], BLOCKS).tolist() == [310, 1410]
    ag = stats["agreement"]
    assert ag["BUILD UP"] == (230, 1.0) and ag["DEFENDING IN MIDDLE THIRD"] == (230, 1.0)
    assert ag["PROGRESSION"] == (40, 0.0) and ag["LONG BALL"] == (40, 0.0)
    assert ag["DEFENSIVE TRANSITION"][0] == 0 and math.isnan(ag["DEFENSIVE TRANSITION"][1])
    assert ag["CREATING CHANCES"][0] == 0


def test_interpolation():
    ctx = parse("--events", EVENTS, need_players=False)
    flag = ctx["flag"]
    assert int(np.sum(flag == 1)) == 15
    assert np.all(flag[150:153] == 1) and np.all(flag[250:262] == 1)
    assert np.all(flag[550:556] == 2) and np.all(flag[1990:2000] == 2)
    assert np.all(flag[1100:1113] == 2) and np.all(flag[500:550] == -1)
    # straight line between the neighbours
    b = ctx["ball_interp"]
    assert b[151].tolist() == pytest.approx(((b[149] + b[153]) / 2).tolist())


def test_degrade_ball_is_deterministic_and_live_only():
    ctx = parse("--events", EVENTS, need_players=False)
    live = ctx["live"]
    ball = ctx["ball"]
    out1, n1 = pc.degrade_ball(ball, live, np.random.default_rng(0), 0.41)
    out2, _ = pc.degrade_ball(ball, live, np.random.default_rng(0), 0.41)
    assert np.array_equal(np.isnan(out1), np.isnan(out2)) and n1 > 0
    miss = ~np.isfinite(out1).all(axis=1)
    share = np.mean(miss[live])
    assert share >= 0.41 and share < 0.41 + 30 * 25 / live.sum()
    assert np.array_equal(miss[~live], ~np.isfinite(ball).all(axis=1)[~live])


def _multiset(players, team, f):
    ok = np.isfinite(players[f]).all(axis=1)
    return sorted((int(t), round(float(x), 3), round(float(y), 3))
                  for (x, y), t in zip(players[f][ok], team[f][ok]))


def test_fragment_tracks():
    ctx = parse("--events", EVENTS)
    players, team = ctx["players"], ctx["team"]
    p0, t0, pieces, dropped = pc.fragment_tracks(players, team, np.random.default_rng(1),
                                                 frag_mean_s=4.0, gap_frames=0)
    assert dropped == 0 and pieces > 11 and p0.shape[1] == 46
    for f in (0, 433, 999, 1500, 1999):
        assert _multiset(p0, t0, f) == _multiset(players, team, f)
    p5, t5, _, _ = pc.fragment_tracks(players, team, np.random.default_rng(1),
                                      frag_mean_s=4.0, gap_frames=5)
    present = np.isfinite(p5).all(axis=2).sum(axis=1)
    assert present.min() < 11 and present.max() == 11


def test_sig3():
    assert pc.sig3(0.12345) == 0.123 and pc.sig3(1850) == 1850 and pc.sig3(12345.6) == 12300.0
    assert math.isnan(pc.sig3(math.nan))
