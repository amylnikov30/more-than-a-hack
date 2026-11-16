"""The step-6 rule on hand-built control sequences, and end to end on the synthetic fixture."""

import os

import numpy as np

from conftest import EDA_FIXTURES
from pipeline import rule

H, A = rule.H, rule.A


def _seq(*parts):
    return np.concatenate([np.full(n, v, dtype=np.int8) for v, n in parts])


def test_one_switch_backdated():
    """20 A control frames after 30 H frames, K = 12: exactly one switch, at the first A frame
    (backdated), not 12 frames into the run."""
    out = rule.rule_spell(_seq((H, 30), (A, 20), (0, 10)), H, 12)
    assert (out[:30] == H).all() and (out[30:] == A).all()


def test_brief_touch_never_switches():
    """5 A frames (< K) between H control keep the spell H."""
    out = rule.rule_spell(_seq((H, 30), (A, 5), (H, 30)), H, 12)
    assert (out == H).all()


def test_interrupted_run_resets():
    """8 + 8 A frames split by a missing / fast ball frame never reach K = 12."""
    out = rule.rule_spell(_seq((A, 8), (0, 1), (A, 8), (0, 10)), H, 12)
    assert (out == H).all()


def test_switch_back_and_forth():
    out = rule.rule_spell(_seq((A, 12), (0, 5), (H, 12), (0, 3)), H, 12)
    assert (out[:17] == A).all() and (out[17:] == H).all()


def test_start_team_order():
    ctrl, near = _seq((0, 3), (A, 2)), _seq((H, 5))
    assert rule.start_team(H, ctrl, near) == H                    # the restart prior wins
    assert rule.start_team(0, ctrl, near) == A                    # then the first control frame
    assert rule.start_team(0, _seq((0, 5)), _seq((0, 2), (A, 3))) == A   # then the nearest team
    assert rule.start_team(0, _seq((0, 5)), _seq((0, 5))) == H    # then H


def test_control_needs_a_slow_visible_ball_on_a_live_frame():
    signals = {"speed": np.array([1.0, 6.0, np.nan, 1.0, 1.0]),
               "nearest": np.array([0, 0, 1, -1, 1])}
    live = np.array([True, True, True, True, False])
    assert rule.control_sign(signals, live).tolist() == [H, 0, 0, 0, 0]


def _run(out_dir):
    argv = [str(out_dir), "--dead", os.path.join(EDA_FIXTURES, "fixture_poss_dead.csv"),
            "--raw", os.path.join(EDA_FIXTURES, "fixture_poss_atd.txt"),
            "--meta", os.path.join(EDA_FIXTURES, "fixture_poss_atd.xml"),
            "--frames", "0-999,1000-1999", "--pitch", "105", "68"]
    rule.main(argv)
    with open(os.path.join(str(out_dir), "rule.csv"), "rb") as handle:
        return handle.read()


def test_end_to_end_and_byte_identical(tmp_path):
    """Same contract as possession.run: one row per dead-file row, D on the 150 dead frames,
    H / A elsewhere, no X; two runs give the same bytes."""
    first = _run(tmp_path / "a")
    assert first == _run(tmp_path / "b")
    lines = first.decode().splitlines()
    assert lines[0] == "frame,label" and len(lines) == 2001
    labels = np.array([line.split(",")[1] for line in lines[1:]])
    assert int(np.sum(labels == "D")) == 150
    assert set(labels[labels != "D"]) <= {"H", "A"}
