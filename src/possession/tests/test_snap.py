"""Switch snapping (§4.4). SNAP_S = 3 s = 75 frames."""

import numpy as np

from possession import config, snap


def _team(switch_at, n=300):
    return np.r_[np.ones(switch_at), -np.ones(n - switch_at)].astype(np.int8)


def test_contest_moves_switch_to_new_team_control():
    """Decoder switch H -> A at 150. H controls 100-109, the ball is loose 110-129 (no
    control), A controls from 130. A also had a touch at 90-94, before H's last control, which
    must be ignored. Window 75-225: H's last control is 109, A's first control after it is 130,
    so the switch moves to 130."""
    control = np.zeros(300, dtype=np.int8)
    control[90:95] = -1
    control[100:110] = 1
    control[130:140] = -1
    out, counts = snap.snap(_team(150), [(1, 0, 300)], control)
    assert out[:130].tolist() == [1] * 130 and (out[130:] == -1).all()
    assert counts == {"moved": 1, "kept": 0, "frames_moved": 20}


def test_no_control_keeps_switch():
    """No A control frame within 75 frames of the switch (a ball gap): it stays at 150."""
    control = np.zeros(300, dtype=np.int8)
    control[100:110] = 1
    out, counts = snap.snap(_team(150), [(1, 0, 300)], control)
    assert (out == _team(150)).all() and counts["kept"] == 1


def test_control_frames_need_k_in_a_row():
    """Ball near a home player and slow on frames 10-14 (5 = CONTROL_K: control) and 30-33
    (4: not control)."""
    n = 50
    near = np.full(n, 10.0)
    near[10:15] = near[30:34] = 0.5
    sig = {"near": near, "speed": np.full(n, 1.0), "nearest": np.zeros(n, dtype=np.int8),
           "flag": np.zeros(n, dtype=np.int8)}
    out = snap.control_frames(sig, np.ones(n, dtype=bool))
    assert config.CONTROL_K == 5
    assert np.flatnonzero(out).tolist() == list(range(10, 15))
