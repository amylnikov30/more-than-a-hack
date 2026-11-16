"""make_submission's file checks on synthetic files, and the aggregates helper."""

import numpy as np
import pytest

import games
from pipeline import aggregates, make_submission as ms
from possession import spells

N = 100
BLOCKS = [(1, 10, 39), (2, 50, 89)]
KICKOFFS = {2: 55}


def _labels():
    lab = np.full(N, "X", dtype="<U1")
    lab[10:40] = "H"
    lab[20:30] = "A"
    lab[50:55] = "D"
    lab[55:90] = "A"
    lab[70:75] = "D"
    return lab


def _write(path, labels, header="frame,label", frames=None):
    frames = range(len(labels)) if frames is None else frames
    with open(path, "w") as handle:
        handle.write(header + "\n" + "".join("{},{}\n".format(f, v)
                                             for f, v in zip(frames, labels)))
    return str(path)


def test_valid_file_passes(tmp_path):
    assert ms.check_file(_write(tmp_path / "ok.csv", _labels()), N, BLOCKS, KICKOFFS) == []


@pytest.mark.parametrize("change, expect", [
    (lambda lab: dict(labels=lab, header="Frame,Label"), "header"),
    (lambda lab: dict(labels=lab[:-1]), "data rows"),
    (lambda lab: dict(labels=lab, frames=[0] + list(range(N - 1))), "row 2"),
    (lambda lab: dict(labels=np.where(np.arange(N) == 15, "Z", lab)), "outside H/A/D/X"),
    (lambda lab: dict(labels=np.where(np.arange(N) == 15, "", lab)), "outside H/A/D/X"),
    (lambda lab: dict(labels=np.where(np.arange(N) == 15, "X", lab)), "graded frames are X"),
    (lambda lab: dict(labels=np.where(np.arange(N) == 95, "H", lab)), "ungraded frames"),
    (lambda lab: dict(labels=np.where(np.arange(N) == 52, "A", lab)), "before the kickoff"),
])
def test_format_failures(tmp_path, change, expect):
    fails = ms.check_file(_write(tmp_path / "bad.csv", **change(_labels())), N, BLOCKS, KICKOFFS)
    assert fails and any(expect in f for f in fails), fails


def test_match_shaped_file_passes(tmp_path):
    """A BAR-ATH-shaped file from the hard mask (199,319 rows, pre-kickoff D) passes."""
    n = games.source_frames("tactical")
    blocks = games.half_video_blocks("BAR-ATH")
    kick = games.anchor_video_frame(games.GAMES["BAR-ATH"]["kickoff_2nd"])
    dead = np.full(n, -1, dtype=np.int8)
    for _, a, b in blocks:
        dead[a:b + 1] = 0
    dead = spells.apply_kickoffs(dead, blocks, {2: kick})
    team = np.where(np.arange(n) % 1000 < 500, 1, -1).astype(np.int8)
    path = tmp_path / "match.csv"
    spells.write_labels(str(path), spells.labels_from(dead, team))
    assert ms.check_file(str(path), n, blocks, {2: kick}) == []


def test_aggregates():
    s = aggregates.summary(_labels(), fps=1)
    assert s["possessions"] == 5           # H, A, H | A | A
    assert s["live_spells"] == 3
    assert s["dead_share"] == round(10 / 70.0, 3)
