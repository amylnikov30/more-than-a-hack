"""Hard mask (§4.0) on a BAR-ATH-shaped dead array, plus the reference metadata round trip."""

import os

import numpy as np

import games
import ball_gaps as bg
from possession import reference, spells

N = games.source_frames("tactical")
BLOCKS = games.half_video_blocks("BAR-ATH")
KICKOFF_2ND = games.anchor_video_frame(games.GAMES["BAR-ATH"]["kickoff_2nd"])


def _dead():
    """-1 off the graded halves, 0 inside, plus a dead spell 20000-20999."""
    dead = np.full(N, -1, dtype=np.int8)
    for _, a, b in BLOCKS:
        dead[a:b + 1] = 0
    dead[20000:21000] = 1
    return dead


def test_mask_and_length():
    """199,319 rows; X exactly where dead == -1, D where dead == 1 (incl. 2nd-half pre-kickoff
    frames, which apply_kickoffs forces dead), H / A on every other frame."""
    assert N == 199319
    dead = spells.apply_kickoffs(_dead(), BLOCKS, {2: KICKOFF_2ND})
    first2 = BLOCKS[1][1]
    assert KICKOFF_2ND > first2
    assert (dead[first2:KICKOFF_2ND] == 1).all()
    team = np.where(np.arange(N) % 2 == 0, 1, -1).astype(np.int8)
    labels = spells.labels_from(dead, team)
    assert len(labels) == N
    assert set(labels[dead == -1]) == {"X"}
    assert set(labels[dead == 1]) == {"D"}
    assert set(labels[dead == 0]) == {"H", "A"}
    assert (labels[first2:KICKOFF_2ND] == "D").all()


def test_live_spells_inside_halves():
    """Half 1 is split by the dead spell; half 2 starts at the kickoff: three spells."""
    dead = spells.apply_kickoffs(_dead(), BLOCKS, {2: KICKOFF_2ND})
    sp = spells.live_spells(dead, BLOCKS)
    assert sp == [(1, BLOCKS[0][1], 20000), (1, 21000, BLOCKS[0][2] + 1),
                  (2, KICKOFF_2ND, BLOCKS[1][2] + 1)]


def test_reference_metadata_round_trip(tmp_path):
    """The written metadata maps slots 1-23 to home and 24-46 to away, for every frame."""
    path = os.path.join(str(tmp_path), "metadata.xml")
    reference.write_metadata(path, ["Alpha", "Beta"], {0: 1, 1: -1}, 100, (105.0, 68.0))
    (first, last, teams), = bg.read_slot_teams(path)
    assert (first, last) == (1, 100)
    assert teams.tolist() == [0] * 23 + [1] * 23
