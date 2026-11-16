"""Restart prior (§4.1). TAG_LINK_S = 3 s = 75 frames."""

import os

import numpy as np

from possession import restart

BLOCKS = [(1, 0, 999)]
SPELLS = [(1, 100, 300), (1, 400, 600), (1, 700, 800)]
NO_BALL = {"speed": np.full(1000, np.nan), "nearest": np.full(1000, -1, dtype=np.int8)}


def test_link_windows():
    """Spell 100 (first of the half): [0 - 75, 100 + 75]. Spell 400: previous spell ends at 300,
    so [225, 475]. Spell 700: [525, 775]."""
    assert restart.link_windows(SPELLS, BLOCKS) == {100: (-75, 175), 400: (225, 475),
                                                    700: (525, 775)}


def test_priors_link_latest_tag_in_the_stoppage():
    """Tags (frame, team): (50, H) -> spell 100, 2.0 s before it. (210, A) falls inside live
    spell 100 before 225: not linked. (240, A) and (250, H) both sit in spell 400's stoppage;
    the latest, (250, H), wins, 6.0 s before the spell (the old +-3 s rule missed it).
    (900, A) is after every window. Spell 700 has no tag and no fast ball: no prior."""
    tags = [(50, 1), (210, -1), (240, -1), (250, 1), (900, -1)]
    out, counts, offsets = restart.priors(SPELLS, NO_BALL, tags, BLOCKS)
    assert out == {100: 1, 400: 1, 700: 0}
    assert counts == {"tag": 2, "fast_ball": 0, "none": 1}
    assert offsets == [2.0, 6.0]


def test_tag_restarts_maps_teams_and_flips_defending(tmp_path):
    """Three SET PIECES tags: Alpha (home) Throw-in -> +1 at frame 25; Beta (away) with a Type
    starting "Defending" -> the other team, +1, at frame 50; N/A -> unmapped. A BUILD UP tag is
    not a set piece."""
    xml = """<file><ALL_INSTANCES>
<instance><code>SET PIECES</code><start>1.0</start><end>2.0</end>
 <label><group>Team</group><text>Alpha</text></label>
 <label><group>Type</group><text>Throw-in</text></label></instance>
<instance><code>SET PIECES</code><start>2.0</start><end>3.0</end>
 <label><group>Team</group><text>Beta Club</text></label>
 <label><group>Type</group><text>Defending Corner</text></label></instance>
<instance><code>SET PIECES</code><start>3.0</start><end>4.0</end>
 <label><group>Team</group><text>N/A</text></label></instance>
<instance><code>BUILD UP</code><start>4.0</start><end>5.0</end>
 <label><group>Team</group><text>Alpha</text></label></instance>
</ALL_INSTANCES></file>"""
    path = os.path.join(str(tmp_path), "tags.xml")
    with open(path, "w") as handle:
        handle.write(xml)
    out, stats = restart.tag_restarts(path, ["Alpha FC", "Beta Club"])
    assert out == [(25, 1), (50, 1)]
    assert (stats["set_pieces"], stats["mapped"], stats["unmapped"]) == (3, 2, ["N/A"])
    assert stats["team_map"] == {"Alpha": "home", "Beta Club": "away"}
