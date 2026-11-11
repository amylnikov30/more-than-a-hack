"""Generate the synthetic fixtures in tests/fixtures/. Deterministic; re-run to rebuild them.

    python tests/make_fixtures.py

Everything here is invented from the schema in the public repo's DATA.md and the open Metrica
sample format. No competition data is used or needed.

Files written
    fixture_atd.xml            FIFA/EPTS metadata: 2 teams, 46 track slots, channels x/y/speed
    fixture_atd.txt.gz         raw tracking, 3,000 frames (frame_id 1..3000), gzipped because
                               46 slots x 3,000 frames of plain text is ~2 MB
    fixture_smart_tagging.xml  SportsCode XML with SET PIECES (mixed case) and one BUILD UP
    fixture_tags.xml           SportsCode XML with several codes and set-piece Types, for
                               tag_gaps.py (whole-frame times)

The design (gaps, tags, crowding) is in the constants below; the hand-derived answers it implies
are in tests/fixtures/README.md and asserted in tests/test_ball_gaps.py.
"""

import gzip
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")

N_FRAMES = 3000                      # video frames 0..2999, ATD frame_id = video + 1
FPS = 25
PITCH = (105.3, 68.0)                # metres, as in games.py
N_SLOTS = 46
HOME_SLOTS = list(range(1, 24))      # slots 1..23 belong to team 1 (home), 24..46 to team 2
PRESENT = list(range(1, 12)) + list(range(24, 35))   # 22 slots are tracked; the other 24 are NaN

# Graded blocks used in the tests: --frames 100-1499,1600-2899 (video frames, inclusive).
GRADED = [(1, 100, 1499), (2, 1600, 2899)]

# Ball missing (NaN), inclusive video-frame ranges.
BALL_NAN = [
    (0, 49),          # before the first graded frame: ignored
    (200, 204),       # G1   5 fr
    (300, 311),       # G2  12 fr  (0.48 s, last frame of the < 0.5 s bin)
    (400, 412),       # G3  13 fr  (0.52 s, first frame of the 0.5-2 s bin)
    (500, 549),       # G4  50 fr  (2.00 s exactly -> 2-5 s bin)
    (700, 899),       # G5 200 fr
    (1450, 1530),     # G6  81 fr, but graded only to 1499 -> clipped to 50 fr, touches half end
    (1550, 1559),     # halftime only: ignored
    (1590, 1619),     # G7  30 fr, graded from 1600 -> clipped to 20 fr, touches half start
    (1800, 1830),     # G8  31 fr
    (2000, 2099),     # G9 100 fr
    (2500, 2649),     # G10 150 fr
    (2700, 2701),     # G11  2 fr
    (2950, 2999),     # after the last graded frame: ignored
]

# Players within 5 m of the ball on the last frame the ball is seen before each gap (that frame
# is the one before the *unclipped* gap start). Every other ball-visible frame has exactly 1.
CROWD_AT_LAST_SEEN = {199: 2, 299: 2, 399: 3, 499: 4, 699: 6, 1449: 5, 1589: 1,
                      1799: 3, 1999: 4, 2499: 6, 2699: 2}

# Offsets from the ball, in metres.
ALWAYS_NEAR = (1.0, 0.0)                       # slot 1, every frame -> the "1" above
DECOYS = {2: (5.2, 0.0), 3: (0.0, 5.1)}        # just outside 5 m, every frame
EXTRA_NEAR = [(0.0, 3.0), (-3.0, 1.0), (2.0, -4.0), (-4.5, -2.0), (3.5, 3.4)]  # 3.0..4.92 m
EXTRA_SLOTS = [4, 5, 6, 7, 8]                  # moved next to the ball on last-seen frames

# Smart Tagging instances: (code, start_s, end_s, team, type). Frames = round(s * 25),
# end exclusive. Codes deliberately in mixed case to test case-insensitive matching.
TAGS = [
    ("SET PIECES", 1.0, 3.0, "N/A", "Kick Off (Start)"),   # [25, 75): ungraded, no gap
    ("SET PIECES", 19.0, 21.0, "Team A", "Free Kick"),     # [475, 525): overlaps G4
    ("set pieces", 30.0, 33.0, "Team B", "Goal Kick"),     # [750, 825): inside G5
    ("SET PIECES", 56.0, 58.0, "N/A", "Throw-in"),         # [1400, 1450): ends where G6 starts
    ("BUILD UP", 79.0, 85.0, "Team A", None),              # [1975, 2125): overlaps G9, wrong code
    ("Set Pieces", 84.0, 86.0, "Team B", "Corner Kick"),   # [2100, 2150): starts right after G9
    ("SET PIECES", 105.0, 110.0, "N/A", "Throw-in"),       # [2625, 2750): overlaps G10
]

# tag_gaps.py fixture: (code, start_frame, end_frame, team, type). Times are written as
# frame / 25 s, so round(s * 25) gives the frame back exactly. See fixtures/README.md (T1-T10).
TAG_FRAMES = [
    ("SET PIECES", 905, 1000, "Team B", "Goal Kick"),          # T1 links G5, no overlap
    ("Set Pieces", 560, 600, "N/A", "Throw-in"),               # T2 links G4
    ("SET PIECES", 2100, 2200, "Team A", "Corner Kick"),       # T3 links G9
    ("set pieces", 1250, 1300, "Team A", "Free Kick"),         # T4 control
    ("SET PIECES", 25, 75, "N/A", "Kick Off (Start)"),         # T5 ungraded anchor
    ("PROGRESSION", 680, 800, "Team A", None),                 # T6 overlaps G5
    ("Long Ball", 1490, 1650, "Team B", None),                 # T7 crosses halftime
    ("BUILD UP", 1835, 1900, "Team B", None),                  # T8 links G8
    ("SET PIECES", 2660, 2700, "N/A", "Throw-in"),             # T9 links G10, G11
    ("GOALS", 2300, 2400, "Team A", None),                     # T10 control
]


def ball_path(video_frame):
    """Normalised ball position (also used as the reference point while the ball is missing)."""
    t = video_frame / float(N_FRAMES - 1)
    return 0.3 + 0.4 * t, 0.5 + 0.1 * math.sin(video_frame / 200.0)


def background_offset(index):
    """Fixed offset in metres, 12-24 m from the ball, for background slot number `index`."""
    angle = 2 * math.pi * index / 19.0
    radius = 12.0 + 6.0 * (index % 3)
    return radius * math.cos(angle), radius * math.sin(angle)


def ball_missing(video_frame):
    return any(a <= video_frame <= b for a, b in BALL_NAN)


def frame_line(video_frame):
    bx, by = ball_path(video_frame)
    length, width = PITCH
    n_near_extra = CROWD_AT_LAST_SEEN.get(video_frame, 1) - 1
    moved = dict(zip(EXTRA_SLOTS[:n_near_extra], EXTRA_NEAR))

    groups = []
    background_index = 0
    for slot in range(1, N_SLOTS + 1):
        if slot not in PRESENT:
            groups.append("NaN,NaN,NaN")
            continue
        if slot == 1:
            dx, dy = ALWAYS_NEAR
        elif slot in DECOYS:
            dx, dy = DECOYS[slot]
        elif slot in moved:
            dx, dy = moved[slot]
        else:
            dx, dy = background_offset(background_index)
        if slot not in (1,) and slot not in DECOYS:
            background_index += 1
        x = bx + dx / length
        y = by + dy / width
        groups.append("{:.5f},{:.5f},{:.2f}".format(x, y, 1.5))

    ball = "NaN,NaN,NaN" if ball_missing(video_frame) else "{:.5f},{:.5f},{:.2f}".format(
        bx, by, 8.0)
    return "{}:{}:{}".format(video_frame + 1, ";".join(groups), ball)


def metadata_xml():
    players = []
    for slot in range(1, N_SLOTS + 1):
        team = "T1" if slot in HOME_SLOTS else "T2"
        players.append(
            '      <Player id="P{0}" teamId="{1}">\n'
            '        <Name>Track {0}</Name>\n'
            '        <ShirtNumber>{0}</ShirtNumber>\n'
            '      </Player>'.format(slot, team))
    player_channels = []
    refs = []
    for slot in range(1, N_SLOTS + 1):
        for ch in ("x", "y", "speed"):
            player_channels.append(
                '      <PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'
                .format(slot, ch))
        refs.append(
            '        <SplitRegister separator=",">\n'
            + "\n".join('          <PlayerChannelRef playerChannelId="player{}_{}"/>'.format(
                slot, ch) for ch in ("x", "y", "speed"))
            + "\n        </SplitRegister>")
    return """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FileDate>2026-11-10T00:00:00.000Z</FileDate>
      <FileName>synthetic_fixture</FileName>
      <TrackingType>Optical</TrackingType>
      <ProviderName>Synthetic</ProviderName>
      <FrameRate>{fps}</FrameRate>
      <ProviderGlobalParameters>
        <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
        <ProviderParameter><Name>first_half_end</Name><Value>1550</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_start</Name><Value>1551</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_end</Name><Value>{n}</Value></ProviderParameter>
      </ProviderGlobalParameters>
      <FieldSize><Width>105</Width><Height>68</Height></FieldSize>
    </GlobalConfig>
    <Sessions>
      <Session id="1">
        <SessionType>Match</SessionType>
        <Start>1</Start>
        <End>{n}</End>
        <MatchParameters>
          <Score idLocalTeam="T1" idVisitingTeam="T2">
            <LocalTeamScore>0</LocalTeamScore>
            <VisitingTeamScore>0</VisitingTeamScore>
          </Score>
        </MatchParameters>
      </Session>
    </Sessions>
    <Teams>
      <Team id="T1"><Name>Team A</Name></Team>
      <Team id="T2"><Name>Team B</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
    <Devices>
      <Device id="device1">
        <Name>Device 1</Name>
        <Sensors>
          <Sensor id="position">
            <Name>Position</Name>
            <Channels>
              <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
              <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
              <Channel id="speed"><Name>speed</Name><Unit>meters per second</Unit></Channel>
            </Channels>
          </Sensor>
        </Sensors>
      </Device>
    </Devices>
    <PlayerChannels>
{player_channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification endFrame="{n}" separator=":" startFrame="1">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{refs}
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
        <BallChannelRef channelId="speed"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
""".format(fps=FPS, n=N_FRAMES, players="\n".join(players),
           player_channels="\n".join(player_channels), refs="\n".join(refs))


def smart_tagging_xml(tags=None):
    instances = []
    for i, (code, start, end, team, kind) in enumerate(TAGS if tags is None else tags, 1):
        half = "1st Half" if start * FPS < 1550 else "2nd Half"
        labels = [("Team", team), ("Half", half)] + ([("Type", kind)] if kind else [])
        instances.append(
            "    <instance>\n      <ID>{}</ID>\n      <start>{:.2f}</start>\n"
            "      <end>{:.2f}</end>\n      <code>{}</code>\n{}\n    </instance>".format(
                i, start, end, code,
                "\n".join("      <label>\n        <group>{}</group>\n        <text>{}</text>\n"
                          "      </label>".format(g, t) for g, t in labels)))
    return ('<?xml version="1.0" encoding="utf-8"?>\n<file>\n  <ALL_INSTANCES>\n{}\n'
            '  </ALL_INSTANCES>\n</file>\n'.format("\n".join(instances)))


# ---------------------------------------------------------------- flip fixture, analysis (d)
# Tiny and hand-checkable: 200 frames, 6 slots, ball fixed at the centre. See fixtures/README.md.

FLIP_N_FRAMES = 200
FLIP_TEAMS = {1: "T1", 2: "T2", 3: "T1", 4: "T2", 5: "REF", 6: "T1"}   # REF is not a <Team>
FLIP_DEFAULT = {1: (2.0, 0.0), 2: (0.0, 3.0), 3: (10.0, 0.0), 4: (0.0, 12.0),
                5: (-20.0, 0.0), 6: None}                            # metres from ball; None = NaN
FLIP_BALL_NAN = [(0, 12), (20, 29), (40, 44), (60, 74), (80, 84), (115, 119), (125, 174),
                 (180, 199)]
# Per-frame overrides: slot -> offset in metres, None for NaN, or ("abs", x, y) normalised.
FLIP_OVERRIDES = {
    30: {1: (6.0, 0.0)},
    59: {5: (0.5, 0.0)},
    79: {1: None, 4: None, 2: ("abs", 0.25, 0.5), 3: ("abs", 0.75, 0.5)},
    85: {1: (6.0, 0.0)},
    114: {1: None, 2: None, 3: None, 4: None},
    175: {1: (15.0, 0.0), 2: (0.0, 7.0)},
}


def flip_frame_line(video_frame):
    length, width = PITCH
    bx, by = 0.5, 0.5
    over = FLIP_OVERRIDES.get(video_frame, {})
    groups = []
    for slot in sorted(FLIP_TEAMS):
        pos = over.get(slot, FLIP_DEFAULT[slot]) if slot in over else FLIP_DEFAULT[slot]
        if pos is None:
            groups.append("NaN,NaN,NaN")
        elif pos[0] == "abs":
            groups.append("{:.5f},{:.5f},{:.2f}".format(pos[1], pos[2], 1.5))
        else:
            groups.append("{:.5f},{:.5f},{:.2f}".format(
                bx + pos[0] / length, by + pos[1] / width, 1.5))
    missing = any(a <= video_frame <= b for a, b in FLIP_BALL_NAN)
    ball = "NaN,NaN,NaN" if missing else "{:.5f},{:.5f},{:.2f}".format(bx, by, 8.0)
    return "{}:{}:{}".format(video_frame + 1, ";".join(groups), ball)


def flip_metadata_xml():
    players = "\n".join(
        '      <Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
        '<ShirtNumber>{0}</ShirtNumber></Player>'.format(s, t) for s, t in sorted(FLIP_TEAMS.items()))
    channels = "\n".join(
        '      <PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'.format(s, c)
        for s in sorted(FLIP_TEAMS) for c in ("x", "y", "speed"))
    refs = "\n".join(
        '        <SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="player{}_{}"/>'.format(s, c)
            for c in ("x", "y", "speed")) + "</SplitRegister>" for s in sorted(FLIP_TEAMS))
    return """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig><FrameRate>{fps}</FrameRate><ProviderName>Synthetic</ProviderName></GlobalConfig>
    <Teams>
      <Team id="T1"><Name>Team A</Name></Team>
      <Team id="T2"><Name>Team B</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
    <Devices><Device id="device1"><Sensors><Sensor id="position"><Channels>
      <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
      <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
      <Channel id="speed"><Name>speed</Name><Unit>meters per second</Unit></Channel>
    </Channels></Sensor></Sensors></Device></Devices>
    <PlayerChannels>
{channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification endFrame="{n}" separator=":" startFrame="1">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{refs}
      </SplitRegister>
      <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="speed"/></SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
""".format(fps=FPS, n=FLIP_N_FRAMES, players=players, channels=channels, refs=refs)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fixture_flip_atd.xml"), "w") as handle:
        handle.write(flip_metadata_xml())
    with open(os.path.join(OUT, "fixture_flip_atd.txt"), "w") as handle:
        for video_frame in range(FLIP_N_FRAMES):
            handle.write(flip_frame_line(video_frame) + "\n")
    with open(os.path.join(OUT, "fixture_atd.xml"), "w") as handle:
        handle.write(metadata_xml())
    # mtime=0 keeps the gzip byte-identical across rebuilds
    with open(os.path.join(OUT, "fixture_atd.txt.gz"), "wb") as raw:
        with gzip.GzipFile(filename="fixture_atd.txt", mode="wb", fileobj=raw, mtime=0) as gz:
            for video_frame in range(N_FRAMES):
                gz.write((frame_line(video_frame) + "\n").encode("ascii"))
    with open(os.path.join(OUT, "fixture_smart_tagging.xml"), "w") as handle:
        handle.write(smart_tagging_xml())
    with open(os.path.join(OUT, "fixture_tags.xml"), "w") as handle:
        handle.write(smart_tagging_xml([(c, a / float(FPS), b / float(FPS), t, k)
                                        for c, a, b, t, k in TAG_FRAMES]))
    for name in sorted(os.listdir(OUT)):
        print("{:<28} {:>8} bytes".format(name, os.path.getsize(os.path.join(OUT, name))))


if __name__ == "__main__":
    main()
