"""touches.py and validate_metrica.py on synthetic data built here (no match data).

The end-to-end match (one graded half, video frames 0-999, 25 fps, pitch 105.3 x 68 m) has
four slots: H0, H1 (team 0, label H) and A2, A3 (team 1, label A). A player is parked at
(10 + 20 k, 3) m, far from the ball, except while acting, when it stands 0.3 m from the ball.
The ball moves at constant velocity between touches (velocity changes AT the touch frame):

    0-59     still on the centre spot                     -> dead ball D [0, 60] (still run)
    60       H0 kicks off, 10 m/s                         -> restart H (last 5 frames of a
                                                             dead interval do not swallow it)
    100      H1 receives, carries at 2 m/s for 2.4 s
    160      H1 passes, 12 m/s
    200      A2 clips it (direction changes, ~10 m/s)     -> a single A touch: no switch
    240      H0 receives, carries 1.5 m/s
    300      H0 passes, 11 m/s
    340      A3 intercepts and carries at 2 m/s ...       -> turnover A at 340 ('carry')
    400-449  ... ball missing (2 s, a long gap) ...
    500      A3 passes, 10 m/s;  540  A2 receives, carries
    600      A2 kicks it over the touchline (15 m/s); > 1 m out from 613 -> D [613, 649]
    650-799  ball missing (6 s)
    800-859  ball still on the touchline at (40, 68)      -> D [800, 860]
    860      H1 throws in                                 -> restart H
    900      H0 receives, carries to the end

Expected (hand-derived): touches at the 12 frames above (+-1, from smoothing), teams
H H H A H H A A A A H H; control switches restart H@60, turnover A@340, restart H@860; naive
turnovers A@200, H@240, A@340. Labels D 0-60, H 61-339, A 340-612 (X on the 400-449 gap, same
team both sides), D 613-649, X 650-799, D 800-860, H 861-999: 3 possessions (5 naive). M2: 9
pairs (blocks split at 400, 613, 650, 800), 3 switches, of the 2 evaluable 1 switches back after
one touch and 1 within 2 s. M3: the A@340 switch spans no gap. M4 tags: transition starts at
330 (DEFENSIVE, Barcelona: expects A), 340 (ATTACKING, Atletic Club: A) and 750 (ATTACKING,
Barcelona: H); within 2 s the control switch hits both nearby tags (precision 1/1, recall 2/3,
team agree 2).
"""

import csv
import json
import math
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(os.path.dirname(PKG), "scripts"), PKG):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import touches as tp  # noqa: E402
import validate_metrica as vm  # noqa: E402

FPS = 25
PITCH = (105.3, 68.0)
N = 1000
BLOCKS = [(1, 0, N - 1)]
START = (52.65, 34.0)
THROW_SPOT = (40.0, 68.0)
VEL = [(0, (0.0, 0.0)), (60, (0.4, 0.0)), (100, (0.08, 0.0)), (160, (0.48, 0.0)),
       (200, (-0.1, 0.4)), (240, (-0.06, 0.0)), (300, (-0.4, -0.2)), (340, (-0.08, 0.0)),
       (500, (0.0, 0.4)), (540, (0.0, 0.06)), (600, (0.0, 0.6)), (650, (0.0, 0.0)),
       (800, (0.0, 0.0)), (860, (0.2, -0.3)), (900, (0.06, 0.0))]
MISSING = [(400, 449), (650, 799)]
ACTORS = [(0, 57, 63), (0, 237, 303), (0, 897, 999), (1, 97, 163), (1, 857, 863),
          (2, 197, 203), (2, 537, 603), (3, 337, 503)]
SLOT_TEAM = [0, 0, 1, 1]
TOUCHES = [60, 100, 160, 200, 240, 300, 340, 500, 540, 600, 860, 900]
TOUCH_TEAMS = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
TAGS = [("DEFENSIVE TRANSITION", 330, "Barcelona"), ("ATTACKING TRANSITION", 340, "Atletic Club"),
        ("ATTACKING TRANSITION", 750, "Barcelona"), ("SET PIECES", 800, "N/A")]
A = pytest.approx


def velocity(f):
    v = VEL[0][1]
    for start, vel in VEL:
        if f >= start:
            v = vel
    return np.array(v)


def ball_metres():
    """(N, 2) ball in metres, NaN on the missing frames, and the positions underneath them."""
    pos = np.zeros((N, 2))
    p = np.array(START)
    for f in range(N):
        if f == 800:
            p = np.array(THROW_SPOT)
        pos[f] = p
        p = p + velocity(f)
    ball = pos.copy()
    for a, b in MISSING:
        ball[a:b + 1] = np.nan
    return ball, pos


def players_metres(pos):
    pl = np.array([[[10.0 + 20 * k, 3.0] for k in range(4)] for _ in range(N)])
    for slot, a, b in ACTORS:
        pl[a:b + 1, slot] = pos[a:b + 1] + np.array([0.3, 0.0])
    return pl


def arrays():
    ball, pos = ball_metres()
    pl = players_metres(pos)
    teams = np.tile(np.array(SLOT_TEAM, dtype=np.int8), (N, 1))
    return ball / np.array(PITCH), (pl / np.array(PITCH)).astype(np.float32), teams


def write_raw(path):
    ball, players, _ = arrays()
    with open(path, "w") as handle:
        for f in range(N):
            pl = ";".join("{:.7f},{:.7f}".format(*players[f, k]) for k in range(4))
            b = "NaN,NaN" if not np.isfinite(ball[f]).all() else "{:.7f},{:.7f}".format(*ball[f])
            handle.write("{}:{}:{}\n".format(f + 1, pl, b))


def write_meta(path):
    players = "".join('<Player id="P{0}" teamId="T{1}"><Name>Track {0}</Name></Player>'.format(
        k + 1, SLOT_TEAM[k] + 1) for k in range(4))
    channels = "".join('<PlayerChannel id="p{0}_x" playerId="P{0}" channelId="x"/>'
                       '<PlayerChannel id="p{0}_y" playerId="P{0}" channelId="y"/>'.format(k + 1)
                       for k in range(4))
    groups = "".join('<SplitRegister separator=","><PlayerChannelRef playerChannelId="p{0}_x"/>'
                     '<PlayerChannelRef playerChannelId="p{0}_y"/></SplitRegister>'.format(k + 1)
                     for k in range(4))
    xml = ('<?xml version="1.0"?><main><Metadata><Teams><Team id="T1"><Name>Home FC</Name></Team>'
           '<Team id="T2"><Name>Away FC</Name></Team></Teams><Players>{}</Players>'
           '<PlayerChannels>{}</PlayerChannels></Metadata><DataFormatSpecifications>'
           '<DataFormatSpecification startFrame="1" endFrame="{}" separator=":">'
           '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
           '<SplitRegister separator=","><BallChannelRef channelId="x"/>'
           '<BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>'
           '</DataFormatSpecifications></main>').format(players, channels, N, groups)
    with open(path, "w") as handle:
        handle.write(xml)


def write_tags(path):
    inst = []
    for i, (code, frame, team) in enumerate(TAGS, 1):
        inst.append("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                    "<label><group>Team</group><text>{}</text></label></instance>".format(
                        i, frame / 25.0, frame / 25.0 + 4, code, team))
    with open(path, "w") as handle:
        handle.write("<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format("".join(inst)))


def gt_labels():
    lab = np.full(N, "X", dtype="<U1")
    lab[0:61] = "D"
    lab[61:340] = "H"
    lab[340:613] = "A"
    lab[613:861] = "D"
    lab[861:] = "H"
    return lab


def write_gt(path):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for f, c in enumerate(gt_labels()):
            handle.write("{},{}\n".format(f, c))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("touch_match")
    out = {k: str(d / v) for k, v in (("raw", "raw.txt"), ("meta", "meta.xml"),
                                      ("tags", "tags.xml"), ("gt", "gt.csv"))}
    write_raw(out["raw"])
    write_meta(out["meta"])
    write_tags(out["tags"])
    write_gt(out["gt"])
    return out


@pytest.fixture(scope="module")
def res(files):
    return tp.analyse(files["raw"], BLOCKS, files["meta"], files["tags"], files["gt"],
                      null_n=0)


# ---------------------------------------------------------------- step 1 pieces

def test_savgol_centre_row_and_quadratic():
    m = tp.savgol_matrix(7, 2)
    assert m[3] * 21 == A([-2, 3, 6, 7, 6, 3, -2])
    t = np.arange(20.0)
    xy = np.column_stack([0.5 * t ** 2 - t, 3 * t + 1])
    assert tp.smooth(xy, 7, 2) == A(xy)          # edges too: their rows are exact fits
    assert tp.smooth(xy[:2], 7, 2) == A(xy[:2])  # too short: unchanged


def kink(v1, v2, n1=30, n2=30):
    """Positions (m) at 25 fps: velocity v1 (m/s) for n1 frames, then v2; the kink at n1."""
    v = np.array([v1] * n1 + [v2] * n2, dtype=float) / FPS
    return np.vstack([[0.0, 0.0], np.cumsum(v, axis=0)])[:-1]


@pytest.mark.parametrize("v1,v2,dv,angle", [
    ((10, 0), (-5, 0), 15.0, 180.0),       # reversal
    ((0, 0), (12, 0), 12.0, math.nan),     # kick from rest: heading undefined before
    ((5, 0), (15, 0), 10.0, 0.0),          # same direction, faster: angle alone misses it
    ((8, 0), (0, 0), 8.0, math.nan),       # first touch kills the ball
])
def test_touch_signal_kinds(v1, v2, dv, angle):
    xy = kink(v1, v2)
    sig, vb, va = tp.touch_signal(xy, 4, FPS)
    peaks = tp.pick_peaks(sig, 1.0, 5)
    assert list(peaks) == [30]
    assert sig[30] == A(dv)
    assert sig[28] == A(dv * 0.5)            # triangular profile, half-width w
    ang = tp.angle_deg(vb[30], va[30])[0]
    assert (math.isnan(ang) and math.isnan(angle)) or ang == A(angle, abs=1e-6)


def test_pick_peaks_min_sep_keeps_the_higher():
    dv = np.zeros(30)
    dv[10], dv[13], dv[25] = 5.0, 7.0, 4.0
    assert list(tp.pick_peaks(dv, 1.0, 5)) == [13, 25]
    assert list(tp.pick_peaks(dv, 1.0, 3)) == [10, 13, 25]
    assert list(tp.pick_peaks(dv, 4.5, 3)) == [10, 13]
    assert list(tp.pick_peaks(np.full(5, np.nan), 1.0, 3)) == []


def test_visible_runs_jumps_and_glitches():
    half_of = np.ones(40, dtype=int)
    bm = np.column_stack([np.arange(40) * 0.2, np.zeros(40)])
    bm[10] = [30.0, 30.0]                     # one-frame glitch: dropped, runs meet over a gap
    bm[20:23] = np.nan
    bm[30:] += [50.0, 0.0]                    # a real jump (both sides long): kept as a split
    runs, n_jumps, glitch = tp.visible_runs(bm, half_of, FPS, 40.0)
    assert [(a, b) for _, a, b in runs] == [(0, 9), (11, 19), (23, 29), (30, 39)]
    assert n_jumps == 1
    assert list(np.flatnonzero(glitch)) == [10]


def test_gap_candidates_straight_vs_bend():
    p = dict(tp.DEFAULTS, dv_half=4, sg_window=1, short_gap_s=0.5)
    half_of = np.ones(60, dtype=int)
    straight = kink((10, 0), (10, 0))
    bend = kink((10, 0), (0, 10))
    for xy, touch in ((straight, False), (bend, True)):
        bm = xy.copy()
        bm[28:33] = np.nan                    # 5 frames missing around the kink at 30
        tr = tp.build_track(bm, half_of, FPS, p)
        cols, lines, counts = tp.gap_candidates(tr, FPS, p)
        assert counts["short"] == 1 and counts["bridged"] == 1
        assert len(cols["frame"]) == 1
        if touch:
            assert cols["dv"][0] == A(math.hypot(10, 10))
            assert cols["frame"][0] == 30     # where the two lines meet
            assert cols["x"][0] == A(bend[30, 0], abs=0.01)
        else:
            assert cols["dv"][0] == A(0.0, abs=1e-9)
            assert cols["miss"][0] == A(0.0, abs=1e-9)
        # the attribution window follows the true track through the gap (both are lines here)
        win = tp.windows_gap(cols["frame"], lines, 3, FPS)
        f = int(cols["frame"][0])
        assert win[0] == A(xy[f - 3:f + 4], abs=1e-6)


def test_attribution_and_contested_margin():
    d = np.array([[0.5, 5.0], [3.0, 2.5], [0.4, 0.7], [0.4, 1.2], [np.inf, np.inf]])
    team, dist, other, contested = tp.attribution(d, 2.0, 0.5)
    assert list(team) == [0, -1, 0, 0, -1]
    assert list(contested) == [False, False, True, False, False]
    assert math.isnan(dist[4])
    _, _, _, literal = tp.attribution(d, 2.0)          # margin inf: both inside the limit
    assert list(literal) == [False, False, True, True, False]


def test_dead_intervals_outside_and_still():
    ball = np.full((200, 2), 0.5)
    ball[:, 0] = 0.3 + 0.001 * np.arange(200)          # moving, in play
    ball[50:90] = [0.5, 0.5]                            # still on the centre spot: dead
    ball[120:150, 1] = 1.05                             # 3.4 m beyond the touchline: dead
    ball[170:199] = [0.3, 0.3]                          # still elsewhere: not dead
    p = dict(tp.DEFAULTS)
    dead, frames = tp.dead_intervals(ball, [(1, 0, 199)], FPS, PITCH, p)
    assert [(iv["start"], iv["end"], iv["signals"]) for iv in dead] == \
        [(50, 89, "still"), (120, 149, "outside")]
    assert frames["outside"] == 30


# ---------------------------------------------------------------- step 4

def tbl(rows):
    """Touch table from (frame, team, contested, carry_team, carry_s, sp_b, sp_a, dist)."""
    keys = ("frame", "team", "contested", "carry_team", "carry_s", "sp_b", "sp_a", "dist")
    T = {k: np.array([r[i] for r in rows]) for i, k in enumerate(keys)}
    T["contested"] = T["contested"].astype(bool)
    T["half"] = np.ones(len(rows), dtype=int)
    T["in_dead"] = np.zeros(len(rows), dtype=bool)
    return T


def touch(frame, team, contested=False, carry_team=-1, carry_s=0.0, sp_b=10.0, sp_a=10.0,
          dist=0.5):
    return (frame, team, contested, carry_team, carry_s, sp_b, sp_a, dist)


P = dict(tp.DEFAULTS)
B = [(1, 0, 10000)]


def turnovers(sw):
    return [(s["frame"], s["team"], s["confirm"]) for s in sw if s["kind"] == "turnover"]


def test_control_single_clip_does_not_switch():
    T = tbl([touch(10, 0), touch(50, 0), touch(90, 1), touch(130, 0), touch(170, 0)])
    sw = tp.control_rule(T, [], [], B, P)
    assert [(s["frame"], s["kind"]) for s in sw] == [(10, "restart")]
    assert turnovers(tp.control_rule(T, [], [], B, P, naive=True)) == \
        [(90, 1, "touch"), (130, 0, "touch")]


def test_control_two_touches_switch_at_the_first():
    T = tbl([touch(10, 0), touch(50, 1), touch(90, 1), touch(130, 1)])
    assert turnovers(tp.control_rule(T, [], [], B, P)) == [(50, 1, "two_touches")]


def test_control_carry_and_receive():
    T = tbl([touch(10, 0), touch(50, 1, carry_team=1, carry_s=1.2)])
    assert turnovers(tp.control_rule(T, [], [], B, P)) == [(50, 1, "carry")]
    T = tbl([touch(10, 0), touch(50, 1, carry_team=1, carry_s=0.8)])
    assert turnovers(tp.control_rule(T, [], [], B, P)) == []
    T = tbl([touch(10, 0), touch(50, 1, sp_b=12.0, sp_a=2.0, dist=0.8)])
    assert turnovers(tp.control_rule(T, [], [], B, P)) == [(50, 1, "receive")]
    T = tbl([touch(10, 0), touch(50, 1, sp_b=12.0, sp_a=2.0, dist=1.8)])   # player too far
    assert turnovers(tp.control_rule(T, [], [], B, P)) == []


def test_control_long_gap_clears_candidate_not_team():
    T = tbl([touch(10, 0), touch(50, 1), touch(300, 1), touch(340, 1)])
    gap = [{"start": 100, "end": 250}]
    sw = tp.control_rule(T, [], gap, B, P)
    assert turnovers(sw) == [(300, 1, "two_touches")]
    assert sw[-1]["prev"] == 0                 # the last H touch, before the gap


def test_control_dead_ball_restart_and_contested():
    T = tbl([touch(10, 0), touch(50, 0), touch(400, 1), touch(440, 0, contested=True),
             touch(480, 0, contested=True, carry_team=0, carry_s=1.5)])
    dead = [{"half": 1, "start": 100, "end": 350}]
    sw = tp.control_rule(T, dead, [], B, P)
    assert [(s["frame"], s["team"], s["kind"], s["confirm"]) for s in sw] == \
        [(10, 0, "restart", "restart"), (400, 1, "restart", "restart"),
         (480, 0, "turnover", "carry")]
    nv = tp.control_rule(T, dead, [], B, P, naive=True)
    assert turnovers(nv) == []                 # naive ignores contested touches


def test_labels_fill_and_runs():
    sw = [{"half": 1, "frame": 10, "team": 0}, {"half": 1, "frame": 50, "team": 1}]
    dead = [{"half": 1, "start": 70, "end": 79}]
    gaps = [{"start": 20, "end": 29}, {"start": 85, "end": 90}]
    lab, counts = tp.make_labels(100, [(1, 0, 99)], sw, dead, gaps)
    assert "".join(lab) == "X" * 10 + "H" * 10 + "X" * 10 + "H" * 20 + "A" * 20 + "D" * 10 + \
        "X" * 20
    assert counts == {"same": 1, "differs": 0, "unknown": 1}
    filled = tp.fill_x(lab, [(1, 0, 99)])
    assert "".join(filled) == "H" * 50 + "A" * 20 + "D" * 30
    assert tp.possession_runs(filled, [(1, 0, 99)]) == [(1, 0, 49, "H"), (1, 50, 69, "A")]


# ---------------------------------------------------------------- measurements

def test_consecutive_and_gap_spans():
    T = tbl([touch(f, t) for f, t in ((0, 0), (10, 1), (20, 0), (30, 0), (40, 1), (60, 1))])
    m = tp.consecutive(T, [], [], [(1, 0, 1000)], FPS, 2.0)
    assert (m["n_pairs"], m["n_switch"], m["n_back_eval"], m["back_1"], m["back_2s"]) == \
        (5, 3, 2, 1, 2)
    m = tp.consecutive(T, [], [{"start": 35, "end": 38}], [(1, 0, 1000)], FPS, 2.0)
    assert (m["n_pairs"], m["n_switch"]) == (4, 2)
    gidx = tp.GapIndex([{"start": 3, "end": 5}, {"start": 100, "end": 200}])
    spans = tp.gap_spans([(0, 10), (10, 20), (90, 210)], gidx, FPS, 12)
    assert (spans["no_gap"], spans["short_gap"], spans["long_gap"]) == (1, 1, 1)
    assert spans["bin:<0.5s"] == 1 and spans["bin:2-5s"] == 1


def test_dip_finds_two_modes():
    edges = np.arange(0, 10.25, 0.25)
    counts = np.zeros(40)
    counts[2:5] = [30, 40, 30]               # touches around 0.75 m
    counts[20:25] = [10, 15, 20, 15, 10]     # noise around 5.5 m
    m1, d, m2, bimodal = tp.dip(counts, edges)
    assert m1 == A(0.875) and m2 == A(5.625) and 1.0 < d < 5.0 and bimodal


def test_tag_alignment_null_identity():
    sw = [{"kind": "turnover", "half": 1, "frame": 340, "team": 1}]
    an = [{"half": 1, "frame": 330, "code": "DEFENSIVE TRANSITION", "expected": 1},
          {"half": 1, "frame": 900, "code": "ATTACKING TRANSITION", "expected": 0}]
    offsets = {1: np.array([0, 0])}          # a shift of 0 is the real thing
    r = tp.tag_alignment(sw, an, [(1, 0, 999)], FPS, (2.0,), offsets)
    cell = r["groups"]["all"]["windows"][2.0]
    assert (cell["precision"], cell["recall"]) == (1.0, 0.5)
    assert cell["precision_null_mean"] == 1.0 and cell["precision_p_emp"] == 1.0
    assert (cell["team_agree"], cell["team_disagree"]) == (1, 0)
    assert r["groups"]["all"]["offsets"] == [A(0.4)]


# ---------------------------------------------------------------- end to end

def test_e2e_touches(res):
    T = res["T"]
    assert len(T["frame"]) == len(TOUCHES)
    assert np.abs(T["frame"] - np.array(TOUCHES)).max() <= 1
    assert list(T["team"]) == TOUCH_TEAMS
    assert not T["contested"].any()
    assert np.nanmax(T["dist"]) < 0.5
    assert res["n_in_dead"] == 0


def test_e2e_dead_and_gaps(res):
    assert [(iv["start"], iv["end"]) for iv in res["dead"]] == [(0, 60), (613, 649), (800, 860)]
    assert [(g["start"], g["end"]) for g in res["long_gaps"]] == [(400, 449), (650, 799)]
    assert res["gap_counts"] == {"same": 1, "differs": 0, "unknown": 1}


def test_e2e_segments(res):
    T = res["T"]
    i340 = int(np.argmin(np.abs(T["frame"] - 340)))
    assert T["seg_type"][i340] == "carry" and T["carry_team"][i340] == 1
    assert T["carry_s"][i340] == A(160 / 25.0, abs=0.1)
    i200 = int(np.argmin(np.abs(T["frame"] - 200)))
    assert T["seg_type"][i200] == "flight"


def test_e2e_switches_and_labels(res):
    sw = [(s["frame"], s["team"], s["kind"], s["confirm"]) for s in res["switches"]]
    assert [(k, t, c) for _, t, k, c in sw] == [("restart", 0, "restart"),
                                              ("turnover", 1, "carry"),
                                              ("restart", 0, "restart")]
    assert abs(sw[1][0] - 340) <= 1
    naive = [(s["team"]) for s in res["switches_naive"] if s["kind"] == "turnover"]
    assert naive == [1, 0, 1]
    lab = res["labels"]
    f340 = sw[1][0]
    assert set(lab[0:61]) == {"D"} and set(lab[61:f340]) == {"H"}
    assert set(lab[f340:400]) == {"A"} and set(lab[400:450]) == {"X"}
    assert set(lab[613:650]) == {"D"} and set(lab[650:800]) == {"X"}
    assert set(lab[800:861]) == {"D"} and set(lab[861:]) == {"H"}
    assert tp.possession_stats(lab, BLOCKS, FPS)["n"] == 3
    assert tp.possession_stats(res["labels_naive"], BLOCKS, FPS)["n"] == 5


def test_e2e_measurements(res):
    m2 = res["m2"]
    assert (m2["n_pairs"], m2["n_switch"], m2["n_back_eval"], m2["back_1"], m2["back_2s"]) == \
        (9, 3, 2, 1, 1)
    assert res["m3_control"]["n"] == 1 and res["m3_control"]["no_gap"] == 1
    ctrl = res["m4"]["control"]["groups"]["all"]["windows"]
    assert (ctrl[2.0]["precision"], ctrl[2.0]["recall"]) == (1.0, A(2 / 3.0))
    assert (ctrl[2.0]["team_agree"], ctrl[2.0]["team_disagree"]) == (2, 0)
    naive = res["m4"]["naive"]["groups"]["all"]["windows"]
    assert naive[2.0]["precision"] == A(1 / 3.0) and naive[5.0]["precision"] == A(2 / 3.0)
    assert res["m1_touch"].size == len(TOUCHES)
    gt = res["gt"]
    assert gt["kinds"]["turnover"]["n"] == 1 and gt["kinds"]["restart"]["n"] == 2
    assert abs(gt["kinds"]["turnover"]["touch"][0]) <= 1.0 / FPS + 1e-9
    assert (gt["poss_truth"], gt["poss_control"], gt["poss_naive"]) == (3, 3, 5)


def test_e2e_summary_rows_are_aggregates(res):
    rows = tp.summary_rows(res)
    keys = {(r[0], r[1], r[2], r[3]) for r in rows}
    assert ("possessions", "control", "", "n") in keys
    assert ("m3_gap_span", "control", "no_gap", "count") in keys
    assert ("timescale", "w=12", "dv>=3", "peaks_per_visible_min") in keys
    assert all(len(r) == 6 for r in rows)


def test_main_writes_outputs(files, tmp_path):
    out = tmp_path / "out"
    tp.main([str(out), "--raw", files["raw"], "--meta", files["meta"], "--tags", files["tags"],
             "--frames", "0-999", "--null-n", "0"])
    with open(out / tp.SUMMARY_NAME) as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["analysis", "scope", "bin", "metric", "value", "unit"]
    with open(out / tp.TOUCHES_NAME) as handle:
        assert len(list(csv.reader(handle))) == len(TOUCHES) + 1
    with open(out / tp.LABELS_NAME) as handle:
        assert len(handle.read().splitlines()) == N + 1


def test_main_no_tables(files, tmp_path):
    out = tmp_path / "out"
    tp.main([str(out), "--raw", files["raw"], "--meta", files["meta"], "--frames", "0-999",
             "--null-n", "0", "--no-tables"])
    assert sorted(os.listdir(out)) == [tp.SUMMARY_NAME]


def test_main_refuses_in_repo_out_dir(files):
    with pytest.raises(SystemExit):
        tp.main([os.path.join(PKG, "out"), "--raw", files["raw"], "--meta", files["meta"],
                 "--frames", "0-999"])


# ---------------------------------------------------------------- validate_metrica

def test_spell_f1_and_macro():
    truth = np.array(list("H" * 100 + "A" * 100 + "D" * 50 + "H" * 100 + "A" * 50))
    blocks = [(1, 0, len(truth) - 1)]
    assert vm.spell_f1(truth.copy(), truth, blocks)[0] == 1.0
    late = np.concatenate([[truth[0]], truth[:-1]])
    assert vm.spell_f1(late, truth, blocks)[0] == 1.0
    assert vm.spell_f1(np.full(len(truth), "H"), truth, blocks)[0] == 0.0
    x = truth.copy()
    x[120:180] = "X"                          # inside a possession: free after the fill
    assert vm.spell_f1(x, truth, blocks)[0] == 1.0
    assert vm.macro_f1(truth.copy(), truth, blocks) == 1.0
    assert vm.macro_f1(x, truth, blocks) < 1.0   # macroF1 charges the X


def test_match_points_one_to_one():
    pairs = vm.match_points([10, 12, 40], [0, 0, 1], [11, 41], [0, 0], 3)
    assert pairs == {0: 0}
    pairs = vm.match_points([10, 12, 40], [0, 0, 1], [11, 41], [0, 0], 3, same_team=False)
    assert pairs == {0: 0, 2: 1}


def test_delete_bursts_reaches_target_on_graded_frames():
    ball = np.full((5000, 2), 0.5)
    blocks = [(1, 100, 2399), (2, 2600, 4899)]
    out, stats = vm.delete_bursts(ball, blocks, 0.41, np.random.default_rng(1))
    graded = np.zeros(5000, dtype=bool)
    graded[100:2400] = graded[2600:4900] = True
    miss = ~np.isfinite(out).all(axis=1)
    assert 0.41 <= miss[graded].mean() < 0.41 + 90 * 25 / 4600.0
    assert not miss[~graded].any()
    assert stats["missing"] == A(miss[graded].mean())
    again, _ = vm.delete_bursts(ball, blocks, 0.41, np.random.default_rng(1))
    assert np.array_equal(np.isnan(out), np.isnan(again))


def write_events(path):
    def ev(kind, team, start, end=None, subs=(), to=False):
        return {"type": {"name": kind}, "team": {"id": team},
                "subtypes": [{"name": s} for s in subs] or None,
                "start": {"frame": start + 1, "x": None, "y": None},
                "end": {"frame": (end if end is not None else start) + 1},
                "to": {"id": "x"} if to else None}
    data = {"data": [ev("SET PIECE", "T1", 10, subs=("KICK OFF",)),
                     ev("PASS", "T1", 10, 50, to=True), ev("CARRY", "T1", 50, 100),
                     ev("BALL LOST", "T1", 100), ev("RECOVERY", "T2", 102),
                     ev("PASS", "T2", 150, 180), ev("BALL OUT", "T2", 180),
                     ev("SET PIECE", "T1", 300, subs=("THROW IN",)),
                     ev("SHOT", "T1", 350, subs=("ON TARGET", "GOAL")),
                     ev("SET PIECE", "T2", 500, subs=("KICK OFF",))]}
    with open(path, "w") as handle:
        json.dump(data, handle)


def test_events_refs_dead_labels(tmp_path):
    path = str(tmp_path / "events.json")
    write_events(path)
    events = vm.read_events(path, ["T1", "T2"])
    blocks = [(1, 0, 599)]
    f, t, k = vm.reference_touches(events, blocks)
    assert list(f) == [10, 50, 100, 102, 150, 300, 350, 500]
    assert list(k) == ["pass_start", "pass_end", "ball_lost", "recovery", "pass_start",
                       "set_piece", "shot", "set_piece"]
    assert list(t) == [0, 0, 0, 1, 1, 0, 0, 1]
    dead = vm.event_dead(events, blocks)
    assert [(d["start"], d["end"]) for d in dead] == [(180, 299), (350, 499)]
    lab = vm.event_labels(events, 600, blocks, dead)
    assert "".join(lab[:10]) == "X" * 10
    assert set(lab[10:102]) == {"H"} and set(lab[102:180]) == {"A"}
    assert set(lab[180:300]) == {"D"} and set(lab[300:350]) == {"H"}
    assert set(lab[350:500]) == {"D"} and set(lab[500:]) == {"A"}
    tf, tt = vm.label_turnovers(lab, blocks)
    assert list(tf) == [102] and list(tt) == [1]
