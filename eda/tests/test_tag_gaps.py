"""tag_gaps.py on the synthetic fixtures. Expected values are derived by hand in
tests/fixtures/README.md ("tag_gaps.py")."""

import os

import numpy as np
import pytest

import ball_gaps as bg
import tag_gaps as tg
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_atd.txt.gz")
TAGS = os.path.join(FIXTURES, "fixture_tags.xml")
OLD_TAGS = os.path.join(FIXTURES, "fixture_smart_tagging.xml")
BLOCKS = [(1, 100, 1499), (2, 1600, 2899)]
W5, W2 = 125, 50

# G1..G11 start frames, in find_gaps order
G = {200: "G1", 300: "G2", 400: "G3", 500: "G4", 700: "G5", 1450: "G6", 1600: "G7",
     1800: "G8", 2000: "G9", 2500: "G10", 2700: "G11"}


@pytest.fixture(scope="module")
def gaps():
    ball, _ = bg.read_ball(RAW)
    out = bg.find_gaps(~np.isfinite(ball).all(axis=1), BLOCKS)
    assert [G[g["start"]] for g in out] == ["G{}".format(i) for i in range(1, 12)]
    return out


@pytest.fixture(scope="module")
def groups():
    return tg.tag_groups(tg.read_tags(TAGS))


def names(gaps, hits):
    return [G[g["start"]] for g, h in zip(gaps, hits) if h]


# ---------------------------------------------------------------- reading

def test_read_tags():
    tags = tg.read_tags(TAGS)
    assert len(tags) == 10
    assert [(t["code"], t["start"], t["end"]) for t in tags[:3]] == [
        ("SET PIECES", 905, 1000), ("SET PIECES", 560, 600), ("SET PIECES", 2100, 2200)]
    assert tags[3]["code"] == "SET PIECES"            # 'set pieces'
    assert tags[6]["code"] == "LONG BALL"             # 'Long Ball'
    assert tags[1]["type"] == "Throw-in" and tags[1]["team"] == "N/A"
    assert tags[5]["type"] is None


def test_groups(groups):
    assert list(groups) == [
        "code:BUILD UP", "code:GOALS", "code:LONG BALL", "code:PROGRESSION", "code:SET PIECES",
        "type:Corner Kick", "type:Free Kick", "type:Goal Kick", "type:Kick Off (Start)",
        "type:Throw-in"]
    assert len(groups["code:SET PIECES"]) == 6
    assert len(groups["type:Throw-in"]) == 2


def test_read_set_pieces_unchanged():
    assert bg.read_set_pieces(OLD_TAGS) == [(25, 75), (475, 525), (750, 825), (1400, 1450),
                                            (2100, 2150), (2625, 2750)]


# ---------------------------------------------------------------- links and overlaps

def test_linkable(gaps):
    assert [G[g["start"]] for g in gaps if not tg.linkable(g, BLOCKS)] == ["G6"]
    assert [g["next_seen"] for g in gaps] == [205, 312, 413, 550, 900, 1531, 1620, 1831, 2100,
                                              2650, 2702]


@pytest.mark.parametrize("group, tol, expected", [
    ("code:SET PIECES", 0, []),
    ("code:SET PIECES", 25, ["G4", "G5", "G9", "G10", "G11"]),
    ("code:PROGRESSION", 0, ["G5"]),
    ("code:LONG BALL", 0, ["G6", "G7"]),
    ("code:BUILD UP", 0, []),
    ("code:GOALS", 0, []),
])
def test_overlap(gaps, groups, group, tol, expected):
    assert names(gaps, tg.overlap_hits(gaps, groups[group], tol)) == expected


@pytest.mark.parametrize("group, w, expected", [
    ("code:SET PIECES", W5, ["G4", "G5", "G9", "G10", "G11"]),
    ("code:SET PIECES", W2, ["G4", "G5", "G9", "G10", "G11"]),
    ("code:PROGRESSION", W5, []),
    ("code:LONG BALL", W5, []),
    ("code:BUILD UP", W5, ["G8"]),
    ("code:BUILD UP", W2, ["G8"]),
    ("code:GOALS", W5, []),
    ("type:Throw-in", W5, ["G4", "G10", "G11"]),
    ("type:Goal Kick", W5, ["G5"]),
    ("type:Corner Kick", W5, ["G9"]),
    ("type:Free Kick", W5, []),
    ("type:Kick Off (Start)", W5, []),
])
def test_link(gaps, groups, group, w, expected):
    anchors, _ = tg.graded_anchors(groups[group], BLOCKS)
    assert names(gaps, tg.link_hits(gaps, anchors, BLOCKS, w)) == expected


def test_dropped_anchor(groups):
    assert tg.graded_anchors(groups["type:Kick Off (Start)"], BLOCKS) == ([], 1)
    assert tg.graded_anchors(groups["code:SET PIECES"], BLOCKS)[1] == 1


# ---------------------------------------------------------------- view B

def test_view_b_counts(gaps, groups):
    b = tg.view_b(gaps, groups, BLOCKS, w_frames=W5)
    assert list(b["code:SET PIECES"]["link"]) == ["all", "0-2s", "2-5s", "5-10s"]
    sp = b["code:SET PIECES"]
    assert {s: (v["real_count"], v["n_gaps"]) for s, v in sp["link"].items()} == {
        "all": (5, 11), "0-2s": (1, 6), "2-5s": (2, 3), "5-10s": (2, 2)}
    assert sp["overlap"]["all"]["real_count"] == 0
    assert (sp["n_tags"], sp["n_dropped"]) == (6, 1)
    assert b["code:PROGRESSION"]["overlap"]["5-10s"]["real_count"] == 1
    assert b["code:LONG BALL"]["overlap"]["2-5s"]["real_count"] == 1
    assert b["code:LONG BALL"]["overlap"]["0-2s"]["real_count"] == 1
    assert b["code:BUILD UP"]["link"]["0-2s"]["real_count"] == 1
    assert b["type:Throw-in"]["link"]["all"]["real_count"] == 3
    assert "null_mean" not in sp["link"]["all"]


@pytest.mark.parametrize("tol", [0, 25])
def test_null_offset_zero_equals_real(gaps, groups, tol):
    zero = {1: np.array([0]), 2: np.array([0])}
    b = tg.view_b(gaps, groups, BLOCKS, tol_frames=tol, w_frames=W5, offsets=zero)
    for name, res in b.items():
        for kind in ("overlap", "link"):
            for s, v in res[kind].items():
                assert v["null_mean"] == pytest.approx(v["real_share"]), (name, kind, s)


def test_link_null_matches_slow_shift(gaps, groups):
    """Shifting anchors by hand and re-running link_hits equals the vectorised matrix."""
    anchors, _ = tg.graded_anchors(groups["code:SET PIECES"], BLOCKS)
    lengths = {h: b - a + 1 for h, a, b in BLOCKS}
    firsts = {h: a for h, a, _ in BLOCKS}
    rng = np.random.default_rng(3)
    offs = {1: rng.integers(0, 1400, size=30), 2: rng.integers(0, 1300, size=30)}
    matrix = tg.link_null_matrix(gaps, anchors, BLOCKS, offs, W5)
    for k in range(30):
        shifted = [(h, firsts[h] + (a - firsts[h] + int(offs[h][k])) % lengths[h])
                   for h, a in anchors]
        assert matrix[k].tolist() == tg.link_hits(gaps, shifted, BLOCKS, W5), k


def test_null_same_seed_identical(gaps, groups):
    def run():
        offs = bg.draw_offsets(BLOCKS, 200, 0, 125)
        return tg.view_b(gaps, groups, BLOCKS, w_frames=W5, offsets=offs)
    assert run() == run()


# ---------------------------------------------------------------- view A

NAN = float("nan")


@pytest.fixture(scope="module")
def missing():
    ball, _ = bg.read_ball(RAW)
    return ~np.isfinite(ball).all(axis=1)


def slow_profile(missing, anchors, window_f, bin_f):
    """Frame-by-frame reference: graded frames of the anchor's own half only."""
    nb = 2 * window_f // bin_f
    miss, counted = [0] * nb, [0] * nb
    halves = {h: (a, b) for h, a, b in BLOCKS}
    for h, t in anchors:
        first, last = halves[h]
        for k in range(nb):
            for f in range(t + (k - nb // 2) * bin_f, t + (k - nb // 2 + 1) * bin_f):
                if first <= f <= last:
                    counted[k] += 1
                    miss[k] += int(f >= len(missing) or bool(missing[f]))
    return miss, counted


def same(a, b):
    return len(a) == len(b) and all((x != x and y != y) or x == pytest.approx(y)
                                    for x, y in zip(a, b))


@pytest.mark.parametrize("group, end, n, miss, counted, share", [
    ("type:Throw-in", False, 2, [50, 30, 0, 2], [50, 50, 50, 50], [1.0, 0.6, 0.0, 0.04]),
    ("type:Throw-in", True, 2, [0, 0, 2, 0], [50, 50, 50, 50], [0.0, 0.0, 0.04, 0.0]),
    ("code:LONG BALL", False, 1, [15, 25, 10, 0], [25, 25, 10, 0], [0.6, 1.0, 1.0, NAN]),
    ("type:Kick Off (Start)", False, 0, [0, 0, 0, 0], [0, 0, 0, 0], [NAN] * 4),
])
def test_view_a_expected(missing, groups, group, end, n, miss, counted, share):
    a = tg.view_a(missing, groups, BLOCKS, window_s=2, bin_s=1, end=end)[group]
    assert a["bins_s"] == [-2.0, -1.0, 0.0, 1.0]
    assert a["n"] == n
    assert a["missing"] == miss and a["counted"] == counted
    assert same(a["share"], share)


def test_view_a_dropped(missing, groups):
    a = tg.view_a(missing, groups, BLOCKS, window_s=2, bin_s=1)
    assert a["type:Kick Off (Start)"]["n_dropped"] == 1
    assert a["code:SET PIECES"]["n_dropped"] == 1 and a["code:SET PIECES"]["n"] == 5


@pytest.mark.parametrize("window_f, bin_f", [(50, 25), (750, 25), (100, 50)])
def test_profile_matches_slow(missing, groups, window_f, bin_f):
    halves = tg.half_arrays(missing, BLOCKS)
    for name, tags in groups.items():
        for end in (False, True):
            anchors, _ = tg.graded_anchors(tags, BLOCKS, end=end)
            m, c = tg.profile_counts(halves, anchors, window_f, bin_f)
            assert (m.tolist(), c.tolist()) == slow_profile(missing, anchors, window_f, bin_f)


def test_profile_null_matches_slow_shift(missing, groups):
    halves = tg.half_arrays(missing, BLOCKS)
    anchors, _ = tg.graded_anchors(groups["code:SET PIECES"], BLOCKS)
    firsts = {h: a for h, a, _ in BLOCKS}
    lengths = {h: b - a + 1 for h, a, b in BLOCKS}
    rng = np.random.default_rng(5)
    offs = {1: rng.integers(0, 1400, size=20), 2: rng.integers(0, 1300, size=20)}
    m, c = tg.profile_counts(halves, anchors, 250, 25, offs, chunk=7)
    for k in range(20):
        shifted = [(h, firsts[h] + (a - firsts[h] + int(offs[h][k])) % lengths[h])
                   for h, a in anchors]
        assert (m[k].tolist(), c[k].tolist()) == slow_profile(missing, shifted, 250, 25), k


def test_view_a_null_offset_zero(missing, groups):
    zero = {1: np.array([0]), 2: np.array([0])}
    for name, a in tg.view_a(missing, groups, BLOCKS, window_s=2, bin_s=1,
                             offsets=zero).items():
        if a["n"]:
            assert same(a["null_mean"], a["share"]), name
            assert same(a["null_p95"], a["share"]), name


def test_view_a_null_same_seed(missing, groups):
    def run():
        offs = bg.draw_offsets(BLOCKS, 200, 0, 125)
        return tg.view_a(missing, groups, BLOCKS, window_s=4, bin_s=1, offsets=offs)
    r1, r2 = run(), run()
    for name in r1:
        for key in ("null_mean", "null_p5", "null_p95"):
            if key in r1[name]:
                assert same(r1[name][key], r2[name][key]), (name, key)


def test_window_not_multiple_of_bin(missing, groups):
    with pytest.raises(ValueError):
        tg.view_a(missing, groups, BLOCKS, window_s=2, bin_s=0.6)


# ---------------------------------------------------------------- view C

@pytest.fixture(scope="module")
def view_c(gaps, groups):
    return tg.view_c(gaps, groups, BLOCKS, 2700, w_frames=W5, thresholds_s=[0, 2, 5])


@pytest.mark.parametrize("L, n_long, linked, prec, hit, rec, frames", [
    (0, 11, 5, 5 / 11.0, 4, 0.8, 633 / 2700.0),
    (2, 5, 4, 0.8, 4, 0.8, 550 / 2700.0),
    (5, 2, 2, 1.0, 2, 0.4, 350 / 2700.0),
])
def test_view_c_set_pieces(view_c, L, n_long, linked, prec, hit, rec, frames):
    st = view_c["groups"]["code:SET PIECES"][L]
    assert (st["n_long"], st["linked"], st["anchors"], st["anchors_hit"]) == (n_long, linked, 5,
                                                                             hit)
    assert st["precision"] == pytest.approx(prec)
    assert st["recall"] == pytest.approx(rec)
    assert view_c["n_long"][L] == n_long
    assert view_c["frame_share"][L] == pytest.approx(frames)


@pytest.mark.parametrize("group, L, linked, prec, hit, anchors", [
    ("type:Throw-in", 2, 2, 0.4, 2, 2),
    ("type:Goal Kick", 2, 1, 0.2, 1, 1),
    ("type:Corner Kick", 2, 1, 0.2, 1, 1),
    ("type:Free Kick", 2, 0, 0.0, 0, 1),
    ("type:Throw-in", 5, 1, 0.5, 1, 2),
])
def test_view_c_types(view_c, group, L, linked, prec, hit, anchors):
    st = view_c["groups"][group][L]
    assert (st["linked"], st["anchors_hit"], st["anchors"]) == (linked, hit, anchors)
    assert st["precision"] == pytest.approx(prec)
    assert st["recall"] == pytest.approx(hit / float(anchors))


def test_view_c_no_graded_anchor(view_c):
    st = view_c["groups"]["type:Kick Off (Start)"][2]
    assert (st["anchors"], st["linked"], st["precision"]) == (0, 0, 0.0)
    assert st["recall"] != st["recall"]                         # NaN


# ---------------------------------------------------------------- CLI

def read_rows(path):
    import csv
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_main_fixture(tmp_path, capsys):
    out = tmp_path / "tg"
    res = tg.main([str(out), "--raw", RAW, "--tags", TAGS, "--frames", "100-1499,1600-2899",
                   "--window", "2", "--bin", "1", "--null-n", "50", "--null-min-shift", "5",
                   "--thresholds", "0,2,5"])
    assert res["n_gaps"] == 11 and res["n_unlinkable"] == 1 and res["null_skipped"] is None
    rows = read_rows(str(out / tg.CSV_NAME))
    assert list(rows[0]) == ["analysis", "scope", "bin", "metric", "value", "unit"]
    assert rows[0]["analysis"] == "params"
    get = {(r["analysis"], r["scope"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert float(get[("overview", "all", "", "baseline_missing_share")]) == \
        pytest.approx(633 / 2700.0, abs=1e-6)
    assert float(get[("threshold", "code:SET PIECES", "2", "precision")]) == 0.8
    assert float(get[("threshold", "code:SET PIECES", "5", "recall")]) == 0.4
    assert float(get[("link", "code:SET PIECES", "2-5s", "real_count")]) == 2
    assert float(get[("profile_start", "type:Throw-in", "-1", "share")]) == 0.6
    assert get[("profile_start", "code:LONG BALL", "1", "share")] == ""          # NaN
    assert ("profile_start", "type:Throw-in", "-1", "null_p95") in get
    printed = capsys.readouterr().out
    assert len(printed.splitlines()) <= 12
    assert "SET PIECES" in printed


def test_main_without_tags(tmp_path, capsys):
    res = tg.main([str(tmp_path / "o"), "--raw", RAW, "--frames", "100-1499,1600-2899"])
    assert res["tags"] is None
    assert "views A-C skipped" in capsys.readouterr().out
    rows = read_rows(str(tmp_path / "o" / tg.CSV_NAME))
    assert {r["analysis"] for r in rows} == {"params", "overview"}


def test_main_null_too_short(tmp_path):
    res = tg.main([str(tmp_path / "o"), "--raw", RAW, "--tags", TAGS, "--frames",
                   "100-1499,1600-2899", "--window", "2", "--bin", "1"])
    assert "shorter" in res["null_skipped"]
    assert "null_mean" not in res["profile_start"]["type:Throw-in"]


def test_main_refuses_repo_and_bad_window(tmp_path):
    from conftest import EDA
    with pytest.raises(SystemExit):
        tg.main([os.path.join(EDA, "x_out"), "--raw", RAW, "--frames", "100-1499"])
    with pytest.raises(SystemExit):
        tg.main([str(tmp_path / "o"), "--raw", RAW, "--frames", "100-1499", "--window", "2",
                 "--bin", "0.6"])
