"""deadball_matching.py and summarise_review.py on a synthetic two-half match built here.

Halves are video frames 0-1999 and 2000-3999 at 25 fps. In play, the ball moves at about 1.3 m/s
along x. Events (inclusive frames):

    200-299    ball missing (4 s)                        -> interval A, "missing"
    500-549    ball 5 m beyond a goal line, moving       \\
    550-599    ball missing (2 s)                         -> interval B, "missing+outside"
    900-959    ball still on the centre spot (2.4 s)     -> interval C, "still"
    1200-1210  ball missing (0.44 s): too short, not dead
    1700-1759  ball still in open play ("elsewhere"): not dead by default
    2500-2599, 2615-2699  missing, 15 visible frames between (< 1 s)
                                                         -> interval D, 2500-2699, "missing"
    3300-3399  ball missing (4 s)                        -> interval E, "missing"

SET PIECES starts: 300 Corner Kick, 650 Throw-in, 1500 Free Kick, 2702 Goal Kick, 3390 Throw-in,
3405 Corner Kick (one coded "set pieces" in lower case), plus a BUILD UP tag at 900 that must be
ignored. Expected one-to-one pairs (window 5 s = 125 frames): A-300 (-1 f), B-650 (-51 f),
D-2702 (-3 f), E-3405 (-6 f; 3390 is 9 f away, so it loses). Unmatched: C; tags 1500 and 3390.
"""

import csv
import math
import os
import sys

import numpy as np
import pytest

# No conftest.py here: eda/tests does `from conftest import FIXTURES`, and a second module named
# conftest would shadow it when both folders are collected in one run.
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(os.path.dirname(PKG), "scripts"), PKG):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import deadball_matching as dm  # noqa: E402
import summarise_review as sr  # noqa: E402

FPS = 25
BLOCKS = [(1, 0, 1999), (2, 2000, 3999)]
N = 4000
TAGS = [("SET PIECES", 300, "Corner Kick"), ("SET PIECES", 650, "Throw-in"),
        ("BUILD UP", 900, None), ("SET PIECES", 1500, "Free Kick"),
        ("SET PIECES", 2702, "Goal Kick"), ("set pieces", 3390, "Throw-in"),
        ("SET PIECES", 3405, "Corner Kick")]
A = pytest.approx


def ball_at(f):
    if 200 <= f <= 299 or 550 <= f <= 599 or 1200 <= f <= 1210 or 2500 <= f <= 2599 or \
            2615 <= f <= 2699 or 3300 <= f <= 3399:
        return None
    if 500 <= f <= 549:
        return 1.05, 0.3 + 0.002 * (f - 500)
    if 900 <= f <= 959:
        return 0.5, 0.5
    if 1700 <= f <= 1759:
        return 0.3, 0.3
    return 0.3 + 0.0005 * (f % 400), 0.4


def write_raw(path):
    with open(path, "w") as handle:
        for f in range(N):
            b = ball_at(f)
            ball = "NaN,NaN" if b is None else "{:.6f},{:.6f}".format(*b)
            handle.write("{}:0.5,0.5;0.6,0.6:{}\n".format(f + 1, ball))


def write_tags(path):
    inst = []
    for i, (code, start, kind) in enumerate(TAGS, 1):
        label = ("<label><group>Type</group><text>{}</text></label>".format(kind)
                 if kind else "")
        inst.append("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end>"
                    "<code>{}</code>{}</instance>".format(i, start / 25.0, start / 25.0 + 4,
                                                         code, label))
    with open(path, "w") as handle:
        handle.write("<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format("".join(inst)))


def write_gt(path):
    """Labelled frames 100-1099: D 200-299 and 480-649, H elsewhere; X outside."""
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for f in range(N):
            if not 100 <= f <= 1099:
                lab = "X"
            elif 200 <= f <= 299 or 480 <= f <= 649:
                lab = "D"
            else:
                lab = "H"
            handle.write("{},{}\n".format(f, lab))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("dm")
    paths = {k: str(d / name) for k, name in
             (("raw", "raw.txt"), ("tags", "tags.xml"), ("gt", "gt.csv"))}
    write_raw(paths["raw"])
    write_tags(paths["tags"])
    write_gt(paths["gt"])
    return paths


@pytest.fixture(scope="module")
def res(files):
    return dm.analyse(files["raw"], BLOCKS, files["tags"], files["gt"], null_n=0)


def spans(res):
    return [(iv["start"], iv["end"], iv["signals"]) for iv in res["intervals"]]


def test_intervals(res):
    assert spans(res) == [(200, 299, "missing"), (500, 599, "missing+outside"),
                          (900, 959, "still"), (2500, 2699, "missing"), (3300, 3399, "missing")]
    assert res["n_edge"] == 0
    assert res["signal_frames"]["still"] == 60          # the open-play still run is excluded


def test_still_elsewhere_counts_with_all_locations(files):
    r = dm.analyse(files["raw"], BLOCKS, files["tags"], still_locations=dm.bs.LOCATIONS,
                   null_n=0)
    assert (1700, 1759, "still") in spans(r)


def test_signal_subset(files):
    r = dm.analyse(files["raw"], BLOCKS, files["tags"], signals=("outside",), null_n=0)
    assert spans(r) == [(500, 549, "outside")]


def test_no_merge_splits_d(files):
    r = dm.analyse(files["raw"], BLOCKS, files["tags"], merge_gap_s=0.4, null_n=0)
    assert (2500, 2599, "missing") in spans(r) and (2615, 2699, "missing") in spans(r)


def test_set_pieces_filter(files):
    tags, dropped = dm.set_piece_tags(files["tags"], BLOCKS)
    assert [t["start"] for t in tags] == [300, 650, 1500, 2702, 3390, 3405]
    assert dropped == 0
    tags, dropped = dm.set_piece_tags(files["tags"], [(1, 0, 1999)])
    assert len(tags) == 3 and dropped == 3


def test_matching(res):
    tags = res["tags"]
    got = {res["intervals"][i]["end"]: tags[j]["start"] for i, j in res["pairs"].items()}
    assert got == {299: 300, 599: 650, 2699: 2702, 3399: 3405}
    assert [tags[j]["start"] for j in res["unmatched_tags"]] == [1500, 3390]
    offs = sorted(iv["offset_s"] for iv in res["intervals"] if iv["tag"] is not None)
    assert offs == [A(-51 / 25.0), A(-6 / 25.0), A(-3 / 25.0), A(-1 / 25.0)]


def test_match_one_to_one_and_half():
    # one end, two starts: the closer wins; a start in the other half never matches
    assert dm.match([(1, 100)], [(1, 90), (1, 104)], 10) == {0: 1}
    assert dm.match([(1, 100)], [(2, 100)], 10) == {}
    # two ends, one start: the closer end wins
    assert dm.match([(1, 100), (1, 120)], [(1, 118)], 30) == {1: 0}
    # outside the window
    assert dm.match([(1, 100)], [(1, 111)], 10) == {}
    # greedy closest-first lets both pair here
    assert dm.match([(1, 100), (1, 110)], [(1, 105), (1, 112)], 10) == {0: 0, 1: 1}


def test_precision_recall(res):
    assert res["precision"]["all"] == [5, 4]
    assert res["precision"]["signals:still"] == [1, 0]
    assert res["precision"]["signals:missing"] == [3, 3]
    assert res["precision"]["length:2-5s"] == [4, 3]
    assert res["recall"]["code:SET PIECES"] == [6, 4]
    assert res["recall"]["type:Corner Kick"] == [2, 2]
    assert res["recall"]["type:Throw-in"] == [2, 1]
    assert res["recall"]["type:Free Kick"] == [1, 0]
    assert res["crosstab"] == {("missing", "Corner Kick"): 2, ("missing", "Goal Kick"): 1,
                               ("missing+outside", "Throw-in"): 1}


def test_gt(res):
    gt = res["gt"]
    assert gt["n_scored"] == 1000 and gt["n_dead"] == 270
    iv = gt["frames"]["intervals"]
    assert (iv["tp"], iv["fp"], iv["fn"]) == (200, 60, 70)
    assert iv["precision"] == A(200 / 260.0) and iv["recall"] == A(200 / 270.0)
    assert gt["frames"]["outside"]["tp"] == 50
    eda, st = gt["restarts"]["eda"], gt["restarts"]["smart_tagging"]
    assert (eda["n_pred"], eda["n_gt"], eda["matched"]) == (3, 2, 2)
    assert (st["n_pred"], st["n_gt"], st["matched"]) == (2, 2, 2)


def test_null(files):
    offsets = {1: np.array([0, 1000, 37]), 2: np.array([0, 1000, 37])}
    r = dm.analyse(files["raw"], BLOCKS, files["tags"], offsets=offsets)
    st = r["null"]["pairs"]
    assert st["real"] == 4
    total, _ = dm.null_counts([(iv["half"], iv["end"]) for iv in r["intervals"]], r["tags"],
                              BLOCKS, offsets, 125, r["types"])
    assert total[0] == 4                      # shift 0 reproduces the real match
    assert st["null_mean"] == A(total.mean())
    assert 0 < st["p_emp"] <= 1


def test_null_skipped_when_halves_short(files):
    r = dm.analyse(files["raw"], BLOCKS, files["tags"])     # 80 s halves, 60 s min shift
    assert r["null"] is None and "shorter" in r["null_skipped"]


def test_review_sample(res):
    rows, counts = dm.review_sample(res, n=8)
    assert counts == {"dead_no_tag": (1, 1, 1), "tag_no_dead": (2, 2, 2)}
    assert [r[0] for r in rows] == ["dead_no_tag", "tag_no_dead", "tag_no_dead"]
    assert rows[0][2:4] == [900, 959] and rows[0][8] == "0:00:33.00"
    assert rows[1][6] == 1500 and rows[1][8] == "0:00:50.00" and rows[1][9] == "0:01:03.00"
    rows, counts = dm.review_sample(res, n=8, exclude=[(1400, 1600)])
    assert counts["tag_no_dead"] == (2, 1, 1)


def test_main_writes_outputs(files, tmp_path):
    out = str(tmp_path / "out")
    dm.main([out, "--raw", files["raw"], "--tags", files["tags"], "--frames",
             "0-1999,2000-3999", "--gt", files["gt"], "--null-min-shift", "10",
             "--null-n", "20"])
    with open(os.path.join(out, dm.SUMMARY_NAME), newline="") as handle:
        summary = list(csv.DictReader(handle))
    prec = [r for r in summary if r["analysis"] == "precision" and r["scope"] == "all" and
            r["metric"] == "precision"]
    assert float(prec[0]["value"]) == A(0.8)
    assert any(r["analysis"] == "null" and r["metric"] == "lift" for r in summary)
    with open(os.path.join(out, dm.MATCHES_NAME), newline="") as handle:
        groups = [r["group"] for r in csv.DictReader(handle)]
    assert groups.count("matched") == 4 and groups.count("dead_no_tag") == 1 and \
        groups.count("tag_no_dead") == 2


def test_main_refuses_in_repo_out_dir(files):
    with pytest.raises(SystemExit):
        dm.main([dm.HERE, "--raw", files["raw"], "--tags", files["tags"], "--frames",
                 "0-1999,2000-3999"])


def test_summarise_review(tmp_path):
    path = tmp_path / "review.csv"
    with open(str(path), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(dm.REVIEW_COLUMNS)
        w.writerow(["dead_no_tag", 1, 900, 959, 2.4, "still", "", "", "a", "b", "Foul", "x"])
        w.writerow(["dead_no_tag", 1, 1900, 1959, 2.4, "missing", "", "", "a", "b", "foul", ""])
        w.writerow(["tag_no_dead", 1, "", "", "", "", 1500, "Free Kick", "a", "b", "", ""])
    rows = sr.main([str(path)])
    assert ["dead_no_tag", "all", "", "foul", 2] in rows
    assert ["tag_no_dead", "tag_type", "Free Kick", sr.UNLABELLED, 1] in rows
    assert not any(isinstance(v, int) and v >= 900 for r in rows for v in r)


def test_frame_scores_nan_safe():
    z = np.zeros(4, dtype=bool)
    sc = dm.frame_scores(z, z, np.ones(4, dtype=bool))
    assert math.isnan(sc["precision"]) and math.isnan(sc["f1"])
