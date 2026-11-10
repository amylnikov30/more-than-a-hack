"""Exact expected numbers on the synthetic fixture. See tests/fixtures/README.md for the design.

    cd eda && python -m pytest -q tests
"""

import csv
import os

import numpy as np
import pytest

import ball_gaps as bg
from conftest import FIXTURES

RAW = os.path.join(FIXTURES, "fixture_atd.txt.gz")
TAGS = os.path.join(FIXTURES, "fixture_smart_tagging.xml")
BLOCKS = [(1, 100, 1499), (2, 1600, 2899)]
FRAMES_ARG = "100-1499,1600-2899"


@pytest.fixture(scope="module")
def full():
    """Crowding baseline over every ball-visible graded frame (no sampling)."""
    return bg.analyse(RAW, BLOCKS, TAGS, sample_size=10 ** 6)


def test_gap_list(full):
    ball, _ = bg.read_ball(RAW)
    gaps = bg.find_gaps(~np.isfinite(ball).all(axis=1), BLOCKS)
    got = [(g["half"], g["start"], g["end"], g["edge"], g["last_seen"]) for g in gaps]
    assert got == [
        (1, 200, 204, False, 199), (1, 300, 311, False, 299), (1, 400, 412, False, 399),
        (1, 500, 549, False, 499), (1, 700, 899, False, 699), (1, 1450, 1499, True, 1449),
        (2, 1600, 1619, True, 1589), (2, 1800, 1830, False, 1799),
        (2, 2000, 2099, False, 1999), (2, 2500, 2649, False, 2499),
        (2, 2700, 2701, False, 2699),
    ]
    assert full["n_gaps"] == 11 and full["edge_gaps"] == 2
    assert full["no_record_frames"] == 0


def test_length_tables(full):
    L = full["lengths"]
    assert L["overall"] == {"<0.5s": (3, 19), "0.5-2s": (3, 64), "2-5s": (3, 200),
                            ">=5s": (2, 350), "all": (11, 633)}
    assert L["half 1"] == {"<0.5s": (2, 17), "0.5-2s": (1, 13), "2-5s": (2, 100),
                           ">=5s": (1, 200), "all": (6, 330)}
    assert L["half 2"] == {"<0.5s": (1, 2), "0.5-2s": (2, 51), "2-5s": (1, 100),
                           ">=5s": (1, 150), "all": (5, 303)}
    assert full["graded_frames"] == {1: 1400, 2: 1300}


def test_bin_boundaries():
    assert bg.bin_label(12) == "<0.5s"      # 0.48 s
    assert bg.bin_label(13) == "0.5-2s"     # 0.52 s
    assert bg.bin_label(49) == "0.5-2s"
    assert bg.bin_label(50) == "2-5s"       # exactly 2 s
    assert bg.bin_label(124) == "2-5s"
    assert bg.bin_label(125) == ">=5s"      # exactly 5 s


def test_set_pieces(full):
    sp = full["set_pieces"]
    assert sp["n_intervals"] == 6            # BUILD UP excluded, mixed-case codes included
    assert (sp["n_long"], sp["n_overlap"]) == (5, 3)
    assert sp["by_bin"] == {"2-5s": (3, 1), ">=5s": (2, 2)}


def test_set_pieces_tolerance():
    res = bg.analyse(RAW, BLOCKS, TAGS, sample_size=10, tag_tolerance_s=1.0)
    assert (res["set_pieces"]["n_long"], res["set_pieces"]["n_overlap"]) == (5, 5)


def test_crowding_exact(full):
    c = full["crowding"]
    assert c["gap_start_all"]["n"] == 11
    assert c["gap_start_all"]["mean"] == pytest.approx(38 / 11)
    assert c["gap_start_all"]["median"] == 3.0
    assert c["gap_start_long"]["n"] == 5 and c["gap_start_long"]["mean"] == pytest.approx(5.0)
    assert c["gap_start_short"]["n"] == 6 and c["gap_start_short"]["mean"] == pytest.approx(13 / 6)
    assert c["gaps_without_last_seen"] == 0
    assert c["visible_population"] == 2067
    assert c["visible"]["n"] == 2067
    assert c["visible"]["mean"] == pytest.approx(2094 / 2067)


def test_crowding_sample_is_seeded():
    a = bg.analyse(RAW, BLOCKS, None, sample_size=500, seed=0)["crowding"]["visible"]
    b = bg.analyse(RAW, BLOCKS, None, sample_size=500, seed=0)["crowding"]["visible"]
    assert a == b and a["n"] == 500
    assert 1.0 <= a["mean"] <= 1.0 + 27 / 500


def test_no_tags_skips_b():
    assert bg.analyse(RAW, BLOCKS, None, sample_size=10)["set_pieces"] is None


def test_crowd_count_threshold_and_nan():
    players = np.array([[0.5 + 4.99 / 105.3, 0.5], [0.5, 0.5 + 5.01 / 68.0]])
    assert bg.crowd_count(players, (0.5, 0.5)) == 1
    assert bg.crowd_count(np.empty((0, 2)), (0.5, 0.5)) == 0


def test_cli_writes_csv_and_prints_few_lines(tmp_path, capsys):
    bg.main([str(tmp_path), "--raw", RAW, "--tags", TAGS, "--frames", FRAMES_ARG,
             "--sample-size", "100000"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) <= 14
    rows = list(csv.DictReader(open(os.path.join(str(tmp_path), bg.CSV_NAME))))
    assert list(rows[0]) == ["analysis", "scope", "bin", "metric", "value", "unit"]
    get = {(r["analysis"], r["scope"], r["bin"], r["metric"]): r["value"] for r in rows}
    assert get[("gap_length", "overall", "all", "missing_frames")] == "633"
    assert float(get[("gap_length", "overall", ">=5s", "share_of_missing_frames")]) == \
        pytest.approx(100 * 350 / 633, abs=1e-5)
    assert get[("set_pieces", "overall", ">=long", "n_overlap")] == "3"
    assert float(get[("crowding", "visible", "", "mean")]) == pytest.approx(2094 / 2067, abs=1e-6)


def test_cli_requires_frames_with_raw(tmp_path):
    with pytest.raises(SystemExit):
        bg.main([str(tmp_path), "--raw", RAW])


def test_parse_frames():
    assert bg.parse_frames("0-10,20-30") == [(1, 0, 10), (2, 20, 30)]
