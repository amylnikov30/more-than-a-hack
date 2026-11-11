"""sample_long_gaps.py on the synthetic fixtures. Expected values: tests/fixtures/README.md,
section `sample_long_gaps.py`."""

import csv
import os

import pytest

import sample_long_gaps as slg
from conftest import FIXTURES, EDA

RAW = os.path.join(FIXTURES, "fixture_atd.txt.gz")
FRAMES = "100-1499,1600-2899"
COLUMNS = ["half", "start_frame", "end_frame", "gap_s", "view_from", "view_to", "category",
           "notes"]


def run(tmp_path, *extra, name="out"):
    out = tmp_path / name
    slg.main([str(out), "--raw", RAW, "--frames", FRAMES] + list(extra))
    with open(out / slg.CSV_NAME, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return header, rows


def halves(rows):
    return [sum(r[0] == str(h) for r in rows) for h in (1, 2)]


def test_candidates_drop_edge_gaps():
    res = slg.analyse(RAW, slg.parse_frames(FRAMES))
    assert res["n_gaps"] == 11 and res["n_candidates"] == 9


@pytest.mark.parametrize("extra, threshold, n_eligible, per_half", [
    (["--quantile", "0.9"], 160.0, 1, [1, 0]),
    (["--quantile", "0.5"], 31.0, 5, [2, 3]),
    (["--quantile", "0.5", "--n", "3"], 31.0, 5, [2, 1]),
    (["--quantile", "0.5", "--n", "4"], 31.0, 5, [2, 2]),
    (["--quantile", "0.5", "--exclude", "450-920", "--n", "2"], 31.0, 3, [0, 2]),
])
def test_quantile_cases(tmp_path, extra, threshold, n_eligible, per_half):
    res = slg.analyse(RAW, slg.parse_frames(FRAMES), **slg.options_from(extra))
    assert res["threshold_frames"] == pytest.approx(threshold)
    assert len(res["eligible"]) == n_eligible
    header, rows = run(tmp_path, *extra)
    assert header == COLUMNS
    assert halves(rows) == per_half


def test_range_half_open(tmp_path):
    _, rows = run(tmp_path, "--min-s", "2", "--max-s", "6")
    assert [(r[1], r[2]) for r in rows] == [("500", "549"), ("2000", "2099")]   # G4, G9; not G10


def test_min_only(tmp_path):
    _, rows = run(tmp_path, "--min-s", "2")
    assert sorted(r[1] for r in rows) == sorted(["500", "700", "2000", "2500"])
    assert halves(rows) == [2, 2]


def test_empty(tmp_path):
    header, rows = run(tmp_path, "--min-s", "10")
    assert header == COLUMNS and rows == []


def test_g5_row(tmp_path):
    _, rows = run(tmp_path, "--quantile", "0.9")
    assert rows == [["1", "700", "899", "8.0", "0:00:25.00", "0:00:38.96", "", ""]]


def test_sorted_and_deterministic(tmp_path):
    run(tmp_path, "--quantile", "0.5", "--n", "3", name="a")
    run(tmp_path, "--quantile", "0.5", "--n", "3", name="b")
    a = (tmp_path / "a" / slg.CSV_NAME).read_bytes()
    assert a == (tmp_path / "b" / slg.CSV_NAME).read_bytes()
    starts = [int(line.split(",")[1]) for line in a.decode().splitlines()[1:]]
    assert starts == sorted(starts)


def test_quotas():
    assert slg.quotas({1: 2, 2: 3}, 3) == {1: 2, 2: 1}
    assert slg.quotas({1: 0, 2: 3}, 2) == {1: 0, 2: 2}
    assert slg.quotas({1: 2, 2: 3}, 99) == {1: 2, 2: 3}


def test_refuses_repo_out_dir():
    with pytest.raises(SystemExit):
        slg.main([os.path.join(EDA, "x_out"), "--raw", RAW, "--frames", FRAMES])


def test_absent_record_not_merged(tmp_path):
    raw = tmp_path / "tiny.txt"
    lines = []
    for fid in range(1, 11):
        if fid == 8:
            continue
        ball = "NaN,NaN" if fid in (4, 5) else "0.5,0.5"
        lines.append("{}:0.4,0.4;0.6,0.6:{}".format(fid, ball))
    raw.write_text("\n".join(lines) + "\n")
    res = slg.analyse(str(raw), slg.parse_frames("0-9"))
    assert [(g["start"], g["end"]) for g in res["gaps"]] == [(3, 4), (7, 7)]


def test_csv_name(tmp_path):
    out = tmp_path / "o"
    slg.main([str(out), "--raw", RAW, "--frames", FRAMES, "--min-s", "2",
              "--csv-name", "gap_test.csv"])
    assert sorted(os.listdir(out)) == ["gap_test.csv"]
    with open(out / "gap_test.csv", newline="") as handle:
        assert len(list(csv.reader(handle))) == 1 + 4


@pytest.mark.parametrize("name", ["sub/x.csv", "x.txt"])
def test_csv_name_rejected(tmp_path, name):
    with pytest.raises(SystemExit):
        slg.main([str(tmp_path / "o"), "--raw", RAW, "--frames", FRAMES, "--csv-name", name])
