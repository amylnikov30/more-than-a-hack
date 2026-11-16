"""run.py end to end on the synthetic possession fixture (eda/tests/fixtures/fixture_poss_*):
2,000 frames, halves 0-999 and 1000-1999, 150 dead frames in the dead file."""

import os

import numpy as np

from conftest import EDA_FIXTURES
from possession import run


def _run(out_dir):
    argv = [str(out_dir), "--dead", os.path.join(EDA_FIXTURES, "fixture_poss_dead.csv"),
            "--raw", os.path.join(EDA_FIXTURES, "fixture_poss_atd.txt"),
            "--meta", os.path.join(EDA_FIXTURES, "fixture_poss_atd.xml"),
            "--frames", "0-999,1000-1999", "--pitch", "105", "68"]
    run.main(argv)
    with open(os.path.join(str(out_dir), "possession.csv"), "rb") as handle:
        return handle.read()


def test_end_to_end_and_byte_identical(tmp_path):
    """One row per dead-file row; D on exactly the 150 dead frames, H / A elsewhere, no X; two
    runs give the same bytes."""
    first = _run(tmp_path / "a")
    assert first == _run(tmp_path / "b")
    lines = first.decode().splitlines()
    assert lines[0] == "frame,label" and len(lines) == 2001
    labels = np.array([line.split(",")[1] for line in lines[1:]])
    assert int(np.sum(labels == "D")) == 150
    assert set(labels[labels != "D"]) <= {"H", "A"}
    assert [int(line.split(",")[0]) for line in lines[1:]] == list(range(2000))
