"""ball_gaps.py robustness fixes. Expected values: tests/fixtures/README.md, section
`ball_gaps.py` robustness fixes."""

import os
import sys
import types

import pytest

import ball_gaps as bg
import sample_long_gaps as slg
from conftest import FIXTURES, EDA

FLIP_RAW = os.path.join(FIXTURES, "fixture_flip_atd.txt")
FLIP_META = os.path.join(FIXTURES, "fixture_flip_atd.xml")
RAW = os.path.join(FIXTURES, "fixture_atd.txt.gz")
META = os.path.join(FIXTURES, "fixture_atd.xml")
BLOCKS = [(1, 100, 1499), (2, 1600, 2899)]


@pytest.fixture
def open_ended_meta(tmp_path):
    with open(FLIP_META) as handle:
        text = handle.read()
    assert 'endFrame="200"' in text
    path = tmp_path / "open_ended.xml"
    path.write_text(text.replace('endFrame="200" ', ""))
    return str(path)


def test_spec_without_end_frame(open_ended_meta):
    specs = bg.read_slot_teams(open_ended_meta)
    for frame in (150, 10 ** 6):
        assert bg.teams_for(specs, frame).tolist() == [0, 1, 0, 1, -1, 0]
    res = bg.analyse(FLIP_RAW, [(1, 10, 99), (2, 110, 189)], None, sample_size=10 ** 6,
                     meta_path=open_ended_meta)
    c = res["flip"]["tables"][("overall", "all")]
    assert (c["n_gaps"], c["n_evaluable"], c["n_flip"]) == (8, 5, 2)


def fake_games(monkeypatch, missing=()):
    calls = []
    paths = {"atd_raw": RAW, "atd_meta": META, "smart_tagging": "/nonexistent/tags.xml"}

    def require(names, game="BAR-ATH", data_dir=None, source="tactical"):
        calls.append(("require", game))
        gone = [n for n in names if n in missing]
        if gone:
            raise IOError("match data not found: {}".format(gone))
        return [paths[n] for n in names]

    def half_video_blocks(game, adjust=0):
        calls.append(("blocks", game))
        return BLOCKS

    mod = types.ModuleType("games")
    mod.require, mod.half_video_blocks = require, half_video_blocks
    monkeypatch.setitem(sys.modules, "games", mod)
    return calls


def test_game_and_optional_tags(monkeypatch, tmp_path, capsys):
    calls = fake_games(monkeypatch, missing=("smart_tagging",))
    res = bg.main([str(tmp_path / "o"), "--game", "XX", "--sample-size", "10"])
    assert calls and all(game == "XX" for _, game in calls)
    assert ("blocks", "XX") in calls
    assert res["set_pieces"] is None and res["set_pieces_null"] is None
    assert res["n_gaps"] == 11
    assert "smart tagging" in capsys.readouterr().out.lower()


def test_missing_tracking_still_exits(monkeypatch, tmp_path):
    fake_games(monkeypatch, missing=("atd_raw",))
    with pytest.raises(SystemExit):
        bg.main([str(tmp_path / "o")])


def test_refusal_names_script(capsys):
    with pytest.raises(SystemExit) as err:
        bg.main([os.path.join(EDA, "x_out"), "--raw", RAW, "--frames", "100-1499"])
    assert "~/eda_out/ball_gaps" in str(err.value)
    with pytest.raises(SystemExit) as err:
        slg.main([os.path.join(EDA, "x_out"), "--raw", RAW, "--frames", "100-1499"])
    assert "~/eda_out/long_gaps" in str(err.value)
