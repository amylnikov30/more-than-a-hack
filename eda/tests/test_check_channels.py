import os

import check_channels as cc
from conftest import FIXTURES

META = os.path.join(FIXTURES, "fixture_atd.xml")


def test_fixture_layout(tmp_path):
    summary, full = cc.report(META)
    text = "\n".join(summary)
    assert "players: 46 slots x [x, y, speed]" in text
    assert "ball: [ball_x, ball_y, ball_speed]" in text
    assert "NONE declared" in text
    assert full.count("player1_x") == 1


def test_flags_status_channels(tmp_path):
    xml = tmp_path / "flag.xml"
    xml.write_text(
        '<main><DataFormatSpecifications><DataFormatSpecification startFrame="1" endFrame="2" '
        'separator=":"><StringRegister name="frameCount"/><SplitRegister separator=",">'
        '<BallChannelRef channelId="x"/><StringRegister name="ballOwningTeam"/>'
        '<StringRegister name="ballStatus"/></SplitRegister></DataFormatSpecification>'
        '</DataFormatSpecifications></main>')
    names = [n for _, n in cc.status_candidates(cc.ET.parse(str(xml)).getroot())]
    assert names == ["ballOwningTeam", "ballStatus"]


def test_cli_writes_file(tmp_path, capsys):
    cc.main([str(tmp_path), "--meta", META])
    assert (tmp_path / "channels.txt").exists()
    assert len(capsys.readouterr().out.splitlines()) <= 12
