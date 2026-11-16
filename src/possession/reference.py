"""Build the reference set ("Track B"): Metrica's open sample games, degraded to look like our ATD,
with possession labels derived from Metrica's own event data by code.

    PYTHONPATH="$PWD:$PWD/src" python3 -m possession.reference OUT_DIR --sample-data DIR

DIR is a clone of github.com/metrica-sports/sample-data (data/Sample_Game_N/...) or a folder
holding Sample_Game_3_{tracking.txt,metadata.xml,events.json} directly; games whose files are
missing are skipped. Per game, OUT_DIR/sgN/ gets

    tracking.txt     degraded tracking in the raw ATD format (normalised 0-1, frame_id = video + 1):
                     ball dropped in bursts (poss_common.degrade_ball), player tracks broken and
                     re-slotted (poss_common.fragment_tracks) into 23 home + 23 away slots
    metadata.xml     slot -> team for that file, team names, declared attacking direction
    labels.csv       frame,label: the reference ground truth (H / A / D; X off the graded halves
                     and on live frames before the half's first controlled event)
    dead_ball.csv    frame,dead: the oracle mask from the same events (1 / 0 / -1)
    manifest.json    frames, pitch, seed and aggregate counts

Labels come from poss_reference.reference_labels (events -> dead and team, CHALLENGE.md rules).
Nothing here touches the competition match. Fixed seed: reruns are byte-identical.
"""

import argparse
import glob
import json
import os
from xml.sax.saxutils import escape

import numpy as np

import poss_common as pc
import poss_reference as pr

from possession import config

POOL = 23            # slots per team: 46 in all, as in the match ATD
SG3_FRAMES = [(1, 0, 69660), (2, 69661, 143760)]
SAMPLE_PITCH = (105.0, 68.0)


# ---------------------------------------------------------------- finding and loading games

def _find(root, name):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    return hits[0] if hits else None


def load_epts(root, game):
    """Sample Game 3 (EPTS): {ball, players, team, record, meta, events, blocks, team_ids}."""
    raw = _find(root, "Sample_Game_{}_tracking.txt".format(game))
    meta = _find(root, "Sample_Game_{}_metadata.xml".format(game))
    events = _find(root, "Sample_Game_{}_events.json".format(game))
    if not (raw and meta and events):
        return None
    blocks = SG3_FRAMES
    n = blocks[-1][2] + 1
    trk = pc.load_tracking(raw, meta, n, SAMPLE_PITCH)
    teams = pc.read_teams(meta)
    return {"ball": trk["ball"], "players": trk["players"], "team": trk["team"],
            "record": trk["record"], "events": pr.read_events(events), "blocks": blocks,
            "team_ids": [t[0] for t in teams], "team_names": [t[1] for t in teams],
            "directions": pc.read_meta_directions(meta)}


def _csv_events(path):
    """Metrica CSV events -> the dicts of poss_reference.read_events (team ids Home / Away)."""
    import csv
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(row["Start Frame"]) - 1
                end = int(row["End Frame"]) - 1 if row.get("End Frame") else start
            except (KeyError, ValueError):
                continue
            sub = (row.get("Subtype") or "").strip().upper()
            out.append({"kind": (row.get("Type") or "").strip().upper(),
                        "subtypes": set(s.strip() for s in sub.split("-") if s.strip()),
                        "team": (row.get("Team") or "").strip(),
                        "start": start, "end": max(end, start),
                        "period": int(row.get("Period") or 0)})
    return out


def load_csv_game(root, game):
    """Sample Games 1-2 (CSV): same dict as load_epts. Teams Home / Away; no declared
    direction, so the attacking direction is inferred from the team centroids."""
    stem = "Sample_Game_{}_RawTrackingData_".format(game)
    home_path, away_path = _find(root, stem + "Home_Team.csv"), _find(root, stem + "Away_Team.csv")
    events = _find(root, "Sample_Game_{}_RawEventsData.csv".format(game))
    if not (home_path and away_path and events):
        return None
    import pandas as pd
    home = pd.read_csv(home_path, skiprows=2)
    away = pd.read_csv(away_path, skiprows=2)
    xy_h = home.iloc[:, 3:-2].values.reshape(len(home), -1, 2)
    xy_a = away.iloc[:, 3:-2].values.reshape(len(away), -1, 2)
    frames = home["Frame"].values.astype(int) - 1
    period = home["Period"].values.astype(int)
    n = int(frames.max()) + 1
    players = np.full((n, xy_h.shape[1] + xy_a.shape[1], 2), np.nan, dtype=np.float32)
    players[frames] = np.concatenate([xy_h, xy_a], axis=1)
    ball = np.full((n, 2), np.nan)
    ball[frames] = home.iloc[:, -2:].values
    players[..., 0] *= SAMPLE_PITCH[0]
    players[..., 1] *= SAMPLE_PITCH[1]
    ball *= np.asarray(SAMPLE_PITCH)
    team = np.zeros((n, players.shape[1]), dtype=np.int8)
    team[:, xy_h.shape[1]:] = 1
    record = np.zeros(n, dtype=bool)
    record[frames] = True
    blocks = [(h, int(frames[period == h].min()), int(frames[period == h].max()))
              for h in (1, 2) if np.any(period == h)]
    return {"ball": ball, "players": players, "team": team, "record": record,
            "events": _csv_events(events), "blocks": blocks, "team_ids": ["Home", "Away"],
            "team_names": ["Home", "Away"], "directions": None}


def load_game(root, game):
    return load_epts(root, game) if game == 3 else load_csv_game(root, game)


# ---------------------------------------------------------------- degrading and writing

def degrade(g, live, seed=config.DEGRADE_SEED):
    """(ball, players (n, 2 * POOL, 2), team (n, 2 * POOL), stats): ball bursts, then each
    team's tracks broken into its own POOL slots, so a slot's team is fixed for the file."""
    rng = np.random.default_rng(seed)
    ball, bursts = pc.degrade_ball(g["ball"], live, rng)
    n = len(ball)
    players = np.full((n, 2 * POOL, 2), np.nan, dtype=np.float32)
    team = np.full((n, 2 * POOL), -1, dtype=np.int8)
    pieces = dropped = 0
    for t in (0, 1):
        cols = np.flatnonzero((g["team"] == t).any(axis=0))   # slots that ever hold team t
        mine = g["team"][:, cols] == t
        p = np.where(mine[..., None], g["players"][:, cols], np.nan).astype(np.float32)
        out_p, out_t, k, d = pc.fragment_tracks(p, np.where(mine, t, -1).astype(np.int8), rng,
                                                pool=POOL)
        players[:, t * POOL:(t + 1) * POOL] = out_p[:, :POOL]
        team[:, t * POOL:(t + 1) * POOL] = t
        pieces, dropped = pieces + k, dropped + d
    return ball, players, team, {"bursts": bursts, "pieces": pieces, "pieces_dropped": dropped}


def _fmt(v):
    return "NaN" if not np.isfinite(v) else "{:.5f}".format(v)


def write_tracking(path, ball, players, record, pitch):
    length, width = pitch
    with open(path, "w") as handle:
        for f in np.flatnonzero(record):
            groups = ";".join("{},{}".format(_fmt(x / length), _fmt(y / width))
                              for x, y in players[f].tolist())
            handle.write("{}:{}:{},{}\n".format(f + 1, groups, _fmt(ball[f, 0] / length),
                                                _fmt(ball[f, 1] / width)))


def write_metadata(path, names, directions, n, pitch):
    """Minimal ATD metadata: two teams, 2 * POOL players with fixed teams, one format spec."""
    side = {1: "left_to_right", -1: "right_to_left"}
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "  <Metadata>",
             "    <GlobalConfig><FrameRate>{}</FrameRate><FieldSize><Width>{:g}</Width>"
             "<Height>{:g}</Height></FieldSize></GlobalConfig>".format(config.FPS, *pitch),
             "    <Teams>"]
    for t in (0, 1):
        param = ""
        if directions:
            param = ("<ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half"
                     "</Name><Value>{}</Value></ProviderParameter></ProviderTeamsParameters>"
                     .format(side[directions[t]]))
        lines.append('      <Team id="T{}"><Name>{}</Name>{}</Team>'.format(
            t + 1, escape(names[t]), param))
    lines += ["    </Teams>", "    <Players>"]
    for s in range(2 * POOL):
        lines.append('      <Player id="P{0}" teamId="T{1}"><Name>Track {0}</Name></Player>'
                     .format(s + 1, s // POOL + 1))
    lines += ["    </Players>", "    <PlayerChannels>"]
    for s in range(2 * POOL):
        for c in "xy":
            lines.append('      <PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'
                         .format(s + 1, c))
    lines += ["    </PlayerChannels>", "  </Metadata>", "  <DataFormatSpecifications>",
              '    <DataFormatSpecification endFrame="{}" separator=":" startFrame="1">'.format(n),
              '      <StringRegister name="frameCount"/>', '      <SplitRegister separator=";">']
    for s in range(2 * POOL):
        lines.append('        <SplitRegister separator=","><PlayerChannelRef playerChannelId='
                     '"player{0}_x"/><PlayerChannelRef playerChannelId="player{0}_y"/>'
                     '</SplitRegister>'.format(s + 1))
    lines += ["      </SplitRegister>", '      <SplitRegister separator=",">',
              '        <BallChannelRef channelId="ball_x"/><BallChannelRef channelId="ball_y"/>',
              "      </SplitRegister>", "    </DataFormatSpecification>",
              "  </DataFormatSpecifications>", "</main>"]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def label_strings(labels, dead, graded):
    out = np.full(len(labels), "X", dtype="<U1")
    out[graded & dead] = "D"
    live = graded & ~dead
    out[live & (labels == pc.H)] = "H"
    out[live & (labels == pc.A)] = "A"
    return out


def build_game(root, game, out_dir, seed=config.DEGRADE_SEED):
    g = load_game(root, game)
    if g is None:
        return None
    n, blocks = len(g["ball"]), g["blocks"]
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    labels, dead = pr.reference_labels(g["events"], g["team_ids"], blocks, n)
    live = graded & ~dead
    ball, players, team, stats = degrade(g, live, seed)
    folder = os.path.join(out_dir, "sg{}".format(game))
    os.makedirs(folder, exist_ok=True)
    write_tracking(os.path.join(folder, "tracking.txt"), ball, players, g["record"], SAMPLE_PITCH)
    write_metadata(os.path.join(folder, "metadata.xml"), g["team_names"], g["directions"], n,
                   SAMPLE_PITCH)
    lab = label_strings(labels, dead, graded)
    with open(os.path.join(folder, "labels.csv"), "w") as handle:
        handle.write("frame,label\n" + "".join("{},{}\n".format(i, v) for i, v in enumerate(lab)))
    code = np.where(graded, np.where(dead, 1, 0), -1)
    with open(os.path.join(folder, "dead_ball.csv"), "w") as handle:
        handle.write("frame,dead,p_dead,llr_dead\n" +
                     "".join("{},{},,\n".format(i, v) for i, v in enumerate(code.tolist())))
    frames = ",".join("{}-{}".format(a, b) for _, a, b in blocks)
    manifest = {"game": game, "frames": frames, "pitch": list(SAMPLE_PITCH), "n_frames": n,
                "seed": seed, "degrade": stats,
                "counts": {k: int(np.sum(lab == k)) for k in ("H", "A", "D", "X")},
                "live_missing_share": float(np.mean(~np.isfinite(ball[live]).all(axis=1)))}
    with open(os.path.join(folder, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out_dir", help="folder outside the repository")
    p.add_argument("--sample-data", required=True, help="Metrica sample-data clone or folder")
    p.add_argument("--games", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seed", type=int, default=config.DEGRADE_SEED)
    args = p.parse_args(argv)
    from possession.run import check_out_dir
    out_dir = check_out_dir(args.out_dir)
    built = 0
    for game in args.games:
        m = build_game(args.sample_data, game, out_dir, args.seed)
        if m is None:
            print("sample game {}: files not found, skipped".format(game))
            continue
        built += 1
        c = m["counts"]
        print("sample game {}: frames {}; H {H}, A {A}, D {D}, X {X}; live ball missing {:.2f}; "
              "{} ball bursts, {} track pieces ({} dropped)".format(
                  game, m["frames"], m["live_missing_share"], m["degrade"]["bursts"],
                  m["degrade"]["pieces"], m["degrade"]["pieces_dropped"], **c))
    print("wrote {} reference game(s) to {}".format(built, out_dir))


if __name__ == "__main__":
    main()
