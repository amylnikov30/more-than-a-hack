"""Label every live frame H or A, on top of the dead-ball labeller's D / X mask.

    PYTHONPATH="$PWD:$PWD/src" python3 -m possession.run OUT_DIR --dead DEAD_BALL_CSV
    ... --raw TRACKING.txt --meta METADATA.xml --frames 0-69660,69661-143760 --pitch 105 68

Without --raw it reads the competition match through games.asset_paths() (MTAH_DATA_DIR). With
--raw it runs on any game in the same raw format, e.g. a reference game from possession.reference.
Writes OUT_DIR/possession.csv (`frame,label`: D where dead == 1, X where dead == -1, H / A on
every live frame; one row per row of the dead file, 199,319 on the match). OUT_DIR must be
outside the repository: the CSV is frame-indexed. stdout: aggregates only.

Order (docs/possession_HA_proposal.md §4): live spells from the mask -> features -> clipped
log-odds -> restart prior -> per-spell decoder -> optional switch snapping (config.SNAP).
"""

import argparse
import os
import sys

import numpy as np

import ball_gaps as bg
import poss_common as pc

from possession import config, decode, evidence, features, restart, snap, spells

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_track(raw, meta, n, pitch):
    trk = pc.load_tracking(raw, meta, n, tuple(pitch))
    return {"ball": trk["ball"], "players": trk["players"], "team": trk["team"]}


def label(track, dead, blocks, meta, pitch, model, tag_list=(), do_snap=config.SNAP,
          fps=config.FPS, X=None, info=None, tag_stats=None):
    """Everything run.py computes, as a dict (team, labels, llr, priors, counts, ...).

    X / info: precomputed features.build output (train.py reuses them across settings)."""
    live = spells.live_mask(dead, blocks)
    sp = spells.live_spells(dead, blocks)
    if X is None:
        X, _, info = features.build(track, blocks, live, meta, pitch, fps)
    llr = evidence.log_odds(model, X)
    llr[~live] = 0.0
    priors, prior_counts, tag_offsets = restart.priors(sp, info["signals"], tag_list, blocks,
                                                       fps)
    m = int(round(model["min_seg_s"] * fps))
    team = decode.decode_all(llr, sp, model["lambda"], m, model["beta"], priors,
                             int(round(config.MIN_SPELL_S * fps)))
    snap_counts = None
    if do_snap:
        control = snap.control_frames(info["signals"], live)
        team, snap_counts = snap.snap(team, sp, control, fps)
    return {"team": team, "labels": spells.labels_from(dead, team), "llr": llr, "live": live,
            "spells": sp, "priors": priors, "prior_counts": prior_counts,
            "tag_offsets": tag_offsets, "tag_stats": tag_stats,
            "snap_counts": snap_counts, "info": info}


def summarise(res, dead, fps=config.FPS):
    live, team, sp = res["live"], res["team"], res["spells"]
    n_live = int(live.sum())
    short = sum(1 for _, a, b in sp if b - a < config.MIN_SPELL_S * fps)
    print("frames: {} graded, {} live, {} dead; {} live spells ({} shorter than {:g} s)".format(
        int(np.sum(dead >= 0)), n_live, int(np.sum(dead == spells.DEAD)), len(sp), short,
        config.MIN_SPELL_S))
    info = res["info"]
    print("attacking direction: from {}; centroid inference agrees on {} of (team, half)".format(
        info["dir_source"], "n/a" if info["dir_agree"] != info["dir_agree"]
        else "{:.2f}".format(info["dir_agree"])))
    ts = res["tag_stats"]
    if ts is not None:
        print("set-piece tags: {} in the file, {} with a mapped team; team map {}; unmapped "
              "names {}".format(ts["set_pieces"], ts["mapped"], ts["team_map"] or "none",
                                ts["unmapped"] or "none"))
    print("restart prior: {tag} from tags, {fast_ball} from the first fast ball, {none} none"
          .format(**res["prior_counts"]))
    if res["tag_offsets"]:
        print("linked tag start -> spell start, s: p10/p50/p90 {}, max {:.1f}".format(
            np.percentile(res["tag_offsets"], [10, 50, 90]).round(1).tolist(),
            max(res["tag_offsets"])))
    llr = res["llr"][live]
    print("evidence: mean |l| {:.2f}, at the clip {:.3f}, l > 0 on {:.3f} of live frames".format(
        float(np.mean(np.abs(llr))) if n_live else 0.0,
        float(np.mean(np.abs(llr) >= config.CLIP - 1e-9)) if n_live else 0.0,
        float(np.mean(llr > 0)) if n_live else 0.0))
    poss = []
    for _, a, b in sp:
        seg = team[a:b]
        edges = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        poss += np.diff(edges).tolist()
    lengths = np.array(poss, dtype=float) / fps
    minutes = n_live / fps / 60.0
    print("possessions: {} ({:.2f} per live minute), length s p10/p50/p90 {}; H share of live "
          "frames {:.3f}; switches {}".format(
              len(poss), len(poss) / minutes if minutes else 0.0,
              np.percentile(lengths, [10, 50, 90]).round(1).tolist() if len(poss) else [],
              float(np.mean(team[live] > 0)) if n_live else 0.0, len(poss) - len(sp)))
    if res["snap_counts"] is not None:
        print("snapping: {moved} switches moved, {kept} kept ({frames_moved} frames moved)"
              .format(**res["snap_counts"]))
    else:
        print("snapping: off")


def check_out_dir(out_dir):
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.commonpath([out_dir, REPO]) == REPO:
        sys.exit("OUT_DIR must be outside the repository: the CSV is frame-indexed.")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out_dir", help="folder outside the repository")
    p.add_argument("--dead", required=True, help="dead_ball.csv from `python -m deadball.run`")
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD-format tracking; default games.asset_paths()")
    p.add_argument("--meta", help="ATD metadata XML (required with --raw)")
    p.add_argument("--tags", help="Smart Tagging XML; default the match's, none with --raw")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half (required with --raw)")
    p.add_argument("--pitch", type=float, nargs=2, metavar=("LENGTH", "WIDTH"),
                   help="metres; default games.GAMES[GAME]['pitch'] (required with --raw)")
    p.add_argument("--tag-teams", help="'HOME,AWAY' Smart Tagging team names, when the "
                   "automatic match to the metadata team names fails")
    p.add_argument("--model", default=evidence.MODEL_PATH, help="default possession/ha_model.json")
    p.add_argument("--snap", choices=("config", "on", "off"), default="config",
                   help="switch snapping; default config.SNAP ({})".format(config.SNAP))
    args = p.parse_args(argv)
    out_dir = check_out_dir(args.out_dir)

    kickoffs = {}
    if args.raw:
        if not (args.meta and args.frames and args.pitch):
            p.error("--meta, --frames and --pitch are required with --raw")
        raw, meta, tags, blocks, pitch = args.raw, args.meta, args.tags, args.frames, args.pitch
        n = spells.count_rows(args.dead)
    else:
        import games
        raw, meta = games.require(["atd_raw", "atd_meta"], args.game)
        tags = args.tags or games.asset_paths(args.game)["smart_tagging"]
        tags = tags if os.path.isfile(tags) else None
        blocks = games.half_video_blocks(args.game)
        pitch = args.pitch or games.GAMES[args.game]["pitch"]
        n = games.source_frames("tactical", args.game)
        kickoffs = {2: games.anchor_video_frame(games.GAMES[args.game]["kickoff_2nd"])}
        if spells.count_rows(args.dead) != n:
            sys.exit("{} has {} rows, expected {}".format(args.dead, spells.count_rows(args.dead), n))

    dead = spells.apply_kickoffs(spells.read_dead(args.dead, n), blocks, kickoffs)
    model = evidence.load(args.model)
    tag_list, tag_stats = restart.tag_restarts(tags, [t[1] for t in pc.read_teams(meta)],
                                               args.tag_teams)
    track = load_track(raw, meta, n, pitch)
    do_snap = config.SNAP if args.snap == "config" else args.snap == "on"
    res = label(track, dead, blocks, meta, pitch, model, tag_list, do_snap,
                tag_stats=tag_stats if tags else None)
    summarise(res, dead)
    path = os.path.join(out_dir, "possession.csv")
    spells.write_labels(path, res["labels"])
    print("wrote", path, "({} rows)".format(n))
    return res


if __name__ == "__main__":
    main()
