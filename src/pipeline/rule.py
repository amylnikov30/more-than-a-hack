"""The step-6 rule: the fallback H/A system (docs/possession_HA_proposal.md §6.2, §8.1).

    PYTHONPATH="$PWD:$PWD/src" python3 -m pipeline.rule OUT_DIR --dead DEAD_BALL_CSV
    ... --raw TRACKING.txt --meta METADATA.xml --frames 0-69660,69661-143760 --pitch 105 68

Same inputs and output format as possession.run (`frame,label`: D where dead == 1, X where
dead == -1, H / A on every live frame), written to OUT_DIR/rule.csv. Per live spell:

  1. Start with the restart team: possession.restart.priors (Smart Tagging set piece, else the
     team nearest the ball just before its first fast movement). Without a prior, the team
     nearest the first slow visible ball in the spell, else the first visible ball, else H.
  2. A frame is a control frame for team T when the ball is visible (or inside a short filled
     gap), slower than possession.config.FAST_BALL_MS, and a player of T is the nearest to it.
  3. Switch after K consecutive control frames of the other team, backdated to the first of
     them. Anything else (own team's control, a fast or missing ball) resets the count, so a
     brief touch or a deflection never switches.

No learned weights. K is the only constant (pipeline.config.RULE_K, chosen on the reference set
by pipeline.evaluate_chain). Deterministic. stdout: aggregates only.
"""

import argparse
import os
import sys

import numpy as np

from possession import config as pconfig
from possession import features, restart, run, spells
from pipeline import config

H, A = 1, -1


def control_sign(signals, live):
    """int8 (n,): +1 H / -1 A on control frames (slow visible ball, that team nearest), else 0."""
    speed, nearest = signals["speed"], signals["nearest"]
    with np.errstate(invalid="ignore"):
        slow = np.isfinite(speed) & (speed < pconfig.FAST_BALL_MS)
    out = np.where(nearest == 0, H, np.where(nearest == 1, A, 0)).astype(np.int8)
    out[~(slow & live)] = 0
    return out


def start_team(prior, control, nearest_sign):
    """The spell's first team: the restart prior, else the first control frame's team, else the
    first visible ball's nearest team, else H (deterministic, as the decoder's ties)."""
    if prior:
        return int(prior)
    for values in (control, nearest_sign):
        hit = np.flatnonzero(values != 0)
        if len(hit):
            return int(values[hit[0]])
    return H


def rule_spell(control, first, k):
    """int8 (T,) of +1 / -1 for one spell. control: +1 / -1 / 0 per frame; first: start team."""
    n = len(control)
    out = np.empty(n, dtype=np.int8)
    cur, run_start, run_len, done = first, 0, 0, 0
    for t in range(n):
        if control[t] == -cur:
            if run_len == 0:
                run_start = t
            run_len += 1
            if run_len >= k:
                out[done:run_start] = cur        # backdate: the new team from the run's start
                done, cur, run_len = run_start, -cur, 0
        else:
            run_len = 0
    out[done:] = cur
    return out


def label(sp, signals, live, priors, k):
    """int8 (n,) team per frame: +1 / -1 on live frames, 0 elsewhere."""
    control = control_sign(signals, live)
    nearest = signals["nearest"]
    nearest_sign = np.where(nearest == 0, H, np.where(nearest == 1, A, 0)).astype(np.int8)
    team = np.zeros(len(live), dtype=np.int8)
    for _, a, b in sp:
        first = start_team(priors.get(a, 0), control[a:b], nearest_sign[a:b])
        team[a:b] = rule_spell(control[a:b], first, k)
    return team


def run_rule(track, dead, blocks, tag_list=(), k=config.RULE_K, fps=pconfig.FPS, signals=None):
    """Everything the rule computes, as a dict shaped like possession.run.label's output, so
    possession.run.summarise prints the same aggregates."""
    live = spells.live_mask(dead, blocks)
    sp = spells.live_spells(dead, blocks)
    if signals is None:
        signals = features.ball_signals(track["ball"], track["players"], track["team"], live,
                                        blocks, fps)
    priors, prior_counts, tag_offsets = restart.priors(sp, signals, tag_list, blocks, fps)
    team = label(sp, signals, live, priors, k)
    return {"team": team, "labels": spells.labels_from(dead, team), "live": live, "spells": sp,
            "priors": priors, "prior_counts": prior_counts, "tag_offsets": tag_offsets,
            "signals": signals}


def summarise(res, dead, k, fps=pconfig.FPS):
    live, team, sp = res["live"], res["team"], res["spells"]
    n_live = int(live.sum())
    print("rule: K {} frames ({:g} s); {} live spells, {} live frames".format(
        k, k / float(fps), len(sp), n_live))
    print("restart prior: {tag} from tags, {fast_ball} from the first fast ball, {none} none"
          .format(**res["prior_counts"]))
    poss = []
    for _, a, b in sp:
        seg = team[a:b]
        edges = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        poss += np.diff(edges).tolist()
    lengths = np.array(poss, dtype=float) / fps
    minutes = n_live / fps / 60.0
    print("possessions: {} ({:.2f} per live minute), length s p10/p50/p90 {}; H share of live "
          "frames {:.3f}".format(
              len(poss), len(poss) / minutes if minutes else 0.0,
              np.percentile(lengths, [10, 50, 90]).round(1).tolist() if len(poss) else [],
              float(np.mean(team[live] > 0)) if n_live else 0.0))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out_dir", help="folder outside the repository")
    p.add_argument("--dead", required=True, help="dead_ball.csv from `python -m deadball.run`")
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD-format tracking; default games.asset_paths()")
    p.add_argument("--meta", help="ATD metadata XML (required with --raw)")
    p.add_argument("--tags", help="Smart Tagging XML; default the match's, none with --raw")
    p.add_argument("--frames", type=_parse_frames,
                   help="graded video frames, one inclusive range per half (required with --raw)")
    p.add_argument("--pitch", type=float, nargs=2, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--tag-teams", help="'HOME,AWAY' Smart Tagging team names")
    p.add_argument("--k", type=int, default=config.RULE_K,
                   help="consecutive control frames to switch; default {}".format(config.RULE_K))
    args = p.parse_args(argv)
    out_dir = run.check_out_dir(args.out_dir)
    inputs = load_inputs(args, p)
    res = run_rule(inputs["track"], inputs["dead"], inputs["blocks"], inputs["tag_list"], args.k)
    summarise(res, inputs["dead"], args.k)
    path = os.path.join(out_dir, "rule.csv")
    spells.write_labels(path, res["labels"])
    print("wrote", path, "({} rows)".format(len(res["labels"])))
    return res


def _parse_frames(text):
    import ball_gaps as bg
    return bg.parse_frames(text)


def load_inputs(args, parser):
    """The inputs possession.run.main reads, from the same arguments, the same way."""
    import poss_common as pc
    kickoffs = {}
    if args.raw:
        if not (args.meta and args.frames and args.pitch):
            parser.error("--meta, --frames and --pitch are required with --raw")
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
            sys.exit("{} has {} rows, expected {}".format(args.dead, spells.count_rows(args.dead),
                                                          n))
    dead = spells.apply_kickoffs(spells.read_dead(args.dead, n), blocks, kickoffs)
    tag_list, _ = restart.tag_restarts(tags, [t[1] for t in pc.read_teams(meta)], args.tag_teams)
    return {"track": run.load_track(raw, meta, n, pitch), "dead": dead, "blocks": blocks,
            "tag_list": tag_list, "meta": meta, "pitch": pitch}


if __name__ == "__main__":
    main()
