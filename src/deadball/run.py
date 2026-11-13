"""Label dead ball on the competition match.

    cd src && PYTHONPATH=.. python -m deadball.run ~/eda_out/dead_ball
    ... --check ../path/to/BAR-ATH_possession_gt_public.csv     # pipeline check, not for tuning

Writes OUT_DIR/dead_ball.csv (one row per video frame; frame-indexed, so OUT_DIR must be outside
the repository) and prints aggregates only. Method, in the order it runs:

  1. Player evidence. The model fitted on the open sample games (train_players.py) turns how
     the 22 players are moving into a log-odds that the ball is dead. No ball needed.
  2. Ball evidence. Each frame's ball track is one of 16 categories (signals.BALL_CATEGORIES).
     How likely each visible category is in dead vs live play is learned from this match with no
     labels, by EM on a two-state model anchored by the player evidence. A missing ball carries
     no evidence either way: gaps happen in live play too, so during a gap the players decide.
  3. Decode. Sticky Viterbi per half; tempering and switch cost come from the sample games.
  4. Sharpen. Dead-spell edges move to the ball reappearing (end) or leaving the pitch (start)
     when one of those happens within SNAP_S of the decoded edge.
  5. The second half's graded block starts before its kickoff; everything before the kickoff
     anchor in games.py is dead.
"""

import argparse
import json
import os
import sys

import numpy as np

from deadball import hmm, signals, train_players
import games

EM_STEPS = 25
SNAP_S = 3.0  # how far a dead-spell edge may move to meet a ball event; fixed, not searched
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def graded_blocks(game):
    """[(start, stop, first_live_frame)] half-open graded spans, with the kickoff if it is later."""
    kickoff_2nd = games.anchor_video_frame(games.GAMES[game]["kickoff_2nd"])
    blocks = []
    for half, first, last in games.half_video_blocks(game):
        blocks.append((first, last + 1, max(first, kickoff_2nd) if half == 2 else first))
    return blocks


def ball_log_ratios(llr, category, blocks, temper, log_a):
    """EM for P(category | state) on live spans; returns log P(c | dead) - log P(c | alive).

    Only the visible categories are learned, as a distribution over visible frames. MISSING is
    fixed at zero evidence, which keeps the model from deciding that "no ball" means "dead" and
    then confirming it to itself.
    """
    n_cat = len(signals.BALL_CATEGORIES)
    log_ratio = np.zeros(n_cat)
    for _ in range(EM_STEPS):
        counts = np.ones((2, n_cat))  # one pseudo-frame per category
        for _, stop, live in blocks:
            total = llr[live:stop] + log_ratio[category[live:stop]]
            posterior = hmm.forward_backward(np.column_stack([-total / 2, total / 2]) / temper, log_a)
            for state in (hmm.ALIVE, hmm.DEAD):
                counts[state] += np.bincount(category[live:stop], weights=posterior[:, state],
                                             minlength=n_cat)
        visible = counts[:, 1:] / counts[:, 1:].sum(axis=1, keepdims=True)
        log_ratio[1:] = np.log(visible[hmm.DEAD] / visible[hmm.ALIVE])
    return log_ratio


def sharpen_edges(dead, category, counts):
    """Move dead-spell edges to the ball events that mark them. Returns a new 0/1 array.

    The decoder places edges from evidence averaged over seconds, so it is right about which
    stoppages happened and vague about when. Two ball events are sharp:

      end    The tracker picks the ball up again when it is put back into play. If the ball
             reappears within SNAP_S of the decoded end, on either side, the spell ends there.
      start  The ball crossing a line stops play at once, before anyone slows down. If the ball
             goes outside the pitch within SNAP_S before the decoded start and is not seen back
             inside in between, the spell starts at the first outside frame.

    Spells are processed in time order and never moved past a neighbouring spell.
    """
    out = dead.copy()
    missing = category == signals.MISSING
    outside = category == signals.OUTSIDE
    snap = int(SNAP_S * signals.FPS)
    n = len(dead)
    spells = hmm.spells(dead == hmm.DEAD)
    for index, (start, stop) in enumerate(spells):
        next_start = spells[index + 1][0] if index + 1 < len(spells) else n
        prev_stop = spells[index - 1][1] if index else 0

        new_stop = stop
        if stop < n and missing[stop]:
            # still missing at the decoded end: wait for the ball, but not for long
            ahead = np.flatnonzero(~missing[stop:min(n, stop + snap + 1, next_start)])
            if len(ahead):
                new_stop = stop + int(ahead[0])
                counts["end_extended"] += 1
        else:
            # the ball came back shortly before the decoded end and stayed
            window = missing[max(start, stop - snap - 1):stop]
            gone = np.flatnonzero(window)
            if len(gone) and not window[-1]:
                new_stop = max(start, stop - snap - 1) + int(gone[-1]) + 1
                counts["end_trimmed"] += 1

        new_start = start
        low = max(prev_stop, start - snap)
        before = category[low:start]
        went_out = np.flatnonzero(outside[low:start])
        if len(went_out):
            first = int(went_out[0])
            back_inside = (before[first:] != signals.OUTSIDE) & (before[first:] != signals.MISSING)
            if not back_inside.any():
                new_start = low + first
                counts["start_moved"] += 1

        counts["frames_moved"] += abs(new_stop - stop) + abs(new_start - start)
        out[start:stop] = hmm.ALIVE
        out[new_start:new_stop] = hmm.DEAD
    return out


def label(game=games.GAME, sharpen=True):
    with open(train_players.MODEL_PATH) as handle:
        model = json.load(handle)
    raw_path = games.require(["atd_raw"], game)[0]
    n_frames = games.source_frames("tactical", game)
    pitch = games.GAMES[game]["pitch"]
    players, ball = signals.load_atd(raw_path, n_frames)

    llr_players = train_players.log_odds_ratio(model, signals.player_features(players, pitch))
    category = signals.ball_categories(players, ball, pitch)
    blocks = graded_blocks(game)
    log_a = hmm.sticky_log_transitions(model["switch_cost"])
    ball_ratio = ball_log_ratios(llr_players, category, blocks, model["temper"], log_a)

    llr = llr_players + ball_ratio[category]
    dead = np.full(n_frames, -1, dtype=np.int8)
    p_dead = np.full(n_frames, np.nan)
    edges = {"end_extended": 0, "end_trimmed": 0, "start_moved": 0, "frames_moved": 0}
    for start, stop, live in blocks:
        emission = np.column_stack([-llr[live:stop] / 2, llr[live:stop] / 2]) / model["temper"]
        decoded = hmm.viterbi(emission, log_a)
        if sharpen:
            decoded = sharpen_edges(decoded, category[live:stop], edges)
        dead[live:stop] = decoded
        p_dead[live:stop] = hmm.forward_backward(emission, log_a)[:, hmm.DEAD]
        dead[start:live], p_dead[start:live] = hmm.DEAD, 1.0
    return {"dead": dead, "p_dead": p_dead, "llr_dead": llr, "llr_players": llr_players,
            "category": category, "ball_ratio": ball_ratio, "blocks": blocks, "edges": edges}


def summarise(result):
    dead, category = result["dead"], result["category"]
    graded = dead >= 0
    print("ball category                          frames   log P(c|dead)/P(c|alive)")
    for index, name in enumerate(signals.BALL_CATEGORIES):
        print("  {:<36} {:>7}   {:+.2f}{}".format(
            name, int(np.sum(graded & (category == index))), result["ball_ratio"][index],
            "  (fixed)" if index == signals.MISSING else ""))
    for half, (start, stop, _) in enumerate(result["blocks"], 1):
        lengths = np.array([b - a for a, b in hmm.spells(dead[start:stop] == hmm.DEAD)]) / signals.FPS
        print("half {}: dead share {:.3f}, {} dead spells, length s p10/p50/p90 {}".format(
            half, np.mean(dead[start:stop] == hmm.DEAD), len(lengths),
            np.percentile(lengths, [10, 50, 90]).round(1).tolist()))
    if "edges" in result:
        print("edges sharpened: {end_extended} ends extended to the ball reappearing, {end_trimmed} "
              "trimmed back to it, {start_moved} starts moved to the ball going out "
              "({frames_moved} frames moved in total)".format(**result["edges"]))
    missing = graded & (category == signals.MISSING)
    print("ball-missing frames called dead: {:.3f}; ball-visible frames called dead: {:.3f}".format(
        np.mean(dead[missing] == hmm.DEAD), np.mean(dead[graded & ~missing] == hmm.DEAD)))


def check(result, gt_path):
    """Alive/dead agreement on the public window. A pipeline check: never choose settings on it."""
    with open(gt_path) as handle:
        labels = np.array([line.strip().split(",")[1] for line in handle][1:])
    scored = labels != "X"
    truth, pred = labels[scored] == "D", result["dead"][scored] == hmm.DEAD
    tp, fp, fn = np.sum(truth & pred), np.sum(~truth & pred), np.sum(truth & ~pred)
    print("public window ({} frames): dead F1 {:.3f}, alive-spell F1 {:.3f}, dead spells {} vs {}".format(
        int(scored.sum()), 2.0 * tp / (2 * tp + fp + fn), hmm.alive_spell_f1(~truth, ~pred),
        len(hmm.spells(pred)), len(hmm.spells(truth))))


def write_csv(result, out_dir):
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.commonpath([out_dir, REPO]) == REPO:
        sys.exit("OUT_DIR must be outside the repository: the CSV is frame-indexed match data.")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dead_ball.csv")
    with open(path, "w") as handle:
        handle.write("frame,dead,p_dead,llr_dead\n")
        for frame, (dead, p_dead, llr) in enumerate(zip(result["dead"], result["p_dead"],
                                                         result["llr_dead"])):
            handle.write("{},{},{},{}\n".format(
                frame, dead, "" if np.isnan(p_dead) else "{:.4f}".format(p_dead),
                "" if dead < 0 else "{:.3f}".format(llr)))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", help="folder outside the repository")
    parser.add_argument("--check", metavar="PUBLIC_GT_CSV", help="print agreement on the public window")
    parser.add_argument("--no-sharpen", action="store_true", help="keep the decoder's own edges")
    args = parser.parse_args()
    result = label(sharpen=not args.no_sharpen)
    summarise(result)
    if args.check:
        check(result, args.check)
    print("wrote", write_csv(result, args.out_dir))
