"""Whole-chain validation on the reference set (docs/possession_HA_proposal.md §6.1-6.3).

    MTAH_PUBLIC_REPO=/path/to/more-than-a-hack-2027 PYTHONPATH="$PWD:$PWD/src" \
        python3 -m pipeline.evaluate_chain OUT_DIR --reference REF_DIR [--write-summary]

REF_DIR is the OUT_DIR of possession.reference (degraded open sample games with event-derived
labels). Per reference game:

  1. The dead-ball labeller runs on the degraded tracking, exactly as on the match (player model,
     ball categories with EM on the game itself, sticky Viterbi, edge sharpening). Its D errors
     are part of what is scored.
  2. H / A on its live spells, three ways:
       model             possession.run.label, evidence weights fitted on the other half (or
                         game) only, with the shipped lambda / m / beta
       rule              pipeline.rule, K chosen on the other half (or game) only
       baseline_smooth50 the organisers' starter baseline, nearest player + 50-frame vote
     plus three diagnostic rows: the shipped model scored in-sample, and model / rule on the
     oracle (event-derived) live mask, which measures H / A alone.
  3. The organisers' run_evaluation.py scores every file with --bootstrap BOOTSTRAP_N.

Decision rule (§6.2): the model ships only if its spellF1 bootstrap interval lies wholly above
the rule's, in every reference game. The result is printed next to pipeline.config.SHIP, which a
human sets. --write-summary writes the aggregates (no frame data) to
src/pipeline/reference_summary.json, which make_submission prints next to the match's.

Nothing here reads the competition match or its ground truth. Deterministic: the degradation,
the fits and the bootstrap all use fixed seeds or none.
"""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

from possession import config as pconfig
from possession import evidence, features, run, spells, train
from pipeline import aggregates, config, rule

SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_summary.json")
SYSTEMS = ("model", "rule", "baseline_smooth{}".format(config.BASELINE_WINDOW))
DIAGNOSTIC = ("model_in_sample", "model_oracle_mask", "rule_oracle_mask")


# ---------------------------------------------------------------- the dead-ball labeller

def deadball_on(raw, n, blocks, pitch, sharpen=True):
    """deadball.run.label() on any raw ATD-format game (it reads only the match itself).

    blocks: [(half, first, last)] inclusive graded frames. Returns the same dict as label()."""
    from deadball import hmm, signals, train_players
    from deadball import run as dbrun
    with open(train_players.MODEL_PATH) as handle:
        model = json.load(handle)
    players, ball = signals.load_atd(raw, n)
    llr_players = train_players.log_odds_ratio(model, signals.player_features(players, pitch))
    category = signals.ball_categories(players, ball, pitch)
    db_blocks = [(a, b + 1, a) for _, a, b in blocks]
    log_a = hmm.sticky_log_transitions(model["switch_cost"])
    ball_ratio = dbrun.ball_log_ratios(llr_players, category, db_blocks, model["temper"], log_a)
    llr = llr_players + ball_ratio[category]
    dead = np.full(n, -1, dtype=np.int8)
    p_dead = np.full(n, np.nan)
    edges = {"end_extended": 0, "end_trimmed": 0, "start_moved": 0, "frames_moved": 0}
    for start, stop, live in db_blocks:
        emission = np.column_stack([-llr[live:stop] / 2, llr[live:stop] / 2]) / model["temper"]
        decoded = hmm.viterbi(emission, log_a)
        if sharpen:
            decoded = dbrun.sharpen_edges(decoded, category[live:stop], edges)
        dead[live:stop] = decoded
        p_dead[live:stop] = hmm.forward_backward(emission, log_a)[:, hmm.DEAD]
    return {"dead": dead, "p_dead": p_dead, "llr_dead": llr, "category": category,
            "ball_ratio": ball_ratio, "edges": edges}


def dead_check(chain, oracle):
    """Aggregates of the labeller's D against the reference D on graded frames."""
    from deadball import hmm
    graded = oracle >= 0
    truth, pred = oracle[graded] == 1, chain[graded] == 1
    tp, fp, fn = np.sum(truth & pred), np.sum(~truth & pred), np.sum(truth & ~pred)
    return {"dead_share_labeller": round(float(pred.mean()), 3),
            "dead_share_reference": round(float(truth.mean()), 3),
            "dead_f1": round(2.0 * tp / max(1, 2 * tp + fp + fn), 3),
            "alive_spell_f1": round(float(hmm.alive_spell_f1(~truth, ~pred)), 3),
            "dead_spells_labeller": len(hmm.spells(pred)),
            "dead_spells_reference": len(hmm.spells(truth))}


# ---------------------------------------------------------------- the organisers' baseline

def load_baseline():
    """starter/baseline.py from the public repo, imported by path (not copied)."""
    path = os.path.join(train.PUBLIC_REPO, "starter", "baseline.py")
    if not os.path.isfile(path):
        sys.exit("baseline.py not found under {}; set MTAH_PUBLIC_REPO".format(path))
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("mtah_baseline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


def nearest_codes(track, home=0, away=1, nobody=-1):
    """The baseline's per-frame code (team of the player nearest the raw ball) on a loaded track.

    The organisers' parser expects 'x,y,speed' groups; the reference tracking has 'x,y', so the
    same quantity is computed here from the loaded arrays and smoothed by their smooth()."""
    ball, players, team = track["ball"], track["players"], track["team"]
    d2 = np.sum((players - ball[:, None, :]) ** 2, axis=2)
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    idx = np.argmin(d2, axis=1)
    rows = np.arange(len(ball))
    ok = np.isfinite(d2[rows, idx])
    code = team[rows, idx]
    return np.where(ok & (code == 0), home, np.where(ok & (code == 1), away, nobody)).astype(np.int8)


# ---------------------------------------------------------------- systems on one game

def heldout_model(games, us, g, game, track, dead, X, info, shipped):
    """Labels where each unit's frames come from a model fitted on the other units only."""
    out = np.full(g["n"], "X", dtype="<U1")
    for unit in [u for u in us if u[0] == game]:
        model = evidence.fit(*train.training_rows(games, [u for u in us if u != unit]))
        model.update({k: shipped[k] for k in ("lambda", "min_seg_s", "beta")})
        res = run.label(track, dead, g["blocks"], g["meta"], g["pitch"], model,
                        do_snap=pconfig.SNAP, X=X, info=info)
        mask = train.unit_mask(g, unit[1])
        out[mask] = res["labels"][mask]
    return out


def rule_by_k(g, dead, signals):
    return {k: rule.run_rule(None, dead, g["blocks"], k=k, signals=signals)["labels"]
            for k in config.RULE_KS}


def heldout_rule(pm, games, us, game, by_k):
    """(labels, {unit: K}): each unit decoded with the K that scores best on the other units.
    Also returns the K that is best over all units (the one to ship). Ties: the larger K."""
    g = games[game]
    unit_scores = {k: {u: train.score_unit(pm, games[u[0]], u[1], by_k[k])["spell_f1_mean"]
                       for u in us if u[0] == game} for k in by_k}

    def best(units):
        return max(sorted(by_k), key=lambda k: (np.mean([unit_scores[k][u] for u in units]), k))

    out = np.full(g["n"], "X", dtype="<U1")
    chosen = {}
    mine = [u for u in us if u[0] == game]
    for unit in mine:
        others = [u for u in mine if u != unit] or mine
        k = best(others)
        chosen[unit] = k
        mask = train.unit_mask(g, unit[1])
        out[mask] = by_k[k][mask]
    return out, chosen, best(mine), unit_scores


def baseline_labels(track, n):
    bl = load_baseline()
    return bl.smooth(nearest_codes(track, bl.HOME, bl.AWAY, bl.NOBODY), config.BASELINE_WINDOW)


# ---------------------------------------------------------------- scoring

def score_files(results_dir, gt_path, out_dir, bootstrap_n):
    """run_evaluation.py on every CSV in results_dir; returns {name: scoreboard row}."""
    script = os.path.join(train.PUBLIC_REPO, "evaluation", "run_evaluation.py")
    if not os.path.isfile(script):
        sys.exit("run_evaluation.py not found under {}; set MTAH_PUBLIC_REPO".format(script))
    cmd = [sys.executable, script, "--gt", gt_path, "--results", results_dir, "--out", out_dir,
           "--bootstrap", str(bootstrap_n), "--tau", "12", "25", "--no-detail"]
    done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    print(done.stdout.rstrip())
    if done.returncode not in (0, 1):
        sys.exit("run_evaluation.py failed ({})".format(done.returncode))
    with open(os.path.join(out_dir, "scoreboard.csv")) as handle:
        return {row["submission"]: row for row in csv.DictReader(handle)}


def _f(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def score_row(row):
    keys = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1",
            "trans_precision_12", "trans_recall_12", "trans_precision_25", "trans_recall_25",
            "fragmentation", "team_acc_gt_alive", "alive_dead_f1")
    out = {k: round(_f(row, k), 4) for k in keys}
    out["n_pred_spells"] = int(_f(row, "poss"))
    out["n_gt_spells"] = int(_f(row, "n_gt_spells"))
    return out


def print_scores(scores):
    print("\n{:<20} {:>8} {:>17} {:>8} {:>11} {:>11} {:>6} {:>8} {:>9}".format(
        "system", "spellF1", "90% interval", "macroF1", "trans P/R12", "trans P/R25", "frag",
        "team acc", "poss p/gt"))
    for name, s in scores.items():
        print("{:<20} {:>8.3f} {:>17} {:>8.3f} {:>11} {:>11} {:>6.2f} {:>8.3f} {:>9}".format(
            name, s["spell_f1_mean"],
            "[{:.3f}, {:.3f}]".format(s["spell_bootstrap_low"], s["spell_bootstrap_high"]),
            s["macro_f1"],
            "{:.2f}/{:.2f}".format(s["trans_precision_12"], s["trans_recall_12"]),
            "{:.2f}/{:.2f}".format(s["trans_precision_25"], s["trans_recall_25"]),
            s["fragmentation"], s["team_acc_gt_alive"],
            "{}/{}".format(s["n_pred_spells"], s["n_gt_spells"])))


def write_labels(path, labels):
    spells.write_labels(path, labels)


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out_dir", help="folder outside the repository")
    p.add_argument("--reference", required=True, help="OUT_DIR of possession.reference")
    p.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_N)
    p.add_argument("--write-summary", action="store_true",
                   help="write src/pipeline/reference_summary.json (aggregates only)")
    args = p.parse_args(argv)
    out_dir = run.check_out_dir(args.out_dir)
    pm = train.scorer()
    shipped = evidence.load()

    games = train.load_reference(args.reference)
    if not games:
        sys.exit("no reference games under {}".format(args.reference))
    us = train.units(games)
    summary = {"reference_games": sorted(games), "bootstrap_n": args.bootstrap,
               "held_out_by": "game" if len(games) >= 2 else "half",
               "shipped_model": {k: shipped[k] for k in ("lambda", "min_seg_s", "beta")},
               "notes": ["The H/A evidence weights in ha_model.json and the dead-ball player "
                         "model (Sample Games 1-3) were fitted on this game; the 'model' and "
                         "'rule' rows hold the H/A part out by {}, the dead-ball labeller is "
                         "in-sample.".format("game" if len(games) >= 2 else "half")],
               "games": {}}
    wins = []
    for game in sorted(games):
        g = games[game]
        folder = os.path.join(args.reference, "sg{}".format(game))
        print("\n== reference game {} ==".format(game))
        db = deadball_on(os.path.join(folder, "tracking.txt"), g["n"], g["blocks"],
                         tuple(g["pitch"]))
        dead = db["dead"]
        check = dead_check(dead, g["dead"])
        print("dead-ball labeller on the degraded game: dead share {dead_share_labeller} "
              "(reference {dead_share_reference}); dead F1 {dead_f1}; ball-in-play spell F1 "
              "{alive_spell_f1}; dead spells {dead_spells_labeller} vs {dead_spells_reference}"
              .format(**check))
        game_dir = os.path.join(out_dir, "sg{}".format(game))
        results = os.path.join(game_dir, "results")
        os.makedirs(results, exist_ok=True)
        from deadball import run as dbrun
        dbrun.write_csv(db, game_dir)

        live = spells.live_mask(dead, g["blocks"])
        X, _, info = features.build(g["track"], g["blocks"], live, g["meta"], g["pitch"])
        labels = {}
        labels["model"] = heldout_model(games, us, g, game, g["track"], dead, X, info, shipped)
        by_k = rule_by_k(g, dead, info["signals"])
        labels["rule"], k_used, k_ship, k_scores = heldout_rule(pm, games, us, game, by_k)
        labels[SYSTEMS[2]] = baseline_labels(g["track"], g["n"])
        labels["model_in_sample"] = run.label(g["track"], dead, g["blocks"], g["meta"],
                                              g["pitch"], shipped, X=X, info=info)["labels"]
        labels["model_oracle_mask"] = heldout_model(games, us, g, game, g["track"], g["dead"],
                                                    g["X"], g["info"], shipped)
        oracle_k = rule_by_k(g, g["dead"], g["info"]["signals"])
        labels["rule_oracle_mask"] = heldout_rule(pm, games, us, game, oracle_k)[0]
        for name, lab in labels.items():
            write_labels(os.path.join(results, name + ".csv"), lab)

        print("rule K by held-out unit: {}; K over all units (to ship): {}; spellF1 by K: {}"
              .format(", ".join("half {} -> {}".format(u[1], k) if u[1] else
                                "game {} -> {}".format(u[0], k) for u, k in k_used.items()),
                      k_ship, ", ".join("{} {}".format(k, " ".join("{:.3f}".format(v)
                                                                    for v in s.values()))
                                        for k, s in k_scores.items())))
        rows = score_files(results, os.path.join(folder, "labels.csv"),
                           os.path.join(game_dir, "scored"), args.bootstrap)
        scores = {name: score_row(rows[name]) for name in SYSTEMS + DIAGNOSTIC}
        print_scores(scores)

        m, r = scores["model"], scores["rule"]
        model_wins = m["spell_bootstrap_low"] > r["spell_bootstrap_high"]
        wins.append(model_wins)
        print("\nmodel {:.3f} [{:.3f}, {:.3f}] vs rule {:.3f} [{:.3f}, {:.3f}]: {}".format(
            m["spell_f1_mean"], m["spell_bootstrap_low"], m["spell_bootstrap_high"],
            r["spell_f1_mean"], r["spell_bootstrap_low"], r["spell_bootstrap_high"],
            "the model is above the rule beyond the intervals" if model_wins else
            "the intervals overlap or the rule is ahead"))

        aggs = {"reference labels": aggregates.summary(np.asarray(train.read_labels(
            os.path.join(folder, "labels.csv"))))}
        for name in SYSTEMS:
            aggs[name] = aggregates.summary(labels[name])
        print()
        aggregates.print_table(list(aggs.items()))
        summary["games"][str(game)] = {"dead_ball": check, "scores": scores,
                                       "rule_k": {"held_out": {str(u[1] or u[0]): k
                                                               for u, k in k_used.items()},
                                                  "to_ship": k_ship},
                                       "aggregates": aggs}

    decision = "model" if all(wins) else "rule"
    summary["decision"] = decision
    summary["rule_k_to_ship"] = summary["games"][str(sorted(games)[0])]["rule_k"]["to_ship"]
    print("\ndecision rule: ship the {} (pipeline.config.SHIP is \"{}\"{})".format(
        decision, config.SHIP, "" if decision == config.SHIP else " -- update it"))
    if summary["rule_k_to_ship"] != config.RULE_K:
        print("rule K to ship on the reference set: {} (pipeline.config.RULE_K is {} -- update "
              "it)".format(summary["rule_k_to_ship"], config.RULE_K))
    if args.write_summary:
        with open(SUMMARY_PATH, "w") as handle:
            json.dump(summary, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print("wrote", SUMMARY_PATH)
    return summary


if __name__ == "__main__":
    main()
