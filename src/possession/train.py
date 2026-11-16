"""Fit the evidence model and tune the decoder on the reference set (possession.reference).

    PYTHONPATH="$PWD:$PWD/src" python3 -m possession.train REFERENCE_DIR [--write]

Held-out by game (leave one game out) when the reference set has two or more games, by half
when it has one. On each fold: fit the logistic model on the other units' labelled live frames,
apply it to the held-out unit, decode with every (lambda, m, beta) of the config grids, and
score with the official scorer's spellF1 (possession_metrics.evaluate from the public starter
repo, $MTAH_PUBLIC_REPO). The setting with the best mean held-out spellF1 wins; snapping is
then scored on and off at that setting. The live mask is the reference (event-derived) one.

--write refits on every unit and writes possession/ha_model.json. Prints aggregates only.
Nothing here reads the competition match or its ground truth. Deterministic.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

import poss_common as pc

from possession import config, evidence, features, run, spells

PUBLIC_REPO = os.environ.get("MTAH_PUBLIC_REPO") or os.path.expanduser("~/more-than-a-hack-2027")


def scorer():
    """The official possession_metrics module, imported from the public starter repo."""
    path = os.path.join(PUBLIC_REPO, "evaluation")
    if not os.path.isfile(os.path.join(path, "possession_metrics.py")):
        sys.exit("possession_metrics.py not found under {}; set MTAH_PUBLIC_REPO".format(path))
    if path not in sys.path:
        sys.path.insert(0, path)
    import possession_metrics
    return possession_metrics


def read_labels(path):
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip("\n").split(",")[1] for line in handle])


def load_reference(ref_dir):
    """{game: dict} with features built once on the oracle live mask."""
    games = {}
    for folder in sorted(os.listdir(ref_dir)):
        manifest_path = os.path.join(ref_dir, folder, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path) as handle:
            m = json.load(handle)
        d = os.path.join(ref_dir, folder)
        blocks = [(i + 1,) + tuple(int(v) for v in r.split("-"))
                  for i, r in enumerate(m["frames"].split(","))]
        n = m["n_frames"]
        meta = os.path.join(d, "metadata.xml")
        dead = spells.read_dead(os.path.join(d, "dead_ball.csv"), n)
        live = spells.live_mask(dead, blocks)
        track = run.load_track(os.path.join(d, "tracking.txt"), meta, n, m["pitch"])
        X, names, info = features.build(track, blocks, live, meta, m["pitch"])
        games[m["game"]] = {"blocks": blocks, "n": n, "meta": meta, "dead": dead, "live": live,
                            "labels": read_labels(os.path.join(d, "labels.csv")), "X": X,
                            "names": names, "info": info, "pitch": m["pitch"], "track": track}
        print("reference game {}: {} live frames, direction from {} (centroid agreement {})"
              .format(m["game"], int(live.sum()), info["dir_source"],
                      "n/a" if info["dir_agree"] != info["dir_agree"]
                      else "{:.2f}".format(info["dir_agree"])))
    return games


def units(games):
    """[(game, half or None)]: whole games if there are two or more, else halves."""
    if len(games) >= 2:
        return [(g, None) for g in sorted(games)]
    g = next(iter(games))
    return [(g, half) for half, _, _ in games[g]["blocks"]]


def unit_mask(g, half):
    mask = np.zeros(g["n"], dtype=bool)
    for h, a, b in g["blocks"]:
        if half is None or h == half:
            mask[a:b + 1] = True
    return mask


def training_rows(games, train_units):
    xs, ys = [], []
    for game, half in train_units:
        g = games[game]
        sel = unit_mask(g, half) & g["live"] & np.isin(g["labels"], ("H", "A"))
        xs.append(g["X"][sel])
        ys.append((g["labels"][sel] == "H").astype(float))
    return np.vstack(xs), np.concatenate(ys)


def feature_aucs(games):
    """Univariate AUC of each H - A feature for 'H has the ball', pooled over labelled live
    frames. A direction-dependent feature with the wrong sign would show here (§6 check)."""
    names = next(iter(games.values()))["names"]
    out = {}
    for j, name in enumerate(names):
        score, pos = [], []
        for g in games.values():
            sel = g["live"] & np.isin(g["labels"], ("H", "A")) & np.isfinite(g["X"][:, j])
            score.append(g["X"][sel, j])
            pos.append(g["labels"][sel] == "H")
        out[name] = pc.auc(np.concatenate(score), np.concatenate(pos))
    return out


def score_unit(pm, g, half, labels):
    mask = unit_mask(g, half)
    gt = np.where(mask, g["labels"], "X")
    pred = np.where(mask, labels, "X")
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reference_dir", help="OUT_DIR of possession.reference")
    p.add_argument("--write", action="store_true", help="write possession/ha_model.json")
    args = p.parse_args(argv)
    pm = scorer()
    games = load_reference(args.reference_dir)
    if not games:
        sys.exit("no reference games under {}".format(args.reference_dir))
    us = units(games)
    grid = list(itertools.product(config.LAMBDAS, config.MIN_SEG_S, config.BETAS))
    scores = {key: [] for key in grid}
    held_auc, folds = [], []
    for unit in us:
        model = evidence.fit(*training_rows(games, [u for u in us if u != unit]))
        g = games[unit[0]]
        llr_all = evidence.log_odds(model, g["X"])
        sel = unit_mask(g, unit[1]) & g["live"] & np.isin(g["labels"], ("H", "A"))
        held_auc.append(pc.auc(llr_all[sel], g["labels"][sel] == "H"))
        fold = {"unit": unit, "model": model}
        for lam, m_s, beta in grid:
            trial = dict(model, **{"lambda": lam, "min_seg_s": m_s, "beta": beta})
            res = run.label(g["track"], g["dead"], g["blocks"], g["meta"], g["pitch"], trial,
                            do_snap=False, X=g["X"], info=g["info"])
            scores[(lam, m_s, beta)].append(score_unit(pm, g, unit[1], res["labels"])
                                            ["spell_f1_mean"])
        folds.append(fold)
    name = lambda u: "game {}".format(u[0]) + ("" if u[1] is None else " half {}".format(u[1]))  # noqa
    print("held-out frame AUC of the evidence: " + ", ".join(
        "{} {:.3f}".format(name(u), a) for u, a in zip(us, held_auc)))
    # ties go to the more conservative setting: larger lambda, then larger m, then larger beta
    ranked = sorted(grid, key=lambda k: (-np.mean(scores[k]), -k[0], -k[1], -k[2]))
    print("top settings by mean held-out spellF1 (lambda, m s, beta: per unit -> mean):")
    for key in ranked[:5]:
        print("  {:>6g} {:>4g} {:>4g}: {} -> {:.3f}".format(
            *key, " ".join("{:.3f}".format(s) for s in scores[key]), np.mean(scores[key])))
    lam, m_s, beta = ranked[0]

    snap_rows = {}
    for do_snap in (False, True):
        per = []
        for fold in folds:
            g = games[fold["unit"][0]]
            trial = dict(fold["model"], **{"lambda": lam, "min_seg_s": m_s, "beta": beta})
            res = run.label(g["track"], g["dead"], g["blocks"], g["meta"], g["pitch"], trial,
                            do_snap=do_snap, X=g["X"], info=g["info"])
            r = score_unit(pm, g, fold["unit"][1], res["labels"])
            per.append({"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
                        "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
                        "frag": r["fragmentation"], "team_acc": r["team_acc_gt_alive"]})
        snap_rows["on" if do_snap else "off"] = per
        print("snapping {:>3}: spellF1 {} (mean {:.3f}); macroF1 mean {:.3f}; spells pred/gt {}; "
              "team acc {:.3f}".format(
                  "on" if do_snap else "off", " ".join("{:.3f}".format(r["spell_f1"]) for r in per),
                  np.mean([r["spell_f1"] for r in per]), np.mean([r["macro_f1"] for r in per]),
                  " ".join("{}/{}".format(r["n_pred"], r["n_gt"]) for r in per),
                  np.mean([r["team_acc"] for r in per])))
    aucs = feature_aucs(games)
    print("feature AUC for 'H has the ball' (pooled): " + ", ".join(
        "{} {:.2f}".format(k, v) for k, v in aucs.items()))
    print("chosen: lambda {:g}, m {:g} s, beta {:g}".format(lam, m_s, beta))

    if args.write:
        model = evidence.fit(*training_rows(games, us))
        model.update({
            "trained_on": "metrica-sports/sample-data, Sample Games {} (degraded, seed {}); labels "
                          "from Metrica events (poss_reference)".format(
                              ",".join(str(g) for g in sorted(games)), config.DEGRADE_SEED),
            "features": next(iter(games.values()))["names"],
            "lambda": lam, "min_seg_s": m_s, "beta": beta, "clip": config.CLIP,
            "held_out_units": [name(u) for u in us],
            "held_out_auc": [round(a, 4) for a in held_auc],
            "held_out_spell_f1": {k: [round(r["spell_f1"], 4) for r in v]
                                  for k, v in snap_rows.items()},
            "feature_auc": {k: round(v, 4) for k, v in aucs.items()},
            "direction": {str(g): {"source": games[g]["info"]["dir_source"],
                                   "centroid_agreement": games[g]["info"]["dir_agree"]}
                          for g in sorted(games)}})
        evidence.save(model)
        print("wrote", evidence.MODEL_PATH)


if __name__ == "__main__":
    main()
