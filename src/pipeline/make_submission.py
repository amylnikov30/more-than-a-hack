"""Write the scored file: dead-ball labeller -> chosen H / A system -> OUT_DIR/TEAM_NAME.csv.

    PYTHONPATH="$PWD:$PWD/src" python3 -m pipeline.make_submission OUT_DIR --team TEAM_NAME

Reads the competition match through games.asset_paths() (MTAH_DATA_DIR), so it runs where the
data is (the VM). Steps:

  1. deadball.run: OUT_DIR/dead_ball.csv, with its aggregates and the tagged-restart check.
  2. The system in pipeline.config.SHIP: possession.run (model) or pipeline.rule (rule), on
     dead_ball.csv's live spells.
  3. OUT_DIR/TEAM_NAME.csv, then the checks of SUBMISSION.md: header, 199,319 rows, frames 0..N-1
     in order, labels in H / A / D / X, X exactly on the ungraded frames, D on the second half's
     graded frames before its kickoff. Then the organisers' validate_submission.py, when
     $MTAH_PUBLIC_REPO has it.
  4. Label-free aggregates of the file (§6.4) next to the reference-set values in
     reference_summary.json, and the file's sha256.

OUT_DIR must be outside the repository. stdout: aggregates only, safe to share.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from pipeline import aggregates, config

LABELS = ("H", "A", "D", "X")


def check_file(path, n, blocks, kickoffs):
    """[failure strings] for a frame,label file; empty when it meets the contract.

    blocks: [(half, first, last)] graded video frames; kickoffs: {half: first live frame}."""
    fails = []
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        lines = raw.decode("utf-8").split("\n")
    except UnicodeDecodeError:
        return ["not UTF-8"]
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0] != "frame,label":
        fails.append("header is not exactly 'frame,label'")
        return fails
    rows = lines[1:]
    if len(rows) != n:
        fails.append("{} data rows, expected {}".format(len(rows), n))
        return fails
    labels = []
    for i, line in enumerate(rows):
        parts = line.split(",")
        if len(parts) != 2 or parts[0] != str(i):
            fails.append("row {} is {!r}, expected '{},<label>'".format(i + 1, line[:40], i))
            return fails
        labels.append(parts[1])
    labels = np.array(labels)
    bad = sorted(set(labels) - set(LABELS))
    if bad:
        fails.append("labels outside H/A/D/X: {}".format(bad[:5]))
        return fails
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    if np.any(labels[~graded] != "X"):
        fails.append("{} ungraded frames are not X".format(int(np.sum(labels[~graded] != "X"))))
    if np.any(labels[graded] == "X"):
        fails.append("{} graded frames are X".format(int(np.sum(labels[graded] == "X"))))
    for half, first, _ in blocks:
        kick = kickoffs.get(half)
        if kick is not None and kick > first and np.any(labels[first:kick] != "D"):
            fails.append("half {}: {} frames before the kickoff are not D".format(
                half, int(np.sum(labels[first:kick] != "D"))))
    return fails


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(path):
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip("\n").split(",")[1] for line in handle])


def validate_official(path):
    """Run the organisers' validate_submission.py; None when it is not available."""
    repo = os.environ.get("MTAH_PUBLIC_REPO") or os.path.expanduser("~/more-than-a-hack-2027")
    script = os.path.join(repo, "evaluation", "validate_submission.py")
    if not os.path.isfile(script):
        print("validate_submission.py not found under {}: official check skipped (set "
              "MTAH_PUBLIC_REPO)".format(repo))
        return None
    done = subprocess.run([sys.executable, script, path], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)
    print(done.stdout.rstrip())
    return done.returncode == 0


def reference_columns(path):
    """[(title, aggregates)] from reference_summary.json, or [] if it is missing."""
    if not os.path.isfile(path):
        return []
    with open(path) as handle:
        summary = json.load(handle)
    out = []
    for game, g in sorted(summary.get("games", {}).items()):
        aggs = g.get("aggregates", {})
        if "reference labels" in aggs:
            out.append(("SG{} truth".format(game), aggs["reference labels"]))
        if config.SHIP in aggs:
            out.append(("SG{} {}".format(game, config.SHIP), aggs[config.SHIP]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out_dir", help="folder outside the repository")
    p.add_argument("--team", required=True, help="team name: the file is OUT_DIR/TEAM.csv")
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--system", choices=("config", "model", "rule"), default="config",
                   help="default pipeline.config.SHIP ({})".format(config.SHIP))
    args = p.parse_args(argv)
    if not args.team or os.sep in args.team or args.team.startswith("."):
        p.error("--team must be a plain file name")
    system = config.SHIP if args.system == "config" else args.system

    import games
    from deadball import run as dbrun
    from possession import run as prun
    from pipeline import evaluate_chain, rule

    out_dir = prun.check_out_dir(args.out_dir)
    print("== 1. dead ball ==")
    result = dbrun.label(args.game)
    dbrun.summarise(result)
    dbrun.tag_check(result, args.game)
    dead_path = dbrun.write_csv(result, out_dir)
    print("wrote", dead_path)

    print("\n== 2. H / A: {} ==".format(system))
    if system == "model":
        prun.main([out_dir, "--dead", dead_path, "--game", args.game])
        labelled = os.path.join(out_dir, "possession.csv")
    else:
        rule.main([out_dir, "--dead", dead_path, "--game", args.game])
        labelled = os.path.join(out_dir, "rule.csv")

    print("\n== 3. submission file ==")
    final = os.path.join(out_dir, "{}.csv".format(args.team))
    shutil.copyfile(labelled, final)
    n = games.source_frames("tactical", args.game)
    blocks = games.half_video_blocks(args.game)
    kickoffs = {2: games.anchor_video_frame(games.GAMES[args.game]["kickoff_2nd"])}
    fails = check_file(final, n, blocks, kickoffs)
    for f in fails:
        print("FAIL ", f)
    if not fails:
        print("format checks: OK ({} rows, X only off the graded halves, D before the 2nd-half "
              "kickoff)".format(n))
    official = validate_official(final)

    print("\n== 4. label-free aggregates ==")
    mine = aggregates.summary(read_labels(final))
    aggregates.print_table([("BAR-ATH", mine)] +
                           reference_columns(evaluate_chain.SUMMARY_PATH))
    print("\nsystem {}; sha256 {}".format(system, sha256(final)))
    print("wrote", final)
    if fails or official is False:
        sys.exit(1)
    return final


if __name__ == "__main__":
    main()
