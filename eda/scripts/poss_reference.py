"""Reference labels for an open game: per-frame H / A / dead from Metrica's event data.

Used by every poss_*.py script with --events (Metrica Sample Game 3, from
github.com/metrica-sports/sample-data: Sample_Game_3_events.json, the same game as
Sample_Game_3_tracking.txt / _metadata.xml). Never point it at event data about the competition
match: that is human-produced data about the provided game (RULES.md, rule 2).

The file is {"data": [event, ...]}; each event has type.name, subtypes (null, one {name}, or a
list of them), team.id, start.frame / end.frame (the tracking frame_id, so video frame =
frame - 1) and period. Derivation, in code only:

    dead   The union of three rules, end exclusive:
           (1) restarts. A restart is a SET PIECE event or any event with subtype GOAL KICK
           (goal kicks are PASS, or BALL LOST when they fail, not SET PIECE, in this format).
           A half's first restart is preceded by dead frames from the half's first graded
           frame; every later restart by dead frames from the end of the last event that
           starts strictly before it (CARD events skipped; a goal kick's same-frame RECOVERY
           does not count) up to its start: the ball went out, a foul was given or a goal
           scored there.
           (2) ball out. After a BALL OUT event, or a SHOT with subtype OUT, the frames from its
           end up to the start of the next event that starts later are dead, even when that
           event is not tagged as a restart.
           (3) end of half. After an event with subtype END HALF, the rest of the half is dead.
           Otherwise frames after a half's last event stay live; --frames ends the half.
    labels On a live frame, the team of the most recent controlled event (PASS, CARRY,
           RECOVERY, SHOT, SET PIECE) starting on or before it in the same half. BALL LOST,
           BALL OUT, CHALLENGE (ground, aerial or tackle, won or lost), FAULT RECEIVED and
           CARD never change possession: a touch without control does not count (CHALLENGE.md).
           The team of the first <Team> in the tracking metadata is H.

As a script it checks the derivation against the tracking (in Metrica's open data the ball is
blank while it is dead) and summarises the labels:

    python poss_reference.py OUT_DIR --raw Sample_Game_3_tracking.txt \\
        --meta Sample_Game_3_metadata.xml --events Sample_Game_3_events.json \\
        --frames 0-69660,69661-143760 --pitch 105 68 [--degrade]

Writes OUT_DIR/poss_reference_summary.csv (long format). Aggregates only.
"""

import argparse
import json
import math
import os

import numpy as np

import poss_common as pc

CONTROLLED = ("PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE")
SKIP_BEFORE_RESTART = ("CARD",)
CSV_NAME = "poss_reference_summary.csv"


def _names(sub):
    if not sub:
        return set()
    if isinstance(sub, dict):
        sub = [sub]
    return set(str(s.get("name", "")).strip().upper() for s in sub if isinstance(s, dict))


def _frame(point, fps):
    if not point:
        return None
    if point.get("frame") is not None:
        return int(point["frame"]) - 1
    if point.get("time") is not None:
        return int(round(float(point["time"]) * fps)) - 1
    return None


def read_events(path, fps=pc.FPS):
    """[{kind, subtypes, team, start, end, period}] in file order; frames are video frames.
    Events without a start frame are skipped; a missing end is the start."""
    with open(path) as handle:
        raw = json.load(handle)
    out = []
    for e in raw.get("data", raw) if isinstance(raw, dict) else raw:
        start = _frame(e.get("start"), fps)
        if start is None:
            continue
        end = _frame(e.get("end"), fps)
        out.append({"kind": str((e.get("type") or {}).get("name", "")).strip().upper(),
                    "subtypes": _names(e.get("subtypes")),
                    "team": (e.get("team") or {}).get("id"),
                    "start": start, "end": start if end is None else max(end, start),
                    "period": int(e.get("period") or 0)})
    return out


def is_restart(e):
    return e["kind"] == "SET PIECE" or "GOAL KICK" in e["subtypes"]


def is_out(e):
    return e["kind"] == "BALL OUT" or (e["kind"] == "SHOT" and "OUT" in e["subtypes"])


def reference_labels(events, team_ids, blocks, n):
    """(labels int8 (n,), dead bool (n,)). labels: H / A on every frame from the half's first
    controlled event on (not yet masked by dead); dead: the restart rule above, inside graded
    blocks only. Events are matched to blocks by period = half number."""
    code = {team_ids[0]: pc.H, team_ids[1]: pc.A} if len(team_ids) == 2 else {}
    labels = np.zeros(n, dtype=np.int8)
    dead = np.zeros(n, dtype=bool)
    for half, first, last in blocks:
        evs = [e for e in events if e["period"] == half]
        seen_restart = False

        def mark_dead(lo, stop):
            lo, stop = max(lo, first), min(stop, last + 1)
            if stop > lo:
                dead[lo:stop] = True
        for i, e in enumerate(evs):
            if is_restart(e):
                if not seen_restart:
                    mark_dead(first, e["start"])
                    seen_restart = True
                else:
                    prev = next((p for p in reversed(evs[:i])
                                 if p["kind"] not in SKIP_BEFORE_RESTART and
                                 p["start"] < e["start"]), None)
                    if prev is not None:
                        mark_dead(prev["end"], e["start"])
            if is_out(e):
                nxt = next((q for q in evs[i + 1:] if q["start"] > e["start"]), None)
                if nxt is not None:
                    mark_dead(e["end"], nxt["start"])
            if "END HALF" in e["subtypes"]:
                mark_dead(e["end"], last + 1)
        mark = np.zeros(last - first + 1, dtype=np.int8)
        for e in evs:
            if e["kind"] in CONTROLLED and e["team"] in code and first <= e["start"] <= last:
                mark[e["start"] - first] = code[e["team"]]
        idx = np.where(mark != 0, np.arange(len(mark)), -1)
        idx = np.maximum.accumulate(idx)
        seg = np.where(idx >= 0, mark[np.maximum(idx, 0)], 0)
        labels[first:last + 1] = seg
    return labels, dead


def analyse(ctx, raw_ball):
    """Label and dead-ball counts, possession durations and the ball-blank check."""
    live, labels, graded = ctx["live"], ctx["labels"], ctx["graded"]
    blocks, fps = ctx["blocks"], ctx["args"].fps
    dead = graded & ~live
    missing = ~np.isfinite(raw_ball).all(axis=1)
    poss = pc.possessions(labels, live, blocks)
    lengths = np.array([e - s + 1 for s, e, _ in poss], dtype=float) / fps
    dead_runs = pc.common.find_runs(dead, blocks)
    deg = ~np.isfinite(ctx["ball"]).all(axis=1)
    return {"graded": int(graded.sum()), "live": int(live.sum()), "dead": int(dead.sum()),
            "H": int(np.sum(labels == pc.H)), "A": int(np.sum(labels == pc.A)),
            "unlabelled_live": int(np.sum(live & (labels == pc.NONE))),
            "n_possessions": len(poss),
            "n_turnovers": len(pc.label_changes(labels, live, blocks)),
            "poss_q": pc.common.percentiles(lengths, [10, 50, 90]),
            "n_dead": len(dead_runs),
            "dead_q": pc.common.percentiles([r["length"] / fps for r in dead_runs],
                                            [10, 50, 90]),
            "dead_missing": int(np.sum(dead & missing)),
            "live_missing": int(np.sum(live & missing)),
            "live_missing_used": int(np.sum(live & deg))}


def summary_rows(ctx, res):
    rows = pc.params_rows(ctx)
    share = lambda a, b: a / float(b) if b else math.nan  # noqa: E731
    rows += [("frames", "all", "", "graded", res["graded"], "frames"),
             ("frames", "all", "", "live", res["live"], "frames"),
             ("frames", "all", "", "dead", res["dead"], "frames"),
             ("frames", "H", "", "frames", res["H"], "frames"),
             ("frames", "A", "", "frames", res["A"], "frames"),
             ("frames", "all", "", "unlabelled_live", res["unlabelled_live"], "frames"),
             ("possessions", "all", "", "n", res["n_possessions"], "count"),
             ("possessions", "all", "", "n_turnovers", res["n_turnovers"], "count"),
             ("dead_runs", "all", "", "n", res["n_dead"], "count")]
    for q, v in zip(("p10", "p50", "p90"), res["poss_q"]):
        rows.append(("possessions", "length", q, "length", v, "s"))
    for q, v in zip(("p10", "p50", "p90"), res["dead_q"]):
        rows.append(("dead_runs", "length", q, "length", v, "s"))
    rows += [("ball_check", "dead", "", "missing_share", share(res["dead_missing"], res["dead"]),
              "share of dead frames"),
             ("ball_check", "live", "", "missing_share", share(res["live_missing"], res["live"]),
              "share of live frames"),
             ("ball_check", "live", "", "missing_share_used",
              share(res["live_missing_used"], res["live"]), "share of live frames")]
    return rows


def print_summary(res, path):
    print("graded {graded}, live {live}, dead {dead} frames".format(**res))
    print("live labels: H {H}, A {A}, unlabelled {unlabelled_live}".format(**res))
    print("{} possessions (p50 {:.1f} s), {} turnovers, {} dead runs (p50 {:.1f} s)".format(
        res["n_possessions"], res["poss_q"][1], res["n_turnovers"], res["n_dead"],
        res["dead_q"][1]))
    print("ball missing: {:.3f} of dead frames, {:.3f} of live frames ({:.3f} as used)".format(
        res["dead_missing"] / max(res["dead"], 1), res["live_missing"] / max(res["live"], 1),
        res["live_missing_used"] / max(res["live"], 1)))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pc.add_args(p, CSV_NAME, "~/eda_out/poss_reference")
    args = p.parse_args(argv)
    pc.check_out_dir(args.out_dir, "~/eda_out/poss_reference")
    if not args.events:
        p.error("--events is required: this script describes the reference labels")
    ctx = pc.load_context(args, p, need_players=False)
    raw, _, _, _, _, _ = pc.resolve_paths(args, p)
    ball, _ = pc.bg.read_ball(raw)
    raw_ball = np.full((ctx["n"], 2), np.nan)
    m = min(len(ball), ctx["n"])
    raw_ball[:m] = ball[:m]
    res = analyse(ctx, raw_ball)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, CSV_NAME)
    pc.write_csv(summary_rows(ctx, res), path)
    print_summary(res, path)
    return res


if __name__ == "__main__":
    main()
