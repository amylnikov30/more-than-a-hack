"""Settle the unit of the declared ATD speed column (km/h or m/s) from the data itself.

For players and ball, a derived speed comes from the displacement between consecutive graded
frames of the same half (metres x fps), only where both positions are present. Against it:

  ratio      declared / derived on pairs with derived speed in [--min-ms, cap]: median and IQR
             (about 1 means m/s, about 3.6 means km/h); pairs above the cap (slot swaps, ball
             jumps) are counted, not used
  speeds     p50 / p99 / p99.9 of the derived speed (all pairs, no cap) and of the declared speed
             (every graded frame with a present object), players and ball separately
  verdict    per class and overall, next to the unit the ATD metadata declares

Usage
    # competition match: paths from games.asset_paths() (honours MTAH_DATA_DIR)
    PYTHONPATH=/path/to/team-repo python eda/speed_units.py OUT_DIR

    # any other game in the raw ATD format (Sample Game 3 has no speed field: derived only)
    python speed_units.py OUT_DIR --raw RAW.txt --frames 0-69660,69661-143760 --pitch 105 68

Writes OUT_DIR/speed_units_summary.csv (long format: analysis, scope, bin, metric, value, unit).
Definitions and fixture answers: tests/fixtures/README.md ("speed_units.py"). Aggregates only.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import ball_gaps as bg
import common

FPS = bg.FPS
PITCH = bg.PITCH
MIN_MS = 1.0             # below this, frame-to-frame position noise dominates the ratio
CAP_PLAYER_MS = 12.0     # about the fastest recorded sprint
CAP_BALL_MS = 40.0       # a very hard shot
QS = (50, 99, 99.9)
UNITS = (("m/s", 1.0), ("km/h", 3.6))
VERDICT_TOL = 0.10       # +-10% of the reference ratio
CSV_NAME = "speed_units_summary.csv"
CLASSES = ("players", "ball")


def declared_unit(meta_path):
    """Unit text of the first sensor channel whose id or name contains 'speed', or None."""
    if not meta_path:
        return None
    import check_channels
    root = ET.parse(meta_path).getroot()
    for _, cid, name, unit in check_channels.sensor_channels(root):
        if "speed" in "{} {}".format(cid, name).lower():
            return unit
    return None


def verdict(median):
    if median is None or (isinstance(median, float) and math.isnan(median)):
        return None
    for label, ref in UNITS:
        if abs(median - ref) <= VERDICT_TOL * ref:
            return label
    return "unclear"


def overall_verdict(per_class):
    found = [v for v in per_class if v is not None]
    if not found:
        return "no declared speed"
    return found[0] if all(v == found[0] for v in found) else "conflict"


def analyse(raw_path, blocks, meta_path=None, fps=FPS, pitch=PITCH, min_ms=MIN_MS,
            cap_player=CAP_PLAYER_MS, cap_ball=CAP_BALL_MS):
    """All numbers, as a dict. No per-frame data leaves this function."""
    half_of = common.graded_half_of(blocks)
    caps = {"players": cap_player, "ball": cap_ball}
    derived = {c: [] for c in CLASSES}       # derived speed of every pair
    decl_pair = {c: [] for c in CLASSES}     # declared speed on the later frame of each pair
    decl_all = {c: [] for c in CLASSES}      # declared speed on every graded present frame
    skipped_steps = 0
    prev = None
    for frame, players, ball in common.stream_frames(raw_path):
        if common.half_at(half_of, frame) > 0:
            for c, arr in (("players", players), ("ball", ball[None, :])):
                ok = np.isfinite(arr[:, :2]).all(axis=1) & np.isfinite(arr[:, 2])
                decl_all[c].append(arr[ok, 2])
        if prev is not None and common.is_step(prev[0], frame, half_of):
            _, p_prev, b_prev = prev
            pairs = [("ball", b_prev[None, :], ball[None, :])]
            if len(p_prev) == len(players):
                pairs.append(("players", p_prev, players))
            else:
                skipped_steps += 1
            for c, a, b in pairs:
                v = common.step_speed(a[:, :2], b[:, :2], pitch, fps)
                ok = np.isfinite(v)
                derived[c].append(v[ok])
                decl_pair[c].append(b[ok, 2])
        prev = (frame, players, ball)

    res = {"blocks": list(blocks), "fps": fps, "pitch": tuple(pitch), "min_ms": min_ms,
           "caps": caps, "skipped_steps": skipped_steps,
           "declared_unit": declared_unit(meta_path), "classes": {}}
    for c in CLASSES:
        d = np.concatenate(derived[c]) if derived[c] else np.empty(0)
        s = np.concatenate(decl_pair[c]) if decl_pair[c] else np.empty(0)
        s_all = np.concatenate(decl_all[c]) if decl_all[c] else np.empty(0)
        use = (d >= min_ms) & (d <= caps[c]) & np.isfinite(s)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = s[use] / d[use]
        med, p25, p75 = (common.percentiles(ratio, (50, 25, 75)))
        res["classes"][c] = {
            "n_pairs": int(len(d)), "n_over_cap": int(np.sum(d > caps[c])),
            "n_ratio": int(len(ratio)), "ratio_median": med, "ratio_p25": p25, "ratio_p75": p75,
            "derived_q": common.percentiles(d, QS), "n_declared": int(len(s_all)),
            "declared_q": common.percentiles(s_all, QS), "verdict": verdict(med)}
    res["verdict"] = overall_verdict([res["classes"][c]["verdict"] for c in CLASSES])
    return res


def summary_rows(res):
    rows = []
    add = lambda *r: rows.append(r)  # noqa: E731
    add("params", "all", "", "fps", res["fps"], "frames/s")
    add("params", "all", "", "pitch_length", res["pitch"][0], "m")
    add("params", "all", "", "pitch_width", res["pitch"][1], "m")
    add("params", "all", "", "min_ms", res["min_ms"], "m/s")
    add("params", "players", "", "cap", res["caps"]["players"], "m/s")
    add("params", "ball", "", "cap", res["caps"]["ball"], "m/s")
    for h, a, b in res["blocks"]:
        add("params", "half {}".format(h), "", "first_frame", a, "video frame")
        add("params", "half {}".format(h), "", "last_frame", b, "video frame")
    add("meta", "all", "", "declared_speed_unit", res["declared_unit"] or "", "")
    add("meta", "all", "", "skipped_player_steps", res["skipped_steps"], "count")
    for c in CLASSES:
        r = res["classes"][c]
        add("ratio", c, "", "n_pairs", r["n_pairs"], "count")
        add("ratio", c, "", "n_over_cap", r["n_over_cap"], "count")
        add("ratio", c, "", "n_ratio", r["n_ratio"], "count")
        for m in ("median", "p25", "p75"):
            add("ratio", c, m, "declared_over_derived", r["ratio_" + m], "ratio")
        for q, v in zip(QS, r["derived_q"]):
            add("speed", c, "p{:g}".format(q), "derived", v, "m/s")
        add("speed", c, "", "n_declared", r["n_declared"], "count")
        for q, v in zip(QS, r["declared_q"]):
            add("speed", c, "p{:g}".format(q), "declared", v, "declared unit")
        add("verdict", c, "", "verdict", r["verdict"] or "", "")
    add("verdict", "overall", "", "verdict", res["verdict"], "")
    return rows


def _f(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def print_summary(res, csv_path):
    for c in CLASSES:
        r = res["classes"][c]
        print("{:<7} {} pairs, {} over the {:g} m/s cap, {} ratios: declared/derived median {} "
              "[IQR {}-{}] -> {}".format(c, r["n_pairs"], r["n_over_cap"], res["caps"][c],
                                         r["n_ratio"], _f(r["ratio_median"]), _f(r["ratio_p25"]),
                                         _f(r["ratio_p75"]), r["verdict"] or "-"))
        print("        derived m/s p50/p99/p99.9 {}; declared p50/p99/p99.9 {} (n={})".format(
            "/".join(_f(v) for v in r["derived_q"]), "/".join(_f(v) for v in r["declared_q"]),
            r["n_declared"]))
    if res["skipped_steps"]:
        print("note: {} steps skipped for players (slot count changed)".format(
            res["skipped_steps"]))
    found = {"m/s": "declared speed is in m/s", "km/h": "declared speed is in km/h",
             "unclear": "unclear: the median ratio is near neither 1 nor 3.6",
             "conflict": "conflict: players and ball disagree"}
    print("verdict: {} (metadata declares: {})".format(
        found.get(res["verdict"], res["verdict"]),
        res["declared_unit"] or "no speed channel / no metadata"))
    print("wrote {}".format(csv_path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", help="folder for {} (outside the repo)".format(CSV_NAME))
    p.add_argument("--game", default="BAR-ATH")
    p.add_argument("--raw", help="raw ATD txt (or .gz); default games.asset_paths()['atd_raw']")
    p.add_argument("--meta", help="ATD metadata XML, for the declared unit")
    p.add_argument("--frames", type=bg.parse_frames,
                   help="graded video frames, one inclusive range per half; default "
                        "games.half_video_blocks(GAME)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--pitch", type=float, nargs=2, default=PITCH, metavar=("LENGTH", "WIDTH"))
    p.add_argument("--min-ms", type=float, default=MIN_MS,
                   help="ratio only where derived speed >= this, m/s (default 1)")
    p.add_argument("--cap-player", type=float, default=CAP_PLAYER_MS,
                   help="player plausibility cap, m/s (default 12)")
    p.add_argument("--cap-ball", type=float, default=CAP_BALL_MS,
                   help="ball plausibility cap, m/s (default 40)")
    args = p.parse_args(argv)
    bg.check_out_dir(args.out_dir, example="~/eda_out/speed_units")
    if args.raw:
        if args.frames is None:
            p.error("--frames is required with --raw")
        raw, meta, blocks = args.raw, args.meta, args.frames
    else:
        paths = common.game_paths(args.game, ["atd_raw", "atd_meta"])
        raw, meta = paths["atd_raw"], args.meta or paths["atd_meta"]
        blocks = args.frames or bg._games().half_video_blocks(args.game)
    res = analyse(raw, blocks, meta, fps=args.fps, pitch=tuple(args.pitch), min_ms=args.min_ms,
                  cap_player=args.cap_player, cap_ball=args.cap_ball)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, CSV_NAME)
    bg.write_csv(summary_rows(res), csv_path)
    print_summary(res, csv_path)
    return res


if __name__ == "__main__":
    main()
