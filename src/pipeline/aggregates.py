"""Label-free aggregates of a frame,label file (§6.4): what a submission claims about the match.

Only rounded aggregates leave this module, so its output may be shared and compared with the
reference set and football norms.
"""

import numpy as np

FPS = 25


def possessions(labels):
    """[(start, stop, label)] maximal runs of H or A (D and X end a run)."""
    labels = np.asarray(labels)
    out = []
    n = len(labels)
    t = 0
    while t < n:
        lab = labels[t]
        if lab not in ("H", "A"):
            t += 1
            continue
        s = t
        while t < n and labels[t] == lab:
            t += 1
        out.append((s, t, lab))
    return out


def summary(labels, fps=FPS):
    """Dict of rounded aggregates: shares, live spells, possessions per live minute, lengths."""
    labels = np.asarray(labels)
    graded = labels != "X"
    live = (labels == "H") | (labels == "A")
    n_graded, n_live = int(graded.sum()), int(live.sum())
    runs = possessions(labels)
    lengths = np.array([b - a for a, b, _ in runs], dtype=float) / fps
    edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
    live_spells = int(np.sum(edges == 1))
    minutes = n_live / float(fps) / 60.0
    return {
        "graded_frames": n_graded,
        "dead_share": round(float(np.sum(labels == "D")) / n_graded, 3) if n_graded else None,
        "live_spells": live_spells,
        "possessions": len(runs),
        "possessions_per_live_minute": round(len(runs) / minutes, 2) if minutes else None,
        "possession_s_p10_p50_p90": (np.percentile(lengths, [10, 50, 90]).round(1).tolist()
                                     if len(runs) else []),
        "h_share_of_live": round(float(np.mean(labels[live] == "H")), 3) if n_live else None,
    }


KEYS = ("dead_share", "live_spells", "possessions", "possessions_per_live_minute",
        "possession_s_p10_p50_p90", "h_share_of_live")


def print_table(columns, keys=KEYS):
    """columns: [(title, summary dict)] side by side, one row per aggregate."""
    width = max(12, max(len(t) for t, _ in columns) + 2)
    print("{:<30}".format("aggregate") + "".join("{:>{w}}".format(t, w=width)
                                                for t, _ in columns))
    for key in keys:
        cells = []
        for _, s in columns:
            v = s.get(key) if s else None
            cells.append("--" if v is None else (" ".join("{:g}".format(x) for x in v)
                                                 if isinstance(v, list) else "{:g}".format(v)))
        print("{:<30}".format(key) + "".join("{:>{w}}".format(c, w=width) for c in cells))
