"""The hard mask from the dead-ball labeller, and the live spells it leaves (§4.0).

dead_ball.csv `dead`: 1 dead -> D, -1 not graded -> X, 0 live -> H or A from the decoder. A live
spell is a maximal run of live frames inside one graded half; each is decoded on its own.
"""

import numpy as np

import poss_common as pc

DEAD, LIVE, UNGRADED = 1, 0, -1


def read_dead(path, n):
    """int8 (n,) of 1 / 0 / -1; frames the file does not list are -1."""
    return pc.read_dead_csv(path, n)


def count_rows(path):
    """Data rows in a frame-indexed CSV (the submission length comes from dead_ball.csv)."""
    with open(path) as handle:
        return sum(1 for _ in handle) - 1


def apply_kickoffs(dead, blocks, kickoffs):
    """Copy of dead with every graded frame of a half before its kickoff set to dead.

    kickoffs: {half: first live video frame}. The dead-ball labeller already does this for
    BAR-ATH; repeating it here keeps the H/A output safe if it is fed another dead file."""
    out = dead.copy()
    for half, first, last in blocks:
        kick = kickoffs.get(half)
        if kick is not None and kick > first:
            stop = min(kick, last + 1)
            out[first:stop] = np.where(out[first:stop] == UNGRADED, UNGRADED, DEAD)
    return out


def live_spells(dead, blocks):
    """[(half, start, stop)] half-open maximal runs of dead == 0 inside each graded block."""
    out = []
    for half, first, last in blocks:
        seg = (dead[first:last + 1] == LIVE).astype(np.int8)
        edges = np.diff(np.concatenate(([0], seg, [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            out.append((half, first + int(a), first + int(b)))
    return out


def live_mask(dead, blocks):
    live = np.zeros(len(dead), dtype=bool)
    for _, a, b in live_spells(dead, blocks):
        live[a:b] = True
    return live


def labels_from(dead, team):
    """Label strings: D / X from the mask, H / A from team (+1 H, -1 A) on live frames.

    A live frame with team 0 would be a decoder bug, so it raises instead of writing X."""
    out = np.full(len(dead), "X", dtype="<U1")
    out[dead == DEAD] = "D"
    live = dead == LIVE
    if np.any(live & (team == 0)):
        raise ValueError("{} live frames have no team".format(int(np.sum(live & (team == 0)))))
    out[live & (team > 0)] = "H"
    out[live & (team < 0)] = "A"
    return out


def write_labels(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(i, lab) for i, lab in enumerate(labels)))
