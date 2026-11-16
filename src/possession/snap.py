"""Switch snapping to control (§4.4), behind config.SNAP.

Possession changes only when the other team gains controlled possession; a loose or contested
ball stays with the last team. For each switch the decoder placed:

  1. look within +-SNAP_S of it (never past the neighbouring switches or the spell edges);
  2. find the old team's last control frame in that window, and the new team's first control
     frame after it (after the window start if the old team has none);
  3. move the switch there; with no new-team control frame (a ball gap), keep it.

Control frame: ball seen or filled, within CONTROL_DIST_M of a player, slower than
CONTROL_SPEED_MS, with the nearest player's team the same for CONTROL_K frames in a row. The
first frame of such a run is where control starts.
"""

import numpy as np

from possession import config


def control_frames(signals, live):
    """int8 (n,): +1 home / -1 away on control frames, 0 elsewhere."""
    near, speed, nearest, flag = (signals["near"], signals["speed"], signals["nearest"],
                                  signals["flag"])
    with np.errstate(invalid="ignore"):
        ok = live & (flag >= 0) & (flag <= 1) & (near < config.CONTROL_DIST_M) & \
            (speed < config.CONTROL_SPEED_MS) & (nearest >= 0)
    code = np.where(ok, np.where(nearest == 0, 1, -1), 0).astype(np.int8)
    out = np.zeros(len(code), dtype=np.int8)
    edges = np.flatnonzero(np.diff(code) != 0) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(code)]))
    for a, b in zip(starts, stops):
        if code[a] != 0 and b - a >= config.CONTROL_K:
            out[a:b] = code[a]
    return out


def snap(team, spells, control, fps=config.FPS):
    """(new team array, {moved, kept, frames_moved}). Segments keep their order and teams."""
    out = team.copy()
    width = int(round(config.SNAP_S * fps))
    counts = {"moved": 0, "kept": 0, "frames_moved": 0}
    for _, a, b in spells:
        seg = team[a:b]
        sw = (a + np.flatnonzero(seg[1:] != seg[:-1]) + 1).tolist()
        new_sw = []
        for i, s in enumerate(sw):
            lo = max(a + 1, s - width, (new_sw[-1] + 1) if new_sw else a + 1)
            hi = min(b - 1, s + width, (sw[i + 1] - 1) if i + 1 < len(sw) else b - 1)
            old, new = int(team[s - 1]), int(team[s])
            window = control[lo:hi + 1]
            old_idx = np.flatnonzero(window == old)
            after = int(old_idx[-1]) + 1 if len(old_idx) else 0
            new_idx = np.flatnonzero(window[after:] == new)
            if len(new_idx):
                target = lo + after + int(new_idx[0])
                counts["moved" if target != s else "kept"] += 1
                counts["frames_moved"] += abs(target - s)
            else:
                target = s
                counts["kept"] += 1
            new_sw.append(target)
        if not sw:
            continue
        bounds = [a] + new_sw + [b]
        first = int(team[a])
        for j in range(len(bounds) - 1):
            out[bounds[j]:bounds[j + 1]] = first if j % 2 == 0 else -first
    return out, counts
