"""Sample long ball-tracking gaps for video review (EDA stratum 1).

Run on the VM from the repo root. Writes to ~/eda_out. The output is
frame-indexed: keep it on the VM and never paste it into a hosted AI.
"""
import argparse
import os

import numpy as np
import pandas as pd
from kloppy import metrica

from games import asset_paths, half_video_blocks

FPS = 25
PUBLIC_GT = (11185, 18684)   # public window: excluded from the sample
KICKOFF_2H = 113742          # 2nd half graded from 110808 but kicks off here
PAD_S = 3                    # seconds of context either side


def runs(mask):
    """Inclusive (start, end) index pairs of True runs."""
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def ts(frame):
    """Video frame -> h:mm:ss.ss of the tactical video, for seeking."""
    s = frame / FPS
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--min-s", type=float, default=None)
    ap.add_argument("--max-s", type=float, default=None)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=os.path.expanduser("~/eda_out/long_gap_sample.csv"))
    a = ap.parse_args()

    p = asset_paths()
    df = metrica.load_tracking_epts(meta_data=p["atd_meta"], raw_data=p["atd_raw"]).to_df()
    vf = df["frame_id"].to_numpy() - 1               # ATD frame_id -> video frame
    missing = df["ball_x"].isna().to_numpy()

    rows = []
    for half, lo, hi in half_video_blocks("BAR-ATH"):
        sel = (vf >= lo) & (vf <= hi)
        f = vf[sel]
        starts, ends = runs(missing[sel])
        for i, j in zip(starts, ends):
            rows.append(dict(half=half, start=int(f[i]), end=int(f[j]), n=int(j - i + 1),
                             clipped=bool(i == 0 or j == len(f) - 1)))
    gaps = pd.DataFrame(rows)
    gaps = gaps[~gaps.clipped]                       # drop gaps cut by a half boundary

    if a.min_s is not None or a.max_s is not None:
        lo_n = (a.min_s or 0) * FPS
        hi_n = (a.max_s or np.inf) * FPS
        long_ = gaps[(gaps.n >= lo_n) & (gaps.n <= hi_n)]
        thr = lo_n
    else:
        thr = gaps.n.quantile(a.quantile)
        long_ = gaps[gaps.n >= thr]
    long_ = long_[~((long_.start <= PUBLIC_GT[1]) & (long_.end >= PUBLIC_GT[0]))]
    long_ = long_[~((long_.half == 2) & (long_.start < KICKOFF_2H))]

    # Aggregates only: these are safe to share.
    print(f"gaps: {len(gaps)}  threshold (q={a.quantile}): {thr / FPS:.1f} s  "
          f"eligible long gaps: {len(long_)}")

    k = min(a.n, len(long_))
    # Sample within each half so both are represented.
    sample = (long_.groupby("half", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), (k + 1) // 2), random_state=a.seed))
                   .sort_values("start"))

    out = pd.DataFrame({
        "half": sample.half,
        "start_frame": sample.start,
        "end_frame": sample.end,
        "gap_s": (sample.n / FPS).round(1),
        "view_from": [ts(max(0, s - PAD_S * FPS)) for s in sample.start],
        "view_to": [ts(e + PAD_S * FPS) for e in sample.end],
        "category": "",   # fill in from the checklist while watching
        "notes": "",
    })
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} intervals to {a.out}")


if __name__ == "__main__":
    main()
