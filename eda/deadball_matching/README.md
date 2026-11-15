# deadball_matching: do the EDA dead-ball stretches line up with Smart Tagging?

The EDA scripts flag frames that look like dead ball (ball missing, ball outside the pitch, ball
still on a restart mark). This folder checks that against Smart Tagging's `SET PIECES`. Then you
watch a sample of the disagreements to find out which source is wrong.

```
deadball_matching/
├── README.md
├── deadball_matching.py   builds dead-ball intervals, matches them to set-piece starts, writes the outputs
├── summarise_review.py    turns the hand-filled review sheet into counts
└── tests/                 synthetic match with hand-derived answers (pytest)
```

Both scripts import the EDA scripts from `../scripts` themselves; stdlib + numpy only.

## The timing detail: match ends to starts, not overlaps

A `SET PIECES` instance starts at the restart (when the corner is taken). It does not cover the
build-up. When the two sources agree, the dead-ball stretch **ends** where the tag **starts**:

```
EDA dead ball:   |==== ball out / missing / still ====|
Smart Tagging:                                         |CORNER|
                                                       ^ restart
```

So an overlap test would count every correct stretch as a miss. Instead, each interval's last
frame is paired with the nearest set-piece start within ±5 s. The pairing is one-to-one: closest
pairs first, each interval and each tag used at most once, same half only.

## What it does

1. **Dead-ball mask per graded frame**, the union of the `--signals`:
   - `missing`: a ball-missing gap of ≥ 2 s (`--min-missing-s`)
   - `outside`: the visible ball is more than 1 m beyond a line (`--margin-m`)
   - `still`: a still run of ≥ 1 s (`--min-still-s`) on a restart mark: centre spot, penalty
     spot, corner arc, goal area or touchline band (`--still-locations`; pass `all` to include
     free kicks in open play)
2. **Intervals**: runs of the mask at most 1 s apart are merged (`--merge-gap-s`), and intervals
   under 1 s are dropped (`--min-dead-s`). Each interval records which signals flagged it, e.g.
   `missing+outside`.
3. **Smart Tagging**: `SET PIECES` instances (code matched case-insensitively), start frame
   `round(seconds × 25)`. Tags starting on an ungraded frame are dropped and counted.
4. **Match** with a window of ±5 s (`--link-window`, the same as `tag_gaps.py`). This gives three groups:
   - `matched`: records the tag's Type and the offset (interval end − tag start, in seconds)
   - `dead_no_tag`: the EDA flagged a stoppage that Smart Tagging didn't tag
   - `tag_no_dead`: Smart Tagging has a restart that the EDA missed
5. **Summary**:
   - precision overall, per signal set and per length bin
   - recall for `SET PIECES` and per Type
   - offset histogram (0.5 s bins) and quartiles, overall and per Type
   - signal set × Type table
   - the same match against tag starts shifted circularly within each half (the `tag_gaps`
     null, 1000 shifts, at least 60 s), so you can see whether the match rate beats chance
6. **Review sample**: up to 8 per unmatched group (`--review-n`), seeded and spread over both
   halves, with video seek times. A `dead_no_tag` row shows the interval ±3 s. A `tag_no_dead` row
   shows 10 s of build-up before the tag and 3 s after it.
7. **Optional ground truth** (`--gt`, a `frame,label` CSV): on the labelled frames only, it scores:
   - frame-level precision, recall and F1 of `D`, for each signal, their union and the merged intervals
   - restart-level matching of both the EDA interval ends and the Smart Tagging starts against
     the ends of the labelled `D` runs

## Run on the VM

`MTAH_DATA_DIR` must point at the data (`/srv/mtah/data` on the team VM). Run from the team-repo
root and write outside the repository (the script refuses otherwise):

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/deadball_matching/deadball_matching.py "$HOME/eda_out/deadball_matching"
```

With the public ground truth: it ships with the starter repository as
`evaluation/gt/BAR-ATH_possession_gt_public.csv`. It isn't in this repository, so pass its path:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/deadball_matching/deadball_matching.py "$HOME/eda_out/deadball_matching" --gt /path/to/BAR-ATH_possession_gt_public.csv
```

To check which signal carries the result, run it again with one signal at a time, e.g.
`--signals missing` (compare with `tag_gaps.py` view C) or `--signals outside,still`. Use a
different output folder for each run.

## Outputs, and what may leave the VM

| file | content | may leave the VM? |
|---|---|---|
| `deadball_matching_summary.csv` | aggregates only (long format: analysis, scope, bin, metric, value, unit) | yes |
| `deadball_matches.csv` | **frame-indexed**: one row per interval and per unmatched tag | **no**, never commit it |
| `review_sample.csv` | **frame-indexed**: the review sheet with seek times | **no** |
| output of `summarise_review.py` | category counts per group, signal set and Type | yes |

stdout from both scripts is aggregate only. This follows the team data rule in
[../README.md](../README.md) and `docs/info/RULES.md`.

## Video review

1. Open `review_sample.csv` on the VM and watch each row from `view_from` to `view_to`.
2. Fill in `category` with a short lowercase label. Suggestions:

   | group | categories |
   |---|---|
   | `dead_no_tag` | `foul`, `offside`, `substitution`, `injury`, `goal`, `ball_out_untagged` (Smart Tagging missed it), `in_play` (a false dead ball, e.g. a tracking dropout in play) |
   | `tag_no_dead` | `quick_restart` (dead stretch too short or barely visible), `signal_missed` (the ball was dead but no signal fired), `false_tag` (no restart happened), `window` (the stretch exists but ends more than 5 s from the tag) |

   `notes` is free text and stays on the VM.
3. Summarise and bring back only the counts:

```bash
python3 eda/deadball_matching/summarise_review.py "$HOME/eda_out/deadball_matching/review_sample.csv" --out "$HOME/eda_out/deadball_matching/review_labels_summary.csv"
```

By default, the review sample leaves out the public GT window (frames 11185–18684) and the
half-2 frames before kick-off, the same as `sample_long_gaps.py`. Use `--exclude` to change that.

## Reading the results

- **Smart Tagging is not ground truth.** It is automatic and unreviewed. It lists only 5 free
  kicks for the whole match, while a real match has far more fouls. Many `dead_no_tag` intervals
  are probably correct on the EDA side (fouls, offsides, substitutions, injuries). Low precision
  against Smart Tagging does not by itself mean the EDA is wrong; the video review decides.
- **Offsets:** a tight peak near 0 means the interval ends at the restart. A consistent shift
  (e.g. intervals end about 2 s before throw-ins) means the end boundary needs adjusting for that
  restart type.
- **Recall per Type:** throw-ins and corners should behave very differently. Throw-ins are often
  quick, with a short or invisible dead stretch.
- **Null:** a `lift` well above 1 with a small `p_emp` means the pairing beats chance. At ±5 s with
  long intervals, chance matches are not negligible, so read precision next to `null_mean`.
- **Ground truth** covers about 5 minutes (17 possessions), so its numbers are small and noisy.
  It is for checking, not for tuning: DATA.md and RULES.md say the public GT is not a tuning set,
  and hand labels on this game may not be used to train or validate a model. Use the review labels
  and GT scores to understand the signals, not to fit thresholds.

## Tests

```bash
python3 -m pytest eda/deadball_matching/tests
```

The tests build a synthetic two-half match (missing, outside and still events, a short gap that
must not count, an open-play still run, a pair of runs to merge, two tags competing for one
interval). The expected intervals, pairs, scores and review rows are worked out by hand in the
test file's docstring. `python3 -m pytest eda` runs this suite together with `eda/tests`.
