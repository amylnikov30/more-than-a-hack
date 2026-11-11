# EDA hand-off: ball-track gaps

**For a Claude chat that helps interpret VM results it must never see.** This file contains no
competition data and nothing derived from it. Numbers here come from synthetic fixtures or
Metrica's open Sample Game 3. Do not ask the user for VM outputs, counts or tables. Ask for their
own-words, qualitative answers to section 7.

Context: label every video frame H, A or D; ranked on spellF1. Carrying the previous label
through a ball gap is free inside a possession and costly across a change of possession. The
ATD has 46 anonymous track slots and an often-NaN ball. Graded frames are the two halves from
`games.half_video_blocks("BAR-ATH")`. `video_frame = frame_id - 1`.

## 1. `check_channels.py`

It reads only the metadata XML and prints the sensor channels, each data-format spec (frame
range, frame counter, player slots × channels, ball channels), and whether any declared name
matches status, alive, dead, in-play, state, owner, possess or owning (`NONE declared`
otherwise). `channels.txt` holds the full name list. kloppy's `to_df()` always adds empty
`ball_state` and `ball_owning_team_id` columns; they are not data.

## 2. `ball_gaps.py`

A **gap** is a maximal run of graded frames with the ball x or y NaN, or with no record. Runs are
found per half and clipped to it. The **last-seen** frame is the last ball frame before the
unclipped run; the **reappearance** frame is the first ball frame after it. Either may be
ungraded or absent.

- **(a) Lengths.** Seconds = frames ÷ 25. Bins are half-open: `[0,0.5)`, `[0.5,2)`, `[2,5)`,
  `[5,∞)`. For each bin: number of gaps and share of missing graded frames, overall and per half.
- **(b) Set pieces.** For gaps ≥ 2 s: the share that overlap (share at least one frame with) a
  `set pieces` tag (case-insensitive). Tags are `round(s*25)`, end exclusive. `--tag-tolerance T`
  widens each tag by T s on both sides.
- **(c) Crowding.** Players (finite x and y) within 5 m of the ball on the last-seen frame, in
  metres on 105.3 × 68.0. Compared with a sample of 2,000 ball-visible graded frames
  (`default_rng(0)`; all of them if there are fewer).
- **(d) Flips.** On the last-seen and reappearance frames, find the nearest player with finite
  x, y and a team. Team comes from the metadata: first `<Team>` = H, second = A; slots with no
  team (referee, unknown) are skipped. Ties go to the lowest slot index, in raw-line order.
  - A gap is **evaluable** if both frames exist and have a nearest player. Otherwise the reason is
    recorded: no_last_seen, no_reappearance or no_player. `--flip-max-dist R` also requires both
    nearest players to be within R m (reason too_far).
  - flip_rate = flips ÷ evaluable, overall, per half and per bin.
  - **Reference:** the same test on pairs (f, f + Δ). f runs over the (c) sample, and Δ is the
    bin's median clipped gap length, rounded (ties to even). A pair is kept only if both frames
    are in the same graded half, the ball is present on both, and both have a nearest player
    (and are within R).
- **(e) Set pieces vs chance.** Tags are clipped to each half and shifted together, circularly
  within the half, by an offset drawn uniformly from [min_shift, L − min_shift]. Each half is
  drawn independently. Gaps stay fixed and the tolerance is applied after shifting. K shifts
  give a null of overlap shares. The output is the null mean, p5 and p95, and
  p_emp = (1 + #null ≥ real) / (1 + K), for gaps ≥ 2 s, 2-5 s and ≥ 5 s. It uses interval
  arithmetic; 1,000 shifts of 600 gaps × 60 tags take about 0.02 s.

| threshold | justification | status |
|---|---|---|
| 0.5 / 2 / 5 s bins | One touch / brief occlusion vs long loss / longer than most restarts | arbitrary first choices |
| 5 m (c) | Rough "in contact with play" radius | arbitrary first choice |
| sample 2,000, seed 0 | Standard error ≈ 2% of the SD; the seed only fixes reproducibility | arbitrary |
| tolerance 0 s | Strictest version; ±1 s is the sensitivity run | choice |
| Δ = median gap length (d) | One typical span per bin. The last-seen and reappearance frames are actually length + 1 apart, so Δ is one frame shorter; this only matters for the shortest bin. | choice (as specified) |
| R off; 5 m suggested (d) | Asks whether "nearest" means "in control" | arbitrary |
| K = 1000, null seed 0 (e) | p_emp resolution 0.001 | choice |
| min_shift 60 s (e) | Keeps shifted tags away from their true place without excluding much of a 53-min half | arbitrary |

Without `--raw`, a missing Smart Tagging file prints a note and skips (b) and (e). A metadata
spec with no `endFrame` is treated as open-ended.

### `sample_long_gaps.py` (video-review helper)

It picks a few long gaps to watch by hand. **Its CSV is frame-indexed and stays on the VM; do not
ask for it.** Gaps come from `ball_gaps.find_gaps`, minus edge gaps. A gap is eligible if
length ≥ the q-quantile (default 0.9) of those lengths, or, with `--min-s`/`--max-s`,
min ≤ s < max. Gaps overlapping `--exclude` ranges are dropped; by default these are the public
GT window and the second-half frames before kickoff. The n gaps (default 8) are spread over the
halves as evenly as the eligible gaps allow, with `default_rng(seed)`. The user records a
category per gap while watching. Only their own-words summary of those categories comes back.

## 3. Output

Printed lines, in order:

1. Totals: graded frames, missing frames, gaps and edge gaps. An edge gap's length is a lower
   bound.
2. `note: ... no tracking record`, printed only on a path or alignment problem.
3. Rows `overall`, `half 1` and `half 2`, with `gaps / % of row's missing frames` per bin.
4. (b) `set pieces: k of n ... (%)`, per bin `a/b`, and the tolerance.
5. (c) gap-start crowding (all, ≥ 2 s, < 2 s), then the visible-frame baseline.
6. (d) `flip: overall rate (flips/evaluable), not evaluable n [reason counts]; per bin
   flips/evaluable [max dist]`.
7. (d) `reference: per bin rate (n pairs, delta)`.
8. (e) `set pieces vs chance:` for ≥ 2 s and each bin: real %, null mean % [p5, p95], p_emp,
   then the tolerance, K and min_shift. It says `skipped` if there is no tags file or a half is
   shorter than 2 × min_shift.

`ball_gaps_summary.csv` columns: `analysis, scope, bin, metric, value, unit`.

| analysis / scope / bin | metrics (unit) |
|---|---|
| `params` / `all`, `half h` | fps, pitch_length/width (m), radius (m), long_gap_threshold (s), sample_size, seed, tag_tolerance (s), flip_max_dist (m; empty = off), null_n, null_seed, null_min_shift (s), first/last_frame |
| `gap_length` / `overall`, `half h` / `all` or bin | n_gaps, missing_frames, share_of_missing_frames (%), graded_frames, missing_share_of_graded (%); overall only: edge_gaps, no_record_frames |
| `set_pieces` / `overall` / `>=long`, bin | n_set_piece_intervals, n_gaps, n_overlap, share_overlap (%) |
| `crowding` / `gap_start_all`, `_long`, `_short`, `visible` | n, mean, median (players); gaps_without_last_seen; population |
| `flip` / `overall`, `half h` / `all` or bin | n_gaps, n_evaluable, n_flip (count), flip_rate (ratio 0-1), not_eval_no_last_seen, _no_reappearance, _no_player, _too_far; bins only: ref_delta (frames), ref_n, ref_flip_rate (ratio) |
| `set_pieces_null` / `overall` / `>=long`, `2-5s`, `>=5s` | n_gaps, real_share, null_mean, null_p5, null_p95 (%), p_emp (ratio), K, min_shift (s) |

## 4. Assumptions and limitations

- **Slots are tracklets.** A duplicate counts twice and a missed player zero. Positions are the
  first two fields of each slot group. Referees count in (c) but are skipped in (d).
- **"Nearest team" is a proxy for possession,** not possession itself. A flip can be a contest
  or a pass to an opponent; "no flip" can hide a lost-and-regained ball. R tests this.
- **Edge effects.** Clipped gaps are undercounted. Last-seen or reappearance frames may be
  pre-match or at halftime. Gaps without either are reported as not evaluable, not dropped
  silently.
- **The reference** is not matched on pitch zone or game state. The (c)/(d) sample includes
  frames next to gaps.
- **Smart Tagging** is automatic and accurate to about 1 s; any one-frame overlap counts.
- **Null clipping.** The null clips tags to halves. With tolerance > 0, a tag lying just outside a
  half could touch a gap in (b) but not in the null at offset 0. This is negligible at the
  match's half boundaries.
- **The null keeps each half's tag pattern intact** and only moves it. It tests timing relative
  to gaps, not tag frequency.

## 5. What was tested

These are not competition results. 53 pytest tests pass. Expected values were derived by hand
before the code (`tests/fixtures/README.md`), and every one matched.

- **(a)-(c), main fixture** (3,000 frames, 46 slots): 11 gaps, 633 / 2,700 missing; bins
  3/19, 3/64, 3/200, 2/350; (b) 3 of 5 (5 of 5 at ±1 s); crowding 38/11 vs 2094/2067.
- **(d), flip fixture** (200 frames, 6 slots including a referee): 8 gaps, 5 evaluable, 2 flips
  (0.4). It covers a flip, a no-flip, no last-seen, no reappearance (edge), no player (referee
  only), a referee nearest that must be skipped, and an exact 26.325 m tie won by the lower slot.
  With R = 5: 3 evaluable, 1 flip, 2 too_far. Reference: Δ 5/15/50 give 27 pairs with 2 flips,
  20 with 3, and 10 with 1 (R = 5: 27/2, 19/2, 10/1). A hand-built pairing case gives 7 pairs
  with 2 flips (5 with 2 at R = 5). On the main fixture slot 1 is always nearest, so there are
  0 flips and a reference of 0.
- **(e):** Offset 0 reproduces (b) exactly at tolerance 0 and ±1 s. The vectorised overlap
  equals a brute-force piece-by-piece check on 25 random offsets. Wrap-around: [280,295) + 10 in
  block 200-299 gives [290,300) and [200,205). Tags never cross halves, and the same seed gives
  identical statistics. Tags placed on the gaps give real 100%, null mean 8.1%,
  p_emp = 1/1001. Independently placed tags give real 20%, null mean 28.4%, p_emp 0.79.
- **Sampler:** on the main fixture, 9 of 11 gaps are not edge gaps. Thresholds are q 0.9 →
  160 frames and q 0.5 → 31. `[2, 6)` s keeps a gap of exactly 6.00 s out. `--n 3` gives 2 + 1
  rows, one half is filled when the other has none, an empty selection writes only a header,
  and the same seed gives byte-identical output. A frame with no record splits gaps instead of
  merging them.
- **Robustness:** a spec with no `endFrame` gives the same teams and flips. `--game` reaches every
  path lookup. A missing tags file skips (b) and (e) without stopping the run.

**Metrica open Sample Game 3 (NOT the competition match).** It has no Smart Tagging file, so
(b) and (e) are skipped, and it is graded whole. (a) and (c): 161 gaps, and ≥ 5 s gaps hold 96.9%
of the missing frames. (d): flip rate 0.36 (58/159). By bin, gap vs reference:

| bin | gap flips | reference |
|---|---|---|
| <0.5 s | 19/51 | 0.03 at Δ = 1 |
| 0.5-2 s | 3/8 | 0.32 |
| 2-5 s | 4/11 | 0.35 |
| ≥ 5 s | 32/89 | 0.42 |

This sample's ball looks blanked when out of play. That is no prior for the competition's
uncorrected tracking.

## 6. Which decision each result informs

- **(a) Where do the missing frames sit?** Many short gaps holding few frames: interpolate them
  and focus elsewhere.
- **(b)/(e) Are long gaps stoppages?** Read (b) only against (e). The overlap supports
  "stoppage → D, tags as restart anchors" only if real is above the null band (p_emp small). If
  it is inside the band, set pieces are no more aligned with gaps than chance; treat long gaps as
  occlusions and carry possession through. A big change at ±1 s means timing noise dominates.
- **(c) Contested or not?** Gap starts more crowded than visible frames means the ball is lost in
  contested play.
- **(d) Can the label be carried forward?** A low flip rate in a bin, near or below the
  reference, means carry-forward is safe at that length. A high rate, clearly above the
  reference, means we need a guess or a team-shape signal through those gaps. If R = 5 changes
  the picture, "nearest" is often not "in control".

## 7. Questions for the user (qualitative only; no numbers or counts)

1. Did line 1 show the published graded-frame count with no "no tracking record" note? Did
   `check_channels` declare any status or owner channel?
2. Which bin holds most gaps, and which holds most missing frames? Do the halves look alike?
3. Which bins flip clearly more than the reference, if any? Which look about the same?
4. Is the flip rate fairly flat across bins, or does it rise with gap length? Did
   `--flip-max-dist 5` change that picture or leave it the same?
5. Were many gaps not evaluable? Which reason dominated?
6. Was the real set-piece overlap inside, above or below the null band? Did ±1 s change that? Was
   it different for 2-5 s and ≥ 5 s?
7. Are gap starts more, equally or less crowded than visible frames?
8. Watching the sampled long gaps: what did they mostly look like (ball out of play, occlusion
   in a crowd, high ball, other)? Was that different between halves?

## 8. Moving `eda/` into the team repo

Copy the whole folder: scripts, both `.md` files, `.gitignore`, `tests/` and the synthetic
`tests/fixtures/` (about 490 KB). Leave out caches, output folders (`~/eda_out`), venvs, sample
data and anything from the VM. The scripts refuse to write inside the repo, and `.gitignore`
covers `out/`, `eda_out/`, `*_out/`, `*.csv` (except fixtures) and caches. The tests need numpy
and pytest, not kloppy.

```bash
rsync -a --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.Rhistory' eda/ /path/to/team-repo/eda/
```

```bash
cd /path/to/team-repo/eda && python -m pytest -q -p no:cacheprovider tests
```
