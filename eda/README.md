# eda: exploratory checks on the ATD ball track

Small scripts for the Metrica "More than a Hack 2027" ball-possession task. They print a
few aggregate lines and write one file each. Only `sample_long_gaps.py` writes frame-indexed
rows, a short list of gaps for video review. Like every output, it stays out of the repository
(`*.csv` is ignored).

| script | what it does | writes |
|---|---|---|
| `check_channels.py` | lists the channels and data-format specs the ATD metadata XML declares, and whether any looks like a ball-status / ball-owner channel | `channels.txt` |
| `ball_gaps.py` | profiles ball-missing gaps on graded frames: (a) length bins, (b) overlap of gaps of 2 s or more with Smart Tagging SET PIECES, (c) players within 5 m of the last-seen ball against a seeded sample of ball-visible frames, (d) whether the nearest player's team changes across a gap, against visible frame pairs the same distance apart, (e) the (b) overlap against a circular-shift null | `ball_gaps_summary.csv` |
| `sample_long_gaps.py` | samples a few long ball gaps (not cut by a half edge) spread over both halves, with video seek times, for watching by hand; needs `ball_gaps.py` next to it | `long_gap_sample.csv` (frame-indexed; never committed) |
| `speed_units.py` | settles whether the declared ATD speed column is km/h or m/s: declared / derived speed ratio (median, IQR), speed percentiles for players and ball, a one-line verdict | `speed_units_summary.csv` |
| `ball_outside.py` | clause a: share of ball-visible frames out of bounds at several margins (goal line vs touchline), run lengths, what follows a run (ball missing / back inside / half end), restart link to set-piece starts with a circular-shift null, line x Type cross-tab | `ball_outside_summary.csv` |
| `ball_stationary.py` | clause c: still-ball runs (derived speed below a threshold for a minimum time) classified by pitch mark (centre spot, penalty spot, corner arc, goal area, touchline band, elsewhere); counts, frame share, lengths, restart link with a circular-shift null, location x Type cross-tab, kick-off check | `ball_stationary_summary.csv` |
| `signal_profiles.py` | generalises `tag_gaps.py` view A to six per-frame signals (ball missing, ball outside, ball still, share of slow players, homography reprojection error, tracked slots), profiled around set-piece starts (and ends) per `Type` with a circular-shift null | `signal_profiles_summary.csv` |
| `gap_stoppage.py` | clause d: per ball-missing gap, mean slow-player share, reprojection error and tracked slots; only medians and quartiles per gap-length bin, plus optional k-means sizes and centres | `gap_stoppage_summary.csv` |
| `tag_gaps.py` | lines gaps up with every Smart Tagging code and set-piece `Type`: (A) P(ball missing) per 1 s bin around each tag's start and end, (B) per gap-length band, how often gaps overlap a code or end within 5 s of its start, (C) for "gap >= L", proxy precision / recall against SET PIECES starts and the share of frames covered; A and B against a circular-shift null; needs `ball_gaps.py` next to it | `tag_gaps_summary.csv` |

Needs Python 3.8+ and numpy. `games.py` comes via `PYTHONPATH` (the team repo root has a copy). Match
files come from `games.asset_paths()`, which honours `MTAH_DATA_DIR`. The output folder is
always the first argument and must be outside the repository holding the scripts (they
refuse otherwise); use `~/eda_out/...`. Definitions: each script's docstring and
`tests/fixtures/README.md`; interpretation notes: the "Analyses" section below.

Data access (team decision): an AI assistant may see the outputs, figures and the run notebook,
but never a whole raw match file. Nothing that could reconstruct the match data (raw files,
frame-level tracking exports, clipped video) goes into the repository (`docs/info/RULES.md`).

## Set up on the VM

Each team member keeps a clone of the team repository on the VM and pulls to update it. The
repository root has its own `games.py`, so `PYTHONPATH="$PWD"` from the root is enough.

```bash
git clone https://github.com/amylnikov30/more-than-a-hack.git ~/more-than-a-hack
```

```bash
cd ~/more-than-a-hack && git pull
```

## Run on the VM (competition match, tactical recording)

`MTAH_DATA_DIR` is already set there. Every command runs from the team-repo root
(`~/more-than-a-hack`; change the `cd` if your clone is elsewhere) and writes to `~/eda_out/`,
outside the repository. These are exactly the runs the notebooks read. Each command writes its own
folder (the five long-gap strata share `long_gaps`, one CSV each), and no script reads another's
output, so the order does not matter.
The commented commands in `eda_ball_missing.ipynb` write `python`; use whichever of `python` and
`python3` has numpy on the VM.

### Ball-missing EDA (read by `eda_ball_missing.ipynb`)

Channels declared in the ATD metadata, tactical and broadcast:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/check_channels.py "$HOME/eda_out/channels"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/check_channels.py "$HOME/eda_out/channels_broadcast" --source broadcast
```

Ball gaps (`ball_gaps.py`): the main run, then a ±1 s set-piece tolerance and a 5 m cap on the
flip distance. It uses `games.half_video_blocks("BAR-ATH")` for the graded frames, the tactical
Smart Tagging XML for (b) and (e) and the tactical ATD metadata for the slot teams in (d); without
the Smart Tagging file it prints a note and skips (b) and (e).

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_gaps.py "$HOME/eda_out/ball_gaps"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_gaps.py "$HOME/eda_out/ball_gaps_tol1s" --tag-tolerance 1
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_gaps.py "$HOME/eda_out/ball_gaps_maxdist5" --flip-max-dist 5
```

Long-gap samples for video review (`sample_long_gaps.py`), one file per length stratum in
`~/eda_out/long_gaps`. Each excludes the public GT window and the second-half frames before
kickoff (`--exclude 11185-18684,110808-113741`, the default):

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 0 --max-s 5 --csv-name gap_00_05_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 5 --max-s 10 --csv-name gap_05_10_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 10 --max-s 20 --csv-name gap_10_20_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 20 --max-s 35 --csv-name gap_20_35_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 35 --csv-name gap_35plus_sample.csv
```

Gaps vs Smart Tagging codes (`tag_gaps.py`): the main run, a tighter restart link and a ±1 s tag
tolerance for the overlap view. Without a Smart Tagging file it reports only the gap totals.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/tag_gaps.py "$HOME/eda_out/tag_gaps"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/tag_gaps.py "$HOME/eda_out/tag_gaps_link2" --link-window 2
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/tag_gaps.py "$HOME/eda_out/tag_gaps_tol1s" --tag-tolerance 1
```

### Dead-ball EDA (read by the three dead-ball notebooks)

Speed units (`speed_units.py`, read by `eda_extras.ipynb`). It streams the whole raw file once
and reads the declared unit from the ATD metadata:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/speed_units.py "$HOME/eda_out/speed_units"
```

Ball outside the pitch (`ball_outside.py`, read by `eda_ball_outside_pitch.ipynb`): the main run
and a tighter restart link.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_outside.py "$HOME/eda_out/ball_outside"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_outside.py "$HOME/eda_out/ball_outside_link2" --link-window 2
```

Ball stationary (`ball_stationary.py`, read by `eda_ball_stationary.ipynb`): the main run, a
stricter speed threshold and a wider mark tolerance.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_stationary.py "$HOME/eda_out/ball_stationary"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_stationary.py "$HOME/eda_out/ball_stationary_still03" --still-ms 0.3
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_stationary.py "$HOME/eda_out/ball_stationary_tol2" --spot-tol-m 2
```

Signals around set-piece starts (`signal_profiles.py`) and gap stoppage signals
(`gap_stoppage.py`), both read by `eda_extras.ipynb`. Both read the homography CSV through
`games.asset_paths()`; section 5 of the notebook needs `--clusters 2`.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/signal_profiles.py "$HOME/eda_out/signal_profiles" --ends
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/gap_stoppage.py "$HOME/eda_out/gap_stoppage" --clusters 2
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/gap_stoppage.py "$HOME/eda_out/gap_stoppage_walk15" --walk-ms 1.5 --clusters 2
```

### Notebook (`eda_ball_missing.ipynb`)

The notebook lists the ball-missing run commands above as comments, including the five
long-gap strata (`gap_00_05_sample.csv` ... `gap_35plus_sample.csv` in `~/eda_out/long_gaps`).
Under each command it reads the outputs from `EDA_OUT`
(default `~/eda_out`) and plots them, or shows the sample tables. Section 4 draws the
`tag_gaps.py` charts: profile small multiples, a lift heatmap and the threshold curves. It finds the public repo through
`MTAH_PUBLIC_REPO` (default `~/more-than-a-hack-2027`) and the data through `MTAH_DATA_DIR`.
The committed copy has been run on the VM, so its outputs (aggregates, plots and the
long-gap sample tables) are competition-derived. `eda_ball_missing.pdf` and `eda_ball_missing.docx` are Quarto renders
of it. Never let a cell print raw tracking rows. To share a copy without outputs, first run
`jupyter nbconvert --clear-output --inplace eda_ball_missing.ipynb`. It needs pandas, matplotlib and Jupyter on the VM.

### Dead-ball notebooks

Three notebooks read the summary CSVs of the dead-ball scripts; they are committed **without
outputs**. Run them on the VM after the scripts, from the `eda/` folder (Jupyter's working
directory must be `eda/` so `plot_helpers.py` is found; otherwise set `EDA_DIR`). `EDA_OUT`
defaults to `~/eda_out`. Each section asks one question, lists its run command as a comment,
draws one chart, explains how to read it and leaves a "Takeaway" to fill in.

| notebook | reads (folders under `EDA_OUT`) |
|---|---|
| `eda_ball_outside_pitch.ipynb` | `ball_outside`, `ball_outside_link2` |
| `eda_ball_stationary.ipynb` | `ball_stationary`, `ball_stationary_still03`, `ball_stationary_tol2` |
| `eda_extras.ipynb` | `speed_units`, `signal_profiles`, `gap_stoppage`, `gap_stoppage_walk15` |

Missing sensitivity folders are skipped with a note. `plot_helpers.py` holds the shared look:
one colour per set-piece `Type` in every notebook, dark ink for observed values, the null as a
light grey p5-p95 band with a dashed mean, p-values at the resolution floor written as
"p ≤ 0.001", and grey "n < 5" stubs instead of values from fewer than 5 cases. Before a run copy
leaves the VM: `jupyter nbconvert --clear-output --inplace eda_*.ipynb`.

## Run on Metrica Sample Game 3 (laptop, open data)

```bash
python check_channels.py OUT_DIR --meta /path/to/Sample_Game_3_metadata.xml
```

```bash
python ball_gaps.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --meta /path/to/Sample_Game_3_metadata.xml --frames 0-69660,69661-143760 --pitch 105 68
```

## Options (`ball_gaps.py`)

`--raw`, `--meta` (ATD metadata, for (d); with `--raw`, (d) is skipped without it), `--tags`,
`--frames a-b,c-d` (video frames, one inclusive range per half; required with `--raw`),
`--pitch L W` (m, default 105.3 68.0), `--radius` (m, default 5), `--long-gap` (s, default 2),
`--sample-size` (default 2000), `--seed` (default 0), `--tag-tolerance` (s, default 0),
`--flip-max-dist` (m, default off), `--null-n` (default 1000), `--null-seed` (default 0),
`--null-min-shift` (s, default 60), `--fps` (default 25), `--game` (default BAR-ATH).

```bash
python sample_long_gaps.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760
```

```bash
python tag_gaps.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760
```

Sample Game 3 has no Smart Tagging file, so this only checks that the script runs (views A-C
are skipped).

```bash
python speed_units.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --meta /path/to/Sample_Game_3_metadata.xml --frames 0-69660,69661-143760 --pitch 105 68
```

Sample Game 3 has no speed field, so this reports derived speeds only and the verdict `no
declared speed`.

## Options (`speed_units.py`)

`--raw`, `--meta` (for the declared unit; default `games.asset_paths()['atd_meta']` without
`--raw`), `--frames`, `--game`, `--fps`, `--pitch L W` (m, default 105.3 68.0), `--min-ms`
(ratio floor, m/s, default 1), `--cap-player` (m/s, default 12), `--cap-ball` (m/s, default 40).

## Options (`ball_outside.py`)

`--raw`, `--tags` (with `--raw`, link and cross-tab need it), `--frames`, `--game`, `--fps`,
`--pitch L W`, `--margin-m` (main margin, m, default 1), `--margins-also` (comma-separated m,
default `0,2`), `--link-window` (s, default 5), `--null-n` (default 1000; 0 = off),
`--null-seed` (default 0), `--null-min-shift` (s, default 60).

```bash
python ball_outside.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760 --pitch 105 68
```

## Options (`ball_stationary.py`)

`--raw`, `--meta` (kick-off check), `--tags` (link and cross-tab), `--frames`, `--game`,
`--fps`, `--pitch L W`, `--lag` (frames, default 5), `--still-ms` (m/s, default 0.5),
`--min-still-s` (main, s, default 1), `--min-still-also` (comma-separated s, default `0.5,2`),
`--still-gap` (frames, default 3), `--spot-tol-m` (m, default 1), `--band-m` (m, default 2),
`--link-window` (s, default 5), `--null-n`, `--null-seed`, `--null-min-shift`.

```bash
python ball_stationary.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --meta /path/to/Sample_Game_3_metadata.xml --frames 0-69660,69661-143760 --pitch 105 68
```

## Options (`signal_profiles.py`)

`--raw`, `--tags` (profiles need it), `--homography` (`reproj_error` needs it), `--frames`,
`--game`, `--fps`, `--pitch L W`, `--lag` (frames, default 5), `--walk-ms` (m/s, default 2),
`--margin-m` (m, default 1), `--still-ms` (m/s, default 0.5), `--min-still-s` (s, default 1),
`--window` (s, default 30), `--bin` (s, default 1), `--ends` (also profile around tag ends),
`--null-n` (default 1000; 0 = off), `--null-seed`, `--null-min-shift` (s, default 60).

## Options (`gap_stoppage.py`)

`--raw`, `--homography`, `--frames`, `--game`, `--fps`, `--pitch L W`, `--lag` (frames,
default 5), `--walk-ms` (m/s, default 2), `--clusters` (k, default 0 = off; needs the
homography), `--seed` (default 0).

```bash
python signal_profiles.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760 --pitch 105 68
```

```bash
python gap_stoppage.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760 --pitch 105 68
```

Sample Game 3 has no Smart Tagging and no homography: `signal_profiles.py` reports only the
baselines, and `gap_stoppage.py` leaves `reproj_error` empty and skips the clusters.

## Options (`sample_long_gaps.py`)

`--n` (default 8), `--quantile` (default 0.9; eligible = length >= that quantile of the gap
lengths), `--min-s`/`--max-s` (s, half-open `[min, max)`; replaces the quantile),
`--exclude a-b,...` (drop gaps overlapping these inclusive video-frame ranges; default without
`--raw` is the two ranges above, with `--raw` none), `--seed` (default 2027), `--raw`, `--frames`,
`--game`, `--fps`, `--csv-name` (file name inside OUT_DIR, default
`long_gap_sample.csv`). Sample sizes per half are as even as the eligible gaps allow.

## Options (`tag_gaps.py`)

`--raw`, `--tags` (with `--raw`, views A-C need it), `--frames`, `--game`, `--fps`,
`--window` (A: s either side of the anchor, default 30), `--bin` (A: s, default 1; the window
must be a multiple of it), `--link-window` (B, C: s, default 5), `--tag-tolerance` (B overlap:
s, default 0), `--thresholds` (C: comma-separated s, default
`0,0.5,1,2,3,5,7,10,15,20,30,45,60`), `--null-n` (default 1000; 0 = off), `--null-seed`
(default 0), `--null-min-shift` (s, default 60).

## Tests (laptop only)

```bash
cd eda && python -m pytest -q -p no:cacheprovider tests
```

The fixtures in `tests/fixtures/` are synthetic. `python tests/make_fixtures.py` rebuilds them,
and `tests/fixtures/README.md` lists the expected answers.

## Analyses

One subsection per analysis added for the dead-ball rule: definitions, defaults and why, CSV
schema, tests, and the questions to ask when reading the results.

### Speed units (`speed_units.py`)

**Question.** The ATD metadata declares speed in kph, `DATA.md` says m/s. Which is it? Later
scripts sidestep the answer by deriving speed from positions, but anyone who uses the declared
column needs it.

**Definitions.** A pair is two consecutive video frames graded in the same half, with both
records in the file and the object's x and y present on both. Derived speed = displacement in
metres (normalised x, y times the pitch size) x fps. Declared speed is the third field of each
`x,y,speed` group on the later frame. Ratio = declared / derived, on pairs with derived speed in
`[--min-ms, cap]`. Player pairs compare the same slot position in both lines; a step where the
slot count changes is skipped for players (counted as `skipped_player_steps`).

**Defaults and why.** `--min-ms 1`: below about 1 m/s a few centimetres of position noise per
frame changes the ratio a lot. `--cap-player 12`: about the fastest sprint in football; faster
steps are slot swaps. `--cap-ball 40`: a very hard shot; faster steps are ball-track jumps.
Verdict: median ratio within +-10% of 1 -> `m/s`, of 3.6 -> `km/h`, else `unclear`; the overall
verdict needs players and ball to agree (`conflict` otherwise).

**CSV schema** (`speed_units_summary.csv`, long format as elsewhere):

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| params | all / players / ball / half h | | fps, pitch_length, pitch_width, min_ms, cap, first_frame, last_frame | |
| meta | all | | declared_speed_unit (text), skipped_player_steps | |
| ratio | players / ball | | n_pairs, n_over_cap, n_ratio | count |
| ratio | players / ball | median / p25 / p75 | declared_over_derived | ratio |
| speed | players / ball | p50 / p99 / p99.9 | derived (m/s), declared (declared unit); n_declared | |
| verdict | players / ball / overall | | verdict (text) | |

Derived percentiles use every pair with no cap, so slot swaps show up in the p99.9 tail; the
`n_over_cap` row says how many there are.

**Tests.** `tests/test_speed_units.py` on `fixture_speed_atd.*` (12 frames, declared in km/h
while the metadata says m/s): pair counts, the swap over the cap, ratio quartiles, percentiles,
the halftime step, a file without a speed field, a missing line.

**Questions for interpretation.**
- Is the IQR of the ratio narrow? A wide IQR around 3.6 suggests the declared speed is smoothed
  over several frames, so it lags the frame-to-frame derived speed.
- Do players and ball agree? A ball ratio far from the player ratio would mean the ball speed is
  computed differently (e.g. in 3D).
- How many pairs exceed the cap? Many player swaps point to tracklet switches that a speed-based
  rule (clause d) must guard against.

### Ball outside the pitch (`ball_outside.py`, clause a)

**Question.** When the tracked ball is visible beyond a line, is that a stoppage, and does a
restart follow? Clause a says "ball visible and outside the pitch (with a margin) -> dead".

**Definitions.** Metres beyond the goal lines `ex = max(-x, x - 1) x length`, beyond the
touchlines `ey = max(-y, y - 1) x width`. A ball-visible graded frame is out at margin M if
`max(ex, ey) > M`; its line is the goal line if `ex >= ey`. A run is a maximal stretch of out
frames in one half (a missing ball ends it); its line is the line of its first frame. What
follows a run: `half_end` if it ends on the half's last frame, `missing` if the ball is missing
on the next frame, else `inside`. A run links to a group (SET PIECES or one Type) if a start of
that group in the same half is within W of the run's last frame (either side). The cross-tab
Type is the nearest SET PIECES start within W.

**Defaults and why.** `--margin-m 1`: ball positions near the line carry about a metre of
error, and the ball is out only once wholly over the line (radius 0.11 m), so a 1 m margin
avoids flagging balls on the line. 0 and 2 m show how much depends on it. `--link-window 5`:
the same restart window as `tag_gaps.py`. Null: the `tag_gaps.py` link null, anchors shifted
circularly within each half, at least 60 s.

**CSV schema** (`ball_outside_summary.csv`). `scope` is `m<margin>:<line>` with line `all`,
`goal_line` or `touchline`.

| analysis | bin | metric | unit |
|---|---|---|---|
| params | main / also / | fps, pitch_length, pitch_width, margin, link_window, null_*, first/last_frame | |
| tags | | n_tags (scope = group) | count |
| share | | visible_frames, out_frames, share | frames, share of frames |
| runs | length bin | n_runs, frames | count, frames |
| follows | `` / missing / inside / half_end | n_runs, share | count, share of runs |
| link | group (`code:SET PIECES`, `type:<Type>`) | n_runs, real_count, real_share, null_mean, null_p5, null_p95, p_emp, lift | |
| crosstab | Type, `none` or `(no Type)` | n_runs | count |

**Tests.** `tests/test_ball_outside.py` on `fixture_outside_*` (2,000 frames, 9 runs over both
lines, a corner, a run split by a missing frame, one at the half end): counts at 0, 1 and 2 m,
lengths, follows, links per Type and line, the cross-tab, offset-0 null, seed stability, no tags,
and `common.find_runs` against `ball_gaps.find_gaps`.

**Questions for interpretation.**
- How often is the ball tracked outside at all? On Sample Game 3 it is rare (117 of 97,436
  visible frames at 1 m), which suggests the tracker drops the ball once it leaves the pitch. If
  the match looks the same, clause a covers little time and the out-then-missing pattern
  (`follows: missing`) is the useful part.
- Is the link lift well above 1, and higher for goal lines with corners/goal kicks and for
  touchlines with throw-ins? That supports clause a and its restart typing.
- How much do the shares move between 0, 1 and 2 m? A big drop from 0 to 1 m means most "out"
  frames are line jitter, not stoppages.

### Ball stationary (`ball_stationary.py`, clause c)

**Question.** Does a visible, motionless ball sit on a restart mark before set pieces? Clause c
says "ball visible and stationary for >= T s at a restart location -> dead".

**Definitions.** Speed on frame f is the ball's displacement from f - lag to f, in metres,
x fps / lag, defined only if the ball is visible on both frames and both are graded in the same
half. A window with speed below `--still-ms` marks all its frames; a frame is still if it is
marked and the ball is visible. A single-frame jump breaks only the windows that start or end on
it, so it does not split a run. Still stretches separated by at most `--still-gap` frames merge;
runs shorter than the minimum length are dropped. A run's location comes from its median
position, folded so both ends and both touchlines look the same. First match wins: centre spot
(within the tolerance of the centre), penalty spot (11 m from either goal line), corner arc
(1 m + tolerance from a corner), goal area (5.5 m deep, 9.16 m either side of centre, plus
tolerance), touchline band (within `--band-m` of a touchline, inside or out), elsewhere. Link,
cross-tab and null as in `ball_outside.py`, from the run's last frame. Kick-off check: a
centre-spot run whose last frame has at least one player of each team and every home player on
one side of halfway, every away player on the other (within the tolerance).

**Defaults and why.** `--lag 5` (0.2 s): ball positions jitter by centimetres per frame, which
would read as motion at 25 fps. `--still-ms 0.5`: a placed ball moves less, a rolling one more.
`--min-still-s 1` (also 0.5 and 2): a restart ball typically waits at least a second.
`--still-gap 3`: bridges a dropped frame or a jitter spike. `--spot-tol-m 1`: about the
tracking error. `--band-m 2`: throw-ins are taken from on or just off the line. The marks are
the IFAB distances in metres and do not scale with `--pitch`; only the halfway line and centre
do.

**CSV schema** (`ball_stationary_summary.csv`). `scope` is `s<min still>:<location>` with
location `all` or one of the six classes.

| analysis | bin | metric | unit |
|---|---|---|---|
| params | main / also / | fps, pitch_*, lag, still_ms, min_still, still_gap, spot_tol, band, link_window, null_*, first/last_frame | |
| overview | | graded_frames, visible_frames, still_frames (scope all) | frames |
| tags | | n_tags (scope = group) | count |
| location | | n_runs, frames, share | count, frames, share of graded frames |
| runs | `<1s` `1-2s` `2-5s` `5-10s` `>=10s` | n_runs | count |
| link | group | n_runs, real_count, real_share, null_mean, null_p5, null_p95, p_emp, lift | |
| crosstab | Type, `none` or `(no Type)` | n_runs | count |
| kickoff | half h (scope `s<m>:centre_spot`) | centre_runs, kickoff_runs | count |

**Tests.** `tests/test_ball_stationary.py` on `fixture_still_*` (2,000 frames, 10 still runs:
one per location class, a short one, a centre-spot run with a missing frame and a jitter frame,
a centre-spot run that fails the kick-off check, two runs 10 frames apart): run counts and frames
at 0.5, 1 and 2 s, lengths, the bridged run, links per location and Type, the cross-tab, the
kick-off check, symmetry of the location classes.

**Questions for interpretation.**
- How much graded time do still runs cover? On Sample Game 3 it is under 1% at 1 s (22 runs,
  most "elsewhere"), so the ball there is rarely tracked while it waits. If the match is similar,
  clause c can only label a few seconds per restart.
- Do corner-arc runs link to corner kicks, goal-area runs to goal kicks, touchline-band runs to
  throw-ins, centre-spot runs to kick-offs, with lift well above 1? That supports clause c.
- Are "elsewhere" runs linked to free kicks? If so, location matters less than stillness, and
  clause c could drop the location condition for free kicks.
- Does the kick-off check find one kick-off per half (plus one per goal)? A miss means the ball
  is not tracked at kick-off, so kick-offs need a different signal.

### Signals around set pieces (`signal_profiles.py`)

**Question.** Which per-frame signals change in the seconds before a restart, and by how much
compared with chance? This tells which signals can mark the run-up (clause b) and the stoppage
(clause d).

**Definitions.** One value per graded frame, NaN where undefined (not counted):
`ball_missing` (1/0, as `tag_gaps` view A, frames past the file count as missing);
`ball_outside` (1 if the visible ball is out by more than `--margin-m`, NaN when missing);
`ball_still` (1 if the frame is in a `ball_stationary.py` run of at least `--min-still-s`, NaN
when missing); `slow_players` (share of slots present on f and f - lag whose speed over the lag
is below `--walk-ms`); `reproj_error` (homography `reprojection_error` where `valid` is 1);
`n_tracked` (slots with a position; referees count, slots are anonymous). The profile per bin
is the pooled mean: sum over defined frames / defined frames, over all instances of the group,
with the window clipped to the anchor's half. The null shifts the group's anchors circularly
within each half (`tag_gaps` view A null). Groups: SET PIECES and each `Type`.

**Defaults and why.** `--walk-ms 2`: brisk walking; jogging starts at about 2-3 m/s. `--lag 5`
as in `ball_stationary.py`. Window, bin and null as in `tag_gaps.py`.

**CSV schema** (`signal_profiles_summary.csv`). `scope` is `<signal>:<group>`, e.g.
`slow_players:type:Corner Kick` (split on the first colon).

| analysis | bin | metric | unit |
|---|---|---|---|
| params | | fps, pitch_*, lag, walk_ms, margin, still_ms, min_still, window, bin, null_*, first/last_frame | |
| baseline | | mean, defined_frames (scope = signal) | signal unit, frames |
| tags | | n_tags, n_dropped (scope = group) | count |
| profile_start / profile_end | left edge of the bin, s | n_instances (bin ``), value, counted, null_mean, null_p5, null_p95 | signal unit (share of frames, share of players, px, slots), frames |

**Tests.** `tests/test_signals.py` on `fixture_signal_*` (400 frames, 4 slots with known
one-frame speeds, ball gaps, a homography CSV with invalid rows): per-frame signals, profiles
per Type and pooled, undefined bins. On the main fixture, `ball_missing` reproduces
`tag_gaps.view_a` exactly, observed and null, starts and ends.

**Questions for interpretation.**
- Does `slow_players` rise above its null band in the seconds before corners, free kicks and
  goal kicks? That is the clause d signal; how many seconds before the start it rises tells the
  run-up extent for clause b.
- Does `reproj_error` jump before restarts (camera cuts, close-ups, replays)? If so, it is a
  stoppage signal that does not need the ball.
- Does `n_tracked` drop before restarts? Fewer tracked players means the camera shows only part
  of the pitch, and every player-based signal is then less reliable.
- Signals whose profile sits inside the null band are not worth a rule clause.

### Gaps that look like stoppages (`gap_stoppage.py`, clause d)

**Question.** Are long ball-missing gaps stoppages (players walking, camera off the play) or
live play the tracker lost? Clause d says "gap >= L and most players at walking pace or slower
-> dead".

**Definitions.** Gaps from `ball_gaps.find_gaps`. Per gap, the mean of `slow_players`,
`reproj_error` and `n_tracked` (as in `signal_profiles.py`) over the gap's frames where defined.
Per length bin `[0,2) [2,5) [5,10) [10,20) [20,35) [35,60) [60,inf)` s: the number of gaps,
their length quartiles and, per signal, the number of gaps with a defined mean and its median and
quartiles. `--clusters k`: k-means on standardised (ln length, slow share, reprojection error),
gaps with all three defined, best of 10 seeded starts; sizes and centres only (length as the
geometric mean). No per-gap rows anywhere.

**Defaults and why.** The bins extend the `tag_gaps.py` strata to 60 s, because long gaps hold
most of the missing time. `--walk-ms 2` and `--lag 5` as in `signal_profiles.py`.

**CSV schema** (`gap_stoppage_summary.csv`).

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| params | all / half h | | fps, pitch_*, lag, walk_ms, clusters, seed, first/last_frame | |
| gap_bin | all | bin or `all` | n_gaps | count |
| gap_bin | length | bin or `all` | median, p25, p75 | s |
| gap_bin | slow_players / reproj_error / n_tracked | bin or `all` | n, median, p25, p75 | share of players, px, slots |
| cluster | all | | n_used, inertia | |
| cluster | cluster i (by length) | | size, length, slow_players, reproj_error | count, s, share of players, px |

**Tests.** `tests/test_signals.py`: per-bin quartiles of the per-gap means, empty bins, the two
clusters and their centres, no homography.

**Questions for interpretation.**
- Does the slow-player share climb with gap length? On Sample Game 3 it does: median 0.50 for
  gaps under 2 s, 0.85 at 10-20 s and 0.93 at 35-60 s, against 0.63 over all graded frames. If
  the match shows the same climb, clause d's "most players walking" is a usable test, and the
  bin where the median crosses about 0.8 suggests L.
- Is there a spread within the long bins (wide IQR)? Long gaps with a low slow share are live
  play the tracker lost: clause d must not call them dead.
- Do the clusters split into "short, fast, low error" and "long, slow, high error"? A clean
  split supports one threshold; a third group would need its own clause.
- `n_tracked` is constant (22) on Sample Game 3, so it says nothing there; on the match it may
  drop during stoppages.
