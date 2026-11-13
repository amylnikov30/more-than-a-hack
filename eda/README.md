# eda: exploratory checks on the ATD ball track

Small scripts for the Metrica "More than a Hack 2027" ball-possession task. They print a
few aggregate lines and write one file each. Only `sample_long_gaps.py` writes frame-indexed
rows, a short list of gaps for video review. Like every output, it stays out of the repository
(`*.csv` is ignored).

## Layout

```
eda/
├── README.md          this file
├── scripts/           ball_gaps.py, ball_outside.py, ball_stationary.py, check_channels.py,
│                      common.py, gap_stoppage.py, sample_long_gaps.py, signal_profiles.py,
│                      speed_units.py, tag_gaps.py; possession: poss_common.py,
│                      poss_reference.py, poss_ball_live.py, poss_carries.py,
│                      poss_pass_chains.py
├── notebooks/         the four eda_*.ipynb + plot_helpers.py
├── reports/           the eda_*.pdf and eda_*.docx renders
├── docs/analyses.md   definitions and reading notes per analysis
└── tests/             synthetic fixtures and the test suite
```

The scripts import each other as siblings (`import ball_gaps`), so keep them together in
`scripts/`; Python puts a script's own folder on the path when you run it.

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
| `poss_common.py` | shared preparation for the `poss_*.py` scripts (no output of its own): tracking in metres, slot teams, attacking direction, live mask (graded minus the dead frames of `dead_ball.csv`), ball interpolation, Smart Tagging pseudo-labels with agreement per code, the `--degrade` copy | |
| `poss_reference.py` | reference labels from Metrica's open event data (Sample Game 3 only): dead frames and H / A per live frame, checked against the ball-blank frames of the open tracking; label counts, possessions, dead runs | `poss_reference_summary.csv` |
| `poss_carries.py` | carries (ball within 1.5 m of one player, moving with them, for 0.5 s or more): durations, share of live time, precision of "carry team = possessing team", AUC per smoothing window, timing of carry-team switches against possession changes, carries on fast balls | `poss_carries_summary.csv` |
| `poss_pass_chains.py` | passes (ball leaves one player and moves with a teammate, faster than 5 m/s in between) and chains of consecutive passes by one team: passes per team, chain lengths, live time covered, precision of "a chain of >= 2 passes by T => T has the ball" per chain length, AUC per window, timing of chain-team switches | `poss_pass_chains_summary.csv` |
| `poss_ball_live.py` | ball availability on live frames: observed / interpolated / missing shares, gaps per length bin before and after interpolation, gap-length quantiles, other interpolation caps, missing share on dead frames and within 2 s of possession changes | `poss_ball_live_summary.csv` |

Needs Python 3.8+ and numpy. `games.py` comes via `PYTHONPATH` (the team repo root has a copy). Match
files come from `games.asset_paths()`, which honours `MTAH_DATA_DIR`. The output folder is
always the first argument and must be outside the repository holding the scripts (they
refuse otherwise); use `~/eda_out/...`. Definitions: each script's docstring and
`tests/fixtures/README.md`; interpretation notes: [docs/analyses.md](docs/analyses.md).

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
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/check_channels.py "$HOME/eda_out/channels"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/check_channels.py "$HOME/eda_out/channels_broadcast" --source broadcast
```

Ball gaps (`ball_gaps.py`): the main run, then a ±1 s set-piece tolerance and a 5 m cap on the
flip distance. It uses `games.half_video_blocks("BAR-ATH")` for the graded frames, the tactical
Smart Tagging XML for (b) and (e) and the tactical ATD metadata for the slot teams in (d); without
the Smart Tagging file it prints a note and skips (b) and (e).

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_gaps.py "$HOME/eda_out/ball_gaps"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_gaps.py "$HOME/eda_out/ball_gaps_tol1s" --tag-tolerance 1
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_gaps.py "$HOME/eda_out/ball_gaps_maxdist5" --flip-max-dist 5
```

Long-gap samples for video review (`sample_long_gaps.py`), one file per length stratum in
`~/eda_out/long_gaps`. Each excludes the public GT window and the second-half frames before
kickoff (`--exclude 11185-18684,110808-113741`, the default):

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 0 --max-s 5 --csv-name gap_00_05_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 5 --max-s 10 --csv-name gap_05_10_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 10 --max-s 20 --csv-name gap_10_20_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 20 --max-s 35 --csv-name gap_20_35_sample.csv
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/sample_long_gaps.py "$HOME/eda_out/long_gaps" --min-s 35 --csv-name gap_35plus_sample.csv
```

Gaps vs Smart Tagging codes (`tag_gaps.py`): the main run, a tighter restart link and a ±1 s tag
tolerance for the overlap view. Without a Smart Tagging file it reports only the gap totals.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/tag_gaps.py "$HOME/eda_out/tag_gaps"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/tag_gaps.py "$HOME/eda_out/tag_gaps_link2" --link-window 2
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/tag_gaps.py "$HOME/eda_out/tag_gaps_tol1s" --tag-tolerance 1
```

### Dead-ball EDA (read by the three dead-ball notebooks)

Speed units (`speed_units.py`, read by `eda_extras.ipynb` sections 1 and 2, including the
ball-diagnostics cell under the section 1 chart: declared ball speed, ratio by speed band and
1 s ball displacement). It streams the whole raw file once and reads the declared unit from the
ATD metadata. Those diagnostics need a run with the current version (rows `ball_declared`,
`ratio_by_speed`, `ball_disp`); keep the default `--disp-window-s 1`, which the cell's labels
assume:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/speed_units.py "$HOME/eda_out/speed_units"
```

Ball outside the pitch (`ball_outside.py`, read by `eda_ball_outside_pitch.ipynb`): the main run
and a tighter restart link.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_outside.py "$HOME/eda_out/ball_outside"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_outside.py "$HOME/eda_out/ball_outside_link2" --link-window 2
```

Ball stationary (`ball_stationary.py`, read by `eda_ball_stationary.ipynb`): the main run, a
stricter speed threshold and a wider mark tolerance.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_stationary.py "$HOME/eda_out/ball_stationary"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_stationary.py "$HOME/eda_out/ball_stationary_still03" --still-ms 0.3
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/ball_stationary.py "$HOME/eda_out/ball_stationary_tol2" --spot-tol-m 2
```

Signals around set-piece starts (`signal_profiles.py`) and gap stoppage signals
(`gap_stoppage.py`), both read by `eda_extras.ipynb`. Both read the homography CSV through
`games.asset_paths()`; section 5 of the notebook needs `--clusters 2`.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/signal_profiles.py "$HOME/eda_out/signal_profiles" --ends
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/gap_stoppage.py "$HOME/eda_out/gap_stoppage" --clusters 2
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/gap_stoppage.py "$HOME/eda_out/gap_stoppage_walk15" --walk-ms 1.5 --clusters 2
```

### Possession signals (read by `eda_possession.ipynb`)

The `poss_*.py` scripts run on live frames, so they need the dead frames from the dead-ball
model first. It writes a frame-indexed `dead_ball.csv` outside the repository (it needs pandas,
through `src/deadball/train_players.py`):

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD:$PWD/src" python3 -m deadball.run "$HOME/eda_out/dead_ball"
```

Ball availability on live play (`poss_ball_live.py`), with the Smart Tagging pseudo-labels for
the change rows:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_ball_live.py "$HOME/eda_out/poss_ball_live" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
```

Carries (`poss_carries.py`):

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_carries.py "$HOME/eda_out/poss_carries" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
```

Pass chains (`poss_pass_chains.py`):

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_pass_chains.py "$HOME/eda_out/poss_pass_chains" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
```

The Sample Game 3 runs (clean and `--degrade`) go to `$HOME/eda_out/poss_*_sg3` and
`$HOME/eda_out/poss_*_sg3_deg`, so the notebook finds them next to the match run; their commands
are under "Run on Metrica Sample Game 3" below.

### Notebook (`eda_ball_missing.ipynb`)

The notebook lists the ball-missing run commands above as comments, including the five
long-gap strata (`gap_00_05_sample.csv` ... `gap_35plus_sample.csv` in `~/eda_out/long_gaps`).
Under each command it reads the outputs from `EDA_OUT`
(default `~/eda_out`) and plots them, or shows the sample tables. Section 4 draws the
`tag_gaps.py` charts: profile small multiples, a lift heatmap and the threshold curves. It finds the public repo through
`MTAH_PUBLIC_REPO` (default `~/more-than-a-hack-2027`) and the data through `MTAH_DATA_DIR`.
The committed copy has been run on the VM, so its outputs (aggregates, plots and the
long-gap sample tables) are competition-derived. `reports/eda_ball_missing.pdf` and `reports/eda_ball_missing.docx` are Quarto renders
of it. Never let a cell print raw tracking rows. To share a copy without outputs, first run, from `eda/`,
`jupyter nbconvert --clear-output --inplace notebooks/eda_ball_missing.ipynb`. It needs pandas, matplotlib and Jupyter on the VM.

### Dead-ball notebooks

Three notebooks read the summary CSVs of the dead-ball scripts; they are committed **without
outputs**. Run them on the VM after the scripts, from the `eda/notebooks/` folder (Jupyter's
working directory must be `eda/notebooks/` so `plot_helpers.py` is found; otherwise set
`EDA_DIR` to that folder). `EDA_OUT`
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
leaves the VM, from `eda/`: `jupyter nbconvert --clear-output --inplace notebooks/eda_*.ipynb`.

Path changes in the notebooks are not made by hand:
`tools/update_eda_notebook_paths.py` (run from the repo root) rewrites only cell sources and
leaves outputs byte-for-byte as they are.

## Run on Metrica Sample Game 3 (laptop, open data)

Every command runs from `eda/`; `--meta` and `--raw` point at your copy of Sample Game 3, and
`OUT_DIR` must be outside the repository.

```bash
python scripts/check_channels.py OUT_DIR --meta /path/to/Sample_Game_3_metadata.xml
```

```bash
python scripts/ball_gaps.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --meta /path/to/Sample_Game_3_metadata.xml --frames 0-69660,69661-143760 --pitch 105 68
```

## Options (`ball_gaps.py`)

`--raw`, `--meta` (ATD metadata, for (d); with `--raw`, (d) is skipped without it), `--tags`,
`--frames a-b,c-d` (video frames, one inclusive range per half; required with `--raw`),
`--pitch L W` (m, default 105.3 68.0), `--radius` (m, default 5), `--long-gap` (s, default 2),
`--sample-size` (default 2000), `--seed` (default 0), `--tag-tolerance` (s, default 0),
`--flip-max-dist` (m, default off), `--null-n` (default 1000), `--null-seed` (default 0),
`--null-min-shift` (s, default 60), `--fps` (default 25), `--game` (default BAR-ATH).

```bash
python scripts/sample_long_gaps.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760
```

```bash
python scripts/tag_gaps.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760
```

Sample Game 3 has no Smart Tagging file, so this only checks that the script runs (views A-C
are skipped).

```bash
python scripts/speed_units.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --meta /path/to/Sample_Game_3_metadata.xml --frames 0-69660,69661-143760 --pitch 105 68
```

Sample Game 3 has no speed field, so this reports derived speeds only and the verdict `no
declared speed`.

## Options (`speed_units.py`)

`--raw`, `--meta` (for the declared unit; default `games.asset_paths()['atd_meta']` without
`--raw`), `--frames`, `--game`, `--fps`, `--pitch L W` (m, default 105.3 68.0), `--min-ms`
(ratio floor, m/s, default 1), `--cap-player` (m/s, default 12), `--cap-ball` or
`--ball-cap-ms` (the same option: the ball's cap, used instead of the player cap; m/s, default
40), `--disp-window-s` (ball displacement window, s, default 1).

## Options (`ball_outside.py`)

`--raw`, `--tags` (with `--raw`, link and cross-tab need it), `--frames`, `--game`, `--fps`,
`--pitch L W`, `--margin-m` (main margin, m, default 1), `--margins-also` (comma-separated m,
default `0,2`), `--link-window` (s, default 5), `--null-n` (default 1000; 0 = off),
`--null-seed` (default 0), `--null-min-shift` (s, default 60).

```bash
python scripts/ball_outside.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760 --pitch 105 68
```

## Options (`ball_stationary.py`)

`--raw`, `--meta` (kick-off check), `--tags` (link and cross-tab), `--frames`, `--game`,
`--fps`, `--pitch L W`, `--lag` (frames, default 5), `--still-ms` (m/s, default 0.5),
`--min-still-s` (main, s, default 1), `--min-still-also` (comma-separated s, default `0.5,2`),
`--still-gap` (frames, default 3), `--spot-tol-m` (m, default 1), `--band-m` (m, default 2),
`--link-window` (s, default 5), `--null-n`, `--null-seed`, `--null-min-shift`.

```bash
python scripts/ball_stationary.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --meta /path/to/Sample_Game_3_metadata.xml --frames 0-69660,69661-143760 --pitch 105 68
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
python scripts/signal_profiles.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760 --pitch 105 68
```

```bash
python scripts/gap_stoppage.py OUT_DIR --raw /path/to/Sample_Game_3_tracking.txt --frames 0-69660,69661-143760 --pitch 105 68
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

## Possession scripts on Sample Game 3

Each `poss_*.py` script runs on the reference game twice: clean and with `--degrade`. `--events`
gives the true labels and the dead frames. Run from the team-repo root, with `SG3` set to your
copy of Metrica's open `sample-data/data/Sample_Game_3` folder (or the folder holding the three
files):

```bash
cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_reference.py "$HOME/eda_out/poss_reference_sg3" --raw "$SG3/Sample_Game_3_tracking.txt" --meta "$SG3/Sample_Game_3_metadata.xml" --events "$SG3/Sample_Game_3_events.json" --frames 0-69660,69661-143760 --pitch 105 68
```

```bash
cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_ball_live.py "$HOME/eda_out/poss_ball_live_sg3" --raw "$SG3/Sample_Game_3_tracking.txt" --meta "$SG3/Sample_Game_3_metadata.xml" --events "$SG3/Sample_Game_3_events.json" --frames 0-69660,69661-143760 --pitch 105 68
```

```bash
cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_ball_live.py "$HOME/eda_out/poss_ball_live_sg3_deg" --raw "$SG3/Sample_Game_3_tracking.txt" --meta "$SG3/Sample_Game_3_metadata.xml" --events "$SG3/Sample_Game_3_events.json" --frames 0-69660,69661-143760 --pitch 105 68 --degrade
```

```bash
cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_carries.py "$HOME/eda_out/poss_carries_sg3" --raw "$SG3/Sample_Game_3_tracking.txt" --meta "$SG3/Sample_Game_3_metadata.xml" --events "$SG3/Sample_Game_3_events.json" --frames 0-69660,69661-143760 --pitch 105 68
```

```bash
cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_carries.py "$HOME/eda_out/poss_carries_sg3_deg" --raw "$SG3/Sample_Game_3_tracking.txt" --meta "$SG3/Sample_Game_3_metadata.xml" --events "$SG3/Sample_Game_3_events.json" --frames 0-69660,69661-143760 --pitch 105 68 --degrade
```

```bash
cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_pass_chains.py "$HOME/eda_out/poss_pass_chains_sg3" --raw "$SG3/Sample_Game_3_tracking.txt" --meta "$SG3/Sample_Game_3_metadata.xml" --events "$SG3/Sample_Game_3_events.json" --frames 0-69660,69661-143760 --pitch 105 68
```

```bash
cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_pass_chains.py "$HOME/eda_out/poss_pass_chains_sg3_deg" --raw "$SG3/Sample_Game_3_tracking.txt" --meta "$SG3/Sample_Game_3_metadata.xml" --events "$SG3/Sample_Game_3_events.json" --frames 0-69660,69661-143760 --pitch 105 68 --degrade
```

## Options (all `poss_*.py` scripts)

`--raw`, `--meta` (required with `--raw`), `--tags` (pseudo-labels; default the match's Smart
Tagging without `--raw`), `--events` (reference game only: true labels and dead frames),
`--dead` (`dead_ball.csv`; required unless `--events`; overrides the dead frames of
`--events`), `--frames`, `--game`, `--fps`, `--pitch L W`, `--lag` (frames, default 5),
`--max-interp-frames` (default 12), `--tag-teams HOME,AWAY` (Smart Tagging team names, when the
automatic match fails), `--degrade` with `--seed` (default 0), `--missing-share` (default 0.41),
`--gap-lengths MEDIAN_S,SIGMA,CAP_S` (default `1,1.1,30`), `--frag-mean-s` (default 20),
`--frag-gap-frames` (default 5), `--windows` (AUC smoothing windows, comma-separated s, default
`0.5,1,2,5,10`; the frame itself is always added), `--timing-window-s` (default 5), and
`--null-n`, `--null-seed`, `--null-min-shift` (as above).

`poss_ball_live.py` also takes `--interp-also` (comma-separated frames, default `6,25`) and
`--change-window-s` (default 2). `poss_carries.py` takes `--carry-dist-m` (default 1.5),
`--carry-dv` (m/s, default 2), `--carry-min-s` (default 0.5) and `--fast-ball-ms` (default 15).
`poss_pass_chains.py` takes `--carry-dist-m` and `--carry-dv` (the contact test),
`--touch-min-s` (default 0.2), `--pass-min-ms` (default 5), `--pass-max-s` (default 5) and
`--min-chain` (passes, default 2).

## Tests (laptop only)

```bash
cd eda && python -m pytest -q -p no:cacheprovider tests
```

Run it from the repo root as shown; `tests/conftest.py` puts `scripts/` on `sys.path`, so the tests
import the scripts by name.

The fixtures in `tests/fixtures/` are synthetic. `python tests/make_fixtures.py` rebuilds them,
and `tests/fixtures/README.md` lists the expected answers.

## Analyses

Definitions, defaults, CSV schemas, tests and reading notes for each analysis: [docs/analyses.md](docs/analyses.md).
