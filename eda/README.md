# eda: exploratory checks on the ATD ball track

Four small scripts for the Metrica "More than a Hack 2027" ball-possession task. They print a
few aggregate lines and write one file each. Only `sample_long_gaps.py` writes frame-indexed
rows, a short list of gaps for video review. Like every output, it stays out of the repository
(`*.csv` is ignored).

| script | what it does | writes |
|---|---|---|
| `check_channels.py` | lists the channels and data-format specs the ATD metadata XML declares, and whether any looks like a ball-status / ball-owner channel | `channels.txt` |
| `ball_gaps.py` | profiles ball-missing gaps on graded frames: (a) length bins, (b) overlap of gaps of 2 s or more with Smart Tagging SET PIECES, (c) players within 5 m of the last-seen ball against a seeded sample of ball-visible frames, (d) whether the nearest player's team changes across a gap, against visible frame pairs the same distance apart, (e) the (b) overlap against a circular-shift null | `ball_gaps_summary.csv` |
| `sample_long_gaps.py` | samples a few long ball gaps (not cut by a half edge) spread over both halves, with video seek times, for watching by hand; needs `ball_gaps.py` next to it | `long_gap_sample.csv` (frame-indexed; never committed) |
| `tag_gaps.py` | lines gaps up with every Smart Tagging code and set-piece `Type`: (A) P(ball missing) per 1 s bin around each tag's start and end, (B) per gap-length band, how often gaps overlap a code or end within 5 s of its start, (C) for "gap >= L", proxy precision / recall against SET PIECES starts and the share of frames covered; A and B against a circular-shift null; needs `ball_gaps.py` next to it | `tag_gaps_summary.csv` |

Needs Python 3.8+ and numpy. `games.py` comes via `PYTHONPATH` (the team repo root has a copy). Match
files come from `games.asset_paths()`, which honours `MTAH_DATA_DIR`. The output folder is
always the first argument and must be outside the repository holding the scripts (they
refuse otherwise); use `~/eda_out/...`. Definitions and interpretation notes: `EDA_HANDOFF.md`.

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

`MTAH_DATA_DIR` is already set there. Run from the repository root; outputs go to
`~/eda_out/`, outside the repository.

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/check_channels.py "$HOME/eda_out/channels"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_gaps.py "$HOME/eda_out/ball_gaps"
```

`ball_gaps.py` uses `games.half_video_blocks("BAR-ATH")` for the graded frames and the tactical
Smart Tagging XML for (b) and (e), and the tactical ATD metadata for the slot teams in (d).
If the Smart Tagging file is missing, it prints a note and skips (b) and (e).
Optional sensitivity runs, each to its own folder:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_gaps.py "$HOME/eda_out/ball_gaps_tol1s" --tag-tolerance 1
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/ball_gaps.py "$HOME/eda_out/ball_gaps_maxdist5" --flip-max-dist 5
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/check_channels.py "$HOME/eda_out/channels_broadcast" --source broadcast
```

Long-gap sample for video review. By default it takes 8 gaps at or above the 90th-percentile
length and excludes the public GT window and the second-half frames before kickoff
(`--exclude 11185-18684,110808-113741`):

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/sample_long_gaps.py "$HOME/eda_out/long_gaps"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/sample_long_gaps.py "$HOME/eda_out/long_gaps_2to6" --min-s 2 --max-s 6
```

Gaps vs Smart Tagging codes (`tag_gaps.py`). Without a Smart Tagging
file it reports only the gap totals and skips views A-C. Sensitivity runs: a tighter restart
link, and a ±1 s tag tolerance for the overlap view:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/tag_gaps.py "$HOME/eda_out/tag_gaps"
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/tag_gaps.py "$HOME/eda_out/tag_gaps_link2" --link-window 2
```

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/tag_gaps.py "$HOME/eda_out/tag_gaps_tol1s" --tag-tolerance 1
```

### Notebook (`eda_vm.ipynb`)

The notebook lists every run command above as a comment, including the five long-gap strata
(`gap_00_05_sample.csv` ... `gap_35plus_sample.csv` in `~/eda_out/long_gaps`, made with
`--min-s/--max-s/--csv-name`). Under each command it reads the outputs from `EDA_OUT`
(default `~/eda_out`) and plots them, or shows the sample tables. Section 4 draws the
`tag_gaps.py` charts: profile small multiples, a lift heatmap and the threshold curves. It finds the public repo through
`MTAH_PUBLIC_REPO` (default `~/more-than-a-hack-2027`) and the data through `MTAH_DATA_DIR`.
The committed copy has been run on the VM, so its outputs (aggregates, plots and the
long-gap sample tables) are competition-derived. `eda_vm.pdf` and `eda_vm.docx` are Quarto renders
of it. Never let a cell print raw tracking rows. To share a copy without outputs, first run
`jupyter nbconvert --clear-output --inplace eda_vm.ipynb`. It needs pandas, matplotlib and Jupyter on the VM.

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
