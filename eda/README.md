# eda: exploratory checks on the ATD ball track

Two small scripts for the Metrica "More than a Hack 2027" ball-possession task. They print a few
aggregate lines and write one small summary file each; never per-frame rows.

| script | what it does | writes |
|---|---|---|
| `check_channels.py` | lists the channels and data-format specs the ATD metadata XML declares, and whether any looks like a ball-status / ball-owner channel | `channels.txt` |
| `ball_gaps.py` | profiles ball-missing gaps on graded frames: (a) length bins, (b) overlap of gaps of 2 s or more with Smart Tagging SET PIECES, (c) players within 5 m of the last-seen ball against a seeded sample of ball-visible frames, (d) whether the nearest player's team changes across a gap, against visible frame pairs the same distance apart, (e) the (b) overlap against a circular-shift null | `ball_gaps_summary.csv` |

Needs Python 3.8+ and numpy. `games.py` comes from the public repo via `PYTHONPATH`. Match
files come from `games.asset_paths()`, which honours `MTAH_DATA_DIR`. The output folder is
always the first argument and must be outside the repository holding the scripts (they
refuse otherwise); use `~/eda_out/...`. Interpretation notes: `EDA_HANDOFF.md`.

## Copy to the VM (from the laptop, in the folder that contains `eda/`)

Scripts and this README only; tests and fixtures stay on the laptop.

```bash
ssh mtah 'mkdir -p ~/eda' && scp eda/check_channels.py eda/ball_gaps.py eda/README.md mtah:eda/
```

## Run on the VM (competition match, tactical recording)

`MTAH_DATA_DIR` is already set there. Outputs go to `~/eda_out/`, outside `eda/`.

```bash
cd ~/eda && PYTHONPATH="$HOME/more-than-a-hack-2027" python3 check_channels.py "$HOME/eda_out/channels"
```

```bash
cd ~/eda && PYTHONPATH="$HOME/more-than-a-hack-2027" python3 ball_gaps.py "$HOME/eda_out/ball_gaps"
```

`ball_gaps.py` uses `games.half_video_blocks("BAR-ATH")` for the graded frames and the tactical
Smart Tagging XML for (b) and (e), and the tactical ATD metadata for the slot teams in (d).
Optional sensitivity runs, each to its own folder:

```bash
cd ~/eda && PYTHONPATH="$HOME/more-than-a-hack-2027" python3 ball_gaps.py "$HOME/eda_out/ball_gaps_tol1s" --tag-tolerance 1
```

```bash
cd ~/eda && PYTHONPATH="$HOME/more-than-a-hack-2027" python3 ball_gaps.py "$HOME/eda_out/ball_gaps_maxdist5" --flip-max-dist 5
```

```bash
cd ~/eda && PYTHONPATH="$HOME/more-than-a-hack-2027" python3 check_channels.py "$HOME/eda_out/channels_broadcast" --source broadcast
```

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

## Tests (laptop only)

```bash
cd eda && python -m pytest -q -p no:cacheprovider tests
```

The fixtures in `tests/fixtures/` are synthetic. `python tests/make_fixtures.py` rebuilds them,
and `tests/fixtures/README.md` lists the expected answers.
