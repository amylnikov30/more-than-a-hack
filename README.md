# more-than-a-hack

Frame-level ball possession for **More than a Hack 2027, Phase 1** (FC Barcelona v Athletic Club,
tactical video). The pipeline writes one `frame,label` CSV of 199,319 rows: `D` where the ball is
dead, `H` or `A` for the team in possession on live frames, `X` off the graded halves.

```
tracking (ATD) ─► dead-ball labeller (src/deadball) ─► live spells ─► H/A model (src/possession) ─► <team>.csv
                                                                  └► fallback rule (src/pipeline/rule.py)
```

| Part | Code | Write-up |
|---|---|---|
| Dead ball (`D`) | `src/deadball/` | [docs/dead_ball_rule.md](docs/dead_ball_rule.md) |
| Possession (`H` / `A`) | `src/possession/` | [docs/ha_model.md](docs/ha_model.md), spec [docs/possession_HA_proposal.md](docs/possession_HA_proposal.md) |
| Fallback rule, whole-chain validation, submission | `src/pipeline/` | this file |
| Exploratory analysis | `eda/` | `eda/README.md` |

No hand labels. Both models are trained on Metrica's open sample games (other matches), with
labels derived by code from Metrica's own event and tracking data. Nothing is trained or tuned on
the competition match or its public 5-minute ground truth.

## Install

Python 3.10 (the VM's version), dependencies pinned:

```bash
python3 -m venv ~/mtah-venv
~/mtah-venv/bin/pip install -r requirements.txt            # numpy, pandas, kloppy
~/mtah-venv/bin/pip install pytest==9.2.0                  # only for `make test`
```

Use that interpreter for every command below (`make ... PY=~/mtah-venv/bin/python`, or activate
the venv).

## Data

The match data is not in this repository and must not be. Put the organisers' bundle anywhere
and point `MTAH_DATA_DIR` at it:

```
$MTAH_DATA_DIR/
  tactical/
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_smart_tagging.xml
    (BAR-ATH_tactical.mp4, BAR-ATH_tactical_homography.csv: not used)
```

For training and validation only (open data, other matches):

```bash
git clone https://github.com/metrica-sports/sample-data ~/sample-data
git clone https://github.com/metrica-sports/more-than-a-hack-2027 ~/more-than-a-hack-2027
export MTAH_PUBLIC_REPO=~/more-than-a-hack-2027    # the official scorer, validator and baseline
```

## Regenerate the submission

```bash
export MTAH_DATA_DIR=/path/to/data MTAH_PUBLIC_REPO=~/more-than-a-hack-2027
make run TEAM=<team_name> PY=~/mtah-venv/bin/python
```

This is `python -m pipeline.make_submission ~/mtah_out/submission --team <team_name>`. It:

1. runs the dead-ball labeller and writes `dead_ball.csv`;
2. runs the H/A system named in `src/pipeline/config.py` (`SHIP`): the model
   (`possession.run`) or the fallback rule (`pipeline.rule`);
3. writes `~/mtah_out/submission/<team_name>.csv` and checks it: header, 199,319 rows,
   frames in order, labels in `H/A/D/X`, `X` exactly off the graded halves, `D` before the
   second-half kickoff, then the organisers' `validate_submission.py`;
4. prints label-free aggregates next to the reference-set values, and the file's sha256.

It uses the committed model files (`src/deadball/player_model.json`,
`src/possession/ha_model.json`) and no random numbers, so every run gives the same bytes.
Outputs are frame-indexed, so every command refuses an output folder inside the repository.

## Retrain from open data (optional)

The committed model files are what the submission uses. To rebuild them:

```bash
make reference PY=...     # possession.reference: degraded Sample Game 3 + event-derived labels
make train PY=...         # deadball.train_players + possession.train --write
```

Retraining on another machine reproduces the H/A weights to about 1e-14 (floating-point
summation order), not to the byte. The submission depends only on the committed JSON files.

## Validate on the reference set

```bash
make reference PY=...
make evaluate PY=...      # pipeline.evaluate_chain, 1000 bootstrap draws
```

`evaluate_chain` runs the whole chain on degraded Sample Game 3: the dead-ball labeller on the
degraded tracking, then H/A on its live spells, scored by the organisers' `run_evaluation.py`.
It compares the model (fitted on the other half), the fallback rule (K chosen on the other
half) and `baseline_smooth50`. The model ships only if its spellF1 interval lies wholly above the
rule's. Results are in `src/pipeline/reference_summary.json` and in the table below.

Whole chain on degraded Sample Game 3 (382 true possessions; 90% block-bootstrap interval over
possessions, 1000 draws):

| System | spellF1 | 90% interval | macroF1 | Team acc. | Possessions (true 382) |
|---|---|---|---|---|---|
| **H/A model** (weights fitted on the other half) | **0.491** | 0.448–0.540 | 0.851 | 0.853 | 223 |
| Fallback rule, K = 12 frames (K chosen on the other half) | 0.373 | 0.331–0.420 | 0.801 | 0.779 | 208 |
| `baseline_smooth50` (organisers' floor) | 0.255 | 0.232–0.278 | 0.754 | 0.755 | 719 |
| *Diagnostic:* model with the true dead-ball mask | 0.613 | 0.561–0.663 | 0.933 | 0.900 | 230 |
| *Diagnostic:* rule with the true dead-ball mask | 0.514 | 0.465–0.563 | 0.894 | 0.842 | 230 |

**Decision: the model ships** (`SHIP = "model"`), because its interval lies above the rule's,
though only just (0.448 against 0.420).

How to read these numbers:

- **Dead-ball errors cost about 0.12 spellF1** (0.613 with the true mask against 0.491 with
  the labeller's). On the degraded game the labeller calls 31.6% of frames dead (truth 33.5%),
  with dead F1 0.87, but it finds 72 stoppages of 103: it misses short ones, which merges
  possessions.
- **Both H/A systems under-segment**: 3.4 possessions per live minute against 6.0 in the
  truth, median length 12.5 s against 5.1 s.
- **In-sample caveats.** The dead-ball player model was trained on Sample Games 1–3, so on this
  game it is in-sample. The shipped H/A weights were fitted on all of Sample Game 3; scored
  in-sample they give 0.494, almost the same as held out (0.491). The decoder constants (λ, m, β)
  were tuned on the same game.
- With one reference game the intervals are about ±0.05 wide, so these are coarse comparisons.

## Tests

```bash
make test PY=...          # synthetic fixtures only, no match data
```

## Runtime

Measured on a laptop (Apple silicon) with the pinned dependencies: `make reference` about 15 s,
`possession.train` about 1.5 min, `make evaluate` about 1.5 min. On the match, the dead-ball
labeller takes under a minute.

## Repository rules

- No match data, per-frame outputs or frame-indexed tables in the repository (outputs go to
  `~/mtah_out` or `~/eda_out`).
- Every constant is a stated football norm or tuned on the open-data reference set; see the
  config files `src/possession/config.py` and `src/pipeline/config.py` and the two write-ups.
- The public 5-minute ground truth is a final pipeline check, never used to choose between
  versions.
