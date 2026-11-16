# H/A possession model

Tech-document section for `src/possession/`. Design: `docs/possession_HA_proposal.md` (the spec).
Every number below is an aggregate on Metrica's open Sample Game 3. None comes from the
competition match or its public ground truth.

## What it does

On every graded frame the dead-ball labeller (`src/deadball/`) marks live, the model writes `H` or
`A`. Frames the labeller marks dead stay `D`. Ungraded frames stay `X`. The model never writes `X`
on a live frame. Each live spell (a maximal run of live frames inside one half) is decoded on its
own:

```
dead_ball.csv ─► live spells ─► restart prior (β) ─┐
                                                    ├─► per-spell decoder ─► [snapping: off] ─► possession.csv
ATD ─► features ─► evidence model (clipped log-odds) ┘
```

| File | Role |
|---|---|
| `spells.py` | Hard mask from `dead_ball.csv`; live spells; D/X/H/A output |
| `features.py` | Per-frame home-minus-away features (reuses `eda/scripts/poss_*.py`) |
| `evidence.py` | Logistic model, clipped log-odds; weights in `ha_model.json` |
| `restart.py` | Who starts each live spell |
| `decode.py` | Exact per-spell decoder |
| `snap.py` | Switch snapping to control (behind `config.SNAP`) |
| `run.py` | End to end; writes `OUT_DIR/possession.csv` |
| `reference.py` | Builds the reference set (degraded sample games + event-derived labels) |
| `train.py` | Fits the evidence model, tunes λ, m and β, writes `ha_model.json` |
| `config.py` | Every constant |

## Features

All features are home minus away. Each is a **centred** mean over 2, 5 and 10 s inside the frame's
live spell: no window lags a change, and none crosses a stoppage. Features that depend on
direction use each team's attacking direction for the half. That direction is read from the
metadata (`attack_direction_first_half`, flipped for the second half) or, when the metadata
declares none, inferred from the team centroids.

| Group | Feature | Pooled AUC for "H has the ball" (degraded SG3) |
|---|---|---|
| Shape | convex-hull area | 0.92 |
| Shape | width | 0.92 |
| Shape | length | 0.84 |
| Shape | share of outfield players goal-side of the ball | 0.14–0.18 (lower for the team in possession, as expected) |
| Ball | carry team (ball within 1.5 m of a player and moving with them for ≥ 0.5 s) | 0.82–0.89 |
| Ball | pass-chain team (chains of ≥ 2 passes) | 0.75–0.78 |
| Ball | nearest team to the visible ball | 0.86–0.92 |
| Missingness | ball missing, and its products with the 2 s ball features | 0.49–0.60 |

Ball features are 0 where the ball gives no evidence. The ball is missing on 41% of live frames
in the degraded copy, the share stated in `DATA.md` for the match.

**Excluded** (from the EDA): raw ball position, team velocity, centroid advance, defensive-line
depth and nearest-team flip density. None of them separates a turnover from a contest.

## Evidence model

Logistic regression for P(H | features). Training frames are labelled live frames of the
reference set. Features are standardised with the training fold's mean and scale, and a
missing feature takes the training mean. The fit is Newton's method with a 1e-3 ridge. The
output is `ℓ_t = clip(logit P(H), −3, +3)`, so no single frame can force a switch. Held-out
frame AUC is **0.965** (half 1) and **0.952** (half 2).

## Decoder

For each live spell, choose `s_t ∈ {H = +1, A = −1}` to maximise

```
Σ_t ℓ_t · s_t  −  λ · (#switches)  +  β · prior · s_0
```

subject to every segment, first and last included, lasting at least `m` frames. An exact
dynamic programme over segment ends solves this in linear time, with deterministic tie-breaks.
A test checks it against brute force. Spells shorter than 2 s get a single team.

**Restart prior** (`prior ∈ {+1, −1, 0}`):

1. A Smart Tagging `SET PIECES` tag with a known team gives that team. A `Type` starting with
   "Defending" gives the other team. The tag is linked to the spell it restarts when it starts
   inside the stoppage before that spell, or at most 3 s into it. Precisely, the window is
   [end of the previous live spell − 3 s, spell start + 3 s]; for a half's first spell it opens
   at the half's first frame. The latest such tag wins.
   **Deviation from the spec**, which linked only tags starting within ±3 s of the spell start.
   On the match, set-piece tags start when the stoppage starts, often well before the ball is
   played again, and that rule linked 3 of 112 spells. The new rule cannot be checked against
   labels: the sample games have no Smart Tagging.
2. Otherwise, take the first frame in the spell's first 3 s where the ball is faster than 4 m/s.
   The prior is the team of the nearest player over the 5 frames before it (majority vote).
3. Otherwise, no prior.

The prior is a bonus, not a constraint, so the evidence can overrule a wrong one. The sample
games have no Smart Tagging, so the reference set exercises rule 2 only.

## Possession rules this encodes

- **Control, not contact.** Shape and carry/chain evidence is averaged over seconds. The switch
  cost and minimum segment length then stop a brief touch, a deflection or a 50/50 from becoming
  a possession.
- **A loose or contested ball stays with the last team**, unless the other team's evidence
  outweighs λ for at least `m` frames.
- **Restarts** belong to the team taking them (the restart prior).
- **Snapping** (switch moved to the new team's first *control frame*: ball seen, within 1.5 m of
  a player, slower than 4 m/s, for 5 frames in a row) is implemented and tested. It is
  **off**, because it did not improve the reference score (see below).

## Constants

| Constant | Value | Origin |
|---|---|---|
| `MIN_SPELL_S` | 2 s | football norm |
| `TAG_LINK_S` | 3 s | Smart Tagging timing looseness (slack on both ends of the stoppage window) |
| `ONSET_S` | 3 s | football norm |
| `FAST_BALL_MS` | 4 m/s | the dead-ball model's slow/played boundary |
| `PRE_KICK_FRAMES` | 5 | equals the 0.2 s speed lag |
| `WINDOWS_S` | 2, 5, 10 s | EDA windows |
| `CARRY_DIST_M`, `CARRY_DV`, `CARRY_MIN_S` | 1.5 m, 2 m/s, 0.5 s | football norms (EDA) |
| `TOUCH_MIN_S`, `PASS_MIN_MS`, `PASS_MAX_S`, `MIN_CHAIN` | 0.2 s, 5 m/s, 5 s, 2 | football norms (EDA) |
| `MAX_INTERP_FRAMES` | 12 (0.48 s) | EDA norm |
| `CLIP` | 3 | fixed in advance |
| `L2` | 1e-3 | as `src/deadball` |
| `CONTROL_DIST_M`, `CONTROL_SPEED_MS`, `CONTROL_K` | 1.5 m, 4 m/s, 5 frames | football norms |
| `SNAP_S` | 3 s | fixed in advance |
| `SNAP` | off | reference set |
| **λ** | **5** | reference set, spellF1 (grid 1–50) |
| **m** | **2 s** | reference set, spellF1 (grid 0.5–6 s) |
| **β** | **50** | reference set, spellF1 (grid 0–50) |

λ, m, β and the evidence weights are stored in `src/possession/ha_model.json` (as `deadball`
stores its decoder constants in `player_model.json`). Their grids are in `config.py`.

## Training data and validation

**Reference set.** Metrica Sample Game 3 (open data). `reference.py` degrades it to look like
our ATD, with a fixed seed (0):

- the ball is dropped in log-normal bursts until 41% of live frames are missing;
- player tracks are broken (mean 20 s) and re-slotted into 23 + 23 anonymous slots.

Possession labels are derived **by code** from Metrica's own event data
(`eda/scripts/poss_reference.py`). The team of the last controlled event (pass, carry,
recovery, shot or set piece) holds the ball. Challenges, lost balls and fouls never change it.
Dead frames come from restarts, ball-out events and half ends. No one on the team labelled
anything. No hand labels of BAR-ATH were used. The public 5-minute window was not used in any
of this.

**Protocol.** Leave one half out on Sample Game 3. Only Sample Game 3 is used: the team
decided not to add Games 1–2, although `reference.py` can load them. On each fold:

1. fit on the other half;
2. decode the held-out half with every (λ, m, β) in the grid;
3. score with the official scorer (`possession_metrics.evaluate`, `spell_f1_mean`).

The live mask is the event-derived one, so these numbers measure H/A alone. The whole chain,
with the dead-ball labeller's own errors, is measured by `src/pipeline/evaluate_chain.py`.

**Whole chain (16 Nov, `make evaluate`).** On degraded Sample Game 3 the dead-ball labeller runs on
the degraded tracking, and H/A runs on its live spells, held out by half. The 90% intervals come
from 1000 bootstrap draws.

| System | spellF1 | 90% interval |
|---|---|---|
| This model | 0.491 | 0.448–0.540 |
| Step-6 rule (`pipeline/rule.py`, K = 12 frames) | 0.373 | 0.331–0.420 |
| `baseline_smooth50` | 0.255 | 0.232–0.278 |
| This model, event-derived live mask | 0.613 | 0.561–0.663 |

The model's interval lies above the rule's, so the model ships (`pipeline/config.py`, `SHIP`).
Dead-ball errors cost about 0.12. The labeller finds 72 of 103 stoppages, and the stoppages it
misses merge possessions.

| Held-out half | spellF1, snapping off | spellF1, snapping on |
|---|---|---|
| Half 1 | 0.641 | 0.645 |
| Half 2 | 0.588 | 0.578 |
| **Mean** | **0.615** | 0.612 |

- **Other metrics (snapping off):** macro-F1 0.933 and team accuracy 0.900. The model
  predicts 111 possessions against 176 true ones in half 1, and 119 against 206 in half 2:
  it under-segments.
- **Bootstrap:** the scorer's 90% block-bootstrap interval on this game is about ±0.05 (from
  a model fitted on both halves). The snapping difference (−0.003) is well inside it, so
  snapping stays off.
- **The grid is flat** for λ between 1 and 5. The minimum segment length m does the work.
  Ties go to the more conservative setting.

**Attacking-direction check (§6).** On the reference game, the direction inferred from team
centroids matches the metadata on 100% of (team, half) pairs. The direction-dependent feature
(goal-side share) separates strongly the expected way (AUC 0.14–0.18). Length is high too
(0.84). Unlike the raw ball position, no direction-dependent feature points the wrong way.

## Reproduction

From the repository root (Python 3.10, `requirements.txt`; numpy only at run time):

```bash
PYTHONPATH="$PWD:$PWD/src" python3 -m deadball.run ~/eda_out/dead_ball
```

```bash
PYTHONPATH="$PWD:$PWD/src" python3 -m possession.run ~/eda_out/possession --dead ~/eda_out/dead_ball/dead_ball.csv
```

To rebuild the reference set and `ha_model.json` from the open data:

```bash
PYTHONPATH="$PWD:$PWD/src" python3 -m possession.reference /tmp/ha_ref --sample-data /path/to/sample-data --games 3
```

```bash
MTAH_PUBLIC_REPO=/path/to/more-than-a-hack-2027 PYTHONPATH="$PWD:$PWD/src" python3 -m possession.train /tmp/ha_ref --write
```

Tests (synthetic fixtures only):

```bash
python3 -m pytest -q -p no:cacheprovider src/possession/tests
```

Both commands are deterministic: two runs give byte-identical CSVs.

## Limitations

- **One reference game.** The constants are tuned on 2 halves of one game. Its style and
  degradation may not match BAR-ATH.
- **Under-segmentation.** The model finds about 0.6× the true number of possessions. Short
  possessions (under m = 2 s) cannot be represented at all.
- **Restart prior untested.** Smart Tagging exists only on the match, so the tag rule of the
  restart prior is untested against labels.
- **Mask errors propagate.** Errors in the dead-ball mask (false `D` inside play, soft
  stoppage edges) carry straight into possessions; the hard mask cannot repair them.
