# The dead-ball rule

How we decide which frames are `D`, why, and how it plugs into the possession HMM.
Code: `src/deadball/`. Evidence: the `eda/` scripts, re-run on the tactical ATD on 13 Nov.

## The rule in one paragraph

**The ball is dead when the players have stopped playing.** A model trained on other matches
reads how the 22 players are moving and says how likely a stoppage is. The ball track then adds
or removes confidence where it is visible: a ball that is outside the pitch, or slow and away
from every player, points to dead; a ball moving fast near a player points to live. **A missing
ball is not evidence of anything** — while it is missing, the players decide. A sticky decoder
turns that per-frame evidence into whole dead-ball spells, and everything before the second-half
kickoff is dead.

## What the data says (and what it ruled out)

| Draft clause | Finding on the match | Verdict |
|---|---|---|
| **a.** Ball visible outside the pitch (> 1 m) → dead | Only 2.7% of visible frames (58 runs); 43 of the 58 are followed by the ball coming back inside. Restart link is real but weak (lift 2.4). | Keep as evidence, not as a rule on its own |
| **c.** Ball still on a restart mark → dead | 50 still runs ≥ 1 s, 2.3% of frames; none on the centre spot or a corner arc; 37 of 50 are "elsewhere" with no restart link (lift 0.3). | Drop the location logic; keep "slow ball far from players" as evidence |
| **d.** Long ball gap + players walking → dead | 99.7% of missing-ball time is in 132 gaps ≥ 5 s. Slow-player share is 0.40 in short gaps, 0.80 at 5–10 s (IQR 0.53–0.91), 0.93 beyond 10 s. | **Keep — this is the core.** But the 5–10 s gaps are mixed: some are live play the tracker lost |
| "Ball missing = dead" | In the public window 44% of missing frames are live play, and two of the four dead-ball spells have the ball visible throughout. | **Rejected** |
| Smart Tagging set pieces as anchors | Tag intervals start several seconds before or after the real restart, and throw-ins carry no team. | Not used for timing |

Speed units, for anyone using the declared column: player speeds are m/s (declared/derived
ratio 0.99); the declared ball speed is blank on 92% of visible frames and unusable.

## How it works

1. **Player evidence** (`train_players.py`, `signals.player_features`). Per frame: the share of
   players under 2 m/s, the share over 4 m/s and the mean speed, each averaged over 1, 3, 6 and
   12 s and over the 3 s before and after. A logistic model turns these 18 numbers into
   log-odds of dead ball. It needs no ball, so it works on all 41% of frames where the ball is
   missing.
2. **Ball evidence** (`signals.ball_categories`, `run.ball_log_ratios`). Each frame's ball is
   one of 14 categories: missing, outside, or visible inside in one of 4 speed bands × 3
   distance-to-nearest-player bands. How much each visible category favours dead or live is
   learned on the match itself **without labels**: EM on a two-state model whose meaning is
   pinned by the player evidence.
3. **Decode** (`hmm.viterbi`). Sticky Viterbi per half, so brief blips do not open or close a
   stoppage.
4. **Before the second-half kickoff** (`games.py` anchor) everything is dead.

What the match taught the ball model (log-odds toward dead; positive = dead):

| Ball | | Ball | |
|---|---|---|---|
| outside the pitch | +3.3 | 4–12 m/s, player within 2 m | −1.5 |
| < 1 m/s, nearest player > 5 m | +4.0 | 4–12 m/s, player 2–5 m | −1.0 |
| < 1 m/s, nearest player 2–5 m | +2.7 | > 12 m/s | −1.7 to −2.3 |
| < 1 m/s, nearest player < 2 m | +1.7 | 1–4 m/s, player within 2 m | +0.3 |
| 1–4 m/s, nearest player > 5 m | +2.1 | missing | 0 (fixed) |

## Where every number comes from

Nothing is tuned on the public 5 minutes, and nothing is hand-labelled.

| Number | Source |
|---|---|
| Player model weights | Metrica open Sample Games 1–3 (other matches). Label: ball position blank ≥ 1 s, which in that data means out of play (blank before 97–98% of set pieces and after 100% of ball-out events). |
| Tempering 10, switch cost 4 | Leave-one-game-out on the same sample games, scored with the spellF1 matching on ball-in-play spells. |
| Ball category log-odds | Unsupervised EM on the competition match. |
| 2 / 4 m/s, 1 m margin, speed and distance bands | Fixed in advance from football (walking, running, tracking error); not searched. |
| Second-half kickoff | Published by the organisers in `games.py`. |

The public window told us two qualitative things about what the labels mean — gaps happen in
live play, and dead ball can have a visible ball — which is its documented purpose. `--check`
prints agreement on it as a pipeline check only.

## How good is it

| | Result |
|---|---|
| Player model alone, held-out sample game | AUC 0.90 / 0.94 / 0.96; ball-in-play spell F1 0.58 / 0.67 / 0.65 |
| Match: dead share | 49.5% (1st half), 52.4% (2nd half) |
| Match: dead spells | 54 + 60, median length 21 s |
| Match: ball-missing frames called dead | 87% (so 13% of gap time is kept as live play) |
| Public window, pipeline check | 4 dead spells found vs 4 true; dead F1 0.85; ball-in-play spell F1 0.88 |

## Known weaknesses

- **Boundaries are soft.** In the public window the spell edges are off by up to about 5 s.
  The player features are averaged over seconds, so they cannot place a whistle or a restart to
  the frame. `spellF1` tolerates this for long spells and punishes it for short ones.
- **Short stoppages are the hard case.** Quick throw-ins and free kicks where nobody slows down
  look like live play to the player model (this is what caps it at 0.65 on the sample games).
  On the match the ball evidence is what catches them, and only when the ball is visible.
- **Dead share looks high** (about 51% against 37–41% in the sample games). Some of that is the
  2,934 dead frames before the second-half kickoff; the rest is probably spells stretched at
  both ends.
- The model has no restart typing (throw-in vs corner vs free kick) and does not say who
  restarts. Neither is needed for `D`.

## How to use it in the possession HMM

`python -m deadball.run OUT_DIR` writes `dead_ball.csv`, one row per video frame:

| Column | Meaning |
|---|---|
| `dead` | 1 dead, 0 live, −1 not graded (write `X`) |
| `p_dead` | posterior probability of dead ball |
| `llr_dead` | per-frame log-likelihood ratio, dead vs live, before tempering |

Two ways to plug it in, simplest first:

1. **Hard mask.** Write `D` wherever `dead == 1` and run the H/A model only on the live spells
   in between. Each live spell then starts with a restart, which is a clean place to
   initialise who has the ball.
2. **Soft evidence.** Add `llr_dead` to the `D` state's emission in the three-state HMM
   (`log b_D = log b_alive + llr_dead`), with the same tempering. The possession model can then
   overrule a weak dead-ball call.

Start with 1: it is one line, and it makes the H/A problem independent of this one.

## Reproduce

```bash
git clone https://github.com/metrica-sports/sample-data ~/sample-data
cd src
PYTHONPATH=.. python -m deadball.train_players ~/sample-data     # rewrites player_model.json
PYTHONPATH=.. python -m deadball.run ~/eda_out/dead_ball          # needs MTAH_DATA_DIR
```

Both steps are deterministic: no random numbers, and two runs give byte-identical output.
Runtime is under a minute. Dependencies: numpy and pandas (`requirements.txt`).

## Next steps, in order of expected gain

1. Sharpen spell edges: snap a dead spell's end to the first frame the ball moves fast again,
   and its start to the last fast ball frame before the players slow.
2. Check the dead share against the sample games after step 1.
3. Feed `llr_dead` into the three-state HMM (option 2) once the H/A emissions exist.
