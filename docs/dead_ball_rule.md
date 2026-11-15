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
   one of 16 categories: missing, outside, slow in a goal area, slow on a touchline, or visible inside in one of 4 speed bands × 3
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

## Update, 13 Nov: restart-type cues and edge sharpening

Rob's signal profiles around tagged set pieces show where the first version is weakest:

| Restart (tags) | Ball missing around it | Players under 2 m/s | Reading |
|---|---|---|---|
| Corner (12) | 100% from −6 s to +2 s | 96% | easy |
| Free kick (5) | 80–100% | 93–97% | easy |
| Kick-off after goal (4) | 100%, reappears 2–3 s after the tag | 93–100% | easy; end at the ball reappearing |
| Throw-in (26) | 60–70% | rises to 85–91% only after the tag | hard: players slow late; visible ball is outside 30–60% of the time |
| Goal kick (10) | 20–50% | 77% | hardest: ball visible, players barely slow |

Three changes follow from that. All constants are fixed in advance from the laws of the game
or the tracking error; none was searched.

1. **Two new ball categories**: a slow ball (< 4 m/s) in a goal area, and a slow ball within
   2 m of a touchline. EM learned +2.35 and +2.00 log-odds toward dead for them, against +0.1 for
   a slow ball at a player's feet elsewhere. This is clause c's location idea, kept only where
   the EDA showed a restart link (goal area lift 4.5, touchline band 3.4).
2. **End at the ball reappearing** (`run.sharpen_edges`). If the ball comes back within 3 s of a
   decoded dead-spell end, on either side, the spell ends at that frame. Applied to 36 of 113
   spells (28 extended, 8 trimmed).
3. **Start at the ball going out.** If the ball leaves the pitch within 3 s before a decoded
   start and is not seen back inside, the spell starts at the first outside frame. Applied once.

**Effect: small.** 1,105 frames moved in total (about 1.2 s per affected spell); 113 dead spells
instead of 114; dead share unchanged at about 51%. The public-window pipeline check is the same
to two decimals (dead F1 0.85, 4 spells of 4). `--no-sharpen` switches change 2 and 3 off.

**What this tells us.** The large edge errors are not at ball gaps. They are in stoppages where
the ball stays visible, where the only timing signal is the player features, and those are
averaged over up to 12 s. The next gain is there: a player model with shorter windows for the
edges, or a restart detector based on the ball's first fast movement after a slow spell.
Kick-off formation (every player in their own half) is still to do and can be scored on the
sample games.

## Experiments that did not make it in (13 Nov)

Four attempts to push the model further, each scored on held-out sample games with the
spellF1 matching on ball-in-play spells. The shipped player model scores 0.633 there.

| Attempt | Held-out sample games | On the match | Decision |
|---|---|---|---|
| 150 player features (more windows, standing, sprinting, flow, spread, acceleration), logistic | 0.656 | not run | not kept |
| Same features, small neural network | 0.665 | — | not kept |
| + positional restart cues (player on a touchline, near a corner) | 0.671 | dead at 42 of 58 tagged set-piece starts (shipped: 51) | not kept |
| + learned edge refinement (short-window model moves each edge up to 4 s) | 0.688 | 46 of 58; public check finds 3 stoppages of 4 (shipped: 4) | not kept |

The gain on the sample games is real but small (+0.055, most of it from one game), and it does
not carry over to the automatic tracking of the match. A model fitted to clean, corrected
tracking with 186 features is more fragile than one with 18. The simpler model stays.

What the error analysis on the sample games shows: stoppages of 10 s or more are found 94% of
the time; stoppages under 6 s only 30%. Those short ones are quick throw-ins and free kicks
where the players never slow down, so no player-only model can see them. On the match the ball
evidence covers them: the ball is called dead at the start of 23 of 26 tagged throw-ins.

Two label-free checks added while doing this:

- **Tagged restarts** (`run.tag_check`, printed on every run): the ball is called dead at the
  start of 51 of 58 tagged set pieces, against a chance level of 0.51. Misses: 3 goal kicks,
  3 throw-ins, 1 corner.
- **Goals**: in all four goal-to-kick-off stretches the model stays dead through the whole
  celebration; the only live spell inside each goal tag is the attack that led to the goal.

On the dead share: 51% is above the 37–41% of the sample games, but those are different matches
and leagues. Without the full ground truth there is no evidence it is wrong, and both checks
above are consistent with it.
