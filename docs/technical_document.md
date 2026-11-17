# Frame-level ball possession from tracking alone

**Team:** Yamalation · More than a Hack 2027, Phase 1 · FC Barcelona v Athletic Club
**Code:** https://github.com/amylnikov30/more-than-a-hack · **File:** `Yamalation.csv` (sha256 `ad3a319c4dcd38d4eb4a09c198bceb79b69b2e31a8ac4bce6a2efb6796b02e8a`)

## 1. Approach

Two problems, solved in order. **Is the ball dead?** Then, on live frames, **which team has it?**

```
tracking (ATD) ─► dead-ball labeller ─► live spells ─► H/A model, one spell at a time ─► frame,label CSV
```

Both stages share one idea: decide per frame from evidence averaged over seconds, then let a
sticky decoder turn those frame decisions into whole spells. The ranking metric scores
possessions as intervals, so a label that flickers inside a possession costs more than one that
is a second late. Every switch has a price; a brief touch, a deflection or a lost ball never
starts a possession on its own.

**Why this, rather than something else.** The tracking loses the ball on 41% of graded frames,
and the stretches where it is lost are not the stretches where the ball is dead: in the public
window 44% of the missing-ball frames are live play, and two of the four stoppages have the ball
visible throughout. Any rule of the form "no ball means dead" fails there, and so does any
possession model that only looks at the ball. Both stages are therefore built on signals that
need no ball (how the 22 players move; how each team is shaped) and use the ball only when it is
there. We did not detect the ball in the video: a learned ball detector could not be validated
without labels on this match, and the player signals carried enough of the answer.

### 1.1 Dead ball (`D`)

The ball is dead when the players have stopped playing.

1. **Player evidence.** For each frame, the share of players under 2 m/s, the share over 4 m/s
   and the mean speed, each averaged over 1, 3, 6 and 12 s and over the 3 s before and after:
   18 numbers, no ball. A logistic model trained on other matches turns them into the log-odds
   of a stoppage.
2. **Ball evidence.** When visible, the ball is in one of 16 categories: outside the pitch,
   slow in a goal area, slow on a touchline, or inside in a speed band × distance-to-nearest-
   player band. How much each category favours dead or live is learned on this match
   *without labels*, by expectation-maximisation on a two-state model anchored by the player
   evidence. A missing ball is fixed at zero evidence.
3. **Decode.** A two-state sticky hidden Markov model per half (Viterbi).
4. **Sharpen.** A stoppage ends where the ball reappears and starts where it went out, when
   either happens within 3 s of the decoded edge. Everything before the second-half kickoff is
   dead.

### 1.2 Possession (`H` / `A`)

Every live spell starts at a restart. Within a spell:

1. **Evidence.** Per frame, home-minus-away features, each a centred mean over 2, 5 and 10 s
   that never crosses a stoppage: team shape (convex-hull area, width, length, players goal-side
   of the ball), and when the ball is visible, which team is nearest it, which team is carrying
   it, and which team's pass chain it is in. A logistic model gives log-odds for "home has the
   ball", clipped to ±3 so no single frame can force a switch. Shape alone separates the teams
   with AUC 0.92 on the reference game, ball or no ball, which is what carries possession
   across the ball gaps.
2. **Restart prior.** The team taking the restart: from a Smart Tagging set-piece tag linked to
   the stoppage (23 of 112 spells), else the team nearest the ball just before its first fast
   movement (81), else none (8). It is a bonus on the first frame, not a constraint.
3. **Decode.** Choose the team sequence maximising summed evidence minus λ per switch plus the
   prior bonus, with every possession at least m = 2 s long. An exact dynamic programme, linear
   time, deterministic.

### 1.3 Fallback rule and the choice between them

A rule with no learned weights was built as the fallback: start with the restart team; switch
after 12 consecutive frames with the other team nearest a slow visible ball, backdated to the
first of them. Both systems were scored on the reference set through the whole chain (section
4). The model ships because its bootstrap interval lies above the rule's.

## 2. Assets used and dropped

| Asset | Used | How |
|---|---|---|
| Tactical ATD tracking (`_atd.txt`) | yes | the only per-frame input to both stages |
| ATD metadata (`_atd.xml`) | yes | which slots are which team; first team listed is home |
| Smart Tagging | partly | set-piece tags give the restart team as a prior; also a label-free check that the labeller is dead at tagged restarts (51 of 58). Not used for timing: tag starts are several seconds off the real restart, and throw-ins carry no team |
| Homography | no | explored in the EDA (reprojection error around set pieces); no signal beyond the tracking |
| Tactical video | no | no ball detection; see section 1 |
| Broadcast video | no | not aligned to the tactical timeline |
| Public 5-minute ground truth | pipeline check only | read twice, qualitatively (gaps occur in live play; dead ball can have a visible ball); agreement printed at the end as a check, never used to tune or to choose between versions |

Explored and dropped as features: raw ball position (points the wrong way on the reference
game, confounded by build-up and pressing), team velocity, centroid advance, defensive-line
depth, nearest-team flip density, and 150 richer player-movement features (better on clean
reference tracking, worse on the automatic tracking of the match).

## 3. Training data

No hand labels were made, and nothing was trained or tuned on the provided game or its ground
truth. All labelled data is Metrica's open sample data
(github.com/metrica-sports/sample-data), other matches, with labels derived by code:

| Model | Data | Labels | Who labelled |
|---|---|---|---|
| Dead-ball player model (18 weights, 2 decoder constants) | Sample Games 1–3, full matches | ball position blank for ≥ 1 s = out of play (blank before 97–98% of set pieces and after 100% of ball-out events in that data) | nobody: derived from Metrica's tracking |
| Ball-category log-odds (dead ball) | the provided match | none: unsupervised EM | — |
| H/A evidence model (25 weights) and decoder constants λ, m, β | Sample Game 3 (both halves), degraded to resemble our tracking: ball dropped in bursts until 41% of live frames are missing, player tracks broken and re-slotted into anonymous slots, fixed seed | team of the last controlled event (pass, carry, recovery, shot, set piece) holds the ball; restarts, ball-out events and half ends give dead frames | nobody: derived from Metrica's event data |
| Fallback rule constant K | the same degraded Sample Game 3, other half | as above | — |

## 4. Possession rules implemented

- **Control, not contact.** A possession is a run of frames where the evidence, averaged over
  seconds, favours one team, and switching costs λ. A touch, a deflection, a blocked pass or a
  50/50 that lasts under 2 s cannot become a possession.
- **Loose and contested balls stay with the last team** until the other team's evidence
  outweighs the switch cost for at least 2 s.
- **A missing ball changes nothing by itself.** Possession carries across a tracking gap; team
  shape decides whether it changed during the gap.
- **Restarts** belong to the team taking them (the prior), and every stoppage ends a
  possession: the spell after it is decoded afresh.
- **Dead ball** is when the players have stopped, not when the tracking lost the ball. A goal
  stays dead through the celebration to the kick-off (all four goals in the match).
- **`X`** only off the graded halves; every graded frame gets a label.
- **Second half before the kickoff** is `D` throughout.

## 5. Validation

### 5.1 Reference set (other match, whole chain)

Degraded Sample Game 3, 382 true possessions. The dead-ball labeller runs on the degraded
tracking, then H/A on *its* live spells (so the labeller's errors count), scored by the
organisers' `run_evaluation.py`, 1000 bootstrap draws. The H/A weights, and the rule's K, are
fitted on the other half of the game.

| System | spellF1 | 90% interval | macroF1 | possessions found |
|---|---|---|---|---|
| **Shipped: dead-ball labeller + H/A model** | **0.491** | 0.448–0.540 | 0.851 | 223 |
| Dead-ball labeller + fallback rule | 0.373 | 0.331–0.420 | 0.801 | 208 |
| `baseline_smooth50` | 0.255 | 0.232–0.278 | 0.754 | 719 |
| H/A model with the true dead-ball mask (diagnostic) | 0.613 | 0.561–0.663 | 0.933 | 230 |

Dead-ball errors cost about 0.12 spellF1: on this game the labeller calls 31.6% of frames dead
(truth 33.5%, dead F1 0.87) but finds 72 of 103 stoppages, and each missed stoppage merges two
possessions. Caveats: one reference game, so the intervals are about ±0.05 wide; the dead-ball
player model saw this game in training (it was trained on Games 1–3); the H/A constants were
tuned on it, held out by half.

### 5.2 Label-free checks on the match (aggregates)

| Check | Match | Reference |
|---|---|---|
| Dead share of graded frames | 49.8% (1st half), 52.8% (2nd) | 34% on Sample Game 3; the sample games are other leagues and this match includes 2 min of pre-kickoff `D` |
| Dead spells | 55 + 58, median 21 s | 103 |
| Labeller dead at the start of tagged set pieces | 51 of 58 (chance 0.51) | — |
| Goals | dead through all four celebrations | — |
| Possessions per live minute | 5.1, median 7.2 s | 6.0, median 5.1 s |
| Home share of live frames | 0.63 | — |

### 5.3 Public window (pipeline check)

Alive/dead agreement on the 5 public minutes, printed at the end of the dead-ball run as a
format and alignment check: dead F1 0.85, 4 stoppages found of 4. It was not compared across
versions.

## 6. Limitations

- **Stoppage edges are soft**, by up to about 5 s, because the player evidence is averaged
  over seconds. This moves the first and last possession of every one of the 112 live spells.
- **Short stoppages are missed.** Quick throw-ins and free kicks where nobody slows down look
  like play: stoppages under 6 s are found about 30% of the time on the reference games. A
  missed stoppage merges two possessions into one.
- **Under-segmentation.** Both stages err on the long side: 5.1 possessions per live minute
  against 6.0 in the reference, and nothing under 2 s can be represented.
- **The dead share looks high** (51% against 34–41% on the sample games). Different matches
  and leagues; without the full ground truth we cannot say whether it is wrong.
- **One reference game** for the H/A constants, and the Smart Tagging restart rule could not
  be checked against labels (the sample games have no tags).
- **What we would do with more time**, in order of expected gain: a restart detector based on
  the ball's first fast movement after a slow spell, to sharpen stoppage edges; a short-window
  player model for the edges; more reference games; feeding the dead-ball log-odds into a
  three-state possession decoder instead of a hard mask.

## 7. Reproduction

Python 3.10, dependencies pinned in `requirements.txt` (numpy, pandas, kloppy). No random
numbers at run time; the committed model files (`src/deadball/player_model.json`,
`src/possession/ha_model.json`) are used as they are, so two runs give the same bytes.

```bash
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data            # the organisers' bundle, tactical/ folder inside
export MTAH_PUBLIC_REPO=/path/to/more-than-a-hack-2027   # for validate_submission.py
make run TEAM=Yamalation                        # writes ~/mtah_out/submission/Yamalation.csv
```

Without `make`, the same run is:

```bash
PYTHONPATH="$PWD:$PWD/src" python3 -m pipeline.make_submission ~/mtah_out/submission --team Yamalation
```

Under a minute for the dead-ball stage and a few minutes for the whole file. To retrain from
the open data: `make reference` then `make train` (clone of metrica-sports/sample-data needed;
reproduces the weights to about 1e-14). To rerun the validation of section 5.1: `make evaluate`.
Full details in the repository `README.md`; design write-ups in `docs/dead_ball_rule.md` and
`docs/ha_model.md`.
