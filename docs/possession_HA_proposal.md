# Proposal: H/A possession model on top of the dead-ball labeller

More than a Hack 2027, Phase 1. Status: proposal for implementation, 15 Nov (code freeze day).
This document specifies *what* to build and *how it is validated*. The team decides and reviews;
whoever implements writes the code.

---

## 1. Goal

For every graded frame that the dead-ball labeller marks **live**, output `H` (Barcelona) or `A`
(Athletic), so that the predicted possessions (maximal runs of `H` or `A`) match the true ones as
whole intervals. The ranking metric is `spellF1`: possessions matched one-to-one by temporal IoU,
**same team only**, F1 averaged over IoU thresholds 0.3–0.7. Over-fragmentation and late
transitions are what lose points (1 s late ≈ 0.65).

Out of scope here: deciding `D` (done by `src/deadball/`), broadcast video, video ball detection.

---

## 2. Hard constraints for whoever implements this (read first)

These come from the competition rules and the data terms. They are not negotiable.

1. **Claude Code / any AI assistant works on laptops only**, against synthetic fixtures and
   Metrica's open sample games. Never on the VM, never on match data, never on match outputs.
2. **No match data, per-frame outputs, frame-indexed tables or plots of frames** are pasted into
   any hosted AI. Rounded aggregate statistics are allowed.
3. **Labels come from code.** No hand labelling or hand correction of the match, and no constants
   or rules tied to specific match frames.
4. **The public 5-minute ground truth is a pipeline check only.** Never used to train, tune, or
   choose between versions — including "just checking which of two is better".
5. **Every constant is either a stated football norm or tuned on the open-data reference set.**
   All constants live in one config file.
6. **All data paths go through `games.asset_paths()` / `MTAH_DATA_DIR`.** No hard-coded VM paths.
7. **Deterministic.** No randomness, or fixed seeds; two runs must give byte-identical CSVs.

---

## 3. Inputs

| Input | Source | Used for |
|---|---|---|
| `dead_ball.csv` (`dead`, `p_dead`, `llr_dead` per video frame) | `python -m deadball.run OUT_DIR` | Live-spell mask |
| Tactical ATD (players + ball, normalised 0–1, team per slot) | `asset_paths()["atd_raw"]`, `["atd_meta"]` | Features |
| ATD metadata XML | `asset_paths()["atd_meta"]` | Team identity and attacking direction **per half**, read by tag, never by line position |
| Smart Tagging | `asset_paths()["smart_tagging"]` | Restart-team prior only |
| Reference set (degraded open sample games + event-derived `frame,label`) | Track B | Training and tuning |

Frame alignment: `video_frame = frame_id − 1` (ATD); `video_frame = round(seconds × 25)` (tags).
Pitch: 105.3 × 68.0 m.

---

## 4. Architecture

```
dead_ball.csv ──► live spells ──► restart prior (β) ─┐
                                                      ├─► per-spell decoder ──► switch snapping ──► CSV
ATD ──► per-frame features ──► evidence model (ℓ_t) ─┘
```

### 4.0 Interface with the dead-ball labeller (hard mask)

- `dead == 1` → write `D`. `dead == -1` (ungraded) → write `X`.
- **Live spell** = maximal run of `dead == 0` within one half. Each spell is decoded independently.
- Spells shorter than `MIN_SPELL_S` (≈ 2 s, football norm) get one team, no switch.
- No `X` on graded live frames: the decoder labels every live frame. (`X` is forward-filled by
  `spellF1` and counted as a miss by `macroF1`, so it gains nothing here.)
- Keep `p_dead` available for diagnostics only (e.g. report live frames that sit next to a
  low-confidence `D`); do not feed it into decoding in this version.

### 4.1 Restart prior (who starts each live spell)

Every live spell begins at a restart. Produce `(team, confidence)`:

1. A Smart Tagging `SET PIECES` instance whose start lies within `TAG_LINK_S` of the spell start
   and whose `Team` is not `N/A` → that team. **A `Defending …` tag names the defending team, so
   it implies the other team has the ball.**
2. Otherwise: the first frame in the spell's opening `ONSET_S` seconds where the ball speed
   exceeds `FAST_BALL_MS`; the team of the nearest player in the frames just before it.
3. Otherwise: no prior.

The prior enters the decoder as a bonus **β** on the first state, not as a hard constraint. On the
match, the candidate restart rules agree poorly with each other, so evidence must be able to
override a wrong prior.

### 4.2 Per-frame evidence model

Logistic regression for `P(H | features)`, output as clipped log-odds
`ℓ_t = clip(logit P(H), −CLIP, +CLIP)` with `CLIP ≈ 3`, so one frame cannot force a switch.

Features are **H minus A**, each expressed in that team's attacking direction for the half.

| Group | Features | Windows | Why (from EDA, aggregates) |
|---|---|---|---|
| Shape | convex-hull area, width, length, players goal-side of the ball/centroid | **centred** means over 2, 5, 10 s | AUC ≈ 0.85–0.92 on degraded Sample Game 3, with or without the ball; full coverage |
| Carry | team of a player within `CONTROL_DIST_M` of a slow ball | centred 2, 5, 10 s; 0 when no ball | AUC ≈ 0.89, precision ≈ 0.99 on Sample Game 3 |
| Pass chain | team of the current chain if ≥ 2 passes | same | precision ≥ 0.97 for chains ≥ 2; coverage 42% |
| Nearest team | team of nearest player to the visible ball | same | AUC ≈ 0.92 on degraded Sample Game 3 |
| Missingness | ball-missing indicator (and its interactions with the ball features) | — | ball missing on ~40% of frames |

**Excluded:** raw ball position (points the wrong way on Sample Game 3, confounded by build-up and
pressing), team velocity, centroid advance, joint reversals, nearest-team flip density (none
separate turnover from contest).

**Centred windows only.** Trailing windows build lag into every transition; centred windows place
a change without systematic delay.

Training: degraded sample games, leave-one-game-out. Fit on live frames only (as defined by the
reference labels). Standardise features using training-fold statistics.

### 4.3 Decoder (per live spell)

Two states `s_t ∈ {H, A}`. Maximise

```
Σ_t  ℓ_t · sign(s_t)   −   λ · (#switches)   +   β · sign_prior(s_0)
```

subject to every segment lasting at least `m` frames (enforce with a duration counter in the
state, or equivalently by forbidding a switch within `m` frames of the previous one).

This is Viterbi with a switch cost: linear time, deterministic, and easy to explain in the tech
document. It captures most of what a semi-Markov duration prior would add. A hidden semi-Markov
model with a log-normal duration prior (Sample Game 3: median 5.1 s, mean 10 s, rising hazard) is
an **optional upgrade** only if time allows and it beats this on the reference set.

Tuned on the reference set by `spellF1` (coarse grid): `λ`, `m`, `β`. Nothing else.

### 4.4 Switch snapping to control

The definition says possession changes only when the other team gains **controlled** possession;
a contested or loose ball stays with the last team. For each switch the decoder places:

1. Search `±SNAP_S` (≈ 3 s) around it.
2. Find the old team's last control frame and the new team's first control frame after it.
3. Move the switch to the new team's first control frame.
4. If no control event exists in the window (ball gap), keep the decoder's position.

**Control frame:** ball visible, within `CONTROL_DIST_M` (≈ 1.5 m) of a player of that team, ball
speed below `CONTROL_SPEED_MS`, for at least `CONTROL_K` consecutive frames. Constants from
football norms or the reference set.

Keep snapping only if it improves `spellF1` on the reference set beyond the bootstrap noise.

---

## 5. Constants (one config file)

| Constant | Starting value | Origin |
|---|---|---|
| `MIN_SPELL_S` | 2 s | football norm |
| `TAG_LINK_S` | 3 s | Smart Tagging timing looseness (documented) |
| `ONSET_S` | 3 s | football norm |
| `FAST_BALL_MS` | 4 m/s | same band as the dead-ball model |
| `CONTROL_DIST_M` | 1.5 m | football norm + tracking error |
| `CONTROL_SPEED_MS` | 4 m/s | football norm |
| `CONTROL_K` | 5 frames (0.2 s) | football norm |
| `CLIP` | 3 | fixed in advance |
| `SNAP_S` | 3 s | fixed in advance |
| `λ`, `m`, `β` | grid | **reference set, spellF1** |

---

## 6. Validation

All tuning and model choice happen on the **reference set** (degraded open sample games).

1. **Validate the whole chain, not H/A alone.** Run the dead-ball labeller on the degraded sample
   games, then H/A on its live spells, then the official scorer:
   `python evaluation/run_evaluation.py --gt <reference_labels.csv> --bootstrap 1000`.
   This measures H/A with realistic `D` errors.
2. **Compare three systems on the reference set:**
   - this model;
   - the plan's step-6 rule (restart team, switch after K frames of the other team nearest a
     slow ball, backdated);
   - `baseline_smooth50`.
   Ship the model only if it beats the rule beyond the bootstrap intervals.
3. **Diagnostics from `evaluate()`:** transition precision/recall at ±12 vs ±25 frames (a gap
   means lateness), fragmentation ratio, team accuracy.
4. **Label-free checks on the match (aggregates only):** possessions per live minute, median
   possession length, against the reference set and football norms.
5. **Public window:** format and alignment smoke test at the end. Never compared across versions.

**Check before trusting any direction-dependent feature:** on Sample Game 3, line depth hardly
separates the teams (AUC 0.52) and ball position points the wrong way. Confirm the attacking
direction per half is applied correctly on the reference games first.

---

## 7. Deliverables

- `src/possession/` with: `spells.py` (mask + live spells), `restart.py` (4.1), `features.py`
  (4.2), `evidence.py` (train/apply logistic model), `decode.py` (4.3), `snap.py` (4.4),
  `run.py` (end-to-end, writes `frame,label` CSV of exactly 199,319 rows).
- `config` file holding every constant in section 5.
- Tests on synthetic fixtures:
  - decoder: a hand-built ℓ sequence with one clear change yields exactly one switch at the
    expected frame; noise below λ yields none; segments never shorter than `m`;
  - snapping: a contest between two control events moves the switch to the new team's first
    control frame;
  - mask: `D` and `X` frames are untouched; output length is 199,319; second-half frames before
    kickoff are `D`.
- One command reproduces the CSV byte for byte (`make run` on the VM).
- A short section for the tech doc: features, decoder, constants and their origins, validation.

---

## 8. Order of work today

1. If the step-6 rule is not yet running end to end, do it first — it is the fallback submission.
2. Evidence model (4.2), then decoder (4.3). This is the core gain.
3. Snapping (4.4), kept only if the reference set agrees.
4. If someone is free: sharpen dead-spell **ends** at the ball's first fast movement (dead-ball
   side). It moves the first possession of every live spell and may be worth as much as 4.4.

## 9. Known risks

- **False `D` inside live play** (the dead-ball model calls 87% of ball-gap time dead; 5–10 s
  gaps are mixed) deletes or splits possessions; the hard mask cannot recover them.
- **Missed short stoppages** (under 6 s, found ~30% on sample games) turn into unmarked turnovers
  the decoder must place.
- **Soft `D` edges** (up to ~5 s) shift the first and last possession of every live spell; with
  ~113 live spells this touches a large share of all possessions.
- **Wrong restart team** costs the whole first possession, since spells only match the same team.
