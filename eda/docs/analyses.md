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

Rows added later (after all the rows above, which are unchanged):

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| params | all | | disp_window | s |
| ball_declared | ball | `nan` / `zero` / `positive` | n_frames | frames |
| ratio_by_speed | players / ball | `0-2` `2-5` `5-10` `10-20` `20+` (m/s) | median, p25, p75, n_ratio | ratio, count |
| ball_disp | ball | `0-0.1` `0.1-0.25` `0.25-0.5` `0.5-1` `1-2` `2-5` `5+` (m) | share, n_frames | share of windows, frames |

`ball_declared` counts graded frames with the ball's position present by their declared speed
field (missing/NaN, exactly 0, above 0). `ratio_by_speed` splits the pairs of the main ratio by
derived speed (half-open bands; with the 1 m/s floor, `0-2` holds only 1-2 m/s); a ratio that
drifts with speed means the declared speed is smoothed or capped. `ball_disp` is the ball's
displacement between f - w and f (w = `--disp-window-s` x fps), on windows with the ball present
at both ends and both frames graded in the same half: the share of windows under 0.1 or 0.25 m
calibrates the stationary-ball threshold of `ball_stationary.py` (0.5 m/s over 0.2 s is 0.1 m).

Derived percentiles use every pair with no cap, so slot swaps show up in the p99.9 tail; the
`n_over_cap` row says how many there are.

**Tests.** `tests/test_speed_units.py` on `fixture_speed_atd.*` (12 frames, declared in km/h
while the metadata says m/s): pair counts, the swap over the cap, ratio quartiles, percentiles,
the halftime step, a file without a speed field, a missing line. On `fixture_disp_atd.txt`
(70 frames, the ball in five isolated segments at 0 to 34 m/s): declared-ball counts, the ratio
per speed band, the displacement bands and the `--ball-cap-ms` alias.

**Questions for interpretation.**
- Is the IQR of the ratio narrow? A wide IQR around 3.6 suggests the declared speed is smoothed
  over several frames, so it lags the frame-to-frame derived speed.
- Do players and ball agree? A ball ratio far from the player ratio would mean the ball speed is
  computed differently (e.g. in 3D).
- How many pairs exceed the cap? Many player swaps point to tracklet switches that a speed-based
  rule (clause d) must guard against.
- Is the declared ball speed mostly NaN or zero on visible frames? Then the ball column is not
  usable and only derived ball speed counts.
- What share of 1 s windows moves the ball less than 0.1 m or 0.25 m? On Sample Game 3 it is
  0.07% and 0.3%: the tracked ball is almost never still, which matches how few still runs
  `ball_stationary.py` finds there. A much larger share on the match would leave room for a
  stricter `--still-ms`.

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

## Possession signals (live play)

These scripts ask, for each candidate signal for the live-play possession model, whether it is
available often enough, separates the two teams, and is sharp enough in time. They do not build
the model. Each `poss_*.py` script writes one `*_summary.csv` in the long format above (floats
to 3 significant figures, NaN empty). Match runs and reference-game runs produce the same rows:
rows that do not apply hold NaN.

### Shared preparation (`poss_common.py`, `poss_reference.py`)

`poss_common.py` is a separate module rather than an extension of `common.py`, which serves the
dead-ball scripts; it reuses `ball_gaps`, `common` and `tag_gaps`.

- **Tracking.** One streamed pass over the raw ATD into metres (`--pitch`); video frame =
  `frame_id - 1`; slot teams per data-format spec from the metadata (first `<Team>` = home,
  anything else = -1, e.g. referees).
- **Attacking direction.** From the metadata's `attack_direction_first_half` team parameter
  (Sample Game 3 declares it), flipped each half. Without it, inferred per half from the first
  60 s of live frames: the team with the lower mean x attacks +x. Both are computed and their
  agreement is reported (`params, direction, centroid_agreement`).
- **Live mask.** Graded frames minus dead frames. On the match the dead frames come from the
  dead-ball model (`src/deadball`), read from the `dead_ball.csv` that
  `python -m deadball.run OUT_DIR` writes (`--dead`; `dead == 0` is live, -1 and missing rows
  are not). On the reference game they come from its events (below).
- **Ball interpolation.** Live gaps of at most `--max-interp-frames` (12 = 0.48 s) with an
  observed neighbour on both sides inside the same live run are filled linearly. A gap next to a
  dead frame or a half edge is never filled. Flag per frame: observed / interpolated / missing.
- **Pseudo-labels (match).** `PHASE_TO_POSSESSION` maps each Smart Tagging code to "the tagged
  team has the ball" (BUILD UP, PROGRESSION, ATTACKING TRANSITION, CREATING CHANCES, BALL IN
  FINAL THIRD, LONG BALL), "the tagged team defends" (DEFENDING IN DEFENSIVE / MIDDLE /
  ATTACKING THIRD, DEFENSIVE TRANSITION) or "ignore" (BALL IN THE BOX, PLAYERS IN THE BOX, SET
  PIECES, GOALS). Codes match case-insensitively; `N/A` teams are dropped; frames where two
  mapped tags disagree are dropped. Tag team names are matched to the metadata team names by a
  shared word (`--tag-teams HOME,AWAY` overrides). Every script reports `agreement` per code:
  over the frames where the code and at least one other code label, the share where all others
  agree. **Uncertain mappings**, to check in the agreement rows first: LONG BALL (the ball may
  end with the other team), ATTACKING / DEFENSIVE TRANSITION (they straddle the turnover that
  starts them, so their first seconds may belong to the other team).
- **Changes.** The first frame of the other team after a labelled frame of one team in the same
  live run. A restart after a dead ball is not a change.

**Reference labels (`poss_reference.py`, Metrica Sample Game 3).** Open data from
github.com/metrica-sports/sample-data: `Sample_Game_3_tracking.txt` / `_metadata.xml` (EPTS,
the same format as our ATD, x,y only, no speed) and `Sample_Game_3_events.json` (Metrica JSON:
`type`, `subtypes`, `team.id`, `start.frame` / `end.frame` = tracking `frame_id`, `period`).
Derived by code, never by hand:

- *Dead* is the union of three rules (end exclusive). (1) Restarts: a `SET PIECE`, or any event
  with subtype `GOAL KICK` (goal kicks are `PASS`, or `BALL LOST` when they fail, not
  `SET PIECE`). The half's first restart is preceded by dead frames from the half's start; each
  later one by dead frames from the end of the last event that starts strictly before it
  (`CARD` skipped; the keeper's same-frame `RECOVERY` before a goal kick does not count). (2) After
  `BALL OUT`, or a `SHOT` with subtype `OUT`, dead until the next event that starts later, even
  if it is not a restart (one goal kick in Sample Game 3 is coded `BALL LOST / FORCED`). (3)
  After an `END HALF` event, dead to the end of the half.
- *Possession* on a live frame: the team of the most recent controlled event (`PASS`, `CARRY`,
  `RECOVERY`, `SHOT`, `SET PIECE`). `BALL LOST`, `BALL OUT`, `CHALLENGE` (won or lost),
  `FAULT RECEIVED` and `CARD` never change it, following CHALLENGE.md: a touch, a deflection or
  an aerial duel without control is not a change.
- *Check against the tracking.* In Metrica's open data the ball is blank while it is dead. On
  Sample Game 3 the derived dead frames have the ball blank on 95.5% of frames, the live frames
  on 0.4%. Rules (1) alone (before goal kicks and `END HALF` were handled) left 13 live gaps of
  5 s or more, all stoppages; with the three rules none remain.
- *Result on Sample Game 3:* 95,623 live and 48,138 dead frames (33.5%), 382 possessions (median
  5.1 s, mean 10.0 s), 280 live turnovers, 104 dead runs (median 13.1 s). The public ground
  truth has a mean possession of 14.2 s over 5 minutes, so event-derived possessions are
  somewhat finer: every `RECOVERY` changes the team, including some the ground truth might call
  loose balls.

**Degraded copy (`--degrade`).** Sample Game 3 is corrected tracking: the ball is almost always
present in live play and every player is one track. `--degrade` makes it look like our ATD,
seeded by `--seed`: (a) ball dropped in bursts at uniform live start frames, clipped to live
frames, until `--missing-share` (0.41) of live frames are missing. Burst lengths are log-normal,
`--gap-lengths 1,1.1,30` (median 1 s, sigma 1.1, cap 30 s): about a quarter of bursts are under
0.5 s and a tenth over 4 s. This is a guess, not fitted: compare the degraded run's gap bins
with the match run of `poss_ball_live.py` and adjust if they differ. (b) Every player's
presence runs are cut into pieces of exponential length (`--frag-mean-s 20`), losing
`--frag-gap-frames` (5) frames at each cut, and each piece is re-slotted into a random free slot
of 46. **Caveats.** 0.41 is the missing share on all *graded* frames of the match; on its live
frames it is probably lower (in the public window 44% of missing frames are live), so the
default is pessimistic. And the bursts land uniformly, while the real tracker loses the ball
more where play is contested: on clean Sample Game 3 the ball is missing on 1.3% of frames
within 2 s of a turnover against 0.1% elsewhere, and the degraded copy does not reproduce that
(0.42 vs 0.41).

### Ball availability on live play (`poss_ball_live.py`)

**Question.** How much of live play has a ball at all, how long are the gaps, and does the ball
go missing where possession changes? This sets how much the ball-free signals must carry.

**Definitions.** On live frames: the share observed, interpolated and missing. A gap is a run
of missing ball inside one live run (a gap that runs into a dead stretch is cut there). Gaps per
half-open bin `[0,0.5) [0.5,2) [2,5) [5,inf)` s, before and after interpolation (count, frames,
share of missing live frames); gap-length p10/p50/p90 before interpolation; frames filled at
the main cap and at `--interp-also` (6, 25 frames); the missing share on dead frames; the
missing share within `--change-window-s` (2 s) of a change (true labels on the reference game,
pseudo-label changes on the match) against the rest of live play.

**Defaults and why.** `--max-interp-frames 12` (0.48 s): a ball lost for under half a second
has rarely changed direction more than once, so a straight line is a fair guess; 6 and 25 show
how much depends on it. `--change-window-s 2`: a turnover is a short event; a wider window
would dilute it with mid-possession play. Both are football norms, not tuned.

**CSV schema** (`poss_ball_live_summary.csv`).

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| params | all / degrade / labels / live / direction / half h | | fps, pitch_*, lag, max_interp_frames, degrade, windows, timing_window (every poss_* script), change_window, interp_also; seed, missing_share, gap_*, frag_*, bursts (degrade); source (labels: events / smart_tagging / none; live: events / dead_csv; direction: metadata / centroids / none), centroid_agreement; first/last_frame | |
| labels | all / H / A | | live_frames, labelled_frames, frames, n_changes, dropped_frames | frames, count |
| agreement | `code:<CODE>` (every mapped code) | | n_frames, agree_share | frames, share |
| coverage | all / half h | `` / observed / interpolated / missing / missing_before_interp | live_frames, frames, share | frames, share of live frames |
| gaps_before / gaps_after | all | `<0.5s` `0.5-2s` `2-5s` `>=5s` `all` | n_gaps, frames, share | count, frames, share of missing live frames |
| gap_length | all | p10 / p50 / p90 | length | s |
| interp_cap | all | cap in frames | filled_frames, share | frames, share of missing live frames |
| dead | all | | dead_frames, missing_share | frames, share of dead frames |
| change | all / around / rest | `` / before / after | n_changes, frames, missing_share | count, frames, share of live frames |

**Tests.** `tests/test_poss_common.py` (live mask from events and from the dead CSV, reference
labels and restart rules, slot teams, directions with and without the metadata, tag team map,
pseudo-labels and agreement, interpolation, the degraded copy) and
`tests/test_poss_ball_live.py` (coverage, bins, quantiles, caps, dead share, changes with true
and pseudo-labels, identical rows for both label sources, no labels, `--degrade`) on
`fixture_poss_*`.

**Results on Sample Game 3.** Clean: 99.6% of live frames have the ball, so the open game says
nothing about ball loss on its own; that is what `--degrade` is for. Degraded (41% missing):
after interpolation 39.7% of live frames still have no ball, and almost all of it (99.9% of
missing frames) is in gaps of 0.5 s or more, half of it in gaps of 5 s or more: interpolation
cannot carry possession through those.

**Questions for interpretation.**
- What share of live frames is missing on the match, and how much does interpolation recover?
  That share is the part of live play the ball-free signals (shape, reversal) must label alone.
- Are most missing frames in gaps of 5 s or more? Long gaps mean a model must hold or change a
  label for seconds without a ball: duration priors and ball-free signals matter most.
- Is the missing share within 2 s of a pseudo-label change well above the rest? Then the ball
  vanishes exactly at turnovers, and change timing must come from players, not the ball.
- How far is the dead-frame missing share above the live one? The gap tells how much "ball
  missing" says about dead ball once the dead-ball model has run (the dead-ball rule keeps it at
  zero evidence).

### Separation and timing, shared by the signal scripts

Every signal script reports separation the same way, so the notebook's scale map needs no
extra runs. The signal is a per-frame score toward "home has the ball", NaN where the signal is
absent. For each window w in `--windows` (0.5, 1, 2, 5, 10 s) plus 0 (the frame itself), the
score is averaged over the defined frames of the centred window `[f - h, f + h]` inside the
live run (`h = round(w x fps / 2)`), and three numbers are written to `separation` rows with
`bin` = w:

- `auc`: AUC (Mann-Whitney, ties half) of the smoothed score for home vs away possession over
  all labelled live frames, an absent score counted as 0. This mixes availability and
  separation: a signal that is right but rare stays near 0.5.
- `auc_defined`: the same over frames where the smoothed score is defined. This is separation
  alone.
- `coverage`: share of live frames where the smoothed score is defined.

An AUC below 0.5 means the hypothesis is reversed (the scripts do not fold it). No null is
needed: chance is 0.5 by construction. Labels are the true labels on the reference game and the
pseudo-labels on the match, whose own agreement rows show how noisy they are.

Signals that fire at changes report `timing` rows: each label change takes the nearest signal
event of the same new team within `--timing-window-s` (5 s; ties go to the later event); the
error is event minus change, in s (p25, p50, p75, median absolute). `matched_share` is the share
of changes with an event; `event_precision` the share of events with a same-team change within
the window.

### Carries (`poss_carries.py`)

**Question.** When the ball moves with one player, is that player's team in possession, and how
much of live play does that cover? Feeds the per-frame "who has the ball" emission of the model.

**Definitions.** The ball is the interpolated ball. Velocities are backward differences over
`--lag` (5) frames, defined only inside one live run. A frame is a carry candidate for a player
of either team (referees excluded) if the ball is within `--carry-dist-m` and the velocity
difference is below `--carry-dv`; the carrier is the nearest candidate. A carry is a run of
frames with one carrier slot, at least `--carry-min-s` long; its team is the slot's team.
Score: +1 home carry, -1 away carry. Precision: labelled carry frames whose carry team is the
label (all, home, away). Switches: the first frame of a carry whose team differs from the
previous carry in the same live run; their timing against label changes. Fast ball: live frames
with a ball speed above `--fast-ball-ms`; `near_rate` is the share with the ball within
`--carry-dist-m` of any player (what distance alone would call a touch), `carry_rate` the share
inside carries (what the velocity test lets through).

**Defaults and why** (football norms, not tuned). `--carry-dist-m 1.5`: about a stride; a
dribbled ball stays within it. `--carry-dv 2`: a dribbled ball moves at the player's speed give
or take a touch; a pass passing by does not. `--carry-min-s 0.5`: the shortest controlled
touch-and-go. `--fast-ball-ms 15`: a firm pass; the ball has no height, so a fast ball close to
a player is the best available proxy for a lofted ball passing over them.

**CSV schema** (`poss_carries_summary.csv`), after the shared `params`, `labels` and
`agreement` rows (params add carry_dist, carry_dv, carry_min, fast_ball):

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| coverage | all | | live_frames, n_carries, carry_frames, share | frames, count, share of live frames |
| durations | all | `0.5-1s` `1-2s` `2-5s` `5-10s` `>=10s` | n_carries | count |
| durations | length | p10 / p50 / p90 | length | s |
| precision | all / H / A | | labelled_frames, precision | frames, share of labelled carry frames |
| separation | carry | window s (`0` = frame) | auc, auc_defined, coverage | AUC, share of live frames |
| timing | carry_switch | `` / p25 / p50 / p75 | n_changes, matched, matched_share, n_events, event_precision, error, median_abs_error | count, share, s |
| fast_ball | all | | fast_frames, near_rate, carry_rate | frames, share of fast frames |

**Tests.** `tests/test_poss_carries.py` on `fixture_poss_*`: the 14 carries (one of them a touch
by the defending team), durations, precision, frame-level AUC, switch timing with true and
pseudo-labels, the fast-ball rate against a per-frame loop, `--degrade`, and the shared AUC,
rolling-mean, velocity and matching helpers.

**Results on Sample Game 3.**

| | clean | degraded |
|---|---|---|
| carries, share of live frames | 851, 0.345 | 554, 0.188 |
| length p10 / p50 / p90 (s) | 0.64 / 1.24 / 2.80 | 0.60 / 1.08 / 2.40 |
| precision (carry team = label) | 0.989 | 0.993 |
| AUC at 0 / 1 / 2 / 5 / 10 s | 0.779 / 0.880 / 0.924 / 0.940 / 0.930 | 0.668 / 0.758 / 0.816 / 0.877 / 0.892 |
| `auc_defined` at 0 / 10 s | 0.989 / 0.935 | 0.993 / 0.919 |
| changes matched by a switch within 5 s, median error | 106 of 280, 0.90 s | 62 of 280, 1.08 s |
| switches near a change | 0.845 | 0.766 |
| fast-ball frames near a player, in a carry | 0.164, 0.000 | 0.157, 0.000 |

When a carry exists it names the right team almost always; the limit is coverage. Smoothing
trades sharpness for coverage, and on the degraded copy the AUC keeps rising to 10 s. Fewer than
half of the turnovers are followed by a carry of the new team within 5 s: many possessions start
with a first-time pass, so carries alone cannot place most changes.

**Questions for interpretation.**
- Is precision on the match (against pseudo-labels) close to the reference game's? A much lower
  value points to either noisy pseudo-labels (check `agreement`) or false carries by defenders
  pressing the carrier; the per-team split tells which team suffers.
- What carry coverage does the match reach? Fragmented tracks break carries (a slot change ends
  one), so compare it with the degraded copy rather than the clean one.
- At which window does `auc` peak? That window is the scale at which carries should enter the
  model; `auc_defined` falling with the window shows how much of the gain is coverage.
- Is `carry_rate` on fast balls 0 while `near_rate` is not? Then the velocity test removes the
  balls passing near or over a player, as intended.

### Pass chains (`poss_pass_chains.py`)

**Question.** Is "team T completes two passes in a row" (the organisers' own proxy in
CHALLENGE.md) a reliable sign that T has the ball, how much live time does it cover, and does a
change of chain team mark a turnover? Feeds the emission model and the change detector.

**Definitions.** Contacts: runs of frames in which the ball moves with one player (the carry
test: within `--carry-dist-m`, velocity difference below `--carry-dv`), at least
`--touch-min-s` long; neighbouring contacts of the same slot in one live run merge. A pass links
two consecutive contacts of different players of one team, in one live run, at most
`--pass-max-s` apart, with a defined ball speed above `--pass-min-ms` somewhere between them
(a flight with no defined speed, e.g. the ball missing throughout, is not a pass). A chain is a
maximal run of contacts linked by passes; its frames run from its first contact's start to its
last contact's end, flights included. The proxy: chains of at least `--min-chain` passes, score
+1 home / -1 away. Precision per chain length (`1` ... `5+` passes) and for the proxy. Switches:
the first frame of a qualifying chain whose team differs from the previous qualifying chain in
the same live run.

**Defaults and why** (norms, not tuned). `--pass-min-ms 5`: a soft pass; slower balls are
dribbles or rolling loose balls. `--pass-max-s 5`: a long ball's flight with a bounce.
`--touch-min-s 0.2`: the shortest touch the 0.2 s velocity lag can see. `--min-chain 2`: the
CHALLENGE.md proxy ("more than two consecutive passes" is read here as two completed passes,
i.e. three touches; `--min-chain 3` is the strict reading). One-touch passes never move with
the player, so they break chains: chains are undercounted, never overcounted by this.

**CSV schema** (`poss_pass_chains_summary.csv`), after the shared rows (params add touch_min,
pass_min, pass_max, min_chain, carry_dist, carry_dv):

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| passes | all / H / A | | n_contacts, n_passes | count |
| chains | all | `1` `2` `3` `4` `5+` | n_chains | count |
| coverage | all / chain / min_chain | | live_frames, frames, share | frames, share of live frames |
| precision | length | `1` ... `5+` / `min_chain` | labelled_frames, precision | frames, share |
| separation | chain | window s | auc, auc_defined, coverage | AUC, share |
| timing | chain_switch | `` / p25 / p50 / p75 | as for carries | |

**Tests.** `tests/test_poss_pass_chains.py`: 14 contacts, 6 passes (a flight with no defined
speed is not a pass), chains of 4, 1 and 1 passes, coverage, precision, AUC, the switch with
`--min-chain 1`, pseudo-labels, and the pass rule on small cases.

**Results on Sample Game 3.**

| | clean | degraded |
|---|---|---|
| passes found (home / away) | 312 / 265 (the events list 1,121 `PASS` events) | 176 / 148 |
| chains of >= 2 passes, share of live frames | 0.562 | 0.273 |
| precision, >= 2 passes (1 pass) | 0.977 (0.952) | 0.983 (0.973) |
| AUC at 0 / 2 / 10 s | 0.886 / 0.896 / 0.876 | 0.727 / 0.748 / 0.780 |
| changes matched by a chain switch within 5 s, median error | 27 of 280, 0.60 s | 10 of 280, 0.80 s |

A chain names the right team about 98% of the time, and precision barely depends on its length:
even one completed pass is 95% right. The proxy's limit is coverage, which halves on the degraded
copy, and timing: chains are long and few, so a switch of chain team matches fewer than one
turnover in ten.

**Questions for interpretation.**
- How many passes does the match yield per minute of live play, compared with the degraded
  copy? Far fewer means fragmented tracks break contacts and the proxy covers little.
- Is precision on the match close to 0.95 or more against pseudo-labels? Then "a completed pass
  by T" can act as a near-certain anchor for T, and the model can trust it over weaker signals.
- Does precision rise with chain length? If not (as on Sample Game 3), `--min-chain 1` buys
  coverage at no cost.
- Chain switches are rare and late: use chains to hold a label, not to place a change.

### Team shape (`poss_shape.py`)

**Question.** Does the team in possession spread out, push up and keep fewer players behind the
ball, clearly enough to tell who has the ball without seeing it? Feeds the ball-free part of the
emission model, the part that must carry the 40% of live play without a ball.

**Definitions.** Per team and live frame, in the team's attacking frame (x toward the opponent
goal), over its present players minus the deepest one (taken as the goalkeeper; "outfield"),
with at least `--min-players` (3) of them: `width` (y range), `length` (x range), `hull_area`
(convex hull, m^2), `centroid_advance` (mean x), `centroid_velocity` (median of the outfield
players' own x velocities over `--lag` inside the live run; a centroid's velocity jumps whenever
a fragmented track enters or leaves, a per-player median does not), `line_depth` (x of the deepest outfield player = metres from the own goal
line), `goal_side` (share of outfield players behind the ball, or behind the opponent outfield
centroid when the ball is missing). Hypothesis: the possessing team has the larger value, except
`goal_side` (smaller); the score toward "home has the ball" is that sign times home minus away
(`hypothesis` rows hold the sign). `diff`: possessing minus defending, raw (mean, median, n).
Separation as in "Separation and timing", three times: over all labelled live frames (`:all`),
those with the ball present (`:ball`, observed or interpolated) and those without (`:no_ball`);
the split uses the frame's own ball flag, the window may average over both. `line_depth` p25 /
p50 / p75 per team when it defends and when it attacks.

**CSV schema** (`poss_shape_summary.csv`), after the shared rows (params add min_players):

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| coverage | feature | | share | share of live frames (both teams defined) |
| hypothesis | feature | | sign | +1 possessing larger, -1 smaller |
| diff | feature | all / ball / no_ball | mean, median, n | feature unit, frames |
| separation | `<feature>:<split>` | window s | auc, auc_defined, coverage | AUC, share |
| line_depth | `H:defending` / `H:attacking` / `A:...` | `` / p25 / p50 / p75 | n, depth | frames, m from own goal line |

**Tests.** `tests/test_poss_shape.py`: every feature on two frames (one with the ball, one
without), the hull, coverage, width and hull differences, the width AUC, line-depth quantiles
per team and state, pseudo-labels, `--degrade`.

**Results on Sample Game 3** (AUC at the frame / best window, all frames; then no-ball frames of
the degraded copy at that window):

| feature | median poss - def (clean) | clean | degraded | degraded, no ball |
|---|---|---|---|---|
| width | +15.9 m | 0.919 / 0.923 (10 s) | 0.918 / 0.923 | 0.916 |
| hull_area | +524 m^2 | 0.920 / 0.922 (5 s) | 0.918 / 0.922 | 0.919 |
| length | +4.6 m | 0.838 / 0.847 (10 s) | 0.835 / 0.847 | 0.844 |
| goal_side | -0.4 | 0.815 / 0.850 (10 s) | 0.790 / 0.858 | 0.845 |
| centroid_velocity | +0.7 m/s | 0.649 / 0.655 (10 s) | 0.649 / 0.655 | 0.670 |
| centroid_advance | +8.3 m | 0.606 / 0.620 (10 s) | 0.606 / 0.621 | 0.606 |
| line_depth | +1.5 m | 0.505 / 0.523 (10 s) | 0.506 / 0.524 | 0.509 |

Width and hull area separate the teams at about 0.92 on every frame, with or without the ball,
and need no smoothing: they are the strongest ball-free signal so far. (The clean game's no-ball
frames are only 0.4% of live play, so its no-ball column is not shown.) Line depth hardly
differs: both Sample Game 3 teams hold their line about 33-35 m from goal whether attacking or
defending, so there is no low block there.

**Questions for interpretation.**
- Do width and hull area keep an AUC near 0.9 against the match pseudo-labels? If so they
  are the backbone of the ball-free emission; check `:no_ball` especially.
- Is Athletic's line depth when defending well below Barcelona's? That is the "low block"
  assumption, tested rather than assumed; if both are similar, a model must not rely on it.
- Is `centroid_advance` weak (as on Sample Game 3) because both teams move together? Then
  position on the pitch says little about possession, and `poss_ball_field.py` should confirm it.
- Does `centroid_velocity` peak at a short window? It is a change signal (see
  `poss_reversal.py`), not a holding signal. (It was first the centroid's own velocity; on the
  degraded copy that gave ten times the direction flips of the clean game, because every track
  entering or leaving the set moves the centroid. The per-player median is as stable degraded as
  clean.)

### Ball position vs ball direction vs team drift (`poss_ball_field.py`)

**Question.** Which tells who has the ball better: where the ball is, which way it moves, or
which way both teams drift? If ball position underperforms the two direction signals, it is
confounded (high pressing, build-up from the back and clearances all put the ball in one team's
half while the other has it). Feeds the choice of position vs direction features in the model.

**Definitions.** Scores toward "home has the ball", in home's attacking frame:
`ball_position` = interpolated ball x minus half the pitch length (positive = away half);
`ball_direction` = sign of the ball's x velocity over `--lag`, NaN below `--min-ms`;
`team_drift` = +1 / -1 when both teams (median outfield player x velocity, as in
`poss_shape.py`) move toward the away / home goal at `--min-ms` or more, NaN otherwise (no ball needed). Separation per
window as in "Separation and timing". `thirds`: per team, labelled ball-present live frames by
the ball's third of the pitch in that team's attacking frame (`defensive` [0, 35), `middle`,
`attacking` [70, 105] m), and the share where that team has the ball.

**Defaults and why.** `--min-ms 0.5`: below it a direction is position jitter or a standing
team (a norm, not tuned).

**CSV schema** (`poss_ball_field_summary.csv`), after the shared rows (params add min_ms):

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| coverage | ball_position / ball_direction / team_drift | | share | share of live frames |
| separation | signal | window s | auc, auc_defined, coverage | AUC, share |
| thirds | `H:defensive` ... `A:attacking` | | n, share_team_has_ball | frames, share |

**Tests.** `tests/test_poss_ball_field.py`: team drift by hand (the reversal frames around each
turnover), spot values on three frames, and ball position, ball direction and thirds against an
independent per-frame recomputation from the fixture text (exact agreement). That check found
that the rolling mean's cumulative sums broke exact ties; the frame window now uses the values
themselves and smoothed means are rounded to 1e-9.

**Results on Sample Game 3** (AUC at the frame / at 10 s):

| signal | clean coverage | clean | degraded coverage | degraded |
|---|---|---|---|---|
| ball_position | 0.997 | 0.386 / 0.396 | 0.603 | 0.426 / 0.409 |
| ball_direction | 0.880 | 0.647 / 0.703 | 0.510 | 0.610 / 0.657 |
| team_drift | 0.612 | 0.624 / 0.615 | 0.610 | 0.624 / 0.615 |

Share of ball-present frames on which a team has the ball, by the ball's third in its attacking
frame (clean): home 0.672 / 0.513 / 0.394 and away 0.606 / 0.487 / 0.328 (defensive / middle /
attacking). On Sample Game 3 ball position is *reversed*: a team has the ball more often when
the ball is in its own third (build-up from the back, the keeper, the other team's attack ending
in a clearance) than in the opponent's. As a score it carries some information (0.61 if flipped),
but in the opposite direction from the naive reading and only weakly. Ball direction and team
drift are both weak (0.61-0.70); neither reaches the 0.75 bar. Team drift
is a change signal more than a holding signal: see `poss_reversal.py`.

**Questions for interpretation.**
- Is ball position on the match also below 0.5? If Barcelona press high and Athletic sit deep,
  the confound may differ by team: compare the two teams' `thirds` rows.
- Does ball direction beat ball position at every window? Then the model should use direction
  (or position only together with the team that has just touched it).
- Team drift with a low AUC and a coverage near 0.6 says both teams moving together happens
  often but does not by itself say who has the ball.

### Joint reversals (`poss_reversal.py`)

**Question.** At a turnover both teams turn around: the new possessor starts attacking, the old
one retreats. Can that joint reversal place a change of possession without the ball, and how
sharply? Feeds the change detector (the transition part of the model), which is where spellF1
is won or lost.

**Definitions.** Team velocity: the median x velocity of the outfield players (as in
`poss_shape.py`), in home's attacking frame. Direction state per live run with hysteresis: +1
once the velocity reaches `--min-ms`, -1 once it reaches -`--min-ms`, unchanged in between; a
flip is a change of state (the first state of a run is not one). A joint reversal is a home flip
with an away flip to the same direction within `--reversal-window-s`, placed at the later of
the two; its direction names the implied new possessor (both teams now moving toward the away
goal: home). Reported:

- *rate*: flips per team and joint reversals per live minute; joint count against the away
  flips shifted circularly within each half (`--null-*`): null mean, p5, p95, `p_emp`, lift;
- *timing*: changes matched by a reversal to the same team (as in "Separation and timing");
- *change detection*: AUC of the reversal indicator, smoothed per window, for "a label change
  within `--change-window-s`" (a different target from the other scripts' AUCs: change vs no
  change, not home vs away; `separation, reversal_change`);
- *state*: the possessor implied by the last reversal, held to the next one in the live run;
  separation per window (`separation, reversal_state`);
- *profile*: each team's velocity in the gaining team's attacking frame, pooled per 1 s bin
  over +-10 s around label changes, against the changes shifted circularly within each half
  (random live anchors). `gainer` is the team gaining possession, `loser` the other.

**Defaults and why** (norms, not tuned). `--min-ms 0.5`: a drifting team, not a standing one.
`--reversal-window-s 1`: two teams react to the same turnover within about a second.
`--change-window-s 1`: the change-detection target.

**CSV schema** (`poss_reversal_summary.csv`), after the shared rows (params add min_ms,
reversal_window, change_window, profile_window, profile_bin, null n / seed / min_shift):

| analysis | scope | bin | metric | unit |
|---|---|---|---|---|
| flips | H / A | | n, per_min | count, per live minute |
| reversals | all / null | | n, per_min, joint_share_of_home_flips; null_mean, null_p5, null_p95, p_emp, lift | |
| timing | reversal | `` / p25 / p50 / p75 | as for carries | |
| separation | reversal_change | `` / window s | n_positive, auc | frames, AUC (change vs no change) |
| separation | reversal_state | window s | auc, auc_defined, coverage | AUC, share |
| profile | anchors | | n_H, n_A | count |
| profile | gainer / loser | bin left edge, s | value, counted, null_mean, null_p5, null_p95 | m/s in the gainer's frame, frames |

**Tests.** `tests/test_poss_reversal.py`: the two planted reversals, rate, null (and that it is
skipped when a half is too short), timing with true and pseudo-labels, change-detection AUC,
state AUC, the profile per bin, hysteresis and pairing on small cases, `--degrade`.

**Results on Sample Game 3** (clean / degraded).

| | clean | degraded |
|---|---|---|
| flips per minute, home / away | 5.5 / 5.4 | 5.7 / 5.5 |
| joint reversals per minute; vs null mean | 3.7; 234 vs 21 | 3.8; 240 vs 22 |
| changes matched within 5 s; median error | 111 of 280; +2.0 s | 113 of 280; +1.8 s |
| reversals within 5 s of a same-team change | 0.44 | 0.44 |
| change-detection AUC at 0 / 2 / 10 s | 0.500 / 0.512 / 0.560 | 0.500 / 0.514 / 0.559 |
| implied-state AUC at 0 / 10 s (coverage) | 0.547 / 0.562 (0.69) | 0.569 / 0.586 (0.70) |

The gaining team's velocity around a change (clean, m/s in its attacking frame): -0.56 at
-3 s, -0.83 at -2 s, -1.02 at -1 s, -0.78 in the first second after, -0.20 in the second, +0.26
in the third. Teams do reverse together far more than chance (11 times the null), and the
turnaround is visible in the averaged profile, but it comes late: the gaining team keeps
retreating for about 1.5 s after the change, so a reversal lands about 2 s after the turnover.
Fewer than half of the turnovers produce one, and more than half of the reversals happen with no
turnover (teams also turn when a pass goes backwards). On its own it is neither sharp nor
reliable enough to place changes; it may still help as soft evidence that a change happened in
the last few seconds, combined with ball signals. The degraded copy gives the same numbers, so
the signal survives fragmented tracks once velocities come from players rather than the
centroid.

**Questions for interpretation.**
- Is the joint count on the match also far above its null? If not, the match's tracking breaks
  the signal and it should be dropped.
- Where does the gainer profile cross zero on the match? That lag is how far a reversal-based
  change would land after the real one; with spellF1 about 0.65 for a possession placed 1 s
  late (EVALUATION.md), a 2 s lag is costly unless corrected by a constant shift.
- Is the change-detection AUC near 0.5 at short windows (as on Sample Game 3)? Then reversals
  only help at the scale of several seconds.
