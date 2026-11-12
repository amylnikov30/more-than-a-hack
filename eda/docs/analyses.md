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
