# touch_possession: split the match at touches, decide possession on top

Between two touches the ball moves freely, so each piece of the ball track is one action by one
player. A ball in flight still belongs to the team that played it, so a change of possession
goes on the new team's **first** touch. One touch is not control, though. Take this sequence: H
passes, an A defender clips the ball, and it still reaches an H team-mate. The ground truth
calls all of it H. A "one touch = possession" rule gives H | A | H, three possessions where
there is one, and spellF1 punishes that heavily. So the touches are a list of events first, and
possession is decided in a separate step on top of them.

```
touch_possession/
├── README.md
├── touches.py                 the pipeline + the four measurements, on any EPTS/ATD game
├── validate_metrica.py        the same pipeline on Metrica Sample Game 3, against its events
├── touch_possession.ipynb     reads the two summary CSVs and plots them
└── tests/test_touches.py      synthetic match with hand-derived answers (pytest)
```

Both scripts import the EDA scripts from `../scripts` themselves; stdlib + numpy only. The
notebook needs pandas and matplotlib and borrows `../notebooks/plot_helpers.py` for the shared look.

## The pipeline (`touches.py`)

1. **Touches** are sudden changes in velocity, not in heading. Per graded half, on each stretch
   where the ball is visible, in metres: light Savitzky-Golay smoothing (7 frames), then
   `dv(t) = |mean velocity over the next w frames - mean velocity over the previous w|`. A
   touch is a peak of `dv` of at least `--min-dv-ms` (3 m/s), peaks at least 0.2 s apart.
   Heading is undefined for a slow ball and misses a kick from rest, a first touch that kills
   the ball and a touch onward in the same direction. The angle is still recorded.
   - **Tracker glitches**: a run of 1-2 frames next to a step faster than 40 m/s is the
     tracker on another object. Those frames are dropped, and the track either side meets
     across a short gap.
   - **Short gaps** (up to 0.5 s) are bridged with straight lines. If the ball leaves the gap
     on another line than it entered (velocity change >= `--min-dv-ms`, or a miss >= 1 m), a
     touch goes inside the gap where the two lines come closest.
2. **Attribution**: each team's closest approach to the ball over +-3 frames. The nearest team
   owns the touch if it is within `--max-dist-m` (2 m). Otherwise it is not a touch: a bounce, a
   post, a tracker jump or a lofted ball. **Contested**: the other team is also within the limit
   *and* less than `--contest-margin-m` (0.5 m) further away.
3. **Segments**: each stretch from one touch to the next is a `carry` (one team within 1.5 m on
   at least 60% of the visible frames, median ball speed at most 10 m/s), a `flight`, `still`,
   `out` or `missing` segment. Consecutive carries of one team are merged; the touches inside
   them are dribbles.
4. **Control rule**: keep one value, the last team in control. It switches to B only when B
   shows control:
   - two B touches in a row;
   - a B touch followed by a B carry of at least 1 s;
   - a B touch that slows the ball to 4 m/s or less with the B player within 1.5 m.

   The switch goes at B's **first** touch of that run, never at the moment control was
   confirmed, which is late. A contested touch counts only when a carry of at least 1 s follows
   it (for the carrier's team). A long ball gap clears a pending candidate but keeps the team in
   control. The naive rule (every uncontested touch switches) runs alongside for comparison.
5. **Dead ball**: D comes from `--dead-signals`. The default is the ball more than 1 m outside
   the pitch, or the ball still for 1 s or more on a restart mark. `missing` is available but
   flags about 40% of graded frames, far more than the share of real dead ball. The first
   touch after D, or after a half start, is a restart and sets the team without confirmation.
   Touches in the last 5 frames of a dead interval stay live, because that is where the restart
   kick is.

**Labels** (`touch_labels.csv`) are H/A from the switches and D on dead ball. A long gap
(> 0.5 s) with the same team on both sides becomes X; the scorer fills it with the previous
label, which costs nothing inside a possession on spellF1 but is a miss on macroF1. A long gap
that a turnover spans keeps the old team up to the new team's first touch, which is what the
scorer's fill would do anyway. Where the switch really goes inside the gap is left to the
ball-free features or the HMM.

### Where this departs from the plan, and why

| plan | here | why |
|---|---|---|
| 3-5 frames each side for the mean velocities | **12** (`--dv-half`) | The ATD ball is already low-pass filtered: a kick is spread over about 0.5 s. With w = 4 the match shows about 10 velocity changes per minute of visible ball; the count only levels off at w = 12 (29 per minute, the `timescale` rows). Sample Game 3 is flat from w = 1, with about 30 per minute. The choice comes from the unlabelled track, not the ground truth. |
| both teams inside the limit -> contested | plus a 0.5 m margin | On Sample Game 3 the plan's literal rule marks 23% of touches contested: a defender pressing at about 1 m while the ball sits 0.08 m from the toucher. Most misses of turnovers came from that. `--contest-margin-m inf` restores the literal rule. |
| drop jumps faster than 40 m/s | drop 1-2 frame glitches; `--jump-touch` optional | On the match, all 216 jumps were single-frame glitches (none left after dropping them). On Sample Game 3 the ball is rebuilt from the events and teleports to the recovering player at a recovery, so the validation runs with `--jump-touch`. On the match that flag changes nothing. |
| bridge short gaps with the RTS smoother | straight lines in and out | It is the same test ("did the ball leave on the line it entered?"). Only 20 gaps are short on this match, so a smoother would buy little. |

## Run on the VM (competition match)

`MTAH_DATA_DIR` must point at the data (`/srv/mtah/data` on the team VM). Run from the team-repo
root and write outside the repository (the script refuses otherwise). It takes about 15 s:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/touch_possession/touches.py "$HOME/eda_out/touch_possession"
```

With the public ground truth (it ships with the starter repository, not this one), the summary
also says where the truth switches relative to the touches. This is a convention check, for
example whether the GT switches at the interceptor's first touch. It is not a tuning target:

```bash
cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/touch_possession/touches.py "$HOME/eda_out/touch_possession" --gt /path/to/BAR-ATH_possession_gt_public.csv
```

Every threshold is an option (`--help`). Use a separate output folder per variant.

## Run on Metrica Sample Game 3 (open data)

Get the three files from `github.com/metrica-sports/sample-data` (`data/Sample_Game_3/`) into
a folder of their own, then:

```bash
python3 eda/touch_possession/validate_metrica.py "$HOME/eda_out/touch_validation" --raw /path/Sample_Game_3_tracking.txt --meta /path/Sample_Game_3_metadata.xml --events /path/Sample_Game_3_events.json --jump-touch
```

It runs the pipeline three times. `full` is the ball as given (already 32% missing, mostly dead
ball). `deleted_uniform` and `deleted_crowd` delete the ball in bursts until 41% of the graded
frames are missing. Burst lengths copy the competition match, where 132 of the 150 gaps last
at least 5 s and hold 99.7% of the missing frames; so most bursts are long (log-normal, median
15 s, 5-90 s) and 10% are short (1-12 frames). `deleted_crowd` places bursts more often where
players crowd the ball.

**Reading it.** This ball track is rebuilt from the event positions (median 0.06 m from the
toucher at a touch), so touch detection looks better here than it will on real tracking, and
the distance clusters say nothing about the match. The reference labels follow an event-data
convention: a recovery is a possession even when the ball is lost straight away. That is not
the competition's "controlled possession", so control rule against naive rule is not settled
here. What this game can settle is whether the touch detector, the attribution and the
mechanics of the rule work.

## Outputs, and what may leave the VM

| file | content | may leave the VM? |
|---|---|---|
| `touch_possession_summary.csv` | aggregates only (long format: analysis, scope, bin, metric, value, unit) | yes |
| `touches.csv` | **frame-indexed**: one row per touch | **no**, never commit it |
| `touch_labels.csv` | **frame-indexed**: `frame,label` from the control rule | **no** |
| `touch_validation_summary.csv` | aggregates on open data | yes |

`--no-tables` writes only the summary. stdout is aggregate only. The notebook reads only the
summary CSVs. Once run on the VM its outputs come from the competition match, so clear them
before a copy leaves the VM:
`jupyter nbconvert --clear-output --inplace eda/touch_possession/touch_possession.ipynb`.

## The summary rows

| analysis | what |
|---|---|
| `params` | every parameter, the team names behind H (team 0) and A (team 1), the halves |
| `funnel` | visible frames and runs, glitches dropped, peaks, touches by source (visible / gap / jump), attributed, contested, no player, touches in dead ball, dribbles, "no player but a fast bending ball" (likely lofted) |
| `timescale` | dv peaks per visible minute for w = 2, 4, 8, 12, 25 and dv >= 3, 5, 8, 12 m/s |
| `sweep`, `dv_hist` | touch counts and attributed share at dv >= 2, 3, 4, 6; histogram of every peak's dv |
| `m1_distance`, `m1_by_dv` | nearest-player distance at touches against random ball-visible frames (same +-3 frame closest approach): 0.25 m histogram, quantiles, shares within 1 / 1.5 / 2 / 3 m, modes and dip |
| `segments` | count, seconds and median length per segment type |
| `m2_consecutive` | consecutive uncontested touches with no half start, dead ball or long gap between: share that switch team, and of those, the share that switch back after one touch or within 2 s |
| `m3_gap_span` | for control-rule turnovers (last old-team touch -> first new-team touch) and for raw consecutive-touch switches (only dead ball breaks them here): no gap / short gap / long gap, and the longest gap's length bin |
| `switches`, `possessions`, `labels`, `dead`, `gaps` | turnovers by confirmation (two touches / carry / receive), restarts, confirmation delay; possession count, rate, mean and median length, share under 2 s (after the scorer's X fill); label shares; long gaps: same / differs (a turnover spans it) / unknown |
| `m4_tags`, `m4_offsets` | switches against ATTACKING / DEFENSIVE TRANSITION starts at +-2 s and +-5 s: share of switches near a tag, share of tags near a switch, both against 1000 circular shifts of the tags (`tag_gaps.py` null), team agreement, and the offset histogram (switch - tag start) |
| `gt_convention` | only with `--gt`: offsets from each GT turnover / restart to the nearest touch and switch of the new team; possession counts in the labelled window |

## Tests

```bash
python3 -m pytest eda/touch_possession/tests
```

The synthetic match (kick-off, passes, a clip that must not switch, an interception carried
across a 2 s gap, a ball out and a throw-in) and its expected touches, switches, labels and
measurements are worked out by hand in the test file's docstring. Unit tests cover each kind of
touch (reversal, kick from rest, same-direction speed-up, a killed ball), peak separation,
glitches, gap bridging, contested margins, every branch of the control rule, labels and the
fill, the measurements, and the validation helpers (approximate spellF1, one-to-one matching,
burst deletion, event parsing).
