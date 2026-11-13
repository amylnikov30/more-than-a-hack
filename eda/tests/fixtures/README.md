# Synthetic fixtures

Invented data in the competition's ATD / Smart Tagging formats. **No real match data.**
Rebuild with `python tests/make_fixtures.py` (deterministic, byte-identical).

| file | what |
|---|---|
| `fixture_atd.xml` | FIFA/EPTS metadata: teams T1/T2, 46 slots `Track 1..46`, channels x, y, speed |
| `fixture_atd.txt.gz` | raw tracking, frame_id 1..3000 (video 0..2999), `x,y,speed` per slot, `NaN,NaN,NaN` when missing. 22 slots tracked, 24 always NaN. Gzipped: plain text is ~2 MB |
| `fixture_smart_tagging.xml` | SportsCode XML, 7 instances (6 SET PIECES in mixed case, 1 BUILD UP) |

Loads in kloppy 3.18 (`metrica.load_tracking_epts` on the unzipped txt, `sportscode.load`).

## Design

Graded frames for tests: `--frames 100-1499,1600-2899` (video frames, inclusive) = half 1
(1,400 frames) and half 2 (1,300 frames), 2,700 graded in total.

Ball gaps (video frames, inclusive), after clipping to graded frames:

| gap | frames | length | s | bin | half | touches half edge | SET PIECES overlap | near at last-seen frame |
|---|---|---|---|---|---|---|---|---|
| G1 | 200-204 | 5 | 0.20 | <0.5 | 1 | | | 2 (frame 199) |
| G2 | 300-311 | 12 | 0.48 | <0.5 | 1 | | | 2 (299) |
| G3 | 400-412 | 13 | 0.52 | 0.5-2 | 1 | | | 3 (399) |
| G4 | 500-549 | 50 | 2.00 | 2-5 | 1 | | yes `[475,525)` | 4 (499) |
| G5 | 700-899 | 200 | 8.00 | >=5 | 1 | | yes `[750,825)` lower case | 6 (699) |
| G6 | 1450-1499 (raw 1450-1530) | 50 | 2.00 | 2-5 | 1 | end | no: `[1400,1450)` ends at its start | 5 (1449) |
| G7 | 1600-1619 (raw 1590-1619) | 20 | 0.80 | 0.5-2 | 2 | start | | 1 (1589, ungraded) |
| G8 | 1800-1830 | 31 | 1.24 | 0.5-2 | 2 | | | 3 (1799) |
| G9 | 2000-2099 | 100 | 4.00 | 2-5 | 2 | | no: BUILD UP overlaps, `[2100,2150)` starts after | 4 (1999) |
| G10 | 2500-2649 | 150 | 6.00 | >=5 | 2 | | yes `[2625,2750)` | 6 (2499) |
| G11 | 2700-2701 | 2 | 0.08 | <0.5 | 2 | | | 2 (2699) |

Ignored (not graded): ball NaN at 0-49, 1550-1559 (halftime), 2950-2999.
Every other ball-visible frame has exactly 1 player within 5 m (slot 1 at 1.0 m), plus two
decoys at 5.2 m and 5.1 m on every frame. Background players are 12-24 m away.

Bins are half-open in seconds: `[0, 0.5)`, `[0.5, 2)`, `[2, 5)`, `[5, inf)`; length in
frames / 25. "Over 2 s" means length >= 50 frames. Smart Tagging frames are
`round(seconds * 25)`, end exclusive; a gap `[s, e]` overlaps a tag `[a, b)` iff `a <= e` and
`b > s`.

## Expected answers

**(a) gap lengths** - counts / missing frames / share of missing frames

| | <0.5 s | 0.5-2 s | 2-5 s | >=5 s | total |
|---|---|---|---|---|---|
| overall | 3 / 19 / 3.00% | 3 / 64 / 10.11% | 3 / 200 / 31.60% | 2 / 350 / 55.29% | 11 gaps, 633 of 2,700 graded frames missing (23.44%) |
| half 1 | 2 / 17 / 5.15% | 1 / 13 / 3.94% | 2 / 100 / 30.30% | 1 / 200 / 60.61% | 6 gaps, 330 of 1,400 (23.57%) |
| half 2 | 1 / 2 / 0.66% | 2 / 51 / 16.83% | 1 / 100 / 33.00% | 1 / 150 / 49.50% | 5 gaps, 303 of 1,300 (23.31%) |

Gaps touching a half edge: 2 (G6, G7).

**(b) set pieces** - gaps >= 2 s: 5; overlapping a SET PIECES interval: 3 (60.0%).
By bin: 2-5 s 1 of 3, >=5 s 2 of 2. With a +-1 s tolerance on the tags: 5 of 5.

**(c) crowding (players within 5 m of last-seen ball)**

- at gap start, all 11 gaps: mean 38/11 = 3.4545
- gaps >= 2 s (5): mean 25/5 = 5.0; gaps < 2 s (6): mean 13/6 = 2.1667
- ball-visible graded frames: 2,067. Using all of them (sample size >= 2,067):
  mean 2094/2067 = 1.01306. With a smaller sample the value depends on the seed, but is
  between 1.0 and 1 + 27/n for sample size n (all 10 crowded frames drawn), and identical
  across runs with the same seed.

---

# Flip fixture: analysis (d)

`fixture_flip_atd.xml` + `fixture_flip_atd.txt` (plain text, about 20 KB). Video frames 0..199
(frame_id 1..200), graded `--frames 10-99,110-189`. The ball sits at (0.5, 0.5) when visible.
There are 6 slots; teams come from the metadata (`T1` is the first team, home; `T2` is away):

| slot | team | default offset from ball (m) |
|---|---|---|
| 1 | home | (2, 0) -> nearest by default, 2 m |
| 2 | away | (0, 3) |
| 3 | home | (10, 0) |
| 4 | away | (0, 12) |
| 5 | **referee** (`teamId="REF"`, not a team) | (-20, 0) |
| 6 | home | always NaN |

Per-frame overrides: 30 and 85 put slot 1 at (6, 0), so away is nearest at 3 m. 59 puts the
referee at (0.5, 0); it must be skipped, leaving home at 2 m. 79 is a **tie**: slots 1 and 4 are
NaN, slot 2 (away) is at x = 0.25 and slot 3 (home) at x = 0.75, both 26.325 m away; the lowest
slot wins, so the answer is away. 114 has slots 1-4 NaN and only the referee present, so there is
no player. 175 puts slot 1 at (15, 0) and slot 2 at (0, 7), so away is nearest at 7 m.

Ball NaN: 0-12, 20-29, 40-44, 60-74, 80-84, 115-119, 125-174, 180-199.

| gap (graded) | len | bin | half | last-seen -> team, m | reappearance -> team, m | result | with `--flip-max-dist 5` |
|---|---|---|---|---|---|---|---|
| 10-12 | 3 | <0.5 | 1 | none (run starts at 0) | 13 | not eval: no_last_seen | same |
| 20-29 | 10 | <0.5 | 1 | 19 H 2 | 30 A 3 | **flip** | flip |
| 40-44 | 5 | <0.5 | 1 | 39 H 2 | 45 H 2 | no flip | no flip |
| 60-74 | 15 | 0.5-2 | 1 | 59 H 2 (referee skipped) | 75 H 2 | no flip | no flip |
| 80-84 | 5 | <0.5 | 1 | 79 A 26.3 (tie) | 85 A 3 | no flip | not eval: too_far |
| 115-119 | 5 | <0.5 | 2 | 114 none | 120 H 2 | not eval: no_player | same |
| 125-174 | 50 | 2-5 | 2 | 124 H 2 | 175 A 7 | **flip** | not eval: too_far |
| 180-189 | 10 | <0.5 | 2 | 179 H 2 | none (NaN to end of file) | not eval: no_reappearance | same |

Expected (d), no distance cap:

- overall: 8 gaps, 5 evaluable, 2 flips, rate 0.4; not evaluable: no_last_seen 1,
  no_reappearance 1, no_player 1, too_far 0.
- half 1: 5 gaps, 4 evaluable, 1 flip (0.25). Half 2: 3 gaps, 1 evaluable, 1 flip (1.0).
- by bin: <0.5 s 6 gaps / 3 evaluable / 1 flip; 0.5-2 s 1 / 1 / 0; 2-5 s 1 / 1 / 1;
  >=5 s 0 / 0 / 0.

With `--flip-max-dist 5`: 3 evaluable, 1 flip, too_far 2 (<0.5 s 2 evaluable / 1 flip;
2-5 s 0 evaluable).

**Reference pairs.** All 67 ball-visible graded frames are used (sample size >= 67).
Visible half 1: 13-19, 30-39, 45-59, 75-79, 85-99; half 2: 110-114, 120-124, 175-179. The
nearest team is home everywhere except 30, 79, 85, 175 (away) and 114 (none).
Delta = round(median clipped length of the bin's gaps): <0.5 s median of 3,5,5,5,10,10 = 5;
0.5-2 s = 15; 2-5 s = 50.

| bin | Delta | pairs kept (f -> f + Delta) | flips | rate | with cap 5 m |
|---|---|---|---|---|---|
| <0.5 s | 5 | 13-14, 30-34, 45-54, 85-94: 27 | (30,35), (85,90): 2 | 0.0741 | same, 27 / 2 |
| 0.5-2 s | 15 | 15-19, 30-39, 75-79: 20 | (15,30), (30,45), (79,94): 3 | 0.15 | (79,94) dropped: 19 / 2 |
| 2-5 s | 50 | 35-39, 45-49: 10 | (35,85): 1 | 0.1 | same, 10 / 1 |

Half 2 contributes no pairs. **Main fixture:** slot 1 (home) is nearest at 1 m on every visible
frame, so all 11 gaps are evaluable with 0 flips, and the reference has 0 flips.

**Hand-built reference case** (`test_reference_pairs_hand_case`). Frames 0-19: half 1 is 0-9,
half 2 is 10-19. The ball is missing at 4. The nearest team is H everywhere except 6 (A) and 12
(none). The nearest distance is 1 m except at frame 5 (9 m). Sample [0,1,2,3,5,6,7,8,9,11,12,15,17], Delta 3.
Kept: (0,3), (2,5), (3,6) flip, (5,8), (6,9) flip, (11,14), (15,18), so n = 7 with 2 flips.
Dropped: (1,4) ball missing; (7,10), (8,11), (9,12) cross halves; (12,15) no player; (17,20)
out of range. With the 5 m cap, (2,5) and (5,8) are also dropped: n = 5, 2 flips.

# Analysis (e): set-piece null

- **Offset 0** for every half reproduces (b) exactly: main fixture 3 of 5 (2-5 s 1/3,
  >=5 s 2/2); with +-1 s tolerance 5 of 5.
- **Wrap-around:** a half block of 200-299 (L = 100) and tag [280, 295) shifted by 10 gives
  [290, 300) and [200, 205). A tag [90, 210) across blocks 0-99 and 200-299 is clipped to
  [90, 100) and [200, 210), and each piece only ever moves within its own block.
- **Same seed** gives identical null statistics. On the main fixture `--null-min-shift 5` is
  needed, because the halves are only 56 s and 52 s long.
- **Tags exactly on the gaps** (one half, L = 5000; 5 gaps of 50 frames at irregular starts
  300, 1100, 1750, 3020, 4400; tags equal to the gaps; min_shift 1500 frames; K = 1000): real
  100%. Under a shift each gap is hit with probability about 5 x 99 / 5000 = 0.1, so "all 5 hit"
  should essentially never happen. Expected p_emp = 1 / 1001 (assert < 0.005).
- **Tags independent of the gaps** (same gaps; 10 tags of 100 frames starting at 0, 500, 1050,
  1500, 2000, 2500, 3200, 3700, 4200, 4700): real = 1 of 5. Only the gap at 1100 overlaps, via
  [1050, 1150). [300, 350) misses [0, 100) and [500, 600); [1750, 1800) misses [1500, 1600) and
  [2000, 2100); [3020, 3070) misses [2500, 2600) and [3200, 3300); [4400, 4450) misses
  [4200, 4300) and [4700, 4800). Per-gap hit probability under a shift is about
  10 x 149 / 5000 = 0.3, so P(null >= 1 of 5) is about 1 - 0.7^5 = 0.83. Expected p_emp is
  about 0.8 (assert > 0.5).

# `sample_long_gaps.py`: long-gap sampler

Main fixture, `--frames 100-1499,1600-2899`. Gaps come from `ball_gaps.find_gaps`. Edge gaps
(G6, G7) are dropped because their true length is unknown. That leaves **9 gaps**, with these
lengths in frames: G11 2, G1 5, G2 12, G3 13, G8 31, G4 50, G9 100, G10 150, G5 200.

**Rules (decided before coding)**

- Without `--min-s`/`--max-s`, the threshold is `np.quantile(lengths, q)` (linear) over the 9
  gaps, and a gap is eligible if `length >= threshold`.
- With them, a gap is eligible if `min_s * 25 <= length < max_s * 25`: half-open, like the
  `ball_gaps` bins. A missing bound is 0 or infinity.
- `--exclude a-b,...` drops a gap that shares at least one frame with an inclusive range.
- The sample size is k = min(n, eligible). Quotas per half are filled one gap at a time: the
  next gap goes to the half with the smallest quota that still has unused eligible gaps, with
  ties going to the lower half number. Within a half, `default_rng(seed).choice` picks gaps
  without replacement, halves in order. Rows are sorted by start frame.
- Columns: `half, start_frame, end_frame, gap_s, view_from, view_to, category, notes`.
  `view_from = ts(max(0, start - 75))`, `view_to = ts(end + 75)`, `ts(f) = h:mm:ss.ss` of f / 25.

**Expected**

| case | threshold / range | eligible (half 1 / half 2) | n | rows per half |
|---|---|---|---|---|
| q 0.9 | 0.9 x 8 = 7.2 -> 150 + 0.2 x 50 = 160 fr (6.4 s) | G5 (1 / 0) | 8 | 1 / 0 |
| q 0.5 | position 4 -> 31 fr (1.24 s) | G4, G5 / G8, G9, G10 (2 / 3) | 8 | 2 / 3 (all 5) |
| q 0.5 | same | same | 3 | 2 / 1 (old code: 4 rows) |
| q 0.5 | same | same | 4 | 2 / 2 |
| q 0.5, `--exclude 450-920` | same threshold; G4, G5 excluded | 0 / 3 | 2 | 0 / 2 (old code: 1) |
| `--min-s 2 --max-s 6` | [50, 150) fr | G4 / G9 (G10 = 6.00 s exactly is out) | 8 | 1 / 1 |
| `--min-s 2` | [50, inf) | G4, G5 / G9, G10 | 8 | 2 / 2 |
| `--min-s 10` | [250, inf) | none | 8 | 0 rows: header-only CSV, no crash |

G5's row: `half 1, start 700, end 899, gap_s 8.0, view_from 0:00:25.00` (frame 625),
`view_to 0:00:38.96` (frame 974).

The same seed gives a byte-identical CSV. An output folder inside the repo is refused
(`SystemExit`).

**Records missing from the file.** In a 10-line raw file (frame_id 1..10, graded `0-9`), the
ball is NaN on frame_id 4-5 and the line for frame_id 8 is absent. The gaps are video 3-4
(length 2) and video 7 (length 1). They are separate, and video 7 counts as missing.

# `ball_gaps.py` robustness fixes

- **Spec without `endFrame`** (or `startFrame`): treated as open-ended (`startFrame` 0,
  `endFrame` infinite). With `endFrame="200"` removed from `fixture_flip_atd.xml`, the teams for
  video frame 150 and for video frame 10**6 are both `[0, 1, 0, 1, -1, 0]`. Before the fix,
  `teams_for` returned `None`, so every gap became `no_player`. The flip analysis on that file
  matches the original: 8 gaps, 5 evaluable, 2 flips.
- **`--game` reaches every path lookup.** With a stand-in `games` module, `main([OUT, "--game",
  "XX"])` calls `require` only with `game="XX"`, and `half_video_blocks` with `"XX"`.
- **Smart Tagging is optional in games mode.** If the stand-in `require(["smart_tagging"])`
  raises `IOError`, the run still finishes on the main fixture (`atd_raw` = `fixture_atd.txt.gz`,
  `atd_meta` = `fixture_atd.xml`, blocks `100-1499,1600-2899`) with `set_pieces` and
  `set_pieces_null` both `None`, 11 gaps, and a stdout note that the tags file is missing. If
  `atd_raw` or `atd_meta` is missing, it still exits (`SystemExit`).
- **The refusal message names the calling script's folder:** `ball_gaps.py` suggests
  `~/eda_out/ball_gaps`, and `sample_long_gaps.py` suggests `~/eda_out/long_gaps`.

**`--csv-name`.** `--min-s 2 --csv-name gap_test.csv` writes `OUT/gap_test.csv`, with the same 4
rows as `--min-s 2`, and no `long_gap_sample.csv`. A name containing a path separator
(`sub/x.csv`) or not ending in `.csv` is a usage error (`SystemExit`).

# `tag_gaps.py`: gaps vs Smart Tagging codes

`fixture_tags.xml` (SportsCode XML, 10 instances) on top of the main ATD fixture, graded
`--frames 100-1499,1600-2899`. Frames are `round(s * 25)`, end exclusive. Tag times are chosen as
whole frames (s = f / 25), so rounding never matters.

| # | code (as written) | frames | Type | why |
|---|---|---|---|---|
| T1 | `SET PIECES` | [905, 1000) | Goal Kick | starts 5 fr after G5 reappears (900): **links**, but does not overlap G5 |
| T2 | `Set Pieces` | [560, 600) | Throw-in | starts 10 fr after G4 reappears (550): links |
| T3 | `SET PIECES` | [2100, 2200) | Corner Kick | starts on G9's reappearance (2100): links |
| T4 | `set pieces` | [1250, 1300) | Free Kick | no gap near: control |
| T5 | `SET PIECES` | [25, 75) | Kick Off (Start) | anchor ungraded: dropped from profiles, links and recall |
| T6 | `PROGRESSION` | [680, 800) | - | overlaps G5; starts 130 fr after G4 reappears: just outside W |
| T7 | `Long Ball` | [1490, 1650) | - | crosses halftime: overlaps G6 and G7 |
| T8 | `BUILD UP` | [1835, 1900) | - | starts 4 fr after G8 reappears (1831): a non-set-piece code that links |
| T9 | `SET PIECES` | [2660, 2700) | Throw-in | links G10 (2650) and G11 (2702); ends exactly where G11 starts: no overlap |
| T10 | `GOALS` | [2300, 2400) | - | nothing near: control |

## Rules (decided before coding)

- Codes are stripped and upper-cased. Groups are `code:<CODE>` for every code, plus
  `type:<Type>` for SET PIECES instances that have a `Type`.
- A tag's **anchor** is its start frame (or its end frame for end-aligned profiles). Its half is
  the graded block holding the anchor. A tag whose anchor is not graded is dropped from profiles,
  links and recall, and counted as `n_dropped`.
- **Reappearance** of a gap = `next_seen` from `find_gaps`. A gap is **linkable** only if
  `next_seen` is a graded frame in the gap's own half. G6 (`next_seen` 1531, halftime) is the only
  unlinkable gap here. Unlinkable gaps stay in the denominators and never link.
- A gap **links** to a group if some anchor a of that group, in the same half, has
  `|next_seen - a| <= W` (W = `--link-window`, default 5 s = 125 fr).
- A gap **overlaps** a group as in (b): inclusive gap `[s, e]` and tag `[a - tol, b + tol)`
  with `a - tol <= e` and `b + tol > s` (`--tag-tolerance`, default 0).
- **View A profile.** Bins are `[t + k*B, t + (k+1)*B)` for k = -W_A/B ... W_A/B - 1
  (B = `--bin`, W_A = `--window`; W_A must be a multiple of B). Only graded frames in the
  anchor's own half count. The value per bin is the pooled share: missing frames / counted
  frames over all instances. With no counted frames it is empty (NaN). Bins are labelled by
  their left edge in seconds.
- **View B strata** (s, half-open): `[0,2) [2,5) [5,10) [10,20) [20,35) [35,inf)`, on the
  clipped length as in (a).
- **View C.** For a threshold L, the gaps with `length >= L * 25` are the "long" gaps.
  Precision = long gaps that link to the group / long gaps. Recall = graded anchors of the group
  with at least one long gap linked / graded anchors. Each gap and each anchor counts once.
  Frame share = frames in long gaps / graded frames (the same for every group).
- **Nulls (A and B)** shift the group's anchors (A) or intervals (B) circularly within each
  half, as in (e). Window frames are not wrapped: they are clipped to the half, exactly as in the
  observed profile. With offset 0 the null equals the observed value.

## Expected

Reappearance frames: G1 205, G2 312, G3 413, G4 550, G5 900, G6 1531 (unlinkable), G7 1620,
G8 1831, G9 2100, G10 2650, G11 2702. Baseline P(missing) on graded frames = 633 / 2700 =
0.23444.

**View B, all 11 gaps** (strata: `[0,2)` G1 G2 G3 G7 G8 G11 = 6; `[2,5)` G4 G6 G9 = 3;
`[5,10)` G5 G10 = 2; the rest 0).

| group | overlap (tol 0) | overlap (tol 1 s) | link (W 5 s) |
|---|---|---|---|
| `code:SET PIECES` | 0 / 11 | 5 / 11 (G4 G5 G9 G10 G11) | 5 / 11: G4 G5 G9 G10 G11. By stratum `[0,2)` 1/6, `[2,5)` 2/3, `[5,10)` 2/2 |
| `code:PROGRESSION` | 1 / 11 (G5, `[5,10)`) | | 0 / 11 |
| `code:LONG BALL` | 2 / 11 (G6 `[2,5)`, G7 `[0,2)`) | | 0 / 11 |
| `code:BUILD UP` | 0 / 11 | | 1 / 11 (G8, `[0,2)`) |
| `code:GOALS` | 0 / 11 | | 0 / 11 |
| `type:Throw-in` | | | 3 / 11 (G4, G10, G11) |
| `type:Goal Kick` | | | 1 / 11 (G5) |
| `type:Corner Kick` | | | 1 / 11 (G9) |
| `type:Free Kick` | | | 0 / 11 |
| `type:Kick Off (Start)` | | | 0 / 11; n_dropped 1 |

With `--link-window 2` (50 fr), SET PIECES links G4 (10), G5 (5), G9 (0), G10 (10), and G11
(42): still 5 / 11. BUILD UP still links G8.

**View A, `--window 2 --bin 1`** (bins -2, -1, 0, 1 s = offsets [-50,-25), [-25,0), [0,25),
[25,50)):

| group, anchor | instances | -2 s | -1 s | 0 s | 1 s |
|---|---|---|---|---|---|
| `type:Throw-in`, start (560, 2660) | 2 | 50/50 = 1.0 | 30/50 = 0.6 | 0/50 = 0.0 | 2/50 = 0.04 |
| `type:Throw-in`, end (600, 2700) | 2 | 0/50 = 0.0 | 0/50 = 0.0 | 2/50 = 0.04 | 0/50 = 0.0 |
| `code:LONG BALL`, start (1490) | 1 | 15/25 = 0.6 | 25/25 = 1.0 | 10/10 = 1.0 (only 1490-1499 graded) | NaN (0 counted) |
| `type:Kick Off (Start)`, start | 0, n_dropped 1 | NaN | NaN | NaN | NaN |

**View C, `code:SET PIECES`, W 5 s.** Graded anchors: 905, 560, 2100, 1250, 2660 (5; 25 dropped).

| L | long gaps | linked | precision | anchors hit | recall | frame share |
|---|---|---|---|---|---|---|
| 0 s | 11 | G4 G5 G9 G10 G11 = 5 | 0.45455 | 905 560 2100 2660 = 4 | 0.8 | 633/2700 = 0.23444 |
| 2 s | G4 G5 G6 G9 G10 = 5 | G4 G5 G9 G10 = 4 | 0.8 | 4 | 0.8 | 550/2700 = 0.20370 |
| 5 s | G5 G10 = 2 | 2 | 1.0 | 905 2660 = 2 | 0.4 | 350/2700 = 0.12963 |

By Type at L = 2 s: Throw-in 2/5 = 0.4 precision, recall 2/2; Goal Kick 1/5 = 0.2, 1/1; Corner
Kick 1/5 = 0.2, 1/1; Free Kick 0/5, 0/1; Kick Off (Start) 0/5, recall NaN (0 graded anchors).
Throw-in at L = 5 s: 1/2 = 0.5, recall 1/2 = 0.5.

**Other checks.** `read_set_pieces` on `fixture_smart_tagging.xml` is unchanged. Null offset 0
equals the observed profile and observed shares. The vectorised profile null equals a slow
per-shift loop. The same seed gives identical null statistics. Like (e), the fixture needs
`--null-min-shift 5`.

# `speed_units.py`: declared vs derived speed

`fixture_speed_atd.xml` + `fixture_speed_atd.txt` (plain text, 12 lines). Video frames 0..11
(frame_id 1..12), graded `--frames 0-5,6-11`, pitch 105.3 x 68.0 m, 25 fps. Three player slots
and the ball, each group `x,y,speed`. Every movement is along y, in steps that are exact at 5
decimals, so derived speeds are whole multiples of 68 x 25 = 1,700 m/s per normalised unit.
The metadata declares the speed channel as `meters per second`, while the data are written in
km/h: the script must report the declared unit and the verdict side by side.

| object | x | y on video frame f | missing | declared speed | derived per step |
|---|---|---|---|---|---|
| slot 1 | 0.30 | 0.40000 + 0.00125 f | never | 7.65 | 0.085 m -> 2.125 m/s (ratio 3.6) |
| slot 2 | 0.50 | 0.50000 | never | 0.00 | 0 m/s (below the 1 m/s floor: no ratio) |
| slot 3 | 0.70 | 0.20000 + 0.0025 f, plus 0.1 from f = 9 on | f = 3 | 17.00 | 0.17 m -> 4.25 m/s (ratio 4.0); step 8->9 is a slot swap: 0.1025 x 1,700 = 174.25 m/s |
| ball | 0.50 | 0.30000 + 0.005 f | f = 6, 10 | 30.60 | 0.34 m -> 8.5 m/s (ratio 3.6) |

## Rules (decided before coding)

- A **pair** is (f - 1, f) with both video frames graded **in the same half**, both records in
  the file, and the object's x and y finite on both. Derived speed = Euclidean displacement in
  metres x fps. Declared speed is the third field on frame f (the later frame).
- **Ratio** = declared / derived, only on pairs with derived speed in `[--min-ms, cap]` and a
  finite declared speed. Defaults: `--min-ms 1` (below it frame noise dominates), cap
  `--cap-player 12` m/s (about the fastest sprint) and `--cap-ball 40` m/s (a very hard shot).
  Above the cap the pair is counted as `n_over_cap` (slot swaps, ball jumps).
- Player pairs only use slots that sit at the same position in both lines; if the two lines
  have different slot counts, the player pairs of that step are skipped (a spec boundary).
- **Percentiles** (numpy, linear): derived over all pairs, no cap (so swaps show in the tail);
  declared over every graded frame with a finite declared speed for a present object.
- **Verdict**, per object class with at least one ratio: `m/s` if the median ratio is within
  +-10% of 1, `km/h` if within +-10% of 3.6, else `unclear`. Overall: the shared answer if
  players and ball agree (or only one has ratios), `conflict` if they differ, `no declared
  speed` if neither has any ratio.

## Expected

Pairs: f in 1..5 and 7..11 (step 5->6 crosses halves).

- slot 1: 10 pairs at 2.125 m/s, ratio 3.6 each.
- slot 2: 10 pairs at 0 m/s, no ratio.
- slot 3: frame 3 is missing, so steps 2->3 and 3->4 drop: 8 pairs (f = 1, 2, 5, 7, 8, 9, 10,
  11). Seven at 4.25 m/s (ratio 4.0), one (8->9) at 174.25 m/s: over the cap.
- ball: frames 6 and 10 missing; pairs f = 1..5, 8, 9 = 7, all 8.5 m/s, ratio 3.6.

**Players.** 28 pairs, 1 over the cap, 17 ratios (ten 3.6, seven 4.0). Sorted ratio positions
0..16: median = position 8 = 3.6; p25 = position 4 = 3.6; p75 = position 12 = 4.0. Verdict
`km/h`.
Derived percentiles over 28 values (ten 0, ten 2.125, seven 4.25, one 174.25; position p x 27):
p50 = position 13.5 = 2.125; p99 = position 26.73 = 4.25 + 0.73 x 170 = 128.35;
p99.9 = position 26.973 = 4.25 + 0.973 x 170 = 169.66.
Declared over 35 present slot-frames (twelve 0.00, twelve 7.65, eleven 17.00; position p x 34):
p50 = position 17 = 7.65; p99 = position 33.66 = 17.0; p99.9 = 17.0.

**Ball.** 7 pairs, 0 over the cap, 7 ratios of 3.6: median 3.6, p25 3.6, p75 3.6. Verdict
`km/h`. Derived p50 = p99 = p99.9 = 8.5. Declared over 10 present frames: all 30.6.

**Overall verdict** `km/h`; declared unit from the metadata: `meters per second`.

**Other checks.** With `--frames 0-11` (one block) step 5->6 counts: slot 1 gains a pair
(11 ratios of 3.6), slot 2 gains one, slot 3 gains one ratio of 4.0 (5->6), and the ball does
not (frame 6 is missing). If the speed field is absent (`x,y` groups only, as in Metrica Sample
Game 3) there are no ratios and the verdict is `no declared speed`; the derived percentiles are
still reported. A missing line (frame_id 4 removed) breaks the steps 2->3 and 3->4 exactly
like a NaN.

# `ball_outside.py`: ball outside the pitch (clause a)

`fixture_outside_atd.txt` (plain text, 2,000 lines, 2 static player slots) and
`fixture_outside_tags.xml`. Video frames 0..1999 (frame_id 1..2000), graded
`--frames 0-999,1000-1999`, pitch 105.3 x 68.0 m. The ball sits at (0.5, 0.5) except:

| run | frames | ball (normalised) | beyond the line by | line | next frame |
|---|---|---|---|---|---|
| R1 | 100-124 (25) | x 1.03 | 3.159 m | goal line | 125: missing (125-199 NaN) |
| R2 | 300-309 (10) | y 1.025 | 1.7 m | touchline | 310: inside |
| R3 | 500-549 (50) | y -0.01 | 0.68 m | touchline | 550: missing (550-599 NaN) |
| R4 | 700-702 (3) | x -0.02 | 2.106 m | goal line | 703: inside |
| R8 | 800-809 (10) | x 1.03 | 3.159 m | goal line | 810: missing (810 NaN) |
| R9 | 811-820 (10) | x 1.03 | 3.159 m | goal line | 821: inside |
| R5 | 1200-1399 (200) | y 1.05 | 3.4 m | touchline | 1400: missing (1400-1449 NaN) |
| R7 | 1600-1604 (5) | x 1.03, y 1.05 | 3.159 m / 3.4 m | touchline (larger excess) | 1605: inside |
| R6 | 1990-1999 (10) | x 1.015 | 1.5795 m | goal line | none: half end |

Ball missing on 125-199, 550-599, 810, 1400-1449 = 176 frames, so 1,824 visible graded frames.

Tags (all `SET PIECES` unless noted, each 50 frames long, times written as frame / 25 s):
T1 Free Kick 20, T2 Goal Kick 200, T3 Throw-in 330, T4 Corner Kick 900, T5 Throw-in 1460,
T6 Free Kick 1900, T7 `BUILD UP` 310 (not a set piece: must be ignored).

## Rules (decided before coding)

- Excess beyond the goal lines `ex = max(-x, x - 1) x 105.3` m, beyond the touchlines
  `ey = max(-y, y - 1) x 68.0` m. A ball-visible graded frame is **out** at margin M if
  `max(ex, ey) > M` (strict). Its **line** is the goal line if `ex >= ey`, else the touchline.
- A **run** is a maximal stretch of out frames inside one graded half; a missing ball or a
  missing record ends it. A run's line is the line of its first frame. Lengths go in the
  `ball_gaps` bins `[0,0.5) [0.5,2) [2,5) [5,inf)` s.
- **What follows** a run ending at e: `half_end` if e is the last frame of its half, else
  `missing` if the ball is missing (or the record is absent) on e + 1, else `inside` (visible
  and not out at this margin).
- **Restart link**: a run links to a group (SET PIECES, or one `Type`) if a start a of that
  group in the same half has `|e - a| <= W` (`--link-window`, default 5 s = 125 fr). The
  **cross-tab Type** is the Type of the SET PIECES start nearest to e within W (ties go to the
  earlier start; no Type -> `(no Type)`; nothing within W -> `none`).
- **Null**: the `tag_gaps` link null (anchors shifted circularly within each half); a run is a
  point at e. Offset 0 reproduces the observed counts.
- All of it is computed for every margin in `--margin-m` (main, default 1) and
  `--margins-also` (default `0,2`).

## Expected

**Out frames and runs**

| margin | runs | out frames (goal / touch) | share of 1,824 visible | follows: missing / inside / half_end | SET PIECES link |
|---|---|---|---|---|---|
| 0 m | 9 (R1-R9) | 323 (58 / 265) | 0.177083 | 4 / 4 / 1 | 6 / 9 |
| 1 m | 8 (all but R3) | 273 (58 / 215) | 0.149671 | 3 / 4 / 1 | 6 / 8 |
| 2 m | 6 (R1 R4 R8 R9 R5 R7) | 253 (48 / 205) | 0.138706 | 3 / 3 / 0 | 4 / 6 |

Run lengths at 1 m: `<0.5s` 6 runs / 48 frames (R2 R4 R8 R9 R7 R6), `0.5-2s` 1 / 25 (R1),
`2-5s` 0 / 0, `>=5s` 1 / 200 (R5). By line at 1 m: goal line 5 runs (R1 R4 R8 R9 R6), touchline
3 (R2 R5 R7).

**Links at 1 m** (run end e -> starts within 125 fr):
R1 124 -> T1 (104), T2 (76): nearest Goal Kick. R2 309 -> T2 (109), T3 (21): Throw-in.
R4 702 -> none (T4 is 198 away). R8 809 -> T4 (91): Corner Kick. R9 820 -> T4 (80): Corner Kick.
R5 1399 -> T5 (61): Throw-in. R7 1604 -> none. R6 1999 -> T6 (99): Free Kick.

| group | links at 1 m | goal line | touchline |
|---|---|---|---|
| `code:SET PIECES` | 6 / 8 | 4 / 5 | 2 / 3 |
| `type:Corner Kick` | 2 (R8 R9) | 2 | 0 |
| `type:Free Kick` | 2 (R1 R6) | 2 | 0 |
| `type:Goal Kick` | 2 (R1 R2) | 1 | 1 |
| `type:Throw-in` | 2 (R2 R5) | 0 | 2 |

Cross-tab at 1 m (nearest Type): goal line Goal Kick 1, Corner Kick 2, Free Kick 1, none 1;
touchline Throw-in 2, none 1.

**Other checks.** `common.find_runs` on the main fixture's missing-ball mask returns the same
(half, start, end) as `ball_gaps.find_gaps`. Without a tags file the link and cross-tab parts
are skipped and the rest is unchanged.

# `ball_stationary.py`: ball stationary at a restart location (clause c)

`fixture_still_atd.xml` + `fixture_still_atd.txt` (2,000 lines; 4 player slots: 1-2 home `T1`,
3-4 away `T2`) and `fixture_still_tags.xml`. Graded `--frames 0-999,1000-1999`, pitch
105.3 x 68.0 m. Outside the still runs the ball moves along y = 0.7,
x = 0.3 + 0.0004 x (f mod 1000), i.e. 1.053 m/s, never near a mark. Players stand at x 0.3, 0.4
(home) and 0.6, 0.7 (away), except slot 3 at x 0.45 on frame 1349.

| run | frames | len (s) | ball | location | notes |
|---|---|---|---|---|---|
| S1 | 0-49 | 50 (2.0) | (0.5, 0.5) | centre spot | kick-off, both teams in own half |
| S2 | 200-262 | 63 (2.52) | (0.005, 0.01) | corner arc | 0.86 m from the corner; also in the touchline band |
| S3 | 400-436 | 37 (1.48) | (0.03, 0.55) | goal area | 3.159 m from the goal line, 3.4 m off centre |
| S4 | 600-674 | 75 (3.0) | (0.89554, 0.5) | penalty spot | right-hand end: 10.9996 m from the goal line |
| S5 | 800-830 | 31 (1.24) | (0.4, 0.99) | touchline band | 0.68 m inside the touchline |
| S6 | 1100-1119 | 20 (0.8) | (0.3, 0.3) | elsewhere | below 1 s |
| S7 | 1300-1349 | 50 (2.0) | (0.5, 0.5) | centre spot | away slot 3 at x 0.45 (5.3 m in the home half) on 1349: not a kick-off |
| S8 | 1500-1549 | 50 (2.0) | (0.5, 0.5) | centre spot | ball missing on 1520, jitter to x 0.502 on 1530: still one run; kick-off |
| S9 | 1700-1729 | 30 (1.2) | (0.6, 0.3) | elsewhere | |
| S10 | 1740-1769 | 30 (1.2) | (0.62, 0.3) | elsewhere | 10 moving frames after S9: a separate run |

Tags (`SET PIECES`, 50 frames each): T1 Kick Off 50, T2 Corner Kick 270, T3 Goal Kick 440,
T4 Throw-in 835, T5 Free Kick 1125, T6 Kick Off 1350, T7 Kick Off 1555.

## Rules (decided before coding)

- **Speed** on frame f = displacement between f - lag and f, in metres, x fps / lag
  (`--lag`, default 5 frames = 0.2 s, so a few centimetres of jitter per frame do not read as
  motion). Defined only if the ball is visible on both and both frames are graded in the same
  half.
- A window [f - lag, f] with speed < `--still-ms` (default 0.5) marks all its frames; a frame is
  **still** if it is marked and the ball is visible on it. A one-frame jump only breaks the
  windows that end or start on it, so the jump frame itself stays still.
- **Runs**: maximal still stretches in a half (`common.find_runs`), then merged when at most
  `--still-gap` frames (default 3) separate them in the same half. A run's length counts the
  bridged frames. Kept if length >= ceil(`--min-still-s` x fps) frames (main 1 s; also 0.5 and
  2 s via `--min-still-also`).
- **Location** of a run: the median x and y over its visible frames, in metres, folded so the
  geometry is symmetric (distance to the nearer goal line `dx`, nearer touchline `dy`; signed,
  negative outside). First match wins:
  1. centre spot: within `--spot-tol-m` (default 1) of (L/2, W/2);
  2. penalty spot: within the tolerance of (11, W/2) from either goal line;
  3. corner arc: within 1 m + tolerance of a corner;
  4. goal area: `dx <= 5.5 + tol` and `|y - W/2| <= 9.16 + tol`;
  5. touchline band: `|dy| <= --band-m` (default 2);
  6. elsewhere.
  IFAB marks are absolute metres (11 m, 5.5 m, 9.16 m = 3.66 + 5.5, 1 m arc); only the halfway
  line and the centre scale with `--pitch`.
- **Frame share** = run frames / graded frames (2,000 here). Length bins (s, half-open):
  `[0,1) [1,2) [2,5) [5,10) [10,inf)`.
- **Link / cross-tab / null**: as in `ball_outside.py`, with the run's last frame as the point.
- **Kick-off check** (needs `--meta`): a centre-spot run whose last frame has at least one
  player of each team and, with t = tol / L in normalised units, every home player at
  x <= 0.5 + t and every away player at x >= 0.5 - t, or the mirror. Counted per half.

## Expected

Window check for S8: valid still windows end at g in 1505..1549 except 1520 and 1525 (missing
ball) and 1530 and 1535 (jitter). Their union covers 1500..1549; dropping the missing 1520
leaves 1500-1519 and 1521-1549, 1 frame apart, so they merge into one run of 50 frames.

| min still | runs | centre / corner / goal area / penalty / touchline / elsewhere (runs, frames) | frames | SET PIECES link |
|---|---|---|---|---|
| 1 s (25 fr) | 9 (all but S6) | 3/150, 1/63, 1/37, 1/75, 1/31, 2/60 | 416 (0.208) | 6 / 9 |
| 0.5 s (13 fr) | 10 | as above, elsewhere 3/80 | 436 (0.218) | 7 / 10 |
| 2 s (50 fr) | 5 (S1 S2 S4 S7 S8) | 3/150, 1/63, 0/0, 1/75, 0/0, 0/0 | 288 (0.144) | 4 / 5 |

Lengths at 1 s: `1-2s` 4 (S3 S5 S9 S10), `2-5s` 5, the rest 0.

Links at 1 s (each run's end is within 125 frames of its own tag only): S1 49 -> T1 (1),
S2 262 -> T2 (8), S3 436 -> T3 (4), S5 830 -> T4 (5), S7 1349 -> T6 (1), S8 1549 -> T7 (6);
S4, S9, S10 none. SET PIECES by location: centre 3/3, corner 1/1, goal area 1/1, penalty 0/1,
touchline 1/1, elsewhere 0/2. By Type: Kick Off 3, Corner Kick 1, Goal Kick 1, Throw-in 1,
Free Kick 0 (S6 links Free Kick only at 0.5 s).

Cross-tab at 1 s: centre spot Kick Off 3; corner arc Corner Kick 1; goal area Goal Kick 1;
penalty spot none 1; touchline band Throw-in 1; elsewhere none 2.

Kick-off check at 1 s: half 1 1 of 1 centre-spot runs (S1); half 2 1 of 2 (S8 yes, S7 no).
Without `--meta` it is skipped.

# `signal_profiles.py` and `gap_stoppage.py`: per-frame signals (clause d)

`fixture_signal_atd.txt` (400 lines, 4 player slots), `fixture_signal_homography.csv` and
`fixture_signal_tags.xml`. Graded `--frames 0-199,200-399`, pitch 105.3 x 68.0 m. The tests
use `--lag 1` so that each player's speed is one frame step (the default lag is 5).

| slot | position on video frame f | speed |
|---|---|---|
| 1 | fixed (0.3, 0.3) | 0 |
| 2 | x 0.5, y 0.1 + 0.002 f | 0.002 x 68 x 25 = 3.4 m/s |
| 3 | x 0.6, y 0.1 + 0.002 m(f), m(f) = moving steps up to f; the steps ending on 101-149 and 301-349 do not move | 3.4 m/s, 0 on 101-149 and 301-349 |
| 4 | x 0.7, y 0.05 + 0.002 f; NaN on 100-149 | 3.4 m/s |

Ball: x 0.2 + 0.002 f, y 0.5 (5.265 m/s, always inside), NaN on the gaps Q3 20-24 (5 fr,
0.2 s), Q1 100-149 (50 fr, 2.0 s), Q4 250-269 (20 fr, 0.8 s), Q2 300-349 (50 fr, 2.0 s).
Homography: `reprojection_error` 0.5 and `valid` 1, except 3.0 on 100-149 and 2.0 on 300-349,
with `valid` 0 on 300-309. Tags: `SET PIECES` Throw-in [150, 200), Corner Kick [350, 400).

## Rules (decided before coding)

- **Signals**, one value per graded frame (NaN = undefined, not counted):
  - `ball_missing`: 1 if the ball is missing or the record is absent, else 0 (as in
    `tag_gaps` view A: frames beyond the file count as missing);
  - `ball_outside`: 1 if the ball is visible and out at `--margin-m` (default 1), else 0
    (NaN when the ball is missing);
  - `ball_still`: 1 if the frame lies in a `ball_stationary.py` still run of at least
    `--min-still-s` (default 1), else 0 (NaN when the ball is missing);
  - `slow_players`: among slots present on f and on f - lag (same half, same slot count), the
    share whose speed is below `--walk-ms` (default 2.0); NaN if there is none;
  - `reproj_error`: homography `reprojection_error` where `valid` is 1, else NaN;
  - `n_tracked`: number of player slots with finite x and y (referees included: slots are
    anonymous); NaN without a record.
- **Profile**: as `tag_gaps` view A, but the value per bin is the pooled mean of the signal:
  sum over defined frames / number of defined frames, over all instances. The null and the
  window clipping are the same. For `ball_missing` this is exactly view A.
- **Gap summary** (`gap_stoppage.py`): per gap from `ball_gaps.find_gaps`, the mean of
  `slow_players`, `reproj_error` and `n_tracked` over the gap's frames where defined (NaN if
  none). Per length bin `[0,2) [2,5) [5,10) [10,20) [20,35) [35,60) [60,inf)` s: n (gaps with a
  defined mean), median, p25, p75 (numpy, linear). `--clusters k`: k-means on standardised
  (ln length in s, slow share, reprojection error) over gaps with all three defined; best of 10
  seeded starts by inertia; clusters sorted by centre length; centres reported in original units
  (length as exp of the mean ln length).

## Expected (`--lag 1`)

Per-frame `slow_players`: 0.25 on most frames (slot 1 slow, 2-4 fast); 1/3 on 100 (slot 4
missing, slot 3's step moves) and 150 (slot 4 missing on 149); 2/3 on 101-149 (slots 1 and 3
slow, slot 4 missing); 0.5 on 301-349; NaN on 0 and 200 (no earlier frame in the half). Slot 3
moves on step 300 and 350, so 300 and 350 are 0.25.
`n_tracked`: 3 on 100-149, 4 elsewhere.

**Profiles, `--window 2 --bin 1`.** Throw-in at 150: bins [100,125) [125,150) [150,175)
[175,200). Corner Kick at 350: [300,325) ... [375,400).

| signal | group | -2 s | -1 s | 0 s | 1 s |
|---|---|---|---|---|---|
| slow_players | Throw-in | (1/3 + 24 x 2/3) / 25 = 0.653333 | 0.666667 | (1/3 + 24 x 0.25) / 25 = 0.253333 | 0.25 |
| slow_players | Corner Kick | (0.25 + 24 x 0.5) / 25 = 0.49 | 0.5 | 0.25 | 0.25 |
| slow_players | SET PIECES | 28.583333 / 50 = 0.571667 | 29.166667 / 50 = 0.583333 | 12.583333 / 50 = 0.251667 | 0.25 |
| n_tracked | Throw-in | 3 | 3 | 4 | 4 |
| reproj_error | Corner Kick | 2.0 (15 counted: 310-324) | 2.0 (25) | 0.5 | 0.5 |
| ball_missing | Throw-in | 1.0 | 1.0 | 0.0 | 0.0 |
| ball_outside, ball_still | SET PIECES | NaN (0 counted: inside Q1 and Q2) | NaN | 0 | 0 |

**Gap summary.** Q1: slow (1/3 + 49 x 2/3) / 50 = 0.66, reproj 3.0, tracked 3. Q2: slow
(0.25 + 49 x 0.5) / 50 = 0.495, reproj 2.0 (310-349 only), tracked 4. Q3 and Q4: 0.25, 0.5, 4.

| bin | n | slow median / p25 / p75 | reproj median / p25 / p75 | tracked median / p25 / p75 |
|---|---|---|---|---|
| 0-2s | 2 (Q3 Q4) | 0.25 / 0.25 / 0.25 | 0.5 / 0.5 / 0.5 | 4 / 4 / 4 |
| 2-5s | 2 (Q1 Q2) | 0.5775 / 0.53625 / 0.61875 | 2.5 / 2.25 / 2.75 | 3.5 / 3.25 / 3.75 |
| the rest | 0 | NaN | NaN | NaN |

`--clusters 2`: sizes 2 and 2. Short cluster: length exp(mean(ln 0.2, ln 0.8)) = 0.4 s, slow
0.25, reproj 0.5. Long cluster: 2.0 s, slow 0.5775, reproj 2.5.

**Reproduction.** On the main fixture with `fixture_tags.xml`, `signal_profiles` for
`ball_missing` equals `tag_gaps.view_a` (observed, and null with the same offsets) for every
SET PIECES group, starts and ends.

## `speed_units.py` additions: declared ball speed, ratio by speed band, ball displacement

`fixture_disp_atd.txt` (70 lines, frame_id 1..70), graded `--frames 0-69` (one half), pitch
105.3 x 68.0 m. One player slot moves along y by 0.004 per frame (0.272 m, 6.8 m/s, declared
24.48 = 6.8 x 3.6). The ball sits at x 0.5 and moves along y in five 10-frame segments separated
by 5-frame gaps (ball NaN), so no window or step crosses a segment:

| segment | frames | y step / frame | metres / frame | speed | declared |
|---|---|---|---|---|---|
| B1 | 0-9 | 0 | 0 | 0 m/s | 0.00 |
| B2 | 15-24 | 0.0005 | 0.034 | 0.85 m/s | 3.06, but NaN on 15 and 16 |
| B3 | 30-39 | 0.002 | 0.136 | 3.4 m/s | 12.24 |
| B4 | 45-54 | 0.01 | 0.68 | 17 m/s | 61.2 |
| B5 | 60-69 | 0.02 | 1.36 | 34 m/s | 122.4 |

Ball NaN on 10-14, 25-29, 40-44, 55-59.

## Rules (decided before coding)

- `--ball-cap-ms` is another name for `--cap-ball` (default 40 m/s): the ball's plausibility
  cap, used instead of the player cap. Existing rows do not change.
- **ball_declared**: on graded frames where the ball's x and y are present, the declared speed
  field is `nan` (missing or NaN), `zero` (exactly 0) or `positive` (> 0). Negative values are
  counted in none of the three (none are expected).
- **ratio_by_speed**: the pairs that enter the main ratio (derived speed in `[--min-ms, cap]`,
  declared finite), grouped by derived speed in half-open bands `[0,2) [2,5) [5,10) [10,20)
  [20,inf)` m/s; per band `median`, `p25`, `p75` (NaN when empty) and `n_ratio`. With the
  default floor of 1 m/s, band `0-2` holds only `[1, 2)`.
- **ball_disp**: for every graded frame f with f - w graded in the same half and the ball
  present on both (w = round(`--disp-window-s` x fps), default 25), the ball's displacement in
  metres between f - w and f, in half-open bands `[0,0.1) [0.1,0.25) [0.25,0.5) [0.5,1) [1,2)
  [2,5) [5,inf)` m; per band `share` of all such windows, plus `n_frames` (an extra metric for
  the n annotations).

## Expected

**ball_declared**: 50 visible frames: `nan` 2 (15, 16), `zero` 10 (B1), `positive` 38.

**ratio_by_speed** (default floor 1 m/s). Ball pairs inside segments: 9 per segment. B1 (0 m/s)
and B2 (0.85 m/s) are below the floor. B3: 9 pairs in `2-5`; B4: 9 in `10-20`; B5: 9 in `20+`
(34 m/s is under the 40 m/s cap); each ratio 3.6, so median = p25 = p75 = 3.6. Bands `0-2` and
`5-10`: n_ratio 0, quartiles NaN. Players: 69 pairs, all in `5-10`, ratio 3.6.

**ball_disp** with `--disp-window-s 0.2` (w = 5): 5 windows per segment (f = start + 5 ... end).
B1 0 m -> `0-0.1`; B2 5 x 0.034 = 0.17 m -> `0.1-0.25`; B3 0.68 m -> `0.5-1`; B4 3.4 m ->
`2-5`; B5 6.8 m -> `5+`. 25 windows: share 0.2 in each of those five bands, 0 in `0.25-0.5` and
`1-2`.

**Unchanged.** On `fixture_speed_atd.*` every existing row is identical, and with the default
1 s window (25 frames) there are no displacement windows (12 frames): every share is NaN and
`n_frames` 0.

# Possession fixture (`poss_*.py`)

`fixture_poss_atd.txt` / `.xml` (2,000 frames, frame_id 1..2000 = video 0..1999), with
`fixture_poss_events.json` (true labels, open Metrica events format),
`fixture_poss_tags.xml` (Smart Tagging pseudo-labels) and `fixture_poss_dead.csv` (the
`deadball.run` output format: `frame,dead,p_dead,llr_dead`). Built by `make_fixtures.py`
(`POSS_*` constants). Run with `--frames 0-999,1000-1999 --pitch 105 68`.

## Design

- **Slots.** 11 groups of `x,y` (no speed field, like Sample Game 3): slots 1-5 home `T1`
  "Alpha FC" (slot 1 the goalkeeper), 6-10 away `T2` "Beta Club" (slot 6 the goalkeeper),
  slot 11 a referee (`teamId="REF"`, not a `<Team>`). The metadata declares
  `attack_direction_first_half`: T1 `left_to_right`, T2 `right_to_left`.
- **Movement.** Goalkeepers stand at x = 5 / 100 m (swapped in half 2), y = 34. Each team's
  four outfielders sit at the team centroid plus (dx, dy) = (-6, -15), (-6, 15), (6, -8),
  (6, 8) m, dy times a width scale: 1.0 for the team in possession, 0.6 for the other, ramped
  linearly over 25 frames after 300, 550 and 1400. Home outfield centroid x (half 1): 40 m
  until 50, +1 m/s to 50 m at 300, -1 m/s to 42 m at 500, still to 549, +1 m/s from 550. Away =
  home + 10 m. Half 2: away centroid 50 m until 1050, +1 m/s to 64 m at 1400, -1 m/s after;
  home = away + 10 m. Referee at the midpoint of the two centroids, y = 60.
- **Ball.** Held 0.5 m in +x of a player, or in flight on a straight line between two hold
  points (`POSS_BALL`). Planted: a 3 s home carry (slot 2, 100-174), a home pass chain
  2 -> 3 -> 5 -> 4 (175-244), an intercepted pass 4 -> away 8 (280-299), an away touch without
  control (away 7 holds the ball 700-729 and passes it to home 5, 730-739; the events keep home
  in possession), and a contested loose ball between home 3 and away 10 (1200-1210: 30% of the
  way from home 3 to away 10 on frames with f // 2 even, 70% on the others, so the nearest
  team flips every two frames while away keeps possession).

| | frames (video, inclusive) |
|---|---|
| dead | 0-49 (before the kick-off), 500-549 (ball out by away, throw-in by home at 550), 1000-1049 (before the second kick-off) |
| home `H` | 50-299, 550-999, 1400-1999 |
| away `A` | 300-499, 1050-1399 |
| ball missing | 0-29, 150-152, 250-261, 400-449, 480-555, 1000-1039, 1100-1112, 1300-1424, 1990-1999 |

**Events** (`POSS_EVENTS`, video frames; the JSON stores `frame_id` = video + 1): kick-off T1
at 50; passes and a carry by T1; `BALL LOST` T1 280-300 then `RECOVERY` T2 at 300; T2 passes;
`BALL OUT` T2 480-500; `CARD` T2 at 520; `SET PIECE / THROW IN` T1 at 550; a ground challenge
won by T1 and lost by T2 at 700; a T1 carry 710-760. Half 2: kick-off T2 at 1050; an aerial
challenge won by T2 at 1200; `BALL LOST` T2 1380-1400 then `RECOVERY` T1 at 1400; a T1 carry.

**Tags** (`POSS_TAGS`, frames, end exclusive; team names in the tags are "Alpha" and "Beta
Club", so the map to the metadata names "Alpha FC" / "Beta Club" goes through shared tokens):

| code | frames | team | pseudo-label |
|---|---|---|---|
| `build up` (lower case) | 60-290 | Alpha | H |
| DEFENDING IN MIDDLE THIRD | 60-290 | Beta Club | H (the tagged team defends) |
| PROGRESSION | 310-490 | Beta Club | A |
| `Long Ball` | 320-360 | Alpha | H: disagrees with PROGRESSION, so 320-359 are dropped |
| SET PIECES | 540-560 | N/A | ignored |
| PROGRESSION | 700-750 | N/A | dropped (no team) |
| DEFENSIVE TRANSITION | 1100-1390 | Alpha | A |
| ATTACKING TRANSITION | 1410-1600 | Alpha | H |
| BALL IN THE BOX | 1500-1550 | Alpha | ignored |

## Rules (decided before coding)

**Live mask.** Live = graded and `dead == 0`. From `--dead CSV`: `dead` is 1 dead, 0 live, -1
not decoded; -1 and frames missing from the CSV are not live. From `--events` (reference game):
the union of three rules, end exclusive. (1) A restart is a `SET PIECE` event or any event
with subtype `GOAL KICK`. A half's first restart is preceded by dead frames from the half's
first graded frame; every later restart by dead frames from the `end` of the last event that
starts strictly before it (skipping `CARD`) up to the restart's `start`. (2) After `BALL OUT`,
or a `SHOT` with subtype `OUT`, dead from its `end` to the start of the next event that starts
later. (3) After an event with subtype `END HALF`, dead to the end of the half. Otherwise
frames after a half's last event stay live (the `--frames` range ends the half). With both
`--events` and `--dead`, `--dead` sets the live mask. In this fixture rule (2) adds nothing
(`BALL OUT` ends at 500, the `CARD` starts at 520, both inside the restart's dead frames).
Rules (1) with a same-frame `RECOVERY` and a goal kick, (2) without a restart, and (3) are
unit-tested on small event lists in `test_poss_common.py`.

**True labels (`--events`).** On a live frame, the team of the most recent controlled event
(`PASS`, `CARRY`, `RECOVERY`, `SHOT`, `SET PIECE`) whose start is on or before it, in the same
half. `BALL LOST`, `BALL OUT`, `CHALLENGE`, `FAULT RECEIVED` and `CARD` never change it.

**Pseudo-labels (`--tags`).** `PHASE_TO_POSSESSION` maps each code to `has` (the tagged team
has the ball), `defends` (the other team has it) or `ignore`; codes match case-insensitively.
`N/A` or unmapped teams are dropped. A live frame gets a label if every mapped tag covering it
says the same team; frames where they disagree are dropped. Agreement per code: over the
frames where the code labels and at least one other code labels too, the share where every
other code gives the same team (NaN when there are none).

**Changes.** A change is the first frame of the other team after a labelled frame of one team,
with no non-live frame in between (the previous labelled frame is in the same live run). For
pseudo-labels the previous labelled frame may be any distance back within the run.

**Interpolation.** A live gap (a run of missing ball inside one live run) of at most
`--max-interp-frames` (12) frames is filled linearly when the ball is observed on both
neighbours, which must belong to the same live run. A gap touching a live-run edge (a dead
frame, a half edge) is never filled.

**`poss_ball_live.py`.** Gaps are runs of missing ball inside live runs; bins are half-open
`[0,0.5) [0.5,2) [2,5) [5,inf)` s on length / fps. "Around changes": live frames within
`--change-window-s` (2 s = 50 frames) of a change, `|f - change| <= 50`; "rest": the other
live frames.

## Expected: shared preparation

| | events (`--events`) | dead CSV (`--dead`) |
|---|---|---|
| live frames | 1,850 (half 1: 900, half 2: 950) | the same |
| dead frames | 150 | the same |
| labels | H 1,300, A 550 on live frames; 5 possessions; changes at 300 and 1400 | (`--tags`) labelled 850, dropped 40; changes at 310 and 1410 |

Slot teams: 5 home, 5 away, 1 other. Attack direction from the metadata: home +x in half 1,
-x in half 2 (away the opposite); inferred from the centroids over the first 60 s of live
frames of each half the same (agreement 1). Tag team map: "Alpha" -> home, "Beta Club" ->
away.

Agreement per code (`--tags`): BUILD UP 230 frames, 1.0; DEFENDING IN MIDDLE THIRD 230, 1.0;
PROGRESSION 40, 0.0; LONG BALL 40, 0.0; DEFENSIVE TRANSITION and ATTACKING TRANSITION 0,
NaN; every other code 0, NaN.

## Expected: `poss_ball_live.py`

Live gaps before interpolation (frames, length, bin): 150-152 (3, <0.5), 250-261 (12, <0.5),
400-449 (50, 2-5), 480-499 (20, 0.5-2; the rest of that raw gap is dead), 550-555 (6, <0.5;
touches the restart), 1100-1112 (13, 0.5-2), 1300-1424 (125, >=5), 1990-1999 (10, <0.5;
touches the half end).

| | observed | interpolated | missing |
|---|---|---|---|
| all live (1,850) | 1,611 (0.871) | 15 (0.00811) | 224 (0.121) |
| half 1 (900) | 809 | 15 | 76 |
| half 2 (950) | 802 | 0 | 148 |

Missing before interpolation: 239 (0.129 of live). Gaps per bin, count / frames / share of
missing frames:

| | <0.5 s | 0.5-2 s | 2-5 s | >=5 s |
|---|---|---|---|---|
| before | 4 / 31 / 0.130 | 2 / 33 / 0.138 | 1 / 50 / 0.209 | 1 / 125 / 0.523 |
| after | 2 / 16 / 0.0714 | 2 / 33 / 0.147 | 1 / 50 / 0.223 | 1 / 125 / 0.558 |

Gap length quantiles before interpolation (lengths 3, 6, 10, 12, 13, 20, 50, 125 frames; numpy
linear): p10 5.1 fr = 0.204 s, p50 12.5 fr = 0.5 s, p90 72.5 fr = 2.9 s.

Interpolation cap sensitivity (`--interp-also 6,25`), frames filled: cap 6 -> 3 (only
150-152; 550-555 has no observed left neighbour); cap 12 -> 15; cap 25 -> 28 (adds 1100-1112;
480-499 ends at a dead frame).

Dead frames: 150, ball missing on 120 (0.8).

Around changes (2 windows of 101 frames, 202 frames; rest 1,648), missing before / after
interpolation:

| labels | around | rest |
|---|---|---|
| events (changes 300, 1400) | 87 (0.431) / 75 (0.371) | 152 (0.0922) / 149 (0.0904) |
| tags (changes 310, 1410) | 67 (0.332) / 65 (0.322) | 172 (0.104) / 159 (0.0965) |

Without labels the change rows are NaN with `n_changes` 0.

**`--degrade`** (seed 0): only live frames lose the ball; the missing share of live frames
reaches at least `--missing-share` and exceeds it by less than one burst; the same seed gives the
same arrays. Track fragmentation with `--frag-gap-frames 0` keeps, on every frame, the same
multiset of (team, x, y) over present slots; with the default gap, every lost position is
within `--frag-gap-frames` frames after a break.

## Expected: `poss_carries.py`

**Rules (decided before coding).** Ball = the interpolated ball (observed or interpolated
frames). Velocities are backward differences over `--lag` (5) frames, `(p[f] - p[f-5]) x 25 / 5`,
defined only if both frames are in the same live run with the position available. A frame is a
carry candidate for a player of either team (not team -1) if the ball is within `--carry-dist-m`
(1.5 m) of the player and `|v_ball - v_player| < --carry-dv` (2 m/s); the carrier is the nearest
candidate. A carry is a maximal run of frames with the same carrier slot, kept if it lasts at
least `--carry-min-s` (0.5 s = 13 frames, `ceil(0.5 x 25)`). The carry team is the slot's team.
Frame score: +1 on home carry frames, -1 on away, NaN elsewhere. Window w: the centred mean of
the score over the defined frames of `[f - h, f + h]` inside the live run,
`h = round(w x fps / 2)`; window `0` is the frame itself. `auc`: AUC of the score (NaN as 0) for
"home has the ball" over labelled live frames, ties counted half; `auc_defined`: the same over
frames with a defined score; `coverage`: share of live frames with a defined score. Precision:
labelled carry frames whose carry team is the label. Switches: the first frame of a carry whose
team differs from the previous carry in the same live run. Timing: for each change, the nearest
switch to the same team within `--timing-window-s` (5 s); error = switch - change (s). Switch
precision: switches with a same-team change within 5 s. Fast ball: live frames with a ball
speed above `--fast-ball-ms` (15 m/s); `near_rate` = share of them with the ball within
1.5 m of a player (distance only), `carry_rate` = share inside carries.

**Why these frames.** Every hold puts the ball 0.5 m in +x of the holder, so on hold frames the
ball moves exactly with the holder (velocity difference 0). The first 5 frames of a hold are
not candidates: the ball at f - 5 was still in a pass flight (at least 0.81 m per frame, so the
velocity difference is at least 3 m/s), or missing, or before the live run. The nearest
opponent to any hold point is 1.80 m away (away 7's hold vs home 4), the rest at least 2.69 m.
Pass flights are at least 20 m/s. The 13-frame interpolated gap 1100-1112 is not filled, so it
splits away 9's hold in two; the filled 250-261 lies on home 4's straight path.

| carry | frames | length (s) | team | true label |
|---|---|---|---|---|
| C1 | 55-89 | 35 (1.40) | H (slot 4) | H |
| C2 | 105-174 | 70 (2.80) | H (slot 2; 150-152 interpolated) | H |
| C3 | 195-209 | 15 (0.60) | H (slot 3) | H |
| - | 230-239 | 10: too short (home 5) | | |
| C4 | 250-279 | 30 (1.20) | H (slot 4) | H |
| C5 | 305-339 | 35 (1.40) | A (slot 8) | A |
| C6 | 360-399 | 40 (1.60) | A (slot 9) | A |
| C7 | 455-479 | 25 (1.00) | A (slot 10; 450-454 have no ball at f - 5) | A |
| C8 | 561-699 | 139 (5.56) | H (slot 3) | H |
| C9 | 705-729 | 25 (1.00) | A (slot 7, the touch without control) | **H** |
| C10 | 745-999 | 255 (10.20) | H (slot 5) | H |
| C11 | 1055-1099 | 45 (1.80) | A (slot 9) | A |
| C12 | 1118-1199 | 82 (3.28) | A (slot 9) | A |
| C13 | 1216-1299 | 84 (3.36) | A (slot 10) | A |
| C14 | 1430-1989 | 560 (22.40) | H (slot 2) | H |

14 carries, 1,440 frames: coverage 1440 / 1850 = 0.778 of live frames. Duration bins
`0.5-1s` 1, `1-2s` 7, `2-5s` 3, `5-10s` 1, `>=10s` 2. Length quantiles (frames 15, 25, 25, 30,
35, 35, 40, 45, 70, 82, 84, 139, 255, 560): p10 25 fr = 1.0 s, p50 42.5 fr = 1.7 s, p90 220.2 fr
= 8.808 s.

**Events labels.** Precision 1415 / 1440 = 0.983; home carries 1,104 frames, precision 1.0;
away carries 336 frames, 311 correct, 0.926. Window 0: H frames 1,300 (score +1 on 1,104, -1 on
25, 0 on 171), A frames 550 (-1 on 311, 0 on 239): `auc` = (1104 x 550 + 171 x 311 + 0.5 x (171
x 239 + 25 x 311)) / (1300 x 550) = 684703 / 715000 = 0.958; `auc_defined` = (1104 + 12.5) /
1129 = 0.989; `coverage` 0.778. Switches: 305 (A), 705 (A), 745 (H), 1430 (H). Changes 300 (to
A) and 1400 (to H) match 305 and 1430: errors +0.2 s and +1.2 s, matched 2 of 2; p25 0.45, p50
0.7, p75 0.95 s, median absolute 0.7 s. Switch precision 2 / 4 = 0.5.

**Tags (pseudo-labels).** Labelled carry frames: C1 30 (60-89), C2 70, C3 15, C4 30, C5 10
(310-319; 320-339 are dropped), C6 40, C7 25, C12 82, C13 84, C14 170 (1430-1599); C8-C11 have
no label. 556 frames, all correct: precision 1.0 (home 315, away 241). Changes 310 (to A) and
1410 (to H): errors -0.2 s and +0.8 s, p50 0.3 s; switch precision 0.5.

**Fast ball.** No carry frame has a fast ball (the ball moves with a player at about 1 m/s):
`carry_rate` 0. `near_rate` is checked against a per-frame loop in the test.

## Expected: `poss_pass_chains.py`

**Rules (decided before coding).** Contacts: runs of one carrier slot (the carry candidate test
of `poss_carries.py`: distance and velocity difference) lasting at least `--touch-min-s` (0.2 s
= 5 frames); consecutive contacts of the same slot in the same live run merge into one (the
ball lost and refound by the same player). A pass links two consecutive contacts in the same live
run when both slots are of the same team, the slots differ, the gap from the first contact's
end to the second's start is at most `--pass-max-s` (5 s), and the largest defined ball speed
on the frames after the first contact up to and including the second contact's start exceeds
`--pass-min-ms` (5 m/s); with no defined speed there, it is not a pass. A chain is a maximal run
of contacts linked by passes; its length is its number of passes, its team the contacts' team,
its frames run from the first contact's start to the last contact's end. The proxy fires on
chains of at least `--min-chain` (2) passes: score +1 (home) / -1 (away) on their frames.
Precision per chain-length bin (`1`, `2`, `3`, `4`, `5+`) and for chains of at least
`--min-chain`: labelled chain frames whose label is the chain team. Switches: the first frame
of a qualifying chain whose team differs from the previous qualifying chain in the same live
run.

**Contacts.** The 14 carries plus home 5's 230-239 (10 frames, long enough for a contact):
K1 55-89 H4, K2 105-174 H2, K3 195-209 H3, K4 230-239 H5, K5 250-279 H4, K6 305-339 A8,
K7 360-399 A9, K8 455-479 A10, K9 561-699 H3, K10 705-729 A7, K11 745-999 H5,
K12 1055-1099 A9 and K13 1118-1199 A9 (merged: same slot, 1055-1199), K14 1216-1299 A10,
K15 1430-1989 H2. 14 contacts after merging.

**Passes.** K1 -> K2, K2 -> K3, K3 -> K4, K4 -> K5 (home; flights of 20 to 67 m/s, gaps of
0.2 to 0.64 s); K6 -> K7 (away); K12/13 -> K14 (away; the ball jumps about 16 m to the loose
ball, over 80 m/s). Not passes: K5 -> K6 (other team), K7 -> K8 (the whole flight
400-449 is missing and 450-454 have no ball at f - 5, so no speed is defined), K9 -> K10 and
K10 -> K11 (other team); K8 -> K9 and K11 -> K12 cross a dead stretch. 6 passes: home 4, away 2.

**Chains.** K1-K5 home, 4 passes, frames 55-279 (225); K6-K7 away, 1 pass, 305-399 (95);
K12/13-K14 away, 1 pass, 1055-1299 (245). Length bins: `1` 2, `4` 1, others 0. Chains of at
least 2: one, 225 frames, coverage 225 / 1850 = 0.122; chains of at least 1: 565 frames
(0.305).

**Events labels.** Precision: chains of at least 2, 225 frames, 1.0; bin `1`, 340 frames, 1.0;
bin `4`, 225 frames, 1.0. Window 0: home frames 1,300 (225 at +1, 1,075 at 0), away 550 (all
0): `auc` = (225 x 550 + 0.5 x 1075 x 550) / (1300 x 550) = 762.5 / 1300 = 0.587;
`auc_defined` NaN (no away frame has a defined score). No switch (one qualifying chain): 0 of 2
changes matched. With `--min-chain 1`: switch at 305 (away; the K12/13-K14 chain is the first
of its run), change 300 matched with error +0.2 s, event precision 1 / 1.

**Tags.** Chains of at least 2: labelled frames 60-279 (220), precision 1.0.

## Expected: `poss_shape.py`

**Rules (decided before coding).** Per team and live frame, over its present players (team 0 or
1, x and y finite), in the team's attacking frame (x toward the opponent goal, `toward_goal`):
the deepest player (smallest attacking-frame x) is taken as the goalkeeper and dropped; the
features use the rest ("outfield"), and need at least `--min-players` (3) of them, else NaN.
`width` = y range; `length` = attacking-frame x range; `hull_area` = convex hull area (m^2);
`centroid_advance` = mean attacking-frame x; `centroid_velocity` = the median, over outfield
players, of each player's x velocity (backward difference over `--lag` of its own slot inside
the live run), in the team's attacking frame (m/s; all players move together here, so it equals
the centroid's velocity); `line_depth` = attacking-frame x of the deepest outfield
player (metres from the own goal line); `goal_side` = share of outfield players with
attacking-frame x below the ball's (interpolated ball), or below the opponent outfield
centroid's when the ball is missing. Hypothesis: the possessing team has the larger value, except
`goal_side` (the smaller). Score toward "home has the ball" = sign x (home - away), sign +1, or
-1 for `goal_side`. `diff` rows: possessing minus defending team, on labelled live frames with
both values defined (mean, median, n), for all frames, ball present (observed or interpolated)
and ball missing. Separation per window as for carries, computed three times: over all
labelled live frames, those with the ball present, and those with it missing (the split uses the
frame's own ball flag; the window may mix both). `line_depth` quantiles per team on labelled
frames where that team defends, and where it attacks.

**Frame 200** (home has the ball, half 1, home centroid 46 m, away 56 m, home width scale 1,
away 0.6; the ball 0.5 m in front of home 3 at (40, 49), so at (40.5, 49)). Home outfield
(40, 19), (40, 49), (52, 26), (52, 42): width 30, length 12, hull 276, advance 46, velocity +1,
line depth 40, goal side 2 of 4 = 0.5. Away outfield (50, 25), (50, 43), (62, 29.2),
(62, 38.8), attacking-frame x 55, 55, 43, 43: width 18, length 12, hull 165.6, advance 49,
velocity -1, line depth 43, goal side 4 of 4 = 1.0 (the ball is at 64.5 in their frame).
Scores: width +12, length 0, hull +110.4, advance -3, velocity +2, line depth -3, goal side
+0.5.

**Frame 420** (away has the ball, ball missing, home centroid 45.2 m, away 55.2 m): goal side
uses the opponent centroid; every outfield player of both teams is goal-side of it (home x
39.2 and 51.2 below 55.2; away attacking-frame x 55.8 and 43.8 below 59.8): 1.0 and 1.0,
score 0.

**Coverage.** Every feature is defined on all 1,850 live frames (all players tracked), except
`centroid_velocity`, which needs the frame 5 earlier in the same live run: undefined on the
first 5 frames of each of the 3 live runs, 1835 / 1850 = 0.992. Positions are written with 5
decimals, so values match within 0.01 m, and hull areas within 0.05 m^2.

**Width and hull.** Off the ramps the possessing team's width is 30 and the defender's 18
(hull 276 and 165.6), so possessing minus defending is +12 (+110.4). On the 75 ramp frames
(three ramps of 25 after 300, 550 and 1400) it is -12 + 0.96 k, k = 0..24, for both label
sources. `diff` width: mean (1775 x 12 + 75 x -0.48) / 1850 = 11.494, median 12; hull: mean
105.75, median 110.4. Length: possessing minus defending is 0 up to position rounding (median
0 within 0.01 m). Window-0 `auc` for width (events): home frames 1,250 at +12 and 2 x 25 ramp
values -12 + 0.96 j; away frames 525 at -12 and 25 ramp values 12 - 0.96 k. Wins 1250 x 549 +
24 x 2 x 525 + 2 x 276 = 712,002 (276 pairs with j + k > 25), ties 1250 + 2 x 525 + 2 x 24 =
2,348 (equal values, which position rounding may break either way): `auc` = (712002 + 1174) /
715000 = 0.9974, within +-0.0017.

**Line depth** (events labels; numpy linear quantiles of the analytic values): home
defending (550 frames: 44 - 0.04 i, i < 200, and 39 - 0.04 j, j < 350): p25 30.53, p50 36.02,
p75 38.76 m. Away defending (1,300 frames: 49 - 0.04 i, i < 250; 47 - 0.04 i, i < 450; 58 -
0.04 i, i < 600): p25 38.03, p50 42.68, p75 47.01 m. Home attacking (34 + 0.04 i, i < 250;
36 + 0.04 i, i < 450; 25 + 0.04 i, i < 600): p25 35.99, p50 40.32, p75 44.97 m. Away attacking
(39 + 0.04 i, i < 200; 44 + 0.04 j, j < 350): p25 44.24, p50 46.98, p75 52.47 m. Within 0.01 m
(position rounding).

## Expected: `poss_ball_field.py`

**Rules (decided before coding).** Scores toward "home has the ball", in home's attacking
frame (x toward the away goal), NaN where undefined:
- `ball_position`: interpolated ball x minus half the pitch length (positive = ball in the away
  half).
- `ball_direction`: the sign of the ball's x velocity (backward difference over `--lag` inside
  the live run), NaN when |vx| < `--min-ms` (0.5 m/s).
- `team_drift`: +1 when both teams (median outfield player x velocity, as in `poss_shape.py`)
  move toward the away goal at `--min-ms` or more, -1 when both move toward the home goal at that speed, NaN
  otherwise.
Separation per window as for carries. `thirds`: for each team T, labelled live frames with the
ball present, by the ball's third in T's attacking frame (`defensive` [0, 35), `middle`
[35, 70), `attacking` [70, 105]; x outside the pitch clipped), the share where T has the ball.

**Team drift (by hand).** Both centroids move together at 1 m/s: toward the away goal (home's
frame) on 50-299, 550-999 and 1400-1999, toward the home goal on 300-499 and 1050-1399. The
velocity over 5 frames is NaN on the first 5 frames of each live run (50-54, 550-554, 1050-1054)
and, around each reversal, takes the values 1, 0.6, 0.2, -0.2, -0.6, -1: in home's frame
+1, +0.6, +0.2, -0.2, -0.6 on frames 300-304, and -1, -0.6, -0.2, +0.2, +0.6 on 1400-1404. So
300-301 score +1 under away labels, 1400-1401 score -1 under home labels, and 302-303, 1402-1403
are NaN. Window 0: home frames 1,300 = 1,286 at +1, 12 at 0 (NaN), 2 at -1; away frames 550 =
541 at -1, 7 at 0, 2 at +1. `auc` = (1286 x 548 + 12 x 541 + 0.5 x (1286 x 2 + 12 x 7 + 2 x 541))
/ 715000 = 713089 / 715000 = 0.9973; `auc_defined` = (1286 x 541 + 0.5 x (1286 x 2 + 2 x 541))
/ (1288 x 543) = 697553 / 699384 = 0.9974; coverage 1831 / 1850 = 0.9897.

**Spot values.** Frame 200 (ball at x 40.5, moving with home 3): `ball_position` -12,
`ball_direction` +1, `team_drift` +1. Frame 420 (ball missing, away possession):
`ball_position` and `ball_direction` NaN, `team_drift` -1. Frame 1500 (half 2, home attacks -x,
ball 0.5 m in front of home 2 at x = cA + 4.5 with cA = 60): ball x in home's frame 105 - 64.5
= 40.5, `ball_position` -12; the ball moves -1 m/s in absolute x, +1 in home's frame.

**Ball position, ball direction, thirds.** Continuous in the fixture, so the test recomputes
them from the fixture text with a plain per-frame loop (its own parser, interpolation of the two
short gaps 150-152 and 250-261, and a pairwise AUC count) and requires exact agreement.

## Expected: `poss_reversal.py`

**Rules (decided before coding).** Per team, the median outfield player x velocity (as in
`poss_ball_field.py`) in home's attacking frame. Direction state per live run, with hysteresis:
+1 once the velocity is at least `--min-ms` (0.5 m/s), -1 once it is at most -0.5, unchanged
in between; the first state in a run is not a flip. A flip is a frame where the state changes.
A home flip is joint when an away flip to the same direction lies within
`--reversal-window-s` (1 s = 25 frames); the joint reversal is at the later of the two (the
nearest away flip), and its new direction names the implied new possessor (+1: home attacks, so
home; -1: away). Rate: joint reversals per minute of live play. Null: away flips shifted
circularly within their half by the `--null-*` offsets, joint reversals recounted; `p_emp` =
(1 + shifts with at least the real count) / (1 + shifts). Timing against label changes as for
carries. Change detection: indicator 1 on reversal frames, 0 on other live frames, smoothed per
window; AUC for "a label change within `--change-window-s` (1 s) of the frame". State: after a
joint reversal, the implied possessor holds until the next one in the live run (NaN before the
first); separation per window as for carries. Profile: each team's centroid velocity in the
attacking frame of the team gaining possession, pooled per 1 s bin over -10 s .. +10 s around
the label changes (window clipped to the half; undefined frames, e.g. dead, not counted), with
the changes shifted circularly within each half as the null.

**By hand.** The centroid velocities (both teams equal) are those of the team-drift rule:
+1 on 55-301 (+0.6 at 301), +0.2 at 302, -0.2 at 303, -0.6 at 304, -1 on 305-499; +1 on
555-999; -1 on 1055-1400, -0.6 at 1401, -0.2, +0.2, +0.6 at 1404, +1 on 1405-1999. Flips:
home and away both at 304 (to -1) and 1404 (to +1). 2 joint reversals: 304 (away), 1404
(home). Rate 2 / (1850 / 1500 min) = 1.62 per minute. With `--null-min-shift 5` every shift
is 125-875 frames in a 1,000-frame half, so no shifted away flip lies within 25 frames of a home
flip: null mean 0, `p_emp` 1 / (N + 1). With the default 60 s the halves are too short and the
null is skipped.

Timing (events: changes 300 to away, 1400 to home): errors +0.16 s and +0.16 s, 2 of 2
matched, event precision 1.0. Tags (changes 310, 1410): -0.24 s and -0.24 s.

Change detection (events; positives 275-325 and 1375-1425, 102 frames; negatives 1,748):
window 0: (2 x 1748 + 0.5 x 100 x 1748) / (102 x 1748) = 52 / 102 = 0.510; window 1 s (25
frames, h = 12): 50 frames at 1/25 (292-316 and 1392-1416), all positives: (50 + 0.5 x 52) /
102 = 0.745.

State (events): away on 304-499, home on 1404-1999, NaN elsewhere. Window 0: home frames
596 at +1, 704 at 0; away frames 196 at -1, 354 at 0: `auc` = (596 x 550 + 704 x 196 + 0.5 x
704 x 354) / 715000 = 590392 / 715000 = 0.826; `auc_defined` 1.0; coverage 792 / 1850 = 0.428.

Profile (events, gainer's frame): before both changes -1; bin [0, 1 s) = (-1 - 0.6 - 0.2 +
0.2 + 0.6 + 20 x 1) / 25 = 0.76 for each change; +1 after. Bins -10 .. -1: -1 (bin -10 counts
20 + 25 = 45 frames: 50-54 have no velocity); bin 0: 0.76; bins 1 .. 9: +1 (bins 8 and 9 count
only the 1400 change's 25 frames each: 500-549 are dead). The same for both teams.

## Expected: `poss_contested.py`

**Rules (decided before coding).** Nearest team on a live frame with the (interpolated) ball:
the team of the nearest player of either team (referees excluded); undefined without a ball.
A flip is a frame whose nearest team differs from the previous frame's, both defined and in the
same live run. Flip density at window w: the flip indicator (1 on flips, 0 on other defined
frames) averaged over the centred window, in flips per second (x fps). Contested: at least 2
flips within the `--flip-window-s` (2 s, h = 25) window around the frame. Contexts on live
frames: `around_change` within `--context-s` (1.5 s = 38 frames, rounded) of a label change;
`loose_no_change` within 1.5 s of a `CHALLENGE` or `BALL LOST` event (events only) with no label
change within `--loose-gap-s` (3 s) of it; `mid` more than `--mid-margin-s` (5 s) from every
change, loose anchor and live-run edge. Per context: frames with a defined density, mean flips
per second, contested share. Separation rows: `nearest_team` (+1 home nearest, -1 away) as for
carries (the organisers' baseline); `contested_change`: AUC of the density for around-change vs
mid frames; `contested_loose`: around-change vs loose-no-change frames (can it tell a turnover
from a contest that changes nothing?).

**By hand.** Changes 300 and 1400 (events). Loose anchors: `BALL LOST` 280 and 1380 are within
3 s of a change, so dropped; the two `CHALLENGE` events at 700 and the one at 1200 stay
(`loose_no_change` = 662-738 and 1162-1238). `around_change` = 262-338 and 1362-1438. With tags
(no events) `loose_no_change` is empty (NaN rows) and the changes are 310 and 1410. Everything
else (nearest team, flips, densities, context means, contested shares and the window-0 AUCs) is
checked against an independent per-frame recomputation from the fixture text in the test.
