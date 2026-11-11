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
