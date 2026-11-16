# Agent prompts

Paste one block per agent. All three can start together.

## 1: H/A model

```
Implement docs/possession_HA_proposal.md in src/possession/. It is the spec: read it fully
first and build its section 7 deliverables as written. If you disagree with it, say so in your
report and do not deviate silently. Deadline 17 Nov 2026: get run.py working end to end
(evidence -> decoder) before snapping or any upgrade.

Section 2 of the proposal binds you. In short: laptop only, synthetic fixtures and Metrica open
sample games (github.com/metrica-sports/sample-data); never read match data or match outputs;
public ground truth is never used to tune or compare; every constant in one config file;
byte-identical reruns.

Context beyond the spec:
- Mirror src/deadball/ (structure, style, JSON weights like player_model.json). Rules and
  metric: docs/info/{CHALLENGE,EVALUATION,RULES,DATA}.md.
- Reuse, don't rewrite: eda/scripts/poss_common.py (loader, slot teams, attack_directions,
  degrade, interpolate_ball), poss_reference.py (event-derived labels), poss_carries.py,
  poss_pass_chains.py, poss_shape.py, poss_contested.py.
- The official scorer is in the public starter repo ($MTAH_PUBLIC_REPO, default
  ~/more-than-a-hack-2027), evaluation/possession_metrics.py. Call it; don't copy it.

Additions to the spec (a parallel agent in src/pipeline/ depends on them; don't change them):
- CLI: `PYTHONPATH="$PWD:$PWD/src" python3 -m possession.run OUT_DIR --dead dead_ball.csv`
  writes OUT_DIR/possession.csv (`frame,label`; D where dead==1, X where dead==-1, H/A on every
  live frame). It must also run on a sample game via --raw/--meta/--frames/--pitch, as the
  poss_*.py scripts do. stdout: aggregates only.
- reference.py: builds the reference set ("Track B"): per sample game, degraded tracking
  (poss_common.degrade, fixed seed) plus an event-derived frame,label CSV.
- Snapping sits behind a config flag.
- docs/ha_model.md: the tech-doc section the spec asks for.

Do not edit src/deadball/ or src/pipeline/. Branch `ha-model`, pin new deps in
requirements.txt, run the tests, open a PR to main.

Report: held-out spellF1 per sample game with and without snapping; chosen lambda/m/beta;
result of the attacking-direction check (spec section 6); the exact VM commands a human should
run and which aggregate lines to paste back; any deviation from the spec.
```

## 2: fallback rule, chain validation, submission command

```
Build src/pipeline/ for our More Than A Hack 2027 submission: the fallback rule, whole-chain
validation, and the command that writes the scored file. Spec: docs/possession_HA_proposal.md
sections 4.0, 6, 7, 8 (it wins over this prompt). Also read docs/info/{SUBMISSION,EVALUATION,
RULES,DATA}.md, docs/dead_ball_rule.md, src/deadball/run.py, games.py. Deadline 17 Nov 2026.

Section 2 of the proposal binds you. In short: laptop only, synthetic fixtures and Metrica open
sample games; never read match data or match outputs; public ground truth is a final smoke
test, never compared across versions; constants in the one config file; byte-identical reruns.

Dependencies:
- Organisers' tools live in the public starter repo ($MTAH_PUBLIC_REPO, default
  ~/more-than-a-hack-2027): evaluation/run_evaluation.py, validate_submission.py,
  possession_metrics.py, starter/baseline.py. Call them; don't copy them.
- A parallel agent (branch `ha-model`) is building src/possession/ with this contract:
  `python -m possession.run OUT_DIR --dead dead_ball.csv` writes OUT_DIR/possession.csv
  (`frame,label`, 199,319 rows, D/X from the hard mask, H/A on every live frame) and also runs
  on a sample game via --raw/--meta/--frames/--pitch; `possession.reference` builds the
  reference set (degraded sample games + event-derived labels). Don't wait for it: until it
  merges, use eda/scripts/poss_reference.py and poss_common.degrade directly, and stub
  possession.csv in tests.

Build, in this order:
1. rule.py: the step-6 rule (the fallback submission). Per live spell: start with the restart
   team; switch after K consecutive frames with the other team nearest a slow ball, backdated
   to the first of them. Same output format as possession.run. Working end to end before
   anything else.
2. evaluate_chain.py: spec section 6 items 1-3. Per reference game: dead-ball labeller on the
   degraded tracking -> H/A -> `run_evaluation.py --gt <labels> --bootstrap 1000`, for the
   model, the rule and baseline_smooth50. Output the comparison with intervals and the
   evaluate() diagnostics. The shipped system is a config value: the model only if it beats
   the rule beyond the bootstrap intervals.
3. make_submission.py: `python3 -m pipeline.make_submission OUT_DIR --team TEAM_NAME` runs
   deadball.run, then the chosen system, and writes OUT_DIR/TEAM_NAME.csv. Assert the format
   in SUBMISSION.md, X only on ungraded frames, and D before the second-half kickoff; call
   validate_submission.py when present; print the label-free aggregates of spec 6.4 next to
   the reference-set values.
4. Makefile: run, reference, evaluate, test.
5. Tests on synthetic inputs: the rule (one switch, backdating, no switch on a brief touch),
   format failures, pre-kickoff D, byte-identical reruns.
6. Root README.md as SUBMISSION.md requires: pinned install, data layout, sample-games
   download, training commands for both models, the regenerate command, runtime.

Do not edit src/deadball/ or src/possession/. Do not feed p_dead or llr_dead into decoding
(spec 4.0). Branch `submission-pipeline`, open a PR to main.

Report: the three-system table; which system ships and why; the exact VM commands a human
should run and which aggregate lines to paste back.
```

## 3: technical document

```
Write the technical document for our More Than A Hack 2027 submission (deadline 17 Nov 2026).
Requirements: docs/info/SUBMISSION.md, "3. A technical document": a few pages covering
approach and why, assets used and dropped, training data, possession rules implemented,
limitations, reproduction. Readers are the organisers, who must be able to understand and
reproduce the pipeline from it.

Rules: work from the repository only. No match data, frame-indexed tables or frame plots in
the document (public repo; data not redistributable); rounded aggregates are fine. Never
invent a result: a number missing from the sources becomes `[TODO: what, which branch]`.

Sources:
- docs/possession_HA_proposal.md: H/A design, constants (5), validation (6), risks (9).
- docs/dead_ball_rule.md: the dead-ball labeller, including dropped experiments.
- docs/hmm_writeup/hmm_writeup.typ: why a switch cost acts as the "control" threshold.
- docs/info/*.md; eda/docs/analyses.md; eda/reports/*.pdf (`pdftotext -layout`);
  eda/reports/eda_summary.docx; the READMEs under eda/.
- Being built now: src/possession/ + docs/ha_model.md (branch `ha-model`), src/pipeline/ +
  root README (branch `submission-pipeline`). Check `git branch -a`. Document what was built,
  not what was proposed: if the fallback rule shipped, or snapping is off, say so and why.
  Where a branch does not exist yet, write the section as a marked stub.

Sections:
1. Summary: one pipeline diagram (ATD -> dead-ball labeller -> live spells -> restart prior +
   evidence model -> decoder -> snapping -> CSV) and one paragraph.
2. Framing: three states, spellF1 and what it punishes, the hard mask, no X on graded frames.
3. Data and what the EDA decided, including the signal scorecard (kept vs excluded, AUCs).
4. Dead-ball labeller.
5. H/A model.
6. Possession rules: the control-frame definition; loose or contested ball stays with the
   last team; deflections and brief touches; restarts.
7. One table of every constant with its value and origin. Training data: Metrica open Sample
   Games 1-3, degraded, labels derived from Metrica's events (not by us). State that no hand
   labels on BAR-ATH were used and the public 5 minutes were a smoke test only.
8. Validation: whole-chain reference-set scores with intervals, the three-system comparison
   and ship decision, diagnostics, label-free match checks.
9. Limitations and next steps.
10. Reproduction: the commands from the root README.

Style: plain short sentences, tables for numbers, every number traceable to a repo file.

Output: docs/technical_document/technical_document.md, rendered to
docs/technical_document/TEAM_NAME.pdf (typst or pandoc; put the render command at the top).
Branch `tech-doc`, open a PR to main. End your report with the list of remaining TODOs.
```
