# Handoff: submission pipeline (`src/pipeline/`)

State on 16 Nov 2026: the H/A possession model is on branch `ha-model` (`src/possession/`,
write-up in `docs/ha_model.md`). It runs on the match on the VM and writes a valid 199,319-row
file. It is not merged into `main`. The prompt below is the original pipeline prompt from
`docs/agent_prompts.md` (branch `docs/agent-prompts`), with an "Update" block added for what now
exists. Paste the whole block into the agent.

```
Build src/pipeline/ for our More Than A Hack 2027 submission: the fallback rule, whole-chain
validation, and the command that writes the scored file. Spec: docs/possession_HA_proposal.md
sections 4.0, 6, 7, 8 (it wins over this prompt). Also read docs/info/{SUBMISSION,EVALUATION,
RULES,DATA}.md, docs/dead_ball_rule.md, docs/ha_model.md, src/deadball/run.py, games.py.
Deadline 17 Nov 2026.

Section 2 of the proposal binds you. In short: laptop only, synthetic fixtures and Metrica open
sample games; never read match data or match outputs; public ground truth is a final smoke
test, never compared across versions; constants in the one config file; byte-identical reruns.

Dependencies:
- Organisers' tools live in the public starter repo ($MTAH_PUBLIC_REPO, default
  ~/more-than-a-hack-2027): evaluation/run_evaluation.py, validate_submission.py,
  possession_metrics.py, starter/baseline.py. Call them; don't copy them.
- Branch `ha-model` has src/possession/ with this contract:
  `python -m possession.run OUT_DIR --dead dead_ball.csv` writes OUT_DIR/possession.csv
  (`frame,label`, 199,319 rows, D/X from the hard mask, H/A on every live frame) and also runs
  on a sample game via --raw/--meta/--frames/--pitch; `possession.reference` builds the
  reference set (degraded sample games + event-derived labels). Stub possession.csv in tests.

Update (16 Nov, read before building):
- Branch `submission-pipeline` from `ha-model` (it is not merged into main). Use
  `possession.reference` and `possession.run` directly, so the model and the rule see exactly
  the same degraded tracking. Reference set: Metrica Sample Game 3 only (team decision).
  `possession.reference OUT --sample-data DIR --games 3` writes OUT/sg3/{tracking.txt,
  metadata.xml, labels.csv, dead_ball.csv (oracle mask), manifest.json}; frames
  0-69660,69661-143760, pitch 105 68.
- rule.py: for the restart team, import `possession.restart.priors` (with
  `restart.tag_restarts`); don't reimplement it. On the match the spec's +-3 s tag link matched
  3 of 112 spells; the shipped link matches 23.
- `possession/ha_model.json` is fitted on all of Sample Game 3, so scoring the shipped model on
  Sample Game 3 is in-sample. Compare the model held out by half: fit on one half with
  `possession.evidence.fit` and the helpers in `possession.train` (`load_reference`,
  `training_rows`, `units`), decode the other with the same lambda/m/beta, score it. Report
  that the dead-ball player model (trained on Sample Games 1-3) is also in-sample on Sample
  Game 3.
- `deadball.run.label()` reads only the match (games.require). On a sample game, call its
  internals: signals.load_atd, signals.player_features, train_players.log_odds_ratio,
  signals.ball_categories, run.ball_log_ratios, hmm.viterbi, run.sharpen_edges.
- With one game the bootstrap intervals are about +-0.05 wide, so "the model beats the rule
  beyond the intervals" may well fail and the rule ship. Report the decision rule's outcome as
  it is.
- On the match the dead-ball labeller calls 51% of graded frames dead (Sample Game 3 reference:
  34%). Report the dead share the labeller gives on the degraded sample game next to the
  reference.

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

Match aggregates from `possession.run` on the VM (16 Nov), for comparison: 112 live spells;
restart prior 23 from tags, 81 from the first fast ball, 8 none; 264 possessions (5.12 per live
minute), median 7.2 s; H share of live frames 0.625.
