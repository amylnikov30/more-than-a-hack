"""Every constant of the submission pipeline, in one place (docs/possession_HA_proposal.md §2.5).

The H/A model's own constants live in possession/config.py and possession/ha_model.json, and the
dead-ball labeller's in deadball/; nothing here overrides them.
"""

# ---------------------------------------------------------------- which system writes H / A
# "model" (possession.run) or "rule" (pipeline.rule). Set from evaluate_chain's decision rule:
# the model ships only if its held-out spellF1 interval on the reference set lies above the
# rule's. reference_summary.json records the run that decided it.
SHIP = "model"

# ---------------------------------------------------------------- step-6 rule (§6.2)
# Per live spell: start with the restart team; switch after RULE_K consecutive frames on which
# the other team is nearest a visible ball slower than possession.config.FAST_BALL_MS, and
# backdate the switch to the first of them. RULE_K is chosen on the reference set by
# evaluate_chain (held out by half) from RULE_KS.
RULE_KS = (5, 12, 25, 50)        # frames: 0.2, 0.5, 1 and 2 s
RULE_K = 12

# ---------------------------------------------------------------- evaluation (§6)
BASELINE_WINDOW = 50             # the organisers' published floor, baseline_smooth50
BOOTSTRAP_N = 1000               # draws for run_evaluation.py's block-bootstrap intervals
