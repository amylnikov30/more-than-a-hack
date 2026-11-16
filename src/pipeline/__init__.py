"""Submission pipeline: the fallback rule, whole-chain validation and the scored file.

    python -m pipeline.rule OUT_DIR --dead dead_ball.csv            the step-6 rule (fallback)
    python -m pipeline.evaluate_chain OUT_DIR --reference REF_DIR    model vs rule vs baseline
    python -m pipeline.make_submission OUT_DIR --team TEAM_NAME      writes OUT_DIR/TEAM_NAME.csv

Run from the repository root with PYTHONPATH="$PWD:$PWD/src" (see README.md and the Makefile).
Importing possession puts eda/scripts on sys.path for the shared EDA signal code.
"""

import possession  # noqa: F401  (sys.path for eda/scripts)
