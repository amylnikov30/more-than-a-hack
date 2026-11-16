"""H/A possession on the live frames the dead-ball labeller leaves (docs/possession_HA_proposal.md).

    python -m possession.reference OUT_DIR --sample-data DIR    build the reference set (Track B)
    python -m possession.train REFERENCE_DIR                    fit evidence + tune lambda/m/beta
    python -m possession.run OUT_DIR --dead dead_ball.csv       label the competition match

Run from the repository root with PYTHONPATH="$PWD:$PWD/src". The EDA modules in eda/scripts
(poss_common, poss_shape, poss_carries, ...) are reused as they are; this package puts their
folder on sys.path, found relative to this file. See docs/ha_model.md.
"""

import os
import sys

EDA_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "eda", "scripts")
if EDA_SCRIPTS not in sys.path:
    sys.path.insert(0, EDA_SCRIPTS)
