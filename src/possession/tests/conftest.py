import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(SRC)
EDA_FIXTURES = os.path.join(REPO, "eda", "tests", "fixtures")
for path in (REPO, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)
