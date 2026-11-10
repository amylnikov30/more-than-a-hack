import os
import sys

EDA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(EDA, "tests", "fixtures")
sys.path.insert(0, EDA)
