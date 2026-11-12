import os
import sys

EDA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(EDA, "scripts")
FIXTURES = os.path.join(EDA, "tests", "fixtures")
sys.path.insert(0, SCRIPTS)
