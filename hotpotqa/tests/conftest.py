import os
import sys

# The package loads its data files by relative path (wrappers.py:125,
# "data/hotpot_dev_v1_simplified.json"), so tests must run from hotpotqa/.
HOTPOTQA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HOTPOTQA_DIR)
sys.path.insert(0, HOTPOTQA_DIR)
