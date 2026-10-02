import os
import sys
import tempfile

# The package loads its data files by relative path (wrappers.py:125,
# "data/hotpot_dev_v1_simplified.json"), so tests must run from hotpotqa/.
HOTPOTQA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HOTPOTQA_DIR)
sys.path.insert(0, HOTPOTQA_DIR)

# The offline fixtures build throwaway corpora in pytest's tmp_path (under
# /tmp), which src/durability.py would otherwise refuse as an ephemeral output.
# Fixtures are not run outputs; see durability.ALLOW_ENV_VAR. Scoped to the test
# session only -- no entry script sets this.
os.environ["SPEC_ALLOW_EPHEMERAL_OUTPUTS"] = "1"

# Keep test runs from writing trajectories into the working tree or onto the
# JuiceFS quota: tests that construct a runner get a tmp output root.
os.environ.setdefault("SPEC_RUNS_DIR",
                      os.path.join(tempfile.gettempdir(), "specmem-test-runs"))
