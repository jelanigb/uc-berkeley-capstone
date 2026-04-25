"""Make `src/capstone` (and its nested `data_processing` package) importable
from tests. The repo has no pyproject.toml / setup.cfg, so we replicate the
sys.path the notebook relies on.
"""

import sys
from pathlib import Path


CAPSTONE_ = Path(__file__).resolve().parents[1] / "src" / "capstone"
for path_ in (CAPSTONE_, CAPSTONE_ / "data_processing"):
    s = str(path_)
    if s not in sys.path:
        sys.path.insert(0, s)
