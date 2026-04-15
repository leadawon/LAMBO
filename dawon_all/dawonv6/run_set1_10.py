from __future__ import annotations

import sys
from pathlib import Path


DAWON_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DAWON_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dawonv6.anchor.run_lambo_set1 import main


if __name__ == "__main__":
    main()
