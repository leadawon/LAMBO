from __future__ import annotations

import sys
from pathlib import Path


SEOHEE_ALL_ROOT = Path(__file__).resolve().parent.parent
if str(SEOHEE_ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(SEOHEE_ALL_ROOT))

from seohee_v1.anchor.run_lambo_set1 import main


if __name__ == "__main__":
    main()
