from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss import plot_simulation_history, run_simulation

if __name__ == "__main__":
    plot_simulation_history(Path(__file__).with_name("MansurVerification.toml"))

