# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS - Atmospheric Refueling Iterative System Solver
# ============================================================================== #
#  Description:
#      Runs the Crandall & Wirz (2022) two-spacecraft cases (3U and 6U)
#      and prints solved Isp values.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022TwoSpacecraftIsp.py
# ============================================================================== #

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path


# ------------------------------------------------------------------------------ #
# Path setup
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

for p in (SRC, VALIDATION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ------------------------------------------------------------------------------ #
# ARISS imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config, logger as simulation_logger, run_sizing_loop
from CrandallWirz2022Validation import CASE_SPECS


MAX_ITERATIONS = 260
MASS_TOLERANCE = 1.0e-3


def run_two_spacecraft_isp() -> None:
    print("Crandall & Wirz 2022 two-spacecraft solved Isp:")

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)
    try:
        for spec in CASE_SPECS:
            spacecraft = load_spacecraft_from_base_config(spec["config"])
            with redirect_stdout(io.StringIO()):
                final_sc, converged, _ = run_sizing_loop(
                    spacecraft,
                    max_iterations=MAX_ITERATIONS,
                    mass_tolerance=MASS_TOLERANCE,
                )

            print(
                f"- {spec['title']}: "
                f"Isp = {float(final_sc.thruster.specific_impulse):.2f} s "
                f"(converged={converged})"
            )
    finally:
        simulation_logger.setLevel(old_level)


if __name__ == "__main__":
    run_two_spacecraft_isp()
