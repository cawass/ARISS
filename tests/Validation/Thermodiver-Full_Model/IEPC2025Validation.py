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
#      Validation entry point for the IEPC 2025 configuration.
#
#  Project:        ARISS
#  Module:         IEPC2025Validation.py
# ============================================================================== #

import sys
from pathlib import Path


# ------------------------------------------------------------------------------ #
# Path setup so the ARISS source can be imported
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ------------------------------------------------------------------------------ #
# ARISS imports
# ------------------------------------------------------------------------------ #

from ariss import run_simulation, plot_simulation_history


# ------------------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).with_name("IEPC2025Validation.toml")

MAX_ITERATIONS = 200
MASS_TOLERANCE = 1e-3

SHOW_UI = True


# ------------------------------------------------------------------------------ #
# Run IEPC 2025 validation
# ------------------------------------------------------------------------------ #

def run_iepc2025_validation():

    final_sc, converged, history = run_simulation(
        CONFIG_PATH,
        max_iterations=MAX_ITERATIONS,
        mass_tolerance=MASS_TOLERANCE,
    )

    print(f"Converged: {converged}")
    print(f"Iterations: {len(history)}")
    print(f"Final altitude [km]: {final_sc.orbit.altitude:.6f}")
    print(f"Final total mass [kg]: {final_sc.mass.Mass_total:.6f}")
    print(f"Final thrust [N]: {final_sc.thruster.thrust:.6e}")
    print(f"Final drag [N]: {final_sc.drag.drag_total:.6e}")
    print("Datapoint MSE: n/a (no digitized datapoint dataset configured for this script)")

    if SHOW_UI:
        plot_simulation_history(
            CONFIG_PATH,
            max_iterations=MAX_ITERATIONS,
            mass_tolerance=MASS_TOLERANCE,
            show=True,
        )

    return final_sc, converged, history


# ------------------------------------------------------------------------------ #
# Run script
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":
    run_iepc2025_validation()
