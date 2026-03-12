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
#      Validation entry point for the Mansur reference configuration.
#
#  Project:        ARISS
#  Module:         MansurVerification.py
# ============================================================================== #

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss import plot_simulation_history, run_simulation


def run_mansur_verification(
    config_path: str | Path | None = None,
    show_ui: bool = False,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-3,
):
    validation_path = Path(config_path) if config_path is not None else Path(__file__).with_name("MansurVerification.toml")
    final_sc, converged, history = run_simulation(
        validation_path,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
    )

    print(f"Converged: {converged}")
    print(f"Iterations: {len(history)}")
    print(f"Final altitude [km]: {final_sc.orbit.altitude:.6f}")
    print(f"Final total mass [kg]: {final_sc.mass.Mass_total:.6f}")
    print(f"Final thrust [N]: {final_sc.thruster.thrust:.6e}")
    print(f"Final drag [N]: {final_sc.drag.drag_total:.6e}")

    plot_simulation_history(
            validation_path,
            max_iterations=max_iterations,
            mass_tolerance=mass_tolerance,
            show=True,
        )

    return final_sc, converged, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Mansur validation case.")
    parser.add_argument("--show-ui", action="store_true", help="Open the Tk history UI after the sizing loop.")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--mass-tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()

    run_mansur_verification(
        show_ui=args.show_ui,
        max_iterations=args.max_iterations,
        mass_tolerance=args.mass_tolerance,
    )
