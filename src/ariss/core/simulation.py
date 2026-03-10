# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS — Atmospheric Refueling Iterative System Solver
# ============================================================================== #
#  Description:
#      Core module for ARISS (Atmospheric Refueling Iterative System Solver).
#      Provides numerical routines and simulation utilities for modeling and
#      solving atmospheric refueling dynamics using iterative methods.
#
#  Project:        ARISS
#  Module:         simulation.py
#  Author:         Carlos Carrasco Requejo, Lucas Calderon del Rio
#
# ============================================================================
import logging
import sys
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Budgets import sizing_model
from ariss.modules.Drag import drag_model
from ariss.modules.Power import power_model
from ariss.modules.Propulsion import propulsion_model
from ariss.modules.Refueling import refueling_model
from ariss.modules.Thermal import thermal_model
from ariss.utils.atmosphere import orbit_updates_from_height

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

residual = 10e10

def compute_drag_diagnostics(sc: SpacecraftState) -> SpacecraftState:

    # Inputs:
    #   sc: spacecraft state at the selected iteration.
    #
    # Outputs:
    #   state: copied spacecraft state after running the drag model.

    # Use a copy so UI diagnostics do not mutate the saved history state.
    state = deepcopy(sc)
    drag_model(state)
    return state

def run_sizing_loop(loop_sc: SpacecraftState, max_iterations: int = 200, mass_tolerance: float = 1e-3) -> Tuple[SpacecraftState, bool, List[SpacecraftState]]:

    # Inputs:
    #   loop_sc: initial SpacecraftState.
    #   max_iterations: maximum number of iterations.
    #   mass_tolerance: convergence tolerance on total mass [kg].
    #
    # Outputs:
    #   loop_sc: final spacecraft state.
    #   converged: True if the loop converged.
    #   history: saved spacecraft state at each iteration.
    #
    # Equations used:
    #   residual_i = |M_i - M_(i-1)|
    #   converged if residual_i <= mass_tolerance, for i > 10
    #   orbit updates from orbit_updates_from_height(h)

    # Initialize the orbit-dependent atmospheric properties from the starting
    # mission altitude before entering the iterative sizing loop.
    orbit_updates = orbit_updates_from_height(
        loop_sc.orbit.altitude,
        msis_date=loop_sc.orbit.msis_date,
        msis_f107=loop_sc.orbit.msis_f107,
        msis_ap=loop_sc.orbit.msis_ap,
    )
    loop_sc = replace(loop_sc, orbit=replace(loop_sc.orbit, **orbit_updates))

    # Prepare the iteration history and convergence trackers.
    logger.info("Starting sizing loop. Initial Total Mass: %.2f kg", loop_sc.mass.Mass_total)
    history = []
    residual = 10e10
    converged = False

    # Each pass updates the spacecraft state by cycling through the subsystem
    # models and re-closing the mass and power budgets between them.
    for i in range(max_iterations):
        print(f"Iteratation Number{i}")
        # Save the pre-iteration state so convergence history can be inspected.
        history.append(deepcopy(loop_sc))
        loop_sc = deepcopy(loop_sc)

        # The sizing model is called repeatedly because propulsion, refueling,
        # drag, and power updates each modify quantities that feed back into the
        # overall spacecraft mass and power closure.
        sizing_model(loop_sc)
        propulsion_model(loop_sc)
        sizing_model(loop_sc)
        refueling_model(loop_sc)
        sizing_model(loop_sc)
        drag_model(loop_sc)
        sizing_model(loop_sc)
        power_model(loop_sc)
        sizing_model(loop_sc)
        thermal_model(loop_sc)
        sizing_model(loop_sc)
        
        # Measure convergence by the change in total spacecraft mass between
        # consecutive saved iterations.
        if i > 0:
            residual = abs(loop_sc.mass.Mass_total - history[i - 1].mass.Mass_total)
        logger.debug("Iter %d: Mass = %.6f kg | Residual = %.6e", i, loop_sc.mass.Mass_total, residual)

        # Require both a small residual and a minimum number of iterations to
        # avoid exiting on an early transient match.
        if residual <= mass_tolerance and i > 10:
            logger.info("Convergence reached at iteration %d. Final Mass: %.2f kg", i, loop_sc.mass.Mass_total)
            converged = True
            history.append(deepcopy(loop_sc))
            break

        # Abort if the solved altitude drifts outside the modeled mission range.
        if loop_sc.orbit.altitude > 900:
            break

    # Warn when the loop exits without satisfying the convergence criterion.
    if not converged:
        logger.warning(
            "Sizing loop FAILED to converge after %d iterations. Final residual: %.6f kg",
            max_iterations,
            residual,
        )
    return loop_sc, converged, history


if __name__ == "__main__":
    final_sc, _, _ = run_sizing_loop(SpacecraftState())
