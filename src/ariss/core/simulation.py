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
import tomllib
from dataclasses import replace
from copy import deepcopy
from os import PathLike
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

def load_spacecraft_from_base_config(case_path: str | PathLike[str] | None = None,*, base_config_path: str | PathLike[str] | None = None, ) -> SpacecraftState:

    # Inputs:
    #   case_path:
    #       Optional TOML file containing case-specific overrides. These values
    #       will overwrite parameters defined in the base configuration.
    #
    #   base_config_path:
    #       Optional path to the base spacecraft configuration TOML file.
    #       If not provided, the default "base_config.toml" located in the same
    #       directory as this file will be used.
    #
    # Outputs:
    #   SpacecraftState object initialized from the base configuration and
    #   updated with any parameters specified in case_path.

    base_path = ( Path(base_config_path) if base_config_path is not None else Path(__file__).with_name("base_config.toml"))

    # Load the base spacecraft configuration
    sc = SpacecraftState.from_toml(base_path)

    if case_path is not None:
        with open(Path(case_path), "rb") as handle:
            overrides = tomllib.load(handle)

        # Recursively walk through the override dictionary and apply values
        stack = [(sc, overrides, "")]

        while stack:
            target, payload, prefix = stack.pop()

            for key, value in payload.items():
                dotted = f"{prefix}.{key}" if prefix else key

                if not hasattr(target, key):
                    raise KeyError(f"Unknown key in spacecraft override: {dotted}")

                current = getattr(target, key)

                # If the value is another dictionary, continue descending
                # into the nested object structure.
                if isinstance(value, dict):
                    stack.append((current, value, dotted))
                else:
                    # Otherwise overwrite the attribute with the new value
                    object.__setattr__(target, key, value)

    return sc


def run_sizing_loop(loop_sc: SpacecraftState, max_iterations: int = 200, mass_tolerance: float = 1e-3, force_tolerance: float = 1e-2, density_tolerance: float = 1e-3) -> Tuple[SpacecraftState, bool, List[SpacecraftState]]:

    # Inputs:
    #   loop_sc: initial SpacecraftState.
    #   max_iterations: maximum number of iterations.
    #   mass_tolerance: convergence tolerance on total mass [kg].
    #   force_tolerance: relative tolerance on thrust-drag balance [-].
    #   density_tolerance: relative tolerance on atmospheric-density change [-].
    #
    # Outputs:
    #   loop_sc: final spacecraft state.
    #   converged: True if the loop converged.
    #   history: saved spacecraft state at each iteration.
    #
    # Residual definitions used in the convergence check:
    #   mass_residual_i = |M_i - M_(i-1)|
    #   force_residual_i =|T_i - L_i| / max(|T_i|, |L_i|, eps)
    #   density_residual_i =|rho_i - rho_(i-1)| / max(|rho_i|, |rho_(i-1)|, eps)
    #
    # Convergence condition:
    #   All residuals must be below their tolerances and i > 10
    #
    # Orbit updates are computed from:
    #   orbit_updates_from_height(h)

    # Initialize the orbit-dependent atmospheric properties from the starting
    # mission altitude before entering the iterative sizing loop.
    orbit_updates = orbit_updates_from_height(loop_sc.orbit.altitude, msis_date=loop_sc.orbit.msis_date, msis_f107=loop_sc.orbit.msis_f107, msis_ap=loop_sc.orbit.msis_ap, latitude=loop_sc.orbit.latitude, longitude=loop_sc.orbit.longitude, use_average=loop_sc.orbit.use_average)
    loop_sc = replace(loop_sc, orbit=replace(loop_sc.orbit, **orbit_updates))

    # Prepare the iteration history and convergence trackers.
    logger.info("Starting sizing loop. Initial Total Mass: %.2f kg", loop_sc.mass.Mass_total)
    history = []
    residual = 1.0e11
    force_residual = 1.0e11
    density_residual = 1.0e11
    converged = False

    loop_sc.check_bounds()

    # Each pass updates the spacecraft state by cycling through the subsystem
    # models and re-closing the mass and power budgets between them.
    for i in range(max_iterations):

        # Save the pre-iteration state so convergence history can be inspected.
        history.append(deepcopy(loop_sc))
        loop_sc = deepcopy(loop_sc)

        # The sizing model is called repeatedly because propulsion, refueling,
        # drag, and power updates each modify quantities that feed back into the
        # overall spacecraft mass and power closure.
        sizing_model(loop_sc)
        drag_model(loop_sc)
        propulsion_model(loop_sc)
        refueling_model(loop_sc)
        power_model(loop_sc)
        thermal_model(loop_sc)

        # Measure convergence by comparing the current state to the previous
        # saved iteration.
        if i > 0:
            previous_state = history[i - 1]

            # Absolute mass change between iterations
            residual = abs(loop_sc.mass.Mass_total - previous_state.mass.Mass_total)

            # Relative density change normalized by the largest magnitude
            density_scale = max(abs(loop_sc.orbit.density), abs(previous_state.orbit.density), 1.0e-20,)
            density_residual = abs(loop_sc.orbit.density - previous_state.orbit.density) / density_scale

        # Relative thrust-load imbalance
        force_scale = max(abs(loop_sc.thruster.thrust),abs(loop_sc.thruster.required_load),1.0e-12)
        force_residual = abs(loop_sc.thruster.thrust - loop_sc.thruster.required_load) / force_scale

        logger.debug("Iter %d: Mass = %.6f kg | Mass residual = %.6e | Force residual = %.6e | Density residual = %.6e", i,  loop_sc.mass.Mass_total, residual, force_residual, density_residual)

        # Require mass stability, force balance, density stability, and a
        # minimum number of iterations to avoid exiting on an early transient
        # match.        
        if loop_sc.orbit.altitude > 345:
            converged = False
            break

        if (residual <= mass_tolerance and force_residual <= force_tolerance and density_residual <= density_tolerance and i > 10) and loop_sc.orbit.altitude < 345:
            logger.info("Convergence reached at iteration %d. Final Mass: %.2f kg", i, loop_sc.mass.Mass_total)
            converged = True
            history.append(deepcopy(loop_sc))
            break

        # Abort if the solved altitude drifts outside the modeled mission range.

    return loop_sc, converged, history


if __name__ == "__main__":
    sc = load_spacecraft_from_base_config()
    final_sc, _, _ = run_sizing_loop(sc)
