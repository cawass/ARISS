"""Main ARISS sizing loop and post-processing."""

import logging
import sys
from dataclasses import replace
from copy import deepcopy
from dataclasses import fields, is_dataclass
from numbers import Real
from pathlib import Path
from typing import List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Budgets import sizing_model
from ariss.modules.Drag import drag_model
from ariss.modules.Power import power_model
from ariss.modules.Propulsion import propulsion_model
from ariss.utils.atmosphere import orbit_updates_from_height

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
residual = 10e10


def run_sizing_loop(
    loop_sc: SpacecraftState,
    max_iterations: int = 200,
    mass_tolerance: float = 1e-3,
    relaxation_factor: float = 0.3,
) -> Tuple[SpacecraftState, bool, List[SpacecraftState]]:

    orbit_updates = orbit_updates_from_height(loop_sc.mission_profile.mission_height)
    loop_sc = replace(loop_sc, orbit=replace(loop_sc.orbit, **orbit_updates))
    relaxation_factor = min(max(float(relaxation_factor), 0.0), 1.0)


    logger.info("Starting sizing loop. Initial Total Mass: %.2f kg", loop_sc.mass.Mass_total)
    history = []
    residual = 10e10
    converged = False
    for i in range(max_iterations):
        history.append(deepcopy(loop_sc))
        loop_sc = deepcopy(loop_sc)        
        sizing_model(loop_sc)
        propulsion_model(loop_sc)
        sizing_model(loop_sc)
        drag_model(loop_sc)
        sizing_model(loop_sc)
        power_model(loop_sc)
        sizing_model(loop_sc)
        if i > 0:
            residual = abs(loop_sc.mass.Mass_total - history[i-1].mass.Mass_total)
        logger.debug("Iter %d: Mass = %.6f kg | Residual = %.6e", i, loop_sc.mass.Mass_total , residual)
        if residual <= mass_tolerance and i > 10:
            logger.info("Convergence reached at iteration %d. Final Mass: %.2f kg", i, loop_sc.mass.Mass_total)
            converged = True
            history.append(deepcopy(loop_sc))
            break
        if loop_sc.orbit.altitude > 900:
            break

    if not converged:
        logger.warning(
            "Sizing loop FAILED to converge after %d iterations. Final residual: %.6f kg",
            max_iterations,
            residual,
        )


    return loop_sc, converged, history


if __name__ == "__main__":
    final_sc, _, _ = run_sizing_loop(SpacecraftState())
