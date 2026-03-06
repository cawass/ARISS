"""Main ARISS sizing loop and post-processing."""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Budjects import BudgetIdx, budjet_model
from ariss.modules.DeltaV import delta_v_model
from ariss.modules.Drag import drag_model
from ariss.modules.Power import power_model
from ariss.modules.Propulsion import propulsion_model
from ariss.utils.atmosphere import orbit_updates_from_height

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _bound_capture_areas(min_refuel_area: float, prop_area_candidate: float) -> tuple[float, float, float]:
    """Return bounded propulsion, refueling, and intake areas."""
    prop_area = max(prop_area_candidate, 1.0e-12)
    refuel_area = max(min_refuel_area, 1.0e-12)
    intake_area = prop_area + refuel_area
    return prop_area, refuel_area, intake_area


def _apply_budget_array(sc: SpacecraftState, values: list[float]) -> SpacecraftState:
    """Map the flat budget array into ``mass`` and ``power`` state updates."""
    return sc.update(
        mass={
            "Mass_in": values[BudgetIdx.MASS_IN],
            "Mass_body": values[BudgetIdx.MASS_BODY],
            "Mass_solar": values[BudgetIdx.MASS_SOLAR],
            "Mass_rad": values[BudgetIdx.MASS_RAD],
            "Mass_prop": values[BudgetIdx.MASS_PROP],
            "Mass_ADCS": values[BudgetIdx.MASS_ADCS],
            "Mass_payload": values[BudgetIdx.MASS_PAYLOAD],
            "Mass_refprop": values[BudgetIdx.MASS_REFPROP],
            "Mass_total": values[BudgetIdx.MASS_TOTAL],
        },
        power={
            "Power_in": values[BudgetIdx.POWER_IN],
            "Power_body": values[BudgetIdx.POWER_BODY],
            "Power_solar": values[BudgetIdx.POWER_SOLAR],
            "Power_rad": values[BudgetIdx.POWER_RAD],
            "Power_prop": values[BudgetIdx.POWER_PROP],
            "Power_ADCS": values[BudgetIdx.POWER_ADCS],
            "Power_payload": values[BudgetIdx.POWER_PAYLOAD],
            "Power_refprop": values[BudgetIdx.POWER_REFPROP],
            "Power_total": values[BudgetIdx.POWER_TOTAL],
        },
    )


def run_sizing_loop(
    initial_sc: SpacecraftState,
    max_iterations: int = 200,
    mass_tolerance: float = 1e-3,
) -> Tuple[SpacecraftState, bool, List[SpacecraftState]]:
    """Run iterative drag-propulsion-power-budget sizing until mass converges."""
    if initial_sc.mission_profile.mission_height > 0.0:
        current_sc = initial_sc.update(
            orbit=orbit_updates_from_height(initial_sc.mission_profile.mission_height),
        )
    else:
        current_sc = initial_sc

    configured_refuel_area = (
        current_sc.mission_profile.refueling_area if current_sc.mission_profile.refueling_area > 0.0 else current_sc.geometry.A_ref
    )
    history: List[SpacecraftState] = [current_sc]
    converged = False
    min_refuel_area = max(configured_refuel_area, 1.0e-12)

    logger.info("Starting sizing loop. Initial Total Mass: %.2f kg", current_sc.total_mass)

    for i in range(max_iterations):
        previous_sc = current_sc

        drag_force = float(sum(drag_model(previous_sc)))
        prop_area_candidate, prop_power, required_thrust = propulsion_model(previous_sc, drag_force)
        prop_area, refuel_area, intake_area = _bound_capture_areas(min_refuel_area, prop_area_candidate)

        pre_budget_sc = previous_sc.update(
            geometry={
                "A_in": intake_area,
                "A_ref": refuel_area,
                "A_prop": prop_area,
            },
            thruster={
                "power_required": prop_power,
                "thrust": required_thrust,
            },
        )

        first_budget = budjet_model(pre_budget_sc)
        deployable_area = power_model(
            pre_budget_sc,
            power_required=first_budget[BudgetIdx.POWER_TOTAL],
            efficiency=pre_budget_sc.power.eff,
            alignment_deg=pre_budget_sc.power.alignment,
        )
        area_budget_sc = pre_budget_sc.update(geometry={"A_solar": deployable_area})

        final_budget = budjet_model(area_budget_sc)
        current_sc = _apply_budget_array(area_budget_sc, final_budget)
        history.append(current_sc)

        residual = abs(current_sc.total_mass - previous_sc.total_mass)
        logger.debug("Iter %d: Mass = %.6f kg | Residual = %.6e", i, current_sc.total_mass, residual)
        if residual <= mass_tolerance:
            logger.info("Convergence reached at iteration %d. Final Mass: %.2f kg", i, current_sc.total_mass)
            converged = True
            break

    if not converged:
        logger.warning(
            "Sizing loop FAILED to converge after %d iterations. Final residual: %.6f kg",
            max_iterations,
            abs(history[-1].total_mass - history[-2].total_mass) if len(history) > 1 else 0.0,
        )

    # Delta-v sizing is a post-process on the converged (or last) state, not part of convergence.
    final_budget = budjet_model(current_sc)
    delta_v_terms = delta_v_model(current_sc, final_budget)
    current_sc = current_sc.update(mission_profile={"refueling_area": min_refuel_area})
    history[-1] = current_sc

    logger.info(
        "Post-sizing delta-v result: required delta-v %.3f m/s, refueling time %.3f days",
        delta_v_terms["delta_v_total"],
        delta_v_terms["refueling_time_required"] / (3600.0 * 24.0),
    )
    return current_sc, converged, history


if __name__ == "__main__":
    final_sc, converged, _ = run_sizing_loop(SpacecraftState())
    if converged:
        logger.info("Convergence achieved. Final Total Mass: %.2f kg", final_sc.total_mass)
    else:
        logger.warning("Maximum iterations reached without convergence.")
