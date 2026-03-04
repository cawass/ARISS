import copy
import logging
from typing import Tuple, List

from ariss.spacecraft.spacecraft import SpacecraftState

# logging tool
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_sizing_loop(
    initial_sc: SpacecraftState, 
    max_iterations: int = 200, 
    mass_tolerance: float = 1e-3
) -> Tuple[SpacecraftState, bool, List[SpacecraftState]]:
    """
    Executes the main ARISS iterative sizing and convergence loop.
    
    This loop iteratively recalculates the physics, geometries, and subsystem
    masses until the total spacecraft mass converges within the given tolerance.

    Args:
        initial_sc: The starting SpacecraftState (often loaded from a template).
        max_iterations: Safety limit to prevent infinite loops.
        mass_tolerance: The acceptable difference in total mass (kg) between iterations.

    Returns:
        final_sc: The converged SpacecraftState object.
        converged: A boolean indicating if the loop successfully converged.
        history: A history of the spacecraft state at every iteration step.
    """
    
    current_sc = initial_sc
    history: List[SpacecraftState] = [current_sc]
    converged = False
    
    logger.info(f"Starting sizing loop. Initial Total Mass: {current_sc.total_mass:.2f} kg")

    for i in range(max_iterations):
        # ---------------------------------------------------------
        # 1. PHYSICS & ENVIRONMENT (Read-only on current_sc)
        # ---------------------------------------------------------
        # Example:
        # env_density = atmosphere_model(current_sc.orbit.altitude)
        # drag_results = drag_model(current_sc.geometry, env_density, current_sc.orbit.velocity)
        
        
        # ---------------------------------------------------------
        # 2. SUBSYSTEM SIZING & BUDGETS
        # ---------------------------------------------------------
        # Example:
        # req_power = propulsion_model(current_sc.thruster, new_drag_force)
        # solar_mass = power_system_sizing(req_power)
        # struct_mass = structural_sizing(current_sc.mass_budget)
        
        # Calculate new total mass based on sized components

        
        # ---------------------------------------------------------
        # 3. STATE UPDATE (Creating the next iteration)
        # ---------------------------------------------------------
        # Feed all the results from steps 1 and 2 into the update method.
        # This keeps the history completely immutable and traceable.
        
        next_sc = current_sc.update(
            # total_mass=new_total_mass,
            
            # # Update nested subsystems via dictionary syntax
            # orbit={
            #     "density": new_density
            # },
            # geometry={
            #     "drag_coeff": new_cd
            # },
            # mass_budget={
            #     "structure": new_structure_mass,
            #     "payload": new_payload_mass
            # }
        )
        
        history.append(next_sc)
        
        
        # ---------------------------------------------------------
        # 4. CONVERGENCE CHECK
        # ---------------------------------------------------------
        # We check if the difference in calculated total mass is 
        # below the acceptable tolerance.
        residual = abs(next_sc.total_mass - current_sc.total_mass)
        
        logger.debug(f"Iter {i}: Mass = {next_sc.total_mass:.2f} kg | Residual = {residual:.6f}")
        
        if residual <= mass_tolerance:
            logger.info(f"Convergence reached at iteration {i}! Final Mass: {next_sc.total_mass:.2f} kg")
            converged = True
            current_sc = next_sc
            break
            
        # If not converged, step forward for the next loop
        current_sc = next_sc
        
        
    if not converged:
        logger.warning(f"Sizing loop FAILED to converge after {max_iterations} iterations. "
                       f"Final residual: {residual:.5f} kg")

    return current_sc, converged, history
