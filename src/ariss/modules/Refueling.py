from ariss.core.spacecraft import SpacecraftState

def refueling_model(sc: SpacecraftState) -> float:
    """
    Calculate the refueling power required.
    It calculates the power required by an active refuelling system if applicable.
    Skips the calculation otherwise.
    
    Args:
        sc (SpacecraftState): Spacecraft state.
    
    Returns:
        float: Refueling power required.
        float: Refueling area.
    """
    # Calculate area needed for refuelling
    A_ref = sc.mass.Mass_prop / (sc.refueling.t_refuel * sc.orbit.density * sc.orbit.velocity * sc.refueling.coll_eff)

    if sc.mission_profile.active_refueling:

        # Calculate the work done on the fluid
        m_flow = sc.orbit.density * sc.orbit.velocity * sc.geometry.A_ref * sc.refueling.coll_eff # Mass flow rate after the intake

        m_dot_b = m_flow +  sc.thruster.m_flow  # Add the propellant mass flow rate to the intake mass flow rate because there is no bypass
        P_ref = 1 / sc.refueling.eta_refuel * 1 / (sc.orbit.gamma - 1) * m_dot_b * sc.orbit.R_spec * sc.thermal.T_des * ((sc.refueling.p_tank / sc.orbit.p_orb) ** ((sc.orbit.gamma - 1) / sc.orbit.gamma) - 1)

        sc.refueling.V_prop = sc.mass.Mass_prop * sc.orbit.R_spec * sc.thermal.T_des / sc.refueling.p_tank
        sc.refueling.m_flow = m_flow
        
    else:
        P_ref = 0

    # Save to spacecraft
    sc.power.P_ref = P_ref
    sc.geometry.A_ref = A_ref
        
    return P_ref, A_ref