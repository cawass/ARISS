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
#      Mass and power budget closure relations used inside the sizing loop.
#
#  Project:        ARISS
#  Module:         Budgets.py
#  Author:         Carlos Carrasco Requejo
# ============================================================================

from ariss.core.spacecraft import SpacecraftState

def sizing_model(sc: SpacecraftState) -> None:
    # Inputs:
    #   sc: spacecraft state with geometry, rate, mass, power, solar, and thruster data.
    #
    # Outputs:
    #   Updates mass and power budgets in place.
    #
    # Equations used:
    #   Mass_in = ((A_in + A_body) / 2) * L_in * R_mass_volume_in
    #   Mass_body = A_body * L_body * R_mass_volume_body
    #   Mass_solar = A_solar * R_mass_surface_solar
    #   Mass_rad = A_rad * R_mass_surface_rad
    #   Power_solar = Power_total * (1 / eta_power - 1)
    #   Power_prop = power_required / thruster_eff
    #   Power_total = sum(subsystem powers)

    # Size the structural masses from simple volumetric and areal scaling laws.
    # The intake mass uses the average of inlet and body areas to approximate a
    # tapered intake volume.
    sc.mass.Mass_in = (sc.geometry.A_in + sc.geometry.A_body) * sc.geometry.L_in * sc.rate.R_mass_volume_in / 2.0
    sc.mass.Mass_body = sc.geometry.A_body * sc.geometry.L_body * sc.rate.R_mass_volume_body
    sc.mass.Mass_solar = sc.geometry.A_solar * sc.rate.R_mass_surface_solar
    sc.mass.Mass_rad = sc.geometry.A_rad * sc.rate.R_mass_surface_rad

    # Close the total mass budget used by the outer sizing loop convergence check.
    sc.mass.Mass_total = sc.mass.Mass_in + sc.mass.Mass_body + sc.mass.Mass_solar + sc.mass.Mass_rad + sc.mass.Mass_prop + sc.mass.Mass_ADCS + sc.mass.Mass_payload + sc.mass.Mass_refprop

    # Convert total delivered electrical demand into the extra solar-generation
    # overhead required by power-chain losses, then compute propulsion bus power
    # from the thruster efficiency.
    sc.power.Power_prop = sc.thruster.power
    sc.power.Power_solar = sc.power.Power_total / (sc.solar.eta_power) -  sc.power.Power_total


    # Rebuild the total spacecraft electrical demand from all subsystem terms.
    sc.power.Power_total = sc.power.Power_in + sc.power.Power_body + sc.power.Power_solar + sc.power.Power_rad + sc.power.Power_prop + sc.power.Power_ADCS + sc.power.Power_payload + sc.power.Power_refprop
