from ariss.core.spacecraft import SpacecraftState



def sizing_model(sc: SpacecraftState) -> list[float]:
    sc.mass.Mass_in = (sc.geometry.A_in + sc.geometry.A_body) * sc.geometry.L_in * sc.rate.R_mass_volume_in / 2.0
    sc.mass.Mass_body = sc.geometry.A_body * sc.geometry.L_body * sc.rate.R_mass_volume_body
    sc.mass.Mass_solar = sc.geometry.A_solar * sc.rate.R_mass_surface_solar
    sc.mass.Mass_rad = sc.geometry.A_rad * sc.rate.R_mass_surface_rad
    
    sc.mass.Mass_total = sc.mass.Mass_in + sc.mass.Mass_body + sc.mass.Mass_solar + sc.mass.Mass_rad + sc.mass.Mass_prop + sc.mass.Mass_ADCS + sc.mass.Mass_payload + sc.mass.Mass_refprop 

    sc.power.Power_solar = sc.power.Power_total / (sc.solar.eta_power) -  sc.power.Power_total
    sc.power.Power_prop = sc.thruster.power_required / (sc.thruster.thruster_eff)

    sc.power.Power_total = sc.power.Power_in + sc.power.Power_body + sc.power.Power_solar + sc.power.Power_rad + sc.power.Power_prop + sc.power.Power_ADCS + sc.power.Power_payload + sc.power.Power_refprop