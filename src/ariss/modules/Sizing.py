import os
import sys
import numpy as np

from iteration.SpacecraftClass import SpacecraftClass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#def power_sizing(sc: SpacecraftClass) -> None:

    # Known powers
#    known_fract = sc._power_frac_dict["prop"] +sc._power_frac_dict["adcs"]+sc._power_frac_dict["ttc&cndh"]
#    known_power = sc._power_budgets["prop"] + sc._power_budgets["adcs"] + sc._power_budgets["ttc&cndh"]
#    P_tot = known_power / known_fract
    
    # Total power
    #P_tot / sc._eta_power_sys*(1 + sc._margin_dict['pow'])


def mass_sizing(sc: SpacecraftClass) -> None:
    # Calculate known fraction
    
    known_fract = sc._mass_frac_dict["intake_ref"]+sc._mass_frac_dict["solar_pow"]+sc._mass_frac_dict["rad_thermal"]+sc._mass_frac_dict["dock"]+sc._mass_frac_dict["adcs"]+sc._mass_frac_dict["ttc&cndh"]
    known_mass = (sc.S_dict['prop'] + sc.S_dict['ref']) * sc._intake_dens  + sc.S_dict['pow'] * sc._solar_array_dens + sc.S_dict['therm'] * sc._rad_dens + sc._mass_budgets["dock"] + sc._mass_budgets["adcs"] + sc._mass_budgets["ttc&cndh"] 
    """
    # Propulsion dry mass
    sc.M_dry_dict['prop'] = sc._mass_budgets['prop']
     
    # Docking dry mass
    sc.M_dry_dict['dock'] = sc._mass_budgets['dock']
     
    # Refueling subsystem mass 
    sc.M_dry_dict['ref'] = (sc.S_dict['prop'] + sc.S_dict['ref']) * sc._intake_dens + sc._mass_budgets['ref'] # Intake calculated + fixed turbopump

    # Power dry mass 
    sc.M_dry_dict['pow'] = sc.S_dict['pow'] * sc._solar_array_dens + sc._mass_budgets['pow']

    # ADCS dry mass
    sc.M_dry_dict['adcs'] = sc._mass_budgets['adcs']

    # Thermal subsystem mass
    sc.M_dry_dict['therm'] = sc.S_dict['therm'] * sc._rad_dens + sc._mass_budgets["thermal"]

    # Structure subsystem mass 
    sc.M_dry_dict["struct"] = sc._mass_budgets["struct"]

    # TTC & CNDH dry mass
    sc.M_dry_dict['ttc&cndh'] = sc._mass_budgets['ttc&cndh']
    
    # calculate total dry mass by dividing mass of known components by their percentage of spacecraft
    known_mass = sum(sc.M_dry_dict.values())
   
    unkown_fraction = sc._mass_frac_dict['prop'] + sc._mass_frac_dict['adcs'] + sc._mass_frac_dict['struct']
    sc.M_dry_tot = known_mass * (1 + sc._margin_dict['mass'])
    """ 

    sc.M_dry_tot = known_mass / (known_fract) * (1 + sc._margin_dict['mass'])
    # calculate mass of subsystems
    for key in sc._mass_frac_dict.keys():
        sc.M_dry_dict[key] = sc._mass_frac_dict[key] * sc.M_dry_tot

def volume_sizing(sc: SpacecraftClass) -> None:

    # get bus volume from wet mass of spacecraft
    V_body = sum(sc.M_dry_dict[key] * sc._volume_frac_dict[key] for key in sc.M_dry_dict if key in sc._volume_frac_dict)
    V_body *= (1 + sc._margin_dict['volume'])
    # get bus dimensions
    S_in = sc.S_dict["ref"] + sc.S_dict["prop"]

    AR = 1.5
    print("D_in")
    sc.D_in = np.sqrt(S_in*4/np.pi)
    print(sc.D_in)
    sc.H_in = np.sqrt(S_in*4/np.pi)
    sc.L_in = AR*sc.D_in
    # assert np.isclose(S_in, sc.D_in * sc.H_in), f"Intake area calculation error, with S_in = {S_in}, D_in = {sc.D_in}, H_in = {sc.H_in}"
    sc.D = 1.1# np.sqrt(S_bus_cross * sc.DoH)
    sc.H = 1.1# sc.D / sc.DoH
    sc.L = 2.8
    #sc.L = V_body / (sc.D * sc.H)
    
    S_bus_cross = V_body / sc.L
    sc.taper = S_bus_cross / S_in


    # assert np.isclose(S_bus_cross, sc.D * sc.H), f"Cross-sectional area calculation error, with S_bus_cross = {S_bus_cross}, D = {sc.D}, H = {sc.H}"
    V_intake = 1 / 3 * np.pi*((sc.D_in/2)**2+(sc.D/2)**2 + (sc.D_in/2)*(sc.D/2))*sc.L_in

    # Calculate the angle of the wake for no body drag
    v_m = sc._sigma * np.sqrt(sc._R * sc.T_orb / sc._M)
    theta_max = np.arctan(v_m / sc.V_orb)
    print(theta_max)
    sc.theta_max = theta_max
    sc.taper =  theta_max
    if theta_max < 1e-6:
        L_max_D = 20
        L_max_H = 20
    else:
        L_max_D = (sc.D_in - sc.D) / 2 / np.tan(theta_max) - sc.L_in
        print(L_max_D)
    #    L_max_H = (sc.H_in - sc.H) / 2 / np.tan(theta_max) - sc.L_in
    if L_max_D < 0:
        L_max_D = 0
    #if L_max_H < 0:
    #    L_max_H = 0

    sc.V_tot = V_intake + V_body
    sc.S_dict['bus'] = sc.L * sc.D * 2 + sc.H * sc.D + sc.L * sc.H * 2  # Total surface area of the bus
    sc.S_dict['bus_top'] = sc.L * sc.D  # Top surface area of the bus

    # Bus logic
    sc.w_bus = 1
    if sc.L > L_max_D:
        sc.w_bus -= ((sc.L - L_max_D) * sc.D * 2) / sc.S_dict['bus']
    #if sc.L > L_max_H:
    #    sc.w_bus -= ((sc.L - L_max_H) * sc.H * 2) / sc.S_dict['bus']
    #sc.w_bus -= sc.H * sc.D / sc.S_dict['bus']  # Subtract the area of the bus backside

    # Solar panel logic
    S_intakeSolar = (sc.D_in + sc.D) / 2 * sc.L_in
    S_bodySolar = sc.D * sc.L + S_intakeSolar

    if sc.S_dict['pow'] <= S_bodySolar:
        sc.S_dict['solar_bus'] = sc.S_dict['pow']
        sc.S_dict['solar_extended'] = 0
        sc.w_solar = 1

    elif sc.S_dict['pow'] > S_bodySolar:
        sc.S_dict['solar_bus'] = S_bodySolar
        sc.S_dict['solar_extended'] = sc.S_dict['pow'] - sc.S_dict['solar_bus']
        #sc.w_solar = 0
        sc.w_solar = L_max_D * (sc.D_in - sc.D) / 2 / sc.S_dict['solar_extended']
        if sc.w_solar > 1:
            sc.w_solar = 1
    
    # Thermal area logic
    sc.w_thermal = L_max_D * (sc.H_in - sc.H) / 4 / sc.S_dict['therm']
    if sc.w_thermal > 1:
        sc.w_thermal = 1

    print(sc.S_dict["therm"])

    # print(f"Angle is {np.degrees(theta_max)} degrees, L_max_D is {L_max_D}, L_max_H is {L_max_H}")
    # print(f"L: {sc.L}, H: {sc.H}, D: {sc.D}, L_in: {sc.L_in}, H_in: {sc.H_in}, D_in: {sc.D_in}")
    # print(sc.w_bus, sc.w_solar, sc.w_thermal)


def sizing_model(sc: SpacecraftClass) -> None:

    # power_sizing(sc)

    mass_sizing(sc)

    volume_sizing(sc)
