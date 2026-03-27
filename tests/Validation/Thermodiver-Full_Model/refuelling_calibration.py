import numpy as np
import pymsis as msis
from ariss.utils.atmosphere import sample_atmosphere_at_height

### Refuelling data from thermodiver

p_tank = 24000000.0 # [Pa]
V_tank = 0.12 # [m^3]
m_refuel = 284.9 # [kg]
T_des = 320 # [K]
collection_efficiency = 0.636 # [-]
h_orb = 188.0151 # [km]

### Atmos/mission data
mission_data = sample_atmosphere_at_height(h_orb)

# Calculate mole fractions to find specific heat ratio (gamma)
n_o = mission_data.o_density / 15.999e-3
n_n2 = mission_data.n2_density / 28.0134e-3
n_o2 = mission_data.o2_density / 31.9988e-3
n_total = n_o + n_n2 + n_o2
x_mono = n_o / n_total if n_total > 0 else 0

# For a mixture of monatomic (O) and diatomic (N2, O2) gases:
# gamma = C_p / C_v = (5/2 * x_mono + 7/2 * x_dia) / (3/2 * x_mono + 5/2 * x_dia)
gamma = (7 - 2 * x_mono) / (5 - 2 * x_mono) # 1.512

R_spec = mission_data.specific_gas_constant
p_orb = mission_data.density * R_spec * mission_data.temperature # [Pa]
rho_orb = mission_data.density # [kg/m^3]
v_orb = mission_data.orbital_velocity # [m/s]

print(f"gamma: {gamma}")
print(f"R_spec: {R_spec}")
print(f"p_orb: {p_orb}")
print(f"rho_orb: {rho_orb}")
print(f"v_orb: {v_orb}")


### Ariss iterables
Intake_area = 4.04 # [m^2]
m_dot_b = Intake_area * rho_orb * v_orb * collection_efficiency # [kg/s]
Power_TD = 336.5 # [W]


### Find eta_refuel that matches the power
eta_refuel_calibrated = 1 / Power_TD * 1 / (gamma - 1) * m_dot_b * gamma * R_spec * T_des * ((p_tank / p_orb) ** ((gamma - 1) / gamma) - 1)


### Ariss Power TD
Power_TD_ariss = 1 / 0.5 * 1 / (gamma - 1) * m_dot_b * gamma * R_spec * T_des * ((p_tank /  p_orb) ** ((gamma - 1) / gamma) - 1)

### Isothermal compression
Power_TD_isothermal = m_dot_b * R_spec * T_des * np.log(p_tank / p_orb)
eta_refuel_isothermal = 1 / Power_TD * m_dot_b * R_spec * T_des * np.log(p_tank / p_orb)


print(f"eta_refuel_calibrated: {eta_refuel_calibrated}")
print(f"Power_TD_ariss: {Power_TD_ariss}")
print(f"Power_TD_isothermal: {Power_TD_isothermal}")
print(f"eta_refuel_isothermal: {eta_refuel_isothermal}")