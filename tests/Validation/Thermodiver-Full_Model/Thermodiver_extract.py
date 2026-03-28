import sys
from pathlib import Path
import pprint
from dataclasses import asdict
import numpy as np

# ------------------------------------------------------------------------------ #
# Path setup so the ARISS source can be imported
# ------------------------------------------------------------------------------ #
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ------------------------------------------------------------------------------ #
# ARISS imports
# ------------------------------------------------------------------------------ #
from ariss import run_simulation

CONFIG_PATH = Path(__file__).with_name("IEPC2025Validation.toml")

def extract_values():
    print(f"Running simulation with config: {CONFIG_PATH.name} ...\n")
    final_sc, converged, history = run_simulation(
        CONFIG_PATH,
        max_iterations=200,
        mass_tolerance=1e-3,
    )
    
    print("\n" + "="*50)
    print(" SIMULATION SUMMARY ")
    print("="*50)
    print(f"Converged: {converged}")
    print(f"Iterations: {len(history)}")
    print(f"Final Altitude [km]: {final_sc.orbit.altitude:.4f}")
    
    print("\n--- MASS COMPONENTS [kg] ---")
    mass_dict = asdict(final_sc.mass)
    for k, v in mass_dict.items():
        print(f"  {k:<20}: {v:.4f}")
        
    print("\n--- POWER COMPONENTS [W] ---")
    power_dict = asdict(final_sc.power)
    for k, v in power_dict.items():
        print(f"  {k:<20}: {v:.4f}")
    
    print("\n--- GEOMETRY/AREAS [m^2] ---")
    geo_dict = asdict(final_sc.geometry)
    for k, v in geo_dict.items():
        if type(v) is str:
            print(f"  {k:<20}: {v}")
        elif type(v) is bool:
            print(f"  {k:<20}: {v}")
        else:
            print(f"  {k:<20}: {v:.4f}")
    # Intake diameter
    print(F"Intake diameter: {np.sqrt(4*geo_dict.get('A_in', 0.0)/np.pi):.4f} [m]")
    # Total spacecraft length
    print(F"Total spacecraft length: {geo_dict.get('L_body', 0.0) + geo_dict.get('L_in', 0.0):.4f} [m]")

    print("\n--- REFUELING ---")
    ref_dict = asdict(final_sc.refueling)
    for k, v in ref_dict.items():
        if k == "m_flow":
            print(f"  {k:<20}: {v}")
        else:
            print(f"  {k:<20}: {v:.4f}")

    # Calculated mass flow rate from intake area, collection efficiency and atmospheric density
    print("\n--- CALCULATED MASS FLOW RATE ---")
    print(f"  m_flow_calc: {geo_dict.get('A_in', 0.0) * final_sc.orbit.velocity * final_sc.orbit.density * final_sc.refueling.coll_eff} [kg/s]")

    print("\n--- PROPULSION ---")
    thruster_dict = asdict(final_sc.thruster)
    for k, v in thruster_dict.items():
        print(f"  {k:<20}: {v:.4f}")

    print("\n--- SOLAR ---")
    solar_dict = asdict(final_sc.solar)
    for k, v in solar_dict.items():
        print(f"  {k:<20}: {v:.4f}")

    print("\n--- ORBIT ---")
    orbit_dict = asdict(final_sc.orbit)
    for k, v in orbit_dict.items():
        if type(v) is float:
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

if __name__ == "__main__":
    extract_values()
