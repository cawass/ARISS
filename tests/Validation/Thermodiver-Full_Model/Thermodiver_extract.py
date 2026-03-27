import sys
from pathlib import Path
import pprint
from dataclasses import asdict

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
        
    print("\n--- REFUELING & VOLUME ---")
    ref_dict = asdict(final_sc.refueling)
    print(f"  Tank Volume (V_prop) [m^3] : {ref_dict.get('V_prop', 0.0):.6f}")
    print(f"  Tank Pressure (p_tank) [Pa]: {ref_dict.get('p_tank', 0.0):.2f}")
    print(f"  Refueling m_flow [kg/s]    : {ref_dict.get('m_flow', 0.0):.6e}")
    print(f"  Compression Power [W]      : {final_sc.power.Power_refprop:.4f}")
    print(f"  Propellant Mass [kg]       : {final_sc.mass.Mass_prop:.4f}")
    
    print("\n--- GEOMETRY/AREAS [m^2] ---")
    geo_dict = asdict(final_sc.geometry)
    print(f"  Intake Area (A_in)  : {geo_dict.get('A_in', 0.0):.4f}")
    print(f"  Body Area (A_body)  : {geo_dict.get('A_body', 0.0):.4f}")
    print(f"  Solar Area (A_solar)  : {geo_dict.get('A_solar', 0.0):.4f}")
    print(f"  Radiator Area (A_rad)  : {geo_dict.get('A_rad', 0.0):.4f}")

    # print("\n" + "="*50)
    # print(" FULL SPACECRAFT STATE COMPILED DICTIONARY ")
    # print("="*50)
    # pprint.pprint(asdict(final_sc), sort_dicts=False, width=100)

if __name__ == "__main__":
    extract_values()
