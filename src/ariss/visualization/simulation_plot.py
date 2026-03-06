"""Main simulation history plotting."""

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("SIM_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState


def plot_simulation_history(
    sc: SpacecraftState | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1e-8,
    show: bool = True,
):
    """Plot basic state history for the sizing loop."""
    if sc is None:
        sc = SpacecraftState()
    _, _, history = run_sizing_loop(sc, max_iterations=max_iterations, mass_tolerance=mass_tolerance)

    it = list(range(len(history)))
    total_mass = [state.total_mass for state in history]
    intake_area = [state.geometry.A_in for state in history]
    solar_area = [state.geometry.A_solar for state in history]
    prop_power = [state.thruster.power_required for state in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(it, total_mass, marker="o")
    axes[0, 0].set_title("Total Mass")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Mass [kg]")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(it, intake_area, marker="o", color="tab:orange")
    axes[0, 1].set_title("Intake Area")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("A_in [m^2]")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(it, solar_area, marker="o", color="tab:green")
    axes[1, 0].set_title("Solar Area")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("A_solar [m^2]")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(it, prop_power, marker="o", color="tab:red")
    axes[1, 1].set_title("Propulsion Power")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Power [W]")
    axes[1, 1].grid(alpha=0.3)

    if show:
        plt.show()
    return fig, axes, history


if __name__ == "__main__":
    plot_simulation_history()
