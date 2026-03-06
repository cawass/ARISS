"""Mass and power component evolution plotting."""

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("BUDGETS_TOTAL_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState

MASS_COMPONENT_KEYS = [
    ("Mass_in", "Inlet"),
    ("Mass_body", "Body"),
    ("Mass_solar", "Solar"),
    ("Mass_rad", "Radiator"),
    ("Mass_prop", "Propulsion"),
    ("Mass_ADCS", "ADCS"),
    ("Mass_payload", "Payload"),
    ("Mass_refprop", "Refueling/Reserve"),
]

POWER_COMPONENT_KEYS = [
    ("Power_in", "Inlet"),
    ("Power_body", "Body"),
    ("Power_solar", "Solar"),
    ("Power_rad", "Radiator"),
    ("Power_prop", "Propulsion"),
    ("Power_ADCS", "ADCS"),
    ("Power_payload", "Payload"),
    ("Power_refprop", "Refueling/Reserve"),
]


def plot_budgets_total(
    sc: SpacecraftState | None = None,
    max_iterations: int = 20,
    mass_tolerance: float = 1e-8,
    show: bool = True,
):
    """Plot evolution of mass/power components and totals."""
    if sc is None:
        sc = SpacecraftState()
    _, _, history = run_sizing_loop(sc, max_iterations=max_iterations, mass_tolerance=mass_tolerance)

    it = list(range(len(history)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    for key, label in MASS_COMPONENT_KEYS:
        axes[0].plot(it, [getattr(state.mass, key) for state in history], marker="o", linewidth=1.2, label=label)
    axes[0].plot(it, [state.mass.Mass_total for state in history], color="black", linewidth=2.2, linestyle="--", label="Total mass")
    axes[0].set_title("Mass Component Evolution")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mass [kg]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    for key, label in POWER_COMPONENT_KEYS:
        axes[1].plot(it, [getattr(state.power, key) for state in history], marker="o", linewidth=1.2, label=label)
    axes[1].plot(it, [state.power.Power_total for state in history], color="black", linewidth=2.2, linestyle="--", label="Total power")
    axes[1].set_title("Power Component Evolution")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Power [W]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)

    if show:
        plt.show()
    return fig, axes, history


if __name__ == "__main__":
    plot_budgets_total()
