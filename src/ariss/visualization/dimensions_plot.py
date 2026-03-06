"""Geometry evolution plotting."""

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("DIMENSIONS_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState

AREA_KEYS = [("A_in", "A_in"), ("A_body", "A_body"), ("A_prop", "A_prop"), ("A_ref", "A_ref"), ("A_solar", "A_solar"), ("A_rad", "A_rad")]
LENGTH_KEYS = [("L_in", "L_in"), ("L_body", "L_body"), ("L_solar", "L_solar"), ("L_rad", "L_rad")]
ASPECT_RATIO_KEYS = [("AR_in", "AR_in"), ("AR_body", "AR_body"), ("AR_solar", "AR_solar"), ("AR_rad", "AR_rad")]


def plot_dimension_evolution(
    sc: SpacecraftState | None = None,
    max_iterations: int = 20,
    mass_tolerance: float = 1e-8,
    show: bool = True,
):
    """Plot area, length, and aspect-ratio evolution."""
    if sc is None:
        sc = SpacecraftState()
    _, _, history = run_sizing_loop(sc, max_iterations=max_iterations, mass_tolerance=mass_tolerance)

    it = list(range(len(history)))
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)

    for key, label in AREA_KEYS:
        axes[0].plot(it, [getattr(state.geometry, key) for state in history], marker="o", linewidth=1.4, label=label)
    axes[0].set_title("Area Evolution")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Area [m^2]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    for key, label in LENGTH_KEYS:
        axes[1].plot(it, [getattr(state.geometry, key) for state in history], marker="o", linewidth=1.4, label=label)
    axes[1].set_title("Length Evolution")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Length [m]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)

    for key, label in ASPECT_RATIO_KEYS:
        axes[2].plot(it, [getattr(state.geometry, key) for state in history], marker="o", linewidth=1.4, label=label)
    axes[2].set_title("Aspect Ratio Evolution")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Aspect ratio [-]")
    axes[2].grid(alpha=0.3)
    axes[2].legend(ncol=2, fontsize=8)

    if show:
        plt.show()
    return fig, axes, history


if __name__ == "__main__":
    plot_dimension_evolution()
