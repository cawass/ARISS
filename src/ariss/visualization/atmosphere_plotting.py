"""Atmosphere profile plotting utilities."""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

if os.environ.get("ATMOS_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.utils.atmosphere import atmos, calculate_orbital_velocity


def plot_atmosphere_profiles(
    height_min_km: float = 80.0,
    height_max_km: float = 1000.0,
    samples: int = 600,
    show: bool = True,
):
    """Plot atmospheric profiles from pymsis-backed data."""
    heights = np.linspace(height_min_km, height_max_km, samples)
    density, temperature, r_specific, o2, n2, o = atmos(heights)
    velocity = calculate_orbital_velocity(heights)
    dynamic_pressure = 0.5 * density * velocity**2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    flat_axes = axes.flatten()

    flat_axes[0].plot(heights, density, color="tab:blue")
    flat_axes[0].set_title("Density vs Height")
    flat_axes[0].set_yscale("log")
    flat_axes[0].set_ylabel("Density [kg/m^3]")
    flat_axes[0].grid(alpha=0.3)

    flat_axes[1].plot(heights, velocity, color="tab:orange")
    flat_axes[1].set_title("Velocity vs Height")
    flat_axes[1].set_ylabel("Velocity [m/s]")
    flat_axes[1].grid(alpha=0.3)

    flat_axes[2].plot(heights, dynamic_pressure, color="tab:green")
    flat_axes[2].set_title("Dynamic Pressure vs Height")
    flat_axes[2].set_yscale("log")
    flat_axes[2].set_ylabel("q [Pa]")
    flat_axes[2].grid(alpha=0.3)

    flat_axes[3].plot(heights, temperature, color="tab:red")
    flat_axes[3].set_title("Temperature vs Height")
    flat_axes[3].set_ylabel("Temperature [K]")
    flat_axes[3].set_xlabel("Height [km]")
    flat_axes[3].grid(alpha=0.3)

    flat_axes[4].plot(heights, r_specific, color="tab:purple")
    flat_axes[4].set_title("Specific Gas Constant vs Height")
    flat_axes[4].set_ylabel("R [J/kg K]")
    flat_axes[4].set_xlabel("Height [km]")
    flat_axes[4].grid(alpha=0.3)

    flat_axes[5].plot(heights, o, label="O", color="tab:blue")
    flat_axes[5].plot(heights, n2, label="N2", color="tab:green")
    flat_axes[5].plot(heights, o2, label="O2", color="tab:orange")
    flat_axes[5].set_title("Species Density vs Height")
    flat_axes[5].set_yscale("log")
    flat_axes[5].set_ylabel("Density [kg/m^3]")
    flat_axes[5].set_xlabel("Height [km]")
    flat_axes[5].grid(alpha=0.3)
    flat_axes[5].legend(fontsize=8)

    if show:
        plt.show()
    return fig, axes


if __name__ == "__main__":
    plot_atmosphere_profiles()
