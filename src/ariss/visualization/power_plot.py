"""Power model plotting utilities."""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

if os.environ.get("POWER_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Power import power_model


def _sample_spacecraft() -> SpacecraftState:
    return SpacecraftState().update(
        geometry={
            "A_in": 3.0,
            "AR_in": 1.0,
            "A_body": 1.21,
            "AR_body": 1.0,
            "L_in": 2.26,
            "L_body": 3.0,
        }
    )


def plot_power_diagnostics(
    sc: SpacecraftState | None = None,
    efficiency: float = 0.2,
    alignment_deg: float = 0.0,
    baseline_power: float = 2000.0,
    show: bool = True,
):
    """Plot solar-area requirements versus power demand."""
    if sc is None:
        sc = _sample_spacecraft()

    power_sweep = np.linspace(200.0, 10000.0, 80)
    required_area_sweep = np.zeros_like(power_sweep)
    deployable_area_sweep = np.zeros_like(power_sweep)

    for idx, power_required in enumerate(power_sweep):
        _, diagnostics = power_model(
            sc,
            float(power_required),
            efficiency=efficiency,
            alignment_deg=alignment_deg,
            return_diagnostics=True,
        )
        required_area_sweep[idx] = diagnostics.required_area
        deployable_area_sweep[idx] = diagnostics.deployable_area

    baseline_deployable, baseline = power_model(
        sc,
        baseline_power,
        efficiency=efficiency,
        alignment_deg=alignment_deg,
        return_diagnostics=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(power_sweep, required_area_sweep, linewidth=2.0, label="Required total area")
    axes[0].plot(power_sweep, deployable_area_sweep, linewidth=2.0, label="Required deployable area")
    axes[0].scatter([baseline_power], [baseline_deployable], color="red", zorder=3, label="Baseline point")
    axes[0].set_title("Solar Area vs Power Demand")
    axes[0].set_xlabel("Power required [W]")
    axes[0].set_ylabel("Area [m^2]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        ["required", "fixed-top", "deployable"],
        [baseline.required_area, baseline.fixed_top_area, baseline.deployable_area],
        color=["tab:blue", "tab:orange", "tab:green"],
        alpha=0.85,
    )
    axes[1].set_title("Baseline Area Breakdown")
    axes[1].set_ylabel("Area [m^2]")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].text(
        0.02,
        0.98,
        f"flux: {baseline.projected_flux:.2f} W/m^2\n"
        f"P: {baseline.power_required:.1f} W\n"
        f"deployable: {baseline.deployable_area:.3f} m^2",
        transform=axes[1].transAxes,
        va="top",
        family="monospace",
    )

    if show:
        plt.show()
    return fig, axes, baseline


if __name__ == "__main__":
    plot_power_diagnostics()
