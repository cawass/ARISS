"""Propulsion model plotting utilities."""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

if os.environ.get("PROP_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Propulsion import propulsion_model


def _sample_spacecraft() -> SpacecraftState:
    return SpacecraftState().update(
        orbit={
            "density": 2.09e-10,
            "velocity": 7791.44,
        },
        geometry={
            "A_ref": 1.21,
        },
        thruster={"specific_impulse": 2000.0},
    )


def plot_propulsion_diagnostics(
    sc: SpacecraftState | None = None,
    baseline_drag: float = 0.2,
    show: bool = True,
):
    """Plot propulsion area sizing versus drag and show baseline terms."""
    if sc is None:
        sc = _sample_spacecraft()

    drag_sweep = np.linspace(0.02, 0.40, 80)
    s_prop_sweep = np.zeros_like(drag_sweep)
    for idx, drag_force in enumerate(drag_sweep):
        s_prop, _, _ = propulsion_model(sc, float(drag_force))
        s_prop_sweep[idx] = s_prop

    baseline_s_prop, baseline_power, diagnostics = propulsion_model(sc, drag_force=baseline_drag, return_diagnostics=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(drag_sweep, s_prop_sweep, linewidth=2.0, label="S_prop")
    axes[0].scatter([baseline_drag], [baseline_s_prop], color="red", zorder=3, label="Baseline point")
    axes[0].set_title("Required S_prop vs Drag Force")
    axes[0].set_xlabel("D [N]")
    axes[0].set_ylabel("S_prop [m^2]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        ["D term", "rho*S_ref*V_inf"],
        [diagnostics.drag_force, diagnostics.rho * diagnostics.s_ref * diagnostics.v_inf],
        color=["tab:orange", "tab:blue"],
        alpha=0.85,
    )
    axes[1].set_title("Numerator Terms (Baseline)")
    axes[1].set_ylabel("Contribution [N]")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].text(
        0.02,
        0.98,
        f"Numerator: {diagnostics.numerator:.3e}\n"
        f"Denominator: {diagnostics.denominator:.3e}\n"
        f"S_prop: {diagnostics.required_prop_area:.3e}\n"
        f"P_prop: {baseline_power:.3e} W",
        transform=axes[1].transAxes,
        va="top",
        family="monospace",
    )

    if show:
        plt.show()
    return fig, axes, diagnostics


if __name__ == "__main__":
    plot_propulsion_diagnostics()
