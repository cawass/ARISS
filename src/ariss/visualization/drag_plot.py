"""Drag diagnostics plotting."""

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("DRAG_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Drag import DragDiagnostics, drag_model
from ariss.utils import constants as const


def _mark_body_inlet_regions(ax, l_body: float, l_in: float) -> None:
    total_length = l_body + l_in
    ax.axvspan(0.0, l_body, color="tab:blue", alpha=0.07, zorder=0)
    ax.axvspan(l_body, total_length, color="tab:orange", alpha=0.07, zorder=0)
    ax.axvline(l_body, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    y_min, y_max = ax.get_ylim()
    y_text = y_min + 0.92 * (y_max - y_min)
    ax.text(l_body / 2.0, y_text, "Body", ha="center", va="top", fontsize=9)
    ax.text(l_body + l_in / 2.0, y_text, "Inlet", ha="center", va="top", fontsize=9)
    ax.set_xlim(0.0, total_length)


def plot_drag_diagnostics(
    sc: SpacecraftState | None = None,
    n_points: int = 64,
    show: bool = True,
):
    """Plot drag model diagnostics along body and inlet."""
    if sc is None:
        sc = SpacecraftState().update(
            temperature_sc=300.0,
            orbit={
                "temperature": 1000.0,
                "molar_mass": const.MOLAR_MASS_AIR,
                "velocity": 7600.0,
                "alpha": 0.1,
            },
        )

    _, diagnostics = drag_model(sc, n_points=n_points, return_diagnostics=True)
    x = diagnostics.x_array
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].plot(x, diagnostics.fz_array, label="f_z")
    axes[0, 0].plot(x, diagnostics.fy_array, label="f_y")
    axes[0, 0].set_title("Boltzmann Fractions")
    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].grid(alpha=0.3)
    _mark_body_inlet_regions(axes[0, 0], sc.geometry.L_body, sc.geometry.L_in)
    axes[0, 0].legend()

    axes[0, 1].plot(x, diagnostics.d_body_array, label="Body")
    axes[0, 1].plot(x, diagnostics.d_in_array, label="Inlet")
    axes[0, 1].set_title("Local Drag Density")
    axes[0, 1].set_xlabel("X [m]")
    axes[0, 1].grid(alpha=0.3)
    _mark_body_inlet_regions(axes[0, 1], sc.geometry.L_body, sc.geometry.L_in)
    axes[0, 1].legend()

    axes[1, 0].plot(x, diagnostics.D_body_cumulative, label="Body")
    axes[1, 0].plot(x, diagnostics.D_in_cumulative, label="Inlet")
    axes[1, 0].set_title("Cumulative Drag")
    axes[1, 0].set_xlabel("X [m]")
    axes[1, 0].grid(alpha=0.3)
    _mark_body_inlet_regions(axes[1, 0], sc.geometry.L_body, sc.geometry.L_in)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    totals = diagnostics.totals
    axes[1, 1].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"D_solar: {totals['solar']:.3e} N",
                f"D_rad:   {totals['rad']:.3e} N",
                f"D_body:  {totals['body']:.3e} N",
                f"D_inlet: {totals['inlet']:.3e} N",
                f"D_total: {totals['total']:.3e} N",
            ]
        ),
        va="top",
        family="monospace",
    )

    if show:
        plt.show()
    return fig, axes, diagnostics


if __name__ == "__main__":
    plot_drag_diagnostics()
