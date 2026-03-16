from pathlib import Path
import sys
from copy import deepcopy
import io
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ariss.core.spacecraft import SpacecraftState
from ariss.core.simulation import run_sizing_loop, logger as simulation_logger

from MansurEnvelopeValidation import (
    _apply_publication_style,
    smooth_by_y,
    crossing_tp_for_level,
    stitch_branches,
)


BASE_CONFIG_PATH = ROOT / "src" / "ariss" / "core" / "base_config.toml"
CONFIG_PATH = HERE / "MansurValidation.toml"
OUTPUT = HERE / "mansur_thruster_map_validation.png"

EFF_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MDOT_LEVELS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
AIN_LEVELS = [1.2, 0.5, 0.3, 0.16]

EFF_COLOR = "#0025F5"
MDOT_COLOR = "#00B5F5"
AIN_COLOR = "#00F5A0"


def run_sweep():
    base = SpacecraftState.from_toml(BASE_CONFIG_PATH)
    base.update_from_toml(CONFIG_PATH)

    efficiencies = np.geomspace(0.05, 1.0, 60)
    isp_grid = np.linspace(1000.0, 6000.0, 60)

    tp_mn_per_kw = np.full((len(isp_grid), len(efficiencies)), np.nan, dtype=float)
    eff_grid = np.full_like(tp_mn_per_kw, np.nan)
    mdot_mg_per_s = np.full_like(tp_mn_per_kw, np.nan)
    ain_m2 = np.full_like(tp_mn_per_kw, np.nan)

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)

    try:
        for i, isp_s in enumerate(isp_grid):
            for j, efficiency in enumerate(efficiencies):
                spacecraft = deepcopy(base)
                spacecraft.thruster.specific_impulse = float(isp_s)
                spacecraft.thruster.eff = float(efficiency)

                with redirect_stdout(io.StringIO()):
                    final_sc, converged, _ = run_sizing_loop(spacecraft)

                if not converged:
                    continue

                tp_mn_per_kw[i, j] = 1.0e6 * float(final_sc.thruster.thrust) / float(final_sc.thruster.power)
                eff_grid[i, j] = float(final_sc.thruster.eff)
                mdot_mg_per_s[i, j] = 1.0e6 * float(final_sc.thruster.m_flow)
                ain_m2[i, j] = float(final_sc.geometry.A_in)
    finally:
        simulation_logger.setLevel(old_level)

    return isp_grid, tp_mn_per_kw, eff_grid, mdot_mg_per_s, ain_m2


def extract_lines(isp_grid, field, tp_grid, levels):
    lines = {}

    for level in levels:
        rows = []
        for i, isp_s in enumerate(isp_grid):
            hits = crossing_tp_for_level(field[i], tp_grid[i], level)
            rows.append((float(isp_s), hits))

        branches = stitch_branches(rows)
        if branches:
            lines[float(level)] = branches

    return lines


def _label_on_curve(axis, x, y, text, color, tp_target, x_offset=0.12, y_offset=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return

    idx = int(np.argmin(np.abs(x - float(tp_target))))
    axis.text(
        float(x[idx]) + x_offset,
        float(y[idx]) + y_offset,
        text,
        color=color,
        fontsize=10,
        va="center",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.08),
        zorder=5,
    )


def _plot_family(axis, lines, levels, color, linestyle, linewidth, label_specs, zorder):
    for level in levels:
        if float(level) not in lines:
            continue

        branches = sorted(lines[float(level)], key=lambda seg: len(seg[0]), reverse=True)
        for index, (x, y) in enumerate(branches):
            xs, ys = smooth_by_y(x, y)
            axis.plot(xs, ys, color=color, ls=linestyle, lw=linewidth, zorder=zorder)

            if index == 0:
                spec = label_specs.get(float(level), {})
                _label_on_curve(
                    axis,
                    xs,
                    ys,
                    f"{level:g}",
                    color,
                    spec.get("tp", 30.0),
                    x_offset=spec.get("x_offset", 0.15),
                    y_offset=spec.get("y_offset", 0.0),
                )


def plot():
    _apply_publication_style()

    isp_grid, tp_mn_per_kw, eff_grid, mdot_mg_per_s, ain_m2 = run_sweep()

    efficiency_lines = extract_lines(isp_grid, eff_grid, tp_mn_per_kw, EFF_LEVELS)
    mass_flow_lines = extract_lines(isp_grid, mdot_mg_per_s, tp_mn_per_kw, MDOT_LEVELS)
    intake_area_lines = extract_lines(isp_grid, ain_m2, tp_mn_per_kw, AIN_LEVELS)

    figure, axis = plt.subplots(figsize=(9.6, 5.4))

    _plot_family(
        axis,
        efficiency_lines,
        EFF_LEVELS,
        color=EFF_COLOR,
        linestyle="-",
        linewidth=1.0,
        label_specs={
            0.2: {"tp": 12.0},
            0.3: {"tp": 17.0},
            0.4: {"tp": 23.0},
            0.5: {"tp": 30.0},
            0.6: {"tp": 36.0},
            0.7: {"tp": 42.0},
            0.8: {"tp": 48.0},
        },
        zorder=3,
    )

    _plot_family(
        axis,
        mass_flow_lines,
        MDOT_LEVELS,
        color=MDOT_COLOR,
        linestyle="-",
        linewidth=0.95,
        label_specs={
            0.25: {"tp": 14.0},
            0.4: {"tp": 20.0},
            0.6: {"tp": 27.0},
            0.8: {"tp": 34.0},
            1.0: {"tp": 40.0},
            1.2: {"tp": 46.0},
            1.5: {"tp": 52.0},
        },
        zorder=2,
    )

    _plot_family(
        axis,
        intake_area_lines,
        AIN_LEVELS,
        color=AIN_COLOR,
        linestyle="-",
        linewidth=0.9,
        label_specs={
            1.2: {"tp": 46.0},
            0.5: {"tp": 30.0},
            0.3: {"tp": 22.0},
            0.16: {"tp": 16.0},
        },
        zorder=1,
    )

    axis.set_xlabel("T/P (mN/kW)")
    axis.set_ylabel("Isp (s)")
    axis.set_xlim(5, 60)
    axis.set_ylim(2500, 6000)

    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
        spine.set_alpha(1.0)

    axis.tick_params(axis="both", which="major", colors="black", width=0.9, length=5)
    axis.tick_params(axis="both", which="minor", colors="black", width=0.7, length=3)
    axis.xaxis.label.set_color("black")
    axis.yaxis.label.set_color("black")

    axis.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=axis.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            alpha=1.0,
            zorder=3,
            clip_on=False,
        )
    )

    axis.legend(
        handles=[
            Line2D([0], [0], color=EFF_COLOR, ls="-", lw=1.0, label=r"$\eta_T$"),
            Line2D([0], [0], color=MDOT_COLOR, ls="-", lw=0.95, label=r"$\dot{m}$ (mg/s) @ $P_t = 1\ \mathrm{kW}$"),
            Line2D([0], [0], color=AIN_COLOR, ls="-", lw=0.9, label=r"$A_i\,(m^2)$ @ $P_t = 1\ \mathrm{kW}$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handlelength=2.8,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=1200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    plot()
    print(f"Saved figure to: {OUTPUT}")
