# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS - Atmospheric Refueling Iterative System Solver
# ============================================================================== #
#  Description:
#      Side-by-side recreation of Crandall & Wirz (2022) Fig. 26 and Fig. 27.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Fig26Fig27Validation.py
# ============================================================================== #

from __future__ import annotations

import csv
import io
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


# ------------------------------------------------------------------------------ #
# Path setup
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

for p in (SRC, VALIDATION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ------------------------------------------------------------------------------ #
# ARISS imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config, logger as simulation_logger, run_sizing_loop
from ariss.utils import constants as const
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent

CONFIG_PATH = HERE / "CrandallWirz2022_6U.toml"
DATASET_SOLAR_EFF_PATH = HERE / "Solar Cell Eff.csv"
DATASET_ACC_COEFF_PATH = HERE / "Acc Coeff.csv"
OUTPUT_PATH = HERE / "crandall_wirz_2022_fig26_fig27_validation.png"

TP_TARGET_MN_PER_KW = 10.0
TP_TARGET_N_PER_W = TP_TARGET_MN_PER_KW * 1.0e-6

MAX_ITERATIONS = 260
MASS_TOLERANCE = 1.0e-3

SOLAR_CASES = [
    {"label": "Solar Maximum", "f107": 200.0, "color": PALETTE["choice_mid"], "marker": "s"},
    {"label": "Mean Solar Activity", "f107": 114.0, "color": PALETTE["l1_teal"], "marker": "^"},
    {"label": "Solar Minimum", "f107": 62.0, "color": PALETTE["sernn_pink"], "marker": "D"},
]


# ------------------------------------------------------------------------------ #
# Dataset loading
# ------------------------------------------------------------------------------ #

def load_wide_xy_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig")))
    if len(rows) < 3:
        raise ValueError(f"Dataset {path} has insufficient rows.")

    header = rows[0]
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for column in range(0, len(header), 2):
        label = header[column].strip() if column < len(header) else ""
        if not label:
            continue

        x_vals: list[float] = []
        y_vals: list[float] = []
        for row in rows[2:]:
            if column + 1 >= len(row):
                continue
            x_text = row[column].strip()
            y_text = row[column + 1].strip()
            if not x_text or not y_text:
                continue
            try:
                x_value = float(x_text)
                y_value = float(y_text)
            except ValueError:
                continue
            if np.isfinite(x_value) and np.isfinite(y_value):
                x_vals.append(x_value)
                y_vals.append(y_value)

        if x_vals:
            x_array = np.asarray(x_vals, dtype=float)
            y_array = np.asarray(y_vals, dtype=float)
            order = np.argsort(x_array)
            curves[label] = (x_array[order], y_array[order])

    return curves


def _dataset_case_bounds(dataset: dict[str, tuple[np.ndarray, np.ndarray]], label: str) -> tuple[float, float]:
    x_vals, _ = dataset[label]
    finite = np.isfinite(x_vals)
    return float(np.min(x_vals[finite])), float(np.max(x_vals[finite]))


# ------------------------------------------------------------------------------ #
# Model runs
# ------------------------------------------------------------------------------ #

def _solve_point(spacecraft, *, f107: float, solar_eff: float | None = None, acc_coeff: float | None = None) -> tuple[float, float]:
    spacecraft.orbit.msis_f107 = float(f107)
    spacecraft.geometry.use_intake_area_ratio = False

    # Keep T/P fixed at the paper condition for Fig. 26/27.
    spacecraft.thruster.specific_impulse = 2.0 * spacecraft.thruster.eff / (const.EARTH_GRAVITY * TP_TARGET_N_PER_W)

    if solar_eff is not None:
        spacecraft = replace(spacecraft, solar=replace(spacecraft.solar, eta_solar=float(solar_eff)))

    if acc_coeff is not None:
        # The drag kernel epsilon parameter behaves opposite to the paper's
        # accommodation-coefficient direction for this comparison setup.
        epsilon_kernel = float(np.clip(1.0 - acc_coeff, 0.0, 1.0))
        spacecraft.geometry.epsilon_body = epsilon_kernel
        spacecraft.geometry.epsilon_solar = epsilon_kernel
        spacecraft.geometry.epsilon_rad = epsilon_kernel
        spacecraft.geometry.epsilon_in = epsilon_kernel

    with redirect_stdout(io.StringIO()):
        final_sc, _, _ = run_sizing_loop(
            spacecraft,
            max_iterations=MAX_ITERATIONS,
            mass_tolerance=MASS_TOLERANCE,
        )

    tp_value = 1.0e6 * float(final_sc.thruster.thrust) / float(final_sc.thruster.power)
    altitude_value = float(final_sc.orbit.altitude)
    return tp_value, altitude_value


def run_solar_efficiency_sweep(dataset: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x_min = min(_dataset_case_bounds(dataset, case["label"])[0] for case in SOLAR_CASES if case["label"] in dataset)
    x_max = max(_dataset_case_bounds(dataset, case["label"])[1] for case in SOLAR_CASES if case["label"] in dataset)
    x_grid = np.linspace(x_min, x_max, 24)

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in SOLAR_CASES:
        sc = load_spacecraft_from_base_config(CONFIG_PATH)
        y_values = []
        for eta_solar in x_grid:
            _, altitude = _solve_point(sc, f107=case["f107"], solar_eff=float(eta_solar))
            y_values.append(altitude)
        results[case["label"]] = (x_grid.copy(), np.asarray(y_values, dtype=float))
    return results


def run_accommodation_sweep() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x_grid = np.linspace(0.0, 1.0, 26)

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in SOLAR_CASES:
        sc = load_spacecraft_from_base_config(CONFIG_PATH)
        y_values = []
        for acc in x_grid:
            _, altitude = _solve_point(sc, f107=case["f107"], acc_coeff=float(acc))
            y_values.append(altitude)
        results[case["label"]] = (x_grid.copy(), np.asarray(y_values, dtype=float))
    return results


# ------------------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------------------ #

def mape_against_reference(model_xy: tuple[np.ndarray, np.ndarray], ref_xy: tuple[np.ndarray, np.ndarray]) -> float:
    model_x, model_y = model_xy
    ref_x, ref_y = ref_xy

    valid_ref = np.isfinite(ref_x) & np.isfinite(ref_y) & (ref_y > 0.0)
    if not np.any(valid_ref) or len(model_x) < 2:
        return float("nan")

    x_low = float(np.min(model_x))
    x_high = float(np.max(model_x))
    overlap = valid_ref & (ref_x >= x_low) & (ref_x <= x_high)
    if not np.any(overlap):
        return float("nan")

    interp_y = np.interp(ref_x[overlap], model_x, model_y)
    rel = np.abs(interp_y - ref_y[overlap]) / ref_y[overlap]
    return float(100.0 * np.mean(rel))


# ------------------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------------------ #

def plot_side_by_side(
    fig26_model: dict[str, tuple[np.ndarray, np.ndarray]],
    fig27_model: dict[str, tuple[np.ndarray, np.ndarray]],
    fig26_dataset: dict[str, tuple[np.ndarray, np.ndarray]],
    fig27_dataset: dict[str, tuple[np.ndarray, np.ndarray]],
    show: bool = True,
) -> Path:
    apply_validation_style()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    figure = plt.figure(figsize=(13.8, 6.0))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.55], wspace=0.08)
    ax_left = figure.add_subplot(grid[0, 0])
    ax_right = figure.add_subplot(grid[0, 1], sharey=ax_left)
    legend_axis = figure.add_subplot(grid[0, 2])
    legend_axis.axis("off")
    axes = [ax_left, ax_right]

    for axis in axes:
        style_axis(axis)
        axis.tick_params(axis="both", which="both", labelsize=12)
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(which="major", color="0.88", linewidth=0.7)
        axis.grid(which="minor", color="0.94", linewidth=0.5)

    axes[0].set_xlabel("Solar-cell efficiency (-)")
    axes[1].set_xlabel("Accommodation coefficient (-)")
    figure.supylabel("Power-limited minimum operating altitude (km)", x=0.04)

    y_all: list[float] = []

    for case in SOLAR_CASES:
        label = case["label"]
        color = case["color"]
        marker = case["marker"]

        if label in fig26_model:
            x_mod, y_mod = fig26_model[label]
            axes[0].plot(x_mod, y_mod, color=color, lw=2.2, zorder=2)
            y_all.extend(y_mod.tolist())
        if label in fig26_dataset:
            x_ref, y_ref = fig26_dataset[label]
            axes[0].plot(
                x_ref,
                y_ref,
                color=color,
                lw=1.5,
                ls=(0, (4, 2)),
                marker=marker,
                markersize=6.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                zorder=3,
            )
            y_all.extend(y_ref.tolist())

        if label in fig27_model:
            x_mod, y_mod = fig27_model[label]
            axes[1].plot(x_mod, y_mod, color=color, lw=2.2, zorder=2)
            y_all.extend(y_mod.tolist())
        if label in fig27_dataset:
            x_ref, y_ref = fig27_dataset[label]
            axes[1].plot(
                x_ref,
                y_ref,
                color=color,
                lw=1.5,
                ls=(0, (4, 2)),
                marker=marker,
                markersize=6.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                zorder=3,
            )
            y_all.extend(y_ref.tolist())

    axes[0].set_xlim(0.24, 0.505)
    axes[1].set_xlim(0.0, 1.0)
    if y_all:
        y_arr = np.asarray(y_all, dtype=float)
        finite = np.isfinite(y_arr)
        if np.any(finite):
            y_min = float(np.min(y_arr[finite]))
            y_max = float(np.max(y_arr[finite]))
            axes[0].set_ylim(y_min - 1.5, y_max + 1.5)

    source_handles = [
        Line2D([0], [0], color=PALETTE["secondary_text"], lw=2.2, label="ARISS full loop"),
        Line2D(
            [0],
            [0],
            color=PALETTE["secondary_text"],
            lw=1.5,
            ls=(0, (4, 2)),
            marker="o",
            markersize=6.2,
            markerfacecolor="white",
            markeredgecolor=PALETTE["secondary_text"],
            markeredgewidth=1.1,
            label="Crandall-Wirz Data",
        ),
    ]
    case_handles = [
        Line2D([0], [0], color=case["color"], lw=2.2, label=case["label"])
        for case in SOLAR_CASES
    ]

    source_legend = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(source_legend)
    source_legend.get_title().set_fontweight("bold")
    legend_axis.add_artist(source_legend)

    case_legend = legend_axis.legend(
        handles=case_handles,
        title="Solar Activity",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.58),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(case_legend)
    case_legend.get_title().set_fontweight("bold")

    figure.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.14, wspace=0.08)
    figure.savefig(OUTPUT_PATH, dpi=1200, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    return OUTPUT_PATH


def run_crandall_wirz_fig26_fig27_validation(show: bool = True) -> Path:
    fig26_dataset = load_wide_xy_dataset(DATASET_SOLAR_EFF_PATH)
    fig27_dataset = load_wide_xy_dataset(DATASET_ACC_COEFF_PATH)

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)
    try:
        fig26_model = run_solar_efficiency_sweep(fig26_dataset)
        fig27_model = run_accommodation_sweep()
    finally:
        simulation_logger.setLevel(old_level)

    print("MAPE against digitized datasets:")
    for case in SOLAR_CASES:
        label = case["label"]
        if label in fig26_dataset and label in fig26_model:
            mape_26 = mape_against_reference(fig26_model[label], fig26_dataset[label])
            print(f"  Fig26 {label:<20} {mape_26:6.2f}%")
        if label in fig27_dataset and label in fig27_model:
            mape_27 = mape_against_reference(fig27_model[label], fig27_dataset[label])
            print(f"  Fig27 {label:<20} {mape_27:6.2f}%")

    output = plot_side_by_side(fig26_model, fig27_model, fig26_dataset, fig27_dataset, show=show)
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    run_crandall_wirz_fig26_fig27_validation(show=True)
