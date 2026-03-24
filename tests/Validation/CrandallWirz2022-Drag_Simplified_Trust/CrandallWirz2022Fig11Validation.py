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
#      Fig. 11 recreation for Crandall and Wirz (2022) using full sizing loop
#      runs while sweeping thruster efficiency.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Fig11Validation.py
# ============================================================================== #

from __future__ import annotations

import csv
import io
import sys
from contextlib import redirect_stdout
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
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent

CONFIG_PATH = HERE / "CrandallWirz2022_6U.toml"
DATASET_PATH = HERE / "Fig 11.csv"
OUTPUT_PATH = HERE / "crandall_wirz_2022_fig11_validation.png"

EFFICIENCY_GRID = np.linspace(0.38, 1.00, 18)

MAX_ITERATIONS = 260
MASS_TOLERANCE = 1.0e-3

SOLAR_CASES = [
    {"label": "Solar Maximum", "f107": 200.0, "color": PALETTE["choice_mid"], "marker": "s"},
    {"label": "Mean Solar Activity", "f107": 114.0, "color": PALETTE["l1_teal"], "marker": "^"},
    {"label": "Solar Minimum", "f107": 62.0, "color": PALETTE["sernn_pink"], "marker": "D"},
]


# ------------------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------------------ #

def load_fig11_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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


# ------------------------------------------------------------------------------ #
# Full-loop sweep
# ------------------------------------------------------------------------------ #

def run_case_for_solar_flux(f107_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp_vals: list[float] = []
    alt_vals: list[float] = []
    isp_vals: list[float] = []
    converged_flags: list[bool] = []

    for efficiency in EFFICIENCY_GRID:
        spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)
        spacecraft.orbit.msis_f107 = float(f107_value)
        spacecraft.thruster.eff = float(efficiency)

        with redirect_stdout(io.StringIO()):
            final_sc, converged, _ = run_sizing_loop(
                spacecraft,
                max_iterations=MAX_ITERATIONS,
                mass_tolerance=MASS_TOLERANCE,
            )

        if final_sc.thruster.power == 0.0:
            continue

        tp_mn_per_kw = 1.0e6 * float(final_sc.thruster.thrust) / float(final_sc.thruster.power)

        if np.isfinite(tp_mn_per_kw) and np.isfinite(final_sc.orbit.altitude):
            tp_vals.append(tp_mn_per_kw)
            alt_vals.append(float(final_sc.orbit.altitude))
            isp_vals.append(float(final_sc.thruster.specific_impulse))
            converged_flags.append(bool(converged))

    if not tp_vals:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=bool)

    tp_array = np.asarray(tp_vals, dtype=float)
    alt_array = np.asarray(alt_vals, dtype=float)
    isp_array = np.asarray(isp_vals, dtype=float)
    converged_array = np.asarray(converged_flags, dtype=bool)
    order = np.argsort(tp_array)
    return tp_array[order], alt_array[order], isp_array[order], converged_array[order]


def run_fig11_sweep() -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)
    try:
        for spec in SOLAR_CASES:
            tp, altitude, isp, converged = run_case_for_solar_flux(spec["f107"])
            results[spec["label"]] = {
                "tp": tp,
                "altitude": altitude,
                "isp": isp,
                "converged": converged,
            }
    finally:
        simulation_logger.setLevel(old_level)

    return results


# ------------------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------------------ #

def mape_altitude_percent(model_tp: np.ndarray, model_altitude: np.ndarray, ref_tp: np.ndarray, ref_altitude: np.ndarray) -> float:
    if len(model_tp) < 2 or len(ref_tp) == 0:
        return float("nan")

    within = (ref_tp >= np.min(model_tp)) & (ref_tp <= np.max(model_tp))
    valid = within & np.isfinite(ref_tp) & np.isfinite(ref_altitude) & (ref_altitude > 0.0)
    if not np.any(valid):
        return float("nan")

    model_interp = np.interp(ref_tp[valid], model_tp, model_altitude)
    rel = np.abs(model_interp - ref_altitude[valid]) / ref_altitude[valid]
    return float(100.0 * np.mean(rel))


# ------------------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------------------ #

def plot_fig11(results: dict[str, dict[str, np.ndarray]], dataset: dict[str, tuple[np.ndarray, np.ndarray]], show: bool = True) -> Path:
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

    figure = plt.figure(figsize=(12.6, 5.9))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.55], wspace=0.05)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    x_all: list[float] = []
    y_all: list[float] = []

    for spec in SOLAR_CASES:
        label = spec["label"]
        color = spec["color"]
        marker = spec["marker"]

        model = results.get(label, {})
        tp = np.asarray(model.get("tp", np.array([])), dtype=float)
        altitude = np.asarray(model.get("altitude", np.array([])), dtype=float)

        valid_model = np.isfinite(tp) & np.isfinite(altitude)
        axis.plot(tp[valid_model], altitude[valid_model], color=color, lw=2.2, zorder=2)

        if label in dataset:
            x_ref, y_ref = dataset[label]
            valid_ref = np.isfinite(x_ref) & np.isfinite(y_ref)
            axis.plot(
                x_ref[valid_ref],
                y_ref[valid_ref],
                color=color,
                lw=1.4,
                ls=(0, (4, 2)),
                marker=marker,
                markersize=6.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                zorder=3,
            )
            x_all.extend(x_ref[valid_ref].tolist())
            y_all.extend(y_ref[valid_ref].tolist())

        x_all.extend(tp[valid_model].tolist())
        y_all.extend(altitude[valid_model].tolist())

    axis.set_xlabel("T/P (mN/kW)")
    axis.set_ylabel("Power-limited minimum operating altitude (km)")
    style_axis(axis)
    axis.tick_params(axis="both", which="both", labelsize=12)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.grid(which="major", color="0.88", linewidth=0.7)
    axis.grid(which="minor", color="0.94", linewidth=0.5)

    if x_all and y_all:
        x_min = float(np.nanmin(x_all))
        x_max = float(np.nanmax(x_all))
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        axis.set_xlim(0.95 * x_min, 1.03 * x_max)
        axis.set_ylim(y_min - 1.0, y_max + 1.0)

    source_handles = [
        Line2D([0], [0], color=PALETTE["secondary_text"], lw=2.2, label="ARISS full loop"),
        Line2D(
            [0],
            [0],
            color=PALETTE["secondary_text"],
            lw=1.4,
            ls=(0, (4, 2)),
            marker="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=PALETTE["secondary_text"],
            markeredgewidth=1.1,
            label="Crandall-Wirz Data",
        ),
    ]
    case_handles = [
        Line2D([0], [0], color=spec["color"], lw=2.2, label=spec["label"])
        for spec in SOLAR_CASES
    ]

    legend_source = legend_axis.legend(
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
    style_legend(legend_source)
    legend_source.get_title().set_fontweight("bold")
    legend_axis.add_artist(legend_source)

    legend_cases = legend_axis.legend(
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
    style_legend(legend_cases)
    legend_cases.get_title().set_fontweight("bold")

    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.95, wspace=0.05)
    figure.savefig(OUTPUT_PATH, dpi=1200, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    return OUTPUT_PATH


def run_crandall_wirz_fig11_validation(show: bool = True) -> Path:
    dataset = load_fig11_dataset(DATASET_PATH)
    results = run_fig11_sweep()

    print("Altitude MAPE against digitized Fig. 11 curves:")
    for spec in SOLAR_CASES:
        label = spec["label"]
        if label not in dataset:
            continue
        tp_model = np.asarray(results[label]["tp"], dtype=float)
        alt_model = np.asarray(results[label]["altitude"], dtype=float)
        isp_model = np.asarray(results[label]["isp"], dtype=float)
        tp_ref, alt_ref = dataset[label]
        mape = mape_altitude_percent(tp_model, alt_model, tp_ref, alt_ref)
        if np.isfinite(mape):
            print(f"  {label:<20} {mape:6.2f}%")
        else:
            print(f"  {label:<20} n/a (no T/P overlap with dataset)")
        if len(isp_model) > 0:
            print(f"    Isp range [s]: {float(np.min(isp_model)):.2f} to {float(np.max(isp_model)):.2f}")

    output = plot_fig11(results, dataset, show=show)
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    run_crandall_wirz_fig11_validation(show=True)
