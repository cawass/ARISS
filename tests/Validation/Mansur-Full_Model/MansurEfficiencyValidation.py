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
#      Efficiency sweep utility for Mansur-style validation curves.
#
#  Project:        ARISS
#  Module:         MansurEfficiencyVerification.py
# ============================================================================== #

import sys
import io
import csv
import logging
import warnings
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import PchipInterpolator


# ------------------------------------------------------------------------------ #
# Path setup so the ARISS source can be imported
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))


# ------------------------------------------------------------------------------ #
# ARISS simulation imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend


logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt", category=RuntimeWarning)


# ------------------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).with_name("MansurValidation.toml")
DATASET_PATH = Path(__file__).with_name("Isp Altitude Dataset.csv")
OUTPUT_PATH = Path(__file__).with_name("mansur_efficiency_validation.png")
VECTOR_OUTPUT_PATH = Path(__file__).with_name("mansur_efficiency_validation.svg")
PDF_OUTPUT_PATH = Path(__file__).with_name("mansur_efficiency_validation.pdf")

COLLECTION_EFFICIENCIES = (0.35, 0.40, 0.45)
PLOT_COLORS = [PALETTE["l1_teal"], PALETTE["sernn_pink"], PALETTE["choice_mid"], PALETTE["cat_green"], PALETTE["cat_yellow"]]

ISP = np.linspace(200, 10000, 40)  # [s]


# ------------------------------------------------------------------------------ #
# Sweep the efficiencies and store converged results
# ------------------------------------------------------------------------------ #

def sweep_mansur_efficiencies():

    base_spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)

    results = {}
    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)

    try:
        for efficiency in COLLECTION_EFFICIENCIES:

            converged_altitudes = []
            converged_isp = []

            for isp in ISP:
                spacecraft = deepcopy(base_spacecraft)

                spacecraft.refueling.coll_eff = efficiency
                spacecraft.thruster.specific_impulse = float(isp)

                with redirect_stdout(io.StringIO()):
                    final_sc, converged, _ = run_sizing_loop(spacecraft)

                if converged:
                    converged_altitudes.append(final_sc.orbit.altitude)
                    converged_isp.append(final_sc.thruster.specific_impulse)

            results[efficiency] = (np.array(converged_altitudes), np.array(converged_isp))
    finally:
        simulation_logger.setLevel(previous_level)

    return results


def load_mansur_paper_dataset():

    dataset = {}

    with DATASET_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        return dataset

    header = rows[0]

    for column_index in range(0, len(header), 2):
        label = header[column_index].strip()

        if not label:
            continue

        try:
            efficiency = float(label.split("=")[1].replace(",", "").strip())
        except (IndexError, ValueError):
            continue

        altitude_km = []
        isp_s = []

        for row in rows[2:]:
            x_value = row[column_index].strip() if column_index < len(row) else ""
            y_value = row[column_index + 1].strip() if column_index + 1 < len(row) else ""

            if not x_value or not y_value:
                continue

            altitude_km.append(float(x_value))
            isp_s.append(float(y_value))

        altitude_km = np.array(altitude_km)
        isp_s = np.array(isp_s)
        order = np.argsort(isp_s)[::-1]

        dataset[efficiency] = (altitude_km[order], isp_s[order])

    return dataset


# ------------------------------------------------------------------------------ #
# Plot the Mansur verification curves
# ------------------------------------------------------------------------------ #

def _plot_sweep_curve(axis, altitude_km: np.ndarray, isp_s: np.ndarray, color: str) -> None:

    # Inputs:
    #   axis: matplotlib axis used for plotting.
    #   altitude_km: converged altitudes in the same order as the Isp sweep.
    #   isp_s: converged specific impulse values in sweep order.
    #   color: curve color.
    #   efficiency: collection efficiency used for the sweep.
    #
    # Outputs:
    #   Plots the sweep curve without folding the descending and ascending
    #   altitude branches onto each other.

    turning_index = int(np.argmin(altitude_km))
    first_branch = slice(0, turning_index + 1)
    second_branch = slice(turning_index, altitude_km.size)

    _plot_soft_curve_with_markers(
        axis,
        altitude_km[first_branch],
        isp_s[first_branch],
        color=color,
        marker="o",
        line_style="-",
        linewidth=1.1,
    )

    if second_branch.stop - second_branch.start > 1:
        _plot_soft_curve_with_markers(
            axis,
            altitude_km[second_branch],
            isp_s[second_branch],
            color=color,
            marker="o",
            line_style="-",
            linewidth=1.1,
        )


def _rebuild_reference_curve(altitude_km: np.ndarray, isp_s: np.ndarray):

    # Inputs:
    #   altitude_km: reference altitudes from the CSV.
    #   isp_s: reference specific impulse values from the CSV.
    #
    # Outputs:
    #   A single continuous reference curve ordered by specific impulse. The
    #   Mansur CSV points are not stored in plotting order, so the reference
    #   trace is rebuilt from highest to lowest Isp.

    if altitude_km.size == 0 or isp_s.size == 0:
        return altitude_km, isp_s

    order = np.argsort(isp_s)[::-1]
    return altitude_km[order], isp_s[order]


def _plot_reference_curve(axis, altitude_km: np.ndarray, isp_s: np.ndarray, color: str) -> None:

    rebuilt_altitude, rebuilt_isp = _rebuild_reference_curve(altitude_km, isp_s)

    _plot_soft_curve_with_markers(
        axis,
        rebuilt_altitude,
        rebuilt_isp,
        color=color,
        marker="s",
        line_style="--",
        linewidth=1.1,
        alpha=1.0,
    )


def _apply_publication_style():

    # Inputs:
    #   None.
    #
    # Outputs:
    #   Applies a compact publication-style Matplotlib theme inspired by the
    #   BeautifulFigures principles: readable serif typography, minimal clutter,
    #   subtle grid lines, and consistent colours.

    apply_validation_style()


def _soft_curve_points(altitude_km: np.ndarray, isp_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Inputs:
    #   altitude_km: x-coordinates of the plotted curve.
    #   isp_s: y-coordinates of the plotted curve.
    #
    # Outputs:
    #   Shape-preserving interpolated coordinates for a smoother visual line.

    if altitude_km.size < 3 or isp_s.size < 3:
        return altitude_km, isp_s

    segment_lengths = np.hypot(np.diff(altitude_km), np.diff(isp_s))
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    if cumulative_length[-1] <= 0.0:
        return altitude_km, isp_s

    parameter = cumulative_length / cumulative_length[-1]
    unique_parameter, unique_indices = np.unique(parameter, return_index=True)

    if unique_parameter.size < 3:
        return altitude_km[unique_indices], isp_s[unique_indices]

    altitude_interpolator = PchipInterpolator(unique_parameter, altitude_km[unique_indices])
    isp_interpolator = PchipInterpolator(unique_parameter, isp_s[unique_indices])

    sample_count = max(140, 24 * (unique_parameter.size - 1))
    smooth_parameter = np.linspace(0.0, 1.0, sample_count)

    return altitude_interpolator(smooth_parameter), isp_interpolator(smooth_parameter)


def _plot_soft_curve_with_markers(
    axis,
    altitude_km: np.ndarray,
    isp_s: np.ndarray,
    color: str,
    marker: str,
    line_style: str,
    linewidth: float,
    label: str | None = None,
    alpha: float = 1.0,
) -> None:

    # Inputs:
    #   axis: matplotlib axis used for plotting.
    #   altitude_km: x-coordinates of the raw points.
    #   isp_s: y-coordinates of the raw points.
    #   color: base curve colour.
    #   marker: outer marker symbol.
    #   line_style: line style for the smoothed trace.
    #   linewidth: trace width.
    #   label: optional legend label.
    #   alpha: line opacity.
    #
    # Outputs:
    #   Draws a smoothed line plus nested markers at the original points.

    smooth_altitude, smooth_isp = _soft_curve_points(altitude_km, isp_s)

    axis.plot(
        smooth_altitude,
        smooth_isp,
        color=color,
        linestyle=line_style,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    axis.plot(
        altitude_km,
        isp_s,
        linestyle="None",
        marker=marker,
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.1,
        alpha=alpha,
    )


def plot_results(results, paper_results=None, save_path: Path = OUTPUT_PATH, show: bool = True) -> Path:

    if paper_results is None:
        paper_results = {}

    _apply_publication_style()

    figure, axis = plt.subplots(figsize=(9.6, 5.4))

    legend_handles = []
    legend_labels = []

    colors = PLOT_COLORS[: len(COLLECTION_EFFICIENCIES)]

    for color, efficiency in zip(colors, COLLECTION_EFFICIENCIES):

        altitude_km, isp_s = results.get(efficiency, (np.array([]), np.array([])))
        paper_altitude_km, paper_isp_s = paper_results.get(efficiency, (np.array([]), np.array([])))

        if altitude_km.size == 0:
            continue

        _plot_sweep_curve(axis, altitude_km, isp_s, color)

        if paper_altitude_km.size > 0:
            _plot_reference_curve(axis, paper_altitude_km, paper_isp_s, color)

        ariss_h_lim = float(np.min(altitude_km))
        mansur_h_lim = float(np.min(_rebuild_reference_curve(paper_altitude_km, paper_isp_s)[0])) if paper_altitude_km.size > 0 else None

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=1.1,
                marker="o",
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
            )
        )
        legend_labels.append(rf"ARISS $\eta_c = {efficiency:.2f}$, $h_{{lim}} = {ariss_h_lim:.1f}\ \mathrm{{km}}$")

        if mansur_h_lim is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="--",
                    linewidth=1.1,
                    marker="s",
                    markersize=5.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                )
            )
            legend_labels.append(rf"Mansur $\eta_c = {efficiency:.2f}$, $h_{{lim}} = {mansur_h_lim:.1f}\ \mathrm{{km}}$")

    axis.set_xlabel("Converged altitude (km)")
    axis.set_ylabel("Isp (s)")

    axis.set_xlim(150, 230)
    axis.set_ylim(2000, 6000)

    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    style_axis(axis)
    axis.tick_params(axis="both", which="major", width=0.9, length=5)
    axis.tick_params(axis="both", which="minor", width=0.7, length=3)
    if legend_handles:
        ariss_handles = legend_handles[0::2]
        ariss_labels = legend_labels[0::2]
        mansur_handles = legend_handles[1::2]
        mansur_labels = legend_labels[1::2]
        ordered_handles = []
        ordered_labels = []

        for ariss_handle, ariss_label, mansur_handle, mansur_label in zip(
            ariss_handles,
            ariss_labels,
            mansur_handles,
            mansur_labels,
        ):
            ordered_handles.extend([ariss_handle, mansur_handle])
            ordered_labels.extend([ariss_label, mansur_label])

        axis.legend(
            ordered_handles,
            ordered_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.04),
            frameon=False,
            ncol=3,
            columnspacing=1.2,
            handlelength=2.8,
            handletextpad=0.5,
            borderaxespad=0.0,
        )
        legend = axis.get_legend()
        style_legend(legend)

    figure.tight_layout()
    figure.savefig(save_path, dpi=1200, bbox_inches="tight")
    figure.savefig(VECTOR_OUTPUT_PATH, bbox_inches="tight")
    figure.savefig(PDF_OUTPUT_PATH, bbox_inches="tight")

    if show and "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(figure)

    return save_path


# ------------------------------------------------------------------------------ #
# Run script
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":

    results = sweep_mansur_efficiencies()
    paper_results = load_mansur_paper_dataset()
    output_path = plot_results(results, paper_results, save_path=OUTPUT_PATH)

    print(f"Saved figure to: {output_path}")
