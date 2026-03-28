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
PAGE_FIGSIZE = (13.2, 5.4)

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

def _paired_model_reference_samples(
    model_tp: np.ndarray,
    model_altitude: np.ndarray,
    ref_tp: np.ndarray,
    ref_altitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model_tp = np.asarray(model_tp, dtype=float)
    model_altitude = np.asarray(model_altitude, dtype=float)
    ref_tp = np.asarray(ref_tp, dtype=float)
    ref_altitude = np.asarray(ref_altitude, dtype=float)

    valid_model = np.isfinite(model_tp) & np.isfinite(model_altitude)
    if np.count_nonzero(valid_model) < 2:
        return None

    model_tp = model_tp[valid_model]
    model_altitude = model_altitude[valid_model]
    order = np.argsort(model_tp)
    model_tp = model_tp[order]
    model_altitude = model_altitude[order]

    model_tp_unique, unique_idx = np.unique(model_tp, return_index=True)
    model_altitude_unique = model_altitude[unique_idx]
    if len(model_tp_unique) < 2:
        return None

    line_ids = np.arange(1, len(ref_altitude) + 1, dtype=int)
    valid_ref = np.isfinite(ref_tp) & np.isfinite(ref_altitude)
    if not np.any(valid_ref):
        return None

    ref_tp = ref_tp[valid_ref]
    ref_altitude = ref_altitude[valid_ref]
    line_ids = line_ids[valid_ref]

    tp_low = float(np.min(model_tp_unique))
    tp_high = float(np.max(model_tp_unique))
    in_range = (ref_tp >= tp_low) & (ref_tp <= tp_high)
    if not np.any(in_range):
        return None

    ref_tp = ref_tp[in_range]
    ref_altitude = ref_altitude[in_range]
    line_ids = line_ids[in_range]

    model_altitude_at_ref = np.interp(ref_tp, model_tp_unique, model_altitude_unique)
    return model_altitude_at_ref, ref_altitude, line_ids


def relative_altitude_and_corr_stats(
    model_tp: np.ndarray,
    model_altitude: np.ndarray,
    ref_tp: np.ndarray,
    ref_altitude: np.ndarray,
) -> tuple[float, float, int, int, float, int] | None:
    paired = _paired_model_reference_samples(model_tp, model_altitude, ref_tp, ref_altitude)
    if paired is None:
        return None

    model_altitude_at_ref, ref_altitude_used, line_ids = paired

    nonzero_ref = np.abs(ref_altitude_used) > 1.0e-12
    if np.any(nonzero_ref):
        relative_error = (
            np.abs(model_altitude_at_ref[nonzero_ref] - ref_altitude_used[nonzero_ref])
            / np.abs(ref_altitude_used[nonzero_ref])
        )
        rel_line_ids = line_ids[nonzero_ref]
        i_max = int(np.argmax(relative_error))
        max_relative_error = float(relative_error[i_max])
        max_rel_line = int(rel_line_ids[i_max])
        mean_relative_error = float(np.mean(relative_error))
        n_rel = int(len(relative_error))
    else:
        max_relative_error = float("nan")
        max_rel_line = -1
        mean_relative_error = float("nan")
        n_rel = 0

    if len(model_altitude_at_ref) >= 2:
        pearson_r = float(np.corrcoef(model_altitude_at_ref, ref_altitude_used)[0, 1])
    else:
        pearson_r = float("nan")

    return (
        max_relative_error,
        mean_relative_error,
        max_rel_line,
        n_rel,
        pearson_r,
        int(len(model_altitude_at_ref)),
    )


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

    figure = plt.figure(figsize=PAGE_FIGSIZE)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.07)
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

    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95, wspace=0.07)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    return OUTPUT_PATH


def run_crandall_wirz_fig11_validation(show: bool = True) -> Path:
    dataset = load_fig11_dataset(DATASET_PATH)
    results = run_fig11_sweep()

    print("Altitude relative-error and correlation against digitized Fig. 11 curves:")
    pearson_values: list[float] = []
    for spec in SOLAR_CASES:
        label = spec["label"]
        if label not in dataset:
            continue
        tp_model = np.asarray(results[label]["tp"], dtype=float)
        alt_model = np.asarray(results[label]["altitude"], dtype=float)
        isp_model = np.asarray(results[label]["isp"], dtype=float)
        tp_ref, alt_ref = dataset[label]
        stats = relative_altitude_and_corr_stats(tp_model, alt_model, tp_ref, alt_ref)
        if stats is not None:
            max_relative_error, mean_relative_error, line_max_rel, n_rel, pearson_r, n_corr = stats
            line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
            print(
                f"  {label:<20} "
                f"max_relative_error={max_relative_error:8.6f} ({100.0 * max_relative_error:6.3f}%) (line {line_text}), "
                f"mean_relative_error={mean_relative_error:8.6f} ({100.0 * mean_relative_error:6.3f}%), "
                f"pearson_r={pearson_r:8.6f}, n_rel={n_rel}, n_corr={n_corr}"
            )
            if np.isfinite(pearson_r):
                pearson_values.append(pearson_r)
        else:
            print(f"  {label:<20} n/a (no T/P overlap with dataset)")
        if len(isp_model) > 0:
            print(f"    Isp range [s]: {float(np.min(isp_model)):.2f} to {float(np.max(isp_model)):.2f}")

    if pearson_values:
        print(f"  Minimum Pearson correlation coefficient: {min(pearson_values):.6f}")
    else:
        print("  Minimum Pearson correlation coefficient: n/a")

    output = plot_fig11(results, dataset, show=show)
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    run_crandall_wirz_fig11_validation(show=True)
