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
#      Recreation of Fig. 5 (GOCE drag coefficient comparison) from
#      1-s2.0-S009457652100607X-main.pdf using digitized CSV data.
#
#  Project:        ARISS
#  Module:         GOCEEValidation.py
# ============================================================================== #

from __future__ import annotations

import csv
import sys
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

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Drag import drag_model
from ariss.modules.Propulsion import _side_areas
from ariss.utils import constants as const
from ariss.utils.atmosphere import orbit_updates_from_height
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "Gocee.csv"
OUTPUT_PATH = HERE / "gocee_fig5_validation.png"
GOCE_CONFIG_PATH = HERE / "GOCEDrag.toml"
PAGE_FIGSIZE = (13.2, 5.4)


# ------------------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------------------ #

def load_wide_xy_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig")))
    if len(rows) < 3:
        raise ValueError(f"Dataset {path} has insufficient rows.")

    header = rows[0]
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for column in range(0, len(header), 2):
        label = header[column].strip()
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
            x_arr = np.asarray(x_vals, dtype=float)
            y_arr = np.asarray(y_vals, dtype=float)
            order = np.argsort(x_arr)
            curves[label] = (x_arr[order], y_arr[order])

    return curves


# ------------------------------------------------------------------------------ #
# ARISS curve
# ------------------------------------------------------------------------------ #

def compute_ariss_body_cd_curve(
    x_values: np.ndarray,
    config_path: Path = GOCE_CONFIG_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    sc = load_spacecraft_from_base_config(config_path)

    # Use a GOCE-like reference atmosphere state (single point), then map S0 -> V.
    orbit_update = orbit_updates_from_height(
        sc.orbit.altitude,
        msis_date=sc.orbit.msis_date,
        msis_f107=sc.orbit.msis_f107,
        msis_ap=sc.orbit.msis_ap,
        latitude=sc.orbit.latitude,
        longitude=sc.orbit.longitude,
        use_average=sc.orbit.use_average,
    )
    sc.orbit.temperature = float(orbit_update["temperature"])
    sc.orbit.molar_mass = float(orbit_update["molar_mass"])

    # Fig. 5 compares body free-molecular drag coefficient referenced to GOCE
    # front area (A_ref), including frontal and skin-friction contributions.
    body_side_area, inlet_side_area = _side_areas(sc.geometry)
    a_ref = float(sc.geometry.A_ref) if float(sc.geometry.A_ref) > 0.0 else 1.0

    x = np.asarray(x_values, dtype=float)
    y = np.zeros_like(x)
    speed_scale = np.sqrt(sc.orbit.molar_mass / (2.0 * const.UNIVERSAL_GAS * sc.orbit.temperature))

    for i, s0 in enumerate(x):
        sc.orbit.velocity = float(s0 / speed_scale)
        drag_model(sc)
        y[i] = (
            float(sc.drag.cd_inlet_front) * float(sc.geometry.A_in_drag)
            + float(sc.drag.cd_inlet_side) * inlet_side_area
            + float(sc.drag.cd_body_side) * body_side_area
            + float(sc.drag.cd_solar) * float(sc.geometry.A_solar)
            + float(sc.drag.cd_rad) * float(sc.geometry.A_rad)
        ) / a_ref

    return x, y


# ------------------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------------------ #

def _model_and_ref_at_reference_points(
    model_x: np.ndarray,
    model_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model_x = np.asarray(model_x, dtype=float)
    model_y = np.asarray(model_y, dtype=float)
    ref_x = np.asarray(ref_x, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)

    valid_model = np.isfinite(model_x) & np.isfinite(model_y)
    if np.count_nonzero(valid_model) < 2:
        return None

    model_x = model_x[valid_model]
    model_y = model_y[valid_model]
    order = np.argsort(model_x)
    model_x = model_x[order]
    model_y = model_y[order]

    model_x_unique, unique_idx = np.unique(model_x, return_index=True)
    model_y_unique = model_y[unique_idx]
    if len(model_x_unique) < 2:
        return None

    line_ids = np.arange(1, len(ref_y) + 1, dtype=int)
    valid_ref = np.isfinite(ref_x) & np.isfinite(ref_y)
    if not np.any(valid_ref):
        return None

    ref_x = ref_x[valid_ref]
    ref_y = ref_y[valid_ref]
    line_ids = line_ids[valid_ref]

    x_low = float(np.min(model_x_unique))
    x_high = float(np.max(model_x_unique))
    in_range = (ref_x >= x_low) & (ref_x <= x_high)
    if not np.any(in_range):
        return None

    ref_x = ref_x[in_range]
    ref_y = ref_y[in_range]
    line_ids = line_ids[in_range]

    model_at_ref = np.interp(ref_x, model_x_unique, model_y_unique)
    return model_at_ref, ref_y, line_ids

def plot_gocee_fig5(dataset: dict[str, tuple[np.ndarray, np.ndarray]], show: bool = True) -> Path:
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

    # Build S0 range from available datasets and evaluate ARISS curve on that range.
    x_ref = []
    for key in ("Mansur", "Koppenwallner"):
        if key in dataset:
            x_ref.extend(dataset[key][0].tolist())
    if x_ref:
        x_min = float(np.nanmin(x_ref))
        x_max = float(np.nanmax(x_ref))
    else:
        x_min, x_max = 8.0, 13.0

    x_ariss = np.linspace(x_min, x_max, 140)
    x_ariss, y_ariss = compute_ariss_body_cd_curve(x_ariss)
    axis.plot(
        x_ariss,
        y_ariss,
        color=PALETTE["sernn_pink"],
        lw=2.2,
        zorder=2,
    )
    x_all.extend(x_ariss.tolist())
    y_all.extend(y_ariss.tolist())

    if "Mansur" in dataset:
        x_m, y_m = dataset["Mansur"]
        axis.plot(
            x_m,
            y_m,
            color=PALETTE["l1_teal"],
            lw=2.2,
            ls=(0, (4, 2)),
            zorder=2,
        )
        x_all.extend(x_m.tolist())
        y_all.extend(y_m.tolist())

    if "Koppenwallner" in dataset:
        x_k, y_k = dataset["Koppenwallner"]
        axis.plot(
            x_k,
            y_k,
            linestyle="None",
            marker="x",
            markersize=6.2,
            markeredgewidth=1.2,
            color=PALETTE["choice_mid"],
            zorder=3,
        )
        x_all.extend(x_k.tolist())
        y_all.extend(y_k.tolist())

    print("Datapoint relative-error and correlation metrics against digitized GOCE curves:")
    for label in ("Mansur", "Koppenwallner"):
        if label not in dataset:
            continue
        x_ref, y_ref = dataset[label]
        paired = _model_and_ref_at_reference_points(x_ariss, y_ariss, x_ref, y_ref)
        if paired is None:
            print(f"  {label:<14} n/a")
            continue

        model_at_ref, ref_y_used, line_ids = paired

        denom_mask = np.abs(ref_y_used) > 1.0e-12
        if np.any(denom_mask):
            rel_error = np.abs(model_at_ref[denom_mask] - ref_y_used[denom_mask]) / np.abs(ref_y_used[denom_mask])
            rel_line_ids = line_ids[denom_mask]
            i_max = int(np.argmax(rel_error))
            max_rel_error = float(rel_error[i_max])
            max_rel_line = int(rel_line_ids[i_max])
            mean_rel_error = float(np.mean(rel_error))
            rel_count = int(len(rel_error))
        else:
            max_rel_error = float("nan")
            max_rel_line = -1
            mean_rel_error = float("nan")
            rel_count = 0

        if len(model_at_ref) >= 2:
            pearson_r = float(np.corrcoef(model_at_ref, ref_y_used)[0, 1])
        else:
            pearson_r = float("nan")

        max_rel_text = f"{max_rel_error:.6f} ({100.0 * max_rel_error:.3f}%)"
        mean_rel_text = f"{mean_rel_error:.6f} ({100.0 * mean_rel_error:.3f}%)"
        max_rel_line_text = str(max_rel_line) if max_rel_line > 0 else "n/a"

        print(
            f"  {label:<14} "
            f"max_relative_error={max_rel_text} (line {max_rel_line_text}), "
            f"mean_relative_error={mean_rel_text}, "
            f"pearson_r={pearson_r:.6f}, "
            f"n_rel={rel_count}, n_corr={len(model_at_ref)}"
        )

    axis.set_xlabel(r"Speed ratio $S_0$")
    axis.set_ylabel(r"Drag coefficient $C_D$")
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
        x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 0.2
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)

    source_handles = [
        Line2D(
            [0], [0],
            color=PALETTE["sernn_pink"],
            lw=2.2,
            label="ARISS drag model",
        ),
        Line2D(
            [0], [0],
            color=PALETTE["l1_teal"],
            lw=2.2,
            ls=(0, (4, 2)),
            label=r"Mansur model $(A_{in}/A_{ref}=28.2)$",
        ),
        Line2D(
            [0], [0],
            color=PALETTE["choice_mid"],
            lw=0.0,
            marker="x",
            markersize=6.2,
            markeredgewidth=1.2,
            label=r"Koppenwallner GOCE $C_D$",
        ),
    ]

    legend = legend_axis.legend(
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
    style_legend(legend)
    legend.get_title().set_fontweight("bold")

    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95, wspace=0.07)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    return OUTPUT_PATH


def run_gocee_fig5_validation(show: bool = True) -> Path:
    dataset = load_wide_xy_dataset(DATASET_PATH)
    output = plot_gocee_fig5(dataset, show=show)
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    run_gocee_fig5_validation(show=True)
