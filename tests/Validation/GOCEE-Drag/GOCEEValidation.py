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

    figure = plt.figure(figsize=(10.8, 5.7))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.50], wspace=0.06)
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

    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.95, wspace=0.06)
    figure.savefig(OUTPUT_PATH, dpi=1200, bbox_inches="tight")

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
