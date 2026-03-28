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
#      Drag-only Fig. 6 recreation for Crandall and Wirz (2022) using 3U/6U cases
#      and digitized CSV datasets, styled with the shared validation plot format.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Validation.py
# ============================================================================== #

from __future__ import annotations

import csv
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


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

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Drag import drag_model
from ariss.modules.Propulsion import _panel_front_area, _side_areas
from ariss.utils import constants as const
from ariss.utils.atmosphere import atmospheric_properties_from_height
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent

CASE_SPECS = [
    {
        "title": "1 x 3U CubeSat",
        "config": HERE / "CrandallWirz2022_3U.toml",
        "dataset": HERE / "3U Drag.csv",
    },
    {
        "title": "1 x 6U CubeSat",
        "config": HERE / "CrandallWirz2022_6U.toml",
        "dataset": HERE / "6U Drag.csv",
    },
]

OUTPUT_PATH = HERE / "crandall_wirz_2022_fig6_validation.png"
PAGE_FIGSIZE = (13.2, 5.4)

ALTITUDE_SAMPLES = 260
NEWTON_TO_MILLINEWTON = 1.0e3

COMPONENT_SPECS = [
    ("Total Drag", "drag_total", PALETTE["primary_text"], "o"),
    ("Frontal Area Drag", "drag_inlet_front", PALETTE["choice_mid"], "s"),
    ("SA Skin Friction Drag", "drag_solar", PALETTE["l1_teal"], "^"),
    ("Body Skin Friction Drag", "drag_body_side", PALETTE["sernn_pink"], "D"),
    ("SA Frontal Area Drag", "drag_solar_front", PALETTE["goal_dark"], "v"),
]


# ------------------------------------------------------------------------------ #
# Dataset loading
# ------------------------------------------------------------------------------ #

def _load_digitized_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig")))
    if len(rows) < 3:
        raise ValueError(f"Dataset {path} has insufficient rows.")

    header = rows[0]
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for column in range(0, len(header), 2):
        label = header[column].strip() if column < len(header) else ""
        if not label:
            continue
        if label == "SA Skin Friction":
            label = "SA Skin Friction Drag"

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
            order = np.argsort(y_array)
            result[label] = (x_array[order], y_array[order])

    return result


def _normalize_altitude_orientation(
    dataset: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], bool]:
    if "Total Drag" not in dataset:
        return dataset, False

    x_total, y_total = dataset["Total Drag"]
    if len(x_total) < 3:
        return dataset, False

    correlation = float(np.corrcoef(x_total, y_total)[0, 1])
    if not np.isfinite(correlation) or correlation <= 0.0:
        return dataset, False

    y_all = np.concatenate([values[1] for values in dataset.values()])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))

    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (x_vals, y_vals) in dataset.items():
        y_flip = y_min + y_max - y_vals
        order = np.argsort(y_flip)
        normalized[label] = (x_vals[order], y_flip[order])
    return normalized, True


def _dataset_altitude_bounds(dataset: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    all_y = []
    for label, _, _, _ in COMPONENT_SPECS:
        if label in dataset:
            all_y.extend(dataset[label][1].tolist())
    if not all_y:
        raise ValueError("No altitude data found in dataset for expected components.")
    return float(np.min(all_y)), float(np.max(all_y))


# ------------------------------------------------------------------------------ #
# Drag-only sweep
# ------------------------------------------------------------------------------ #

def _set_drag_force_outputs(sc) -> None:
    body_side_area, inlet_side_area = _side_areas(sc.geometry)
    solar_front_area = _panel_front_area(sc.geometry.A_solar, sc.geometry.AR_solar, sc.geometry.t_solar)
    rad_front_area = _panel_front_area(sc.geometry.A_rad, sc.geometry.AR_rad, sc.geometry.t_rad)
    q_inf = 0.5 * sc.orbit.density * sc.orbit.velocity**2

    sc.drag.drag_solar = q_inf * sc.drag.cd_solar * sc.geometry.A_solar
    sc.drag.drag_solar_front = q_inf * sc.drag.cd_solar_front * solar_front_area
    sc.drag.drag_rad = q_inf * sc.drag.cd_rad * sc.geometry.A_rad
    sc.drag.drag_rad_front = q_inf * sc.drag.cd_rad_front * rad_front_area
    sc.drag.drag_body_side = q_inf * sc.drag.cd_body_side * body_side_area
    sc.drag.drag_inlet_side = q_inf * sc.drag.cd_inlet_side * inlet_side_area
    sc.drag.drag_inlet_front = q_inf * sc.drag.cd_inlet_front * sc.geometry.A_in_drag

    sc.drag.drag_total = (
        sc.drag.drag_solar
        + sc.drag.drag_solar_front
        + sc.drag.drag_rad
        + sc.drag.drag_rad_front
        + sc.drag.drag_body_side
        + sc.drag.drag_inlet_side
        + sc.drag.drag_inlet_front
    )


def _run_drag_sweep(config_path: Path, altitudes_km: np.ndarray) -> dict[str, np.ndarray]:
    base_sc = load_spacecraft_from_base_config(config_path)
    base_sc.geometry.A_in_drag = base_sc.geometry.A_in

    properties = atmospheric_properties_from_height(
        altitudes_km,
        msis_date=base_sc.orbit.msis_date,
        msis_f107=base_sc.orbit.msis_f107,
        msis_ap=base_sc.orbit.msis_ap,
        latitude=base_sc.orbit.latitude,
        longitude=base_sc.orbit.longitude,
        use_average=base_sc.orbit.use_average,
    )
    density = np.asarray(properties["density"], dtype=float)
    temperature = np.asarray(properties["temperature"], dtype=float)
    velocity = np.asarray(properties["orbital_velocity"], dtype=float)
    molar_mass = np.asarray(properties["molar_mass"], dtype=float)

    output = {
        "altitude": np.asarray(altitudes_km, dtype=float),
        "drag_total": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_inlet_front": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_solar": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_body_side": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_solar_front": np.full_like(altitudes_km, np.nan, dtype=float),
    }

    for i, altitude in enumerate(altitudes_km):
        sc = deepcopy(base_sc)
        sc.orbit.altitude = float(altitude)
        sc.orbit.density = float(density[i])
        sc.orbit.temperature = float(temperature[i])
        sc.orbit.molar_mass = float(molar_mass[i])
        sc.orbit.velocity = float(velocity[i])

        drag_model(sc)
        _set_drag_force_outputs(sc)

        output["drag_total"][i] = float(sc.drag.drag_total) * NEWTON_TO_MILLINEWTON
        output["drag_inlet_front"][i] = float(sc.drag.drag_inlet_front) * NEWTON_TO_MILLINEWTON
        output["drag_solar"][i] = float(sc.drag.drag_solar) * NEWTON_TO_MILLINEWTON
        output["drag_body_side"][i] = float(sc.drag.drag_body_side) * NEWTON_TO_MILLINEWTON
        output["drag_solar_front"][i] = float(sc.drag.drag_solar_front) * NEWTON_TO_MILLINEWTON

    return output


# ------------------------------------------------------------------------------ #
# Plotting and metrics
# ------------------------------------------------------------------------------ #

def _interp_model_at_altitude(model: dict[str, np.ndarray], key: str, y_query: np.ndarray) -> np.ndarray:
    y_model = np.asarray(model["altitude"], dtype=float)
    x_model = np.asarray(model[key], dtype=float)
    order = np.argsort(y_model)
    return np.interp(y_query, y_model[order], x_model[order])


def _paired_model_reference_samples(
    model_altitude: np.ndarray,
    model_values: np.ndarray,
    ref_altitude: np.ndarray,
    ref_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model_altitude = np.asarray(model_altitude, dtype=float)
    model_values = np.asarray(model_values, dtype=float)
    ref_altitude = np.asarray(ref_altitude, dtype=float)
    ref_values = np.asarray(ref_values, dtype=float)

    valid_model = np.isfinite(model_altitude) & np.isfinite(model_values)
    if np.count_nonzero(valid_model) < 2:
        return None

    model_altitude = model_altitude[valid_model]
    model_values = model_values[valid_model]
    order = np.argsort(model_altitude)
    model_altitude = model_altitude[order]
    model_values = model_values[order]

    model_altitude_unique, unique_idx = np.unique(model_altitude, return_index=True)
    model_values_unique = model_values[unique_idx]
    if len(model_altitude_unique) < 2:
        return None

    line_ids = np.arange(1, len(ref_values) + 1, dtype=int)
    valid_ref = np.isfinite(ref_altitude) & np.isfinite(ref_values)
    if not np.any(valid_ref):
        return None

    ref_altitude = ref_altitude[valid_ref]
    ref_values = ref_values[valid_ref]
    line_ids = line_ids[valid_ref]

    altitude_low = float(np.min(model_altitude_unique))
    altitude_high = float(np.max(model_altitude_unique))
    in_range = (ref_altitude >= altitude_low) & (ref_altitude <= altitude_high)
    if not np.any(in_range):
        return None

    ref_altitude = ref_altitude[in_range]
    ref_values = ref_values[in_range]
    line_ids = line_ids[in_range]
    model_at_ref = np.interp(ref_altitude, model_altitude_unique, model_values_unique)

    return model_at_ref, ref_values, line_ids


def _component_relative_and_corr_stats(
    model: dict[str, np.ndarray],
    dataset: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[float, float, int, int, float, int]]:
    stats: dict[str, tuple[float, float, int, int, float, int]] = {}

    for label, key, _, _ in COMPONENT_SPECS:
        if label not in dataset:
            continue
        x_ref, y_ref = dataset[label]

        paired = _paired_model_reference_samples(
            np.asarray(model["altitude"], dtype=float),
            np.asarray(model[key], dtype=float),
            np.asarray(y_ref, dtype=float),
            np.asarray(x_ref, dtype=float),
        )
        if paired is None:
            continue

        model_at_ref, ref_values, line_ids = paired

        nonzero_ref = np.abs(ref_values) > 1.0e-12
        if np.any(nonzero_ref):
            relative_error = np.abs(model_at_ref[nonzero_ref] - ref_values[nonzero_ref]) / np.abs(ref_values[nonzero_ref])
            rel_line_ids = line_ids[nonzero_ref]
            i_max = int(np.argmax(relative_error))
            max_rel_error = float(relative_error[i_max])
            max_rel_line = int(rel_line_ids[i_max])
            mean_rel_error = float(np.mean(relative_error))
            n_rel = int(len(relative_error))
        else:
            max_rel_error = float("nan")
            max_rel_line = -1
            mean_rel_error = float("nan")
            n_rel = 0

        if len(model_at_ref) >= 2:
            pearson_r = float(np.corrcoef(model_at_ref, ref_values)[0, 1])
        else:
            pearson_r = float("nan")

        stats[label] = (
            max_rel_error,
            mean_rel_error,
            max_rel_line,
            n_rel,
            pearson_r,
            int(len(model_at_ref)),
        )

    return stats


def _plot_case(axis, title: str, model: dict[str, np.ndarray], dataset: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    x_lower = np.inf
    x_upper = 0.0

    for label, key, color, marker in COMPONENT_SPECS:
        x_model = np.asarray(model[key], dtype=float)
        y_model = np.asarray(model["altitude"], dtype=float)
        valid_model = np.isfinite(x_model) & np.isfinite(y_model) & (x_model > 0.0)
        axis.plot(x_model[valid_model], y_model[valid_model], color=color, lw=2.0, zorder=2)

        if np.any(valid_model):
            x_lower = min(x_lower, float(np.min(x_model[valid_model])))
            x_upper = max(x_upper, float(np.max(x_model[valid_model])))

        if label in dataset:
            x_ref, y_ref = dataset[label]
            valid_ref = np.isfinite(x_ref) & np.isfinite(y_ref) & (x_ref > 0.0)
            axis.plot(
                x_ref[valid_ref],
                y_ref[valid_ref],
                linestyle=(0, (4, 2)),
                lw=1.2,
                color=color,
                marker=marker,
                markersize=5.4,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                zorder=3,
            )
            if np.any(valid_ref):
                x_lower = min(x_lower, float(np.min(x_ref[valid_ref])))
                x_upper = max(x_upper, float(np.max(x_ref[valid_ref])))

    axis.set_title(title)
    axis.set_xscale("log")
    axis.set_xlabel("Drag [mN]")
    style_axis(axis)
    axis.tick_params(axis="both", which="both", labelsize=12)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.title.set_size(12)
    axis.yaxis.set_major_locator(MultipleLocator(10))
    axis.yaxis.set_minor_locator(MultipleLocator(5))

    if np.isfinite(x_lower) and np.isfinite(x_upper) and x_upper > x_lower > 0.0:
        axis.set_xlim(0.9 * x_lower, 1.15 * x_upper)


def run_crandall_wirz_validation(show: bool = True) -> Path:
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

    datasets: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for spec in CASE_SPECS:
        dataset = _load_digitized_dataset(spec["dataset"])
        dataset, flipped = _normalize_altitude_orientation(dataset)
        if flipped:
            print(f"[info] Mirrored altitude axis detected and corrected for dataset: {spec['dataset'].name}")
        datasets.append(dataset)
    altitude_min = min(_dataset_altitude_bounds(dataset)[0] for dataset in datasets)
    altitude_max = max(_dataset_altitude_bounds(dataset)[1] for dataset in datasets)
    altitudes_km = np.linspace(altitude_min, altitude_max, ALTITUDE_SAMPLES)

    models = [_run_drag_sweep(spec["config"], altitudes_km) for spec in CASE_SPECS]

    figure = plt.figure(figsize=PAGE_FIGSIZE)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=0.07)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    ]
    axes[1].sharey(axes[0])
    legend_axis = figure.add_subplot(grid[0, 2])
    legend_axis.axis("off")
    for axis, spec, model, dataset in zip(axes, CASE_SPECS, models, datasets):
        _plot_case(axis, spec["title"], model, dataset)
        metrics = _component_relative_and_corr_stats(model, dataset)
        print(f"\n{spec['title']} relative-error and correlation by component:")
        pearson_values: list[float] = []
        for label, _, _, _ in COMPONENT_SPECS:
            if label in metrics:
                max_rel_error, mean_rel_error, line_max_rel, n_rel, pearson_r, n_corr = metrics[label]
                line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
                print(
                    f"  {label:<24} "
                    f"max_relative_error={max_rel_error:10.6f} ({100.0 * max_rel_error:7.3f}%) (line {line_text}), "
                    f"mean_relative_error={mean_rel_error:10.6f} ({100.0 * mean_rel_error:7.3f}%), "
                    f"pearson_r={pearson_r:9.6f}, n_rel={n_rel}, n_corr={n_corr}"
                )
                if np.isfinite(pearson_r):
                    pearson_values.append(pearson_r)
        if pearson_values:
            print(f"  Minimum Pearson correlation coefficient: {min(pearson_values):.6f}")
        else:
            print("  Minimum Pearson correlation coefficient: n/a")

    axes[0].set_ylabel("Altitude [km]")
    axes[0].set_ylim(altitude_min - 1.0, altitude_max + 1.0)

    component_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=label)
        for label, _, color, _ in COMPONENT_SPECS
    ]
    source_handles = [
        Line2D([0], [0], color=PALETTE["secondary_text"], lw=2.0, label="ARISS drag-only model"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=PALETTE["secondary_text"],
            markerfacecolor="white",
            markersize=6,
            lw=1.2,
            ls=(0, (4, 2)),
            label="Crandall-Wirz Data",
        ),
    ]

    legend_source = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.4,
        handletextpad=0.6,
    )
    style_legend(legend_source)
    legend_source.get_title().set_fontweight("bold")
    legend_axis.add_artist(legend_source)

    legend_components = legend_axis.legend(
        handles=component_handles,
        title="Components",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.62),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.4,
        handletextpad=0.6,
    )
    style_legend(legend_components)
    legend_components.get_title().set_fontweight("bold")
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95, wspace=0.07)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    print(f"\nSaved figure: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    run_crandall_wirz_validation(show=True)
