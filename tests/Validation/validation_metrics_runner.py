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
#      Unified validation-metrics runner. Executes validation sweeps and
#      computes line-by-line relative metrics against digitized references.
#
#  Project:        ARISS
#  Module:         validation_metrics_runner.py
# ============================================================================== #

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from validation_metrics import paired_model_reference_samples


HERE = Path(__file__).resolve().parent


def _load_module(relative_path: str, module_name: str):
    module_path = HERE / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paired_line_metrics(
    model_x: np.ndarray | list[float],
    model_y: np.ndarray | list[float],
    ref_x: np.ndarray | list[float],
    ref_y: np.ndarray | list[float],
) -> dict[str, float] | None:
    paired = paired_model_reference_samples(model_x, model_y, ref_x, ref_y)
    if paired is None:
        return None

    model_at_ref, ref_y_used, _ = paired
    if len(model_at_ref) == 0:
        return None

    nonzero_ref = np.abs(ref_y_used) > 1.0e-12
    if np.any(nonzero_ref):
        relative_error = np.abs(model_at_ref[nonzero_ref] - ref_y_used[nonzero_ref]) / np.abs(ref_y_used[nonzero_ref])
        max_relative_error = float(np.max(relative_error))
        mean_relative_error = float(np.mean(relative_error))
    else:
        max_relative_error = float("nan")
        mean_relative_error = float("nan")

    if len(model_at_ref) >= 2:
        pearson_r = float(np.corrcoef(model_at_ref, ref_y_used)[0, 1])
    else:
        pearson_r = float("nan")

    return {
        "max_relative_error": max_relative_error,
        "mean_relative_error": mean_relative_error,
        "pearson_r": pearson_r,
    }


def _collect_gocee_fig5_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module("GOCEE-Drag/GOCEEValidation.py", "gocee_fig5_validation_module")
    dataset = mod.load_wide_xy_dataset(mod.DATASET_PATH)

    x_ref: list[float] = []
    for key in ("Mansur", "Koppenwallner"):
        if key in dataset:
            x_ref.extend(np.asarray(dataset[key][0], dtype=float).tolist())
    x_min = float(np.nanmin(x_ref)) if x_ref else 8.0
    x_max = float(np.nanmax(x_ref)) if x_ref else 13.0
    x_ariss = np.linspace(x_min, x_max, 140)
    x_ariss, y_ariss = mod.compute_ariss_body_cd_curve(x_ariss)

    for label in ("Mansur", "Koppenwallner"):
        if label not in dataset:
            continue
        x_ref_vals, y_ref_vals = dataset[label]
        result = _paired_line_metrics(x_ariss, y_ariss, x_ref_vals, y_ref_vals)
        if result is not None:
            metrics[f"GOCEE Fig.5 | {label}"] = result


def _collect_gocee_thermal_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module("GOCEEThermal/GOCEEThermalValidation.py", "gocee_thermal_validation_module")
    spacecraft = mod.load_spacecraft_from_base_config(mod.CONFIG_PATH)
    diagnostics = mod.thermal_model(spacecraft)

    model_temp_c = float(diagnostics.T_max - 273.15)
    ref_temp_c = 100.0
    if abs(ref_temp_c) > 1.0e-12:
        rel_error = abs(model_temp_c - ref_temp_c) / abs(ref_temp_c)
    else:
        rel_error = float("nan")

    metrics["GOCEE Thermal | Steady-state temperature"] = {
        "max_relative_error": float(rel_error),
        "mean_relative_error": float(rel_error),
        "pearson_r": float("nan"),
    }


def _collect_crandall_fig6_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module(
        "CrandallWirz2022-Drag_Simplified_Trust/CrandallWirz2022Validation.py",
        "crandall_fig6_validation_module",
    )

    datasets: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for spec in mod.CASE_SPECS:
        dataset = mod._load_digitized_dataset(spec["dataset"])
        dataset, _ = mod._normalize_altitude_orientation(dataset)
        datasets.append(dataset)

    altitude_min = min(mod._dataset_altitude_bounds(dataset)[0] for dataset in datasets)
    altitude_max = max(mod._dataset_altitude_bounds(dataset)[1] for dataset in datasets)
    altitudes_km = np.linspace(altitude_min, altitude_max, mod.ALTITUDE_SAMPLES)
    models = [mod._run_drag_sweep(spec["config"], altitudes_km) for spec in mod.CASE_SPECS]

    for spec, model, dataset in zip(mod.CASE_SPECS, models, datasets):
        case_title = str(spec["title"])
        for line_label, model_key, _, _ in mod.COMPONENT_SPECS:
            if line_label not in dataset:
                continue
            x_ref, y_ref = dataset[line_label]
            result = _paired_line_metrics(
                np.asarray(model["altitude"], dtype=float),
                np.asarray(model[model_key], dtype=float),
                np.asarray(y_ref, dtype=float),
                np.asarray(x_ref, dtype=float),
            )
            if result is not None:
                metrics[f"Crandall Fig.6 | {case_title} | {line_label}"] = result


def _collect_crandall_fig11_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module(
        "CrandallWirz2022-Drag_Simplified_Trust/CrandallWirz2022Fig11Validation.py",
        "crandall_fig11_validation_module",
    )
    dataset = mod.load_fig11_dataset(mod.DATASET_PATH)
    results = mod.run_fig11_sweep()

    for case in mod.SOLAR_CASES:
        label = case["label"]
        if label not in dataset or label not in results:
            continue
        tp_model = np.asarray(results[label]["tp"], dtype=float)
        alt_model = np.asarray(results[label]["altitude"], dtype=float)
        tp_ref, alt_ref = dataset[label]
        result = _paired_line_metrics(tp_model, alt_model, tp_ref, alt_ref)
        if result is not None:
            metrics[f"Crandall Fig.11 | {label}"] = result


def _collect_crandall_fig26_fig27_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module(
        "CrandallWirz2022-Drag_Simplified_Trust/CrandallWirz2022Fig26Fig27Validation.py",
        "crandall_fig26_fig27_validation_module",
    )
    fig26_dataset = mod.load_wide_xy_dataset(mod.DATASET_SOLAR_EFF_PATH)
    fig27_dataset = mod.load_wide_xy_dataset(mod.DATASET_ACC_COEFF_PATH)

    old_level = mod.simulation_logger.level
    mod.simulation_logger.setLevel(50)
    try:
        fig26_model = mod.run_solar_efficiency_sweep(fig26_dataset)
        fig27_model = mod.run_accommodation_sweep(fig27_dataset)
    finally:
        mod.simulation_logger.setLevel(old_level)

    for case in mod.SOLAR_CASES:
        label = case["label"]
        if label in fig26_dataset and label in fig26_model:
            result_26 = _paired_line_metrics(
                np.asarray(fig26_model[label][0], dtype=float),
                np.asarray(fig26_model[label][1], dtype=float),
                np.asarray(fig26_dataset[label][0], dtype=float),
                np.asarray(fig26_dataset[label][1], dtype=float),
            )
            if result_26 is not None:
                metrics[f"Crandall Fig.26 | {label}"] = result_26

        if label in fig27_dataset and label in fig27_model:
            result_27 = _paired_line_metrics(
                np.asarray(fig27_model[label][0], dtype=float),
                np.asarray(fig27_model[label][1], dtype=float),
                np.asarray(fig27_dataset[label][0], dtype=float),
                np.asarray(fig27_dataset[label][1], dtype=float),
            )
            if result_27 is not None:
                metrics[f"Crandall Fig.27 | {label}"] = result_27


def _collect_mansur_efficiency_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module("Mansur-Full_Model/MansurEfficiencyValidation.py", "mansur_efficiency_validation_module")
    results = mod.sweep_mansur_efficiencies()
    paper = mod.load_mansur_paper_dataset()

    for eff in mod.COLLECTION_EFFICIENCIES:
        key = float(eff)
        if key not in results or key not in paper:
            continue
        model_x, model_y = results[key]
        ref_x, ref_y = paper[key]
        result = _paired_line_metrics(
            np.asarray(model_x, dtype=float),
            np.asarray(model_y, dtype=float),
            np.asarray(ref_x, dtype=float),
            np.asarray(ref_y, dtype=float),
        )
        if result is not None:
            metrics[f"Mansur Efficiency | eta={key:.2f}"] = result


def _collect_mansur_envelope_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module("Mansur-Full_Model/MansurEnvelopeValidation.py", "mansur_envelope_validation_module")
    isp_values, alt, tp = mod.run_sweep()
    paper, _ = mod.load_dataset(mod.DATASET_PATH)
    ariss = mod.extract_lines(isp_values, alt, tp)

    for level in mod.ALT_LEVELS:
        if level not in paper or level not in ariss:
            continue
        branches = sorted(ariss[level], key=lambda b: len(b[0]), reverse=True)
        if not branches:
            continue
        model_x, model_y = branches[0]
        ref_x, ref_y = paper[level]
        result = _paired_line_metrics(model_x, model_y, ref_x, ref_y)
        if result is not None:
            metrics[f"Mansur Envelope | h={int(level)} km"] = result


def _collect_mansur_thruster_map_metrics(metrics: dict[str, dict[str, float]]) -> None:
    mod = _load_module("Mansur-Full_Model/MansurThrusterMapValidation.py", "mansur_thruster_map_validation_module")
    isp_grid, tp_grid, eff_grid, mdot_grid, ain_grid = mod.run_sweep()

    efficiency_lines = mod.extract_lines(isp_grid, eff_grid, tp_grid, mod.EFF_LEVELS)
    mass_flow_lines = mod.extract_lines(isp_grid, mdot_grid, tp_grid, mod.MDOT_LEVELS)
    intake_area_lines = mod.extract_lines(isp_grid, ain_grid, tp_grid, mod.AIN_LEVELS)

    mansur_eta = mod.load_wide_xy_dataset(mod.ETA_DATASET_PATH)
    mansur_mdot = mod.load_wide_xy_dataset(mod.MFLOW_DATASET_PATH)
    mansur_ain = mod.load_wide_xy_dataset(mod.AIN_DATASET_PATH)

    families = [
        ("eta", mod.EFF_LEVELS, efficiency_lines, mansur_eta),
        ("mdot", mod.MDOT_LEVELS, mass_flow_lines, mansur_mdot),
        ("ain", mod.AIN_LEVELS, intake_area_lines, mansur_ain),
    ]

    for family_name, levels, model_lines, dataset in families:
        for level in levels:
            model_key = mod._find_level_key(model_lines, level)
            data_key = mod._find_level_key(dataset, level)
            if model_key is None or data_key is None:
                continue
            branches = sorted(model_lines[model_key], key=lambda seg: len(seg[0]), reverse=True)
            if not branches:
                continue
            model_x, model_y = branches[0]
            ref_x, ref_y = dataset[data_key]
            ref_x, ref_y = mod.sort_points_by_tp(ref_x, ref_y)
            result = _paired_line_metrics(model_x, model_y, ref_x, ref_y)
            if result is not None:
                metrics[f"Mansur Thruster Map | {family_name}={float(level):.2f}"] = result


def run_all_validation_metrics(*, print_output: bool = True) -> dict[str, dict[str, float]]:
    # Inputs:
    #   print_output: if True, print all computed line metrics.
    #
    # Output:
    #   Mapping:
    #     label -> {
    #       "max_relative_error": float,
    #       "mean_relative_error": float,
    #       "pearson_r": float,
    #     }
    metrics: dict[str, dict[str, float]] = {}

    print("Running validation metrics...")
    print("Running GOCEE Fig. 5 metrics...")
    _collect_gocee_fig5_metrics(metrics)
    print("Running GOCEE thermal metrics...")
    _collect_gocee_thermal_metrics(metrics)
    print("Running Crandall & Wirz Fig. 6 metrics...")
    _collect_crandall_fig6_metrics(metrics)
    print("Running Crandall & Wirz Fig. 11 metrics...")
    _collect_crandall_fig11_metrics(metrics)
    print("Running Crandall & Wirz Fig. 26 & 27 metrics...")
    _collect_crandall_fig26_fig27_metrics(metrics)
    print("Running Mansur efficiency metrics...")
    _collect_mansur_efficiency_metrics(metrics)
    print("Running Mansur envelope metrics...")
    _collect_mansur_envelope_metrics(metrics)
    print("Running Mansur thruster map metrics...")
    _collect_mansur_thruster_map_metrics(metrics)

    if print_output:
        print("Validation line metrics:")
        for key in sorted(metrics):
            values = metrics[key]
            print(
                f"{key} | "
                f"max_rel={float(values['max_relative_error']):.6f} | "
                f"mean_rel={float(values['mean_relative_error']):.6f} | "
                f"pearson_r={float(values['pearson_r']):.6f}"
            )

    return metrics


if __name__ == "__main__":
    run_all_validation_metrics(print_output=True)
