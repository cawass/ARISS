from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt

from ariss.core.sensitivity import find_min_refuel_time, get_path, set_path
from ariss.core.simulation import load_spacecraft_from_base_config, run_sizing_loop
from ariss.utils.ploting import (
    PLOT_GEOMETRY_ASPECT_RATIO_BARS,
    PLOT_MULTI_SENSITIVITY_SIDE_BY_SIDE,
    plot_by_index,
    run_geometry_sensitivity_cases,
    run_original_sensitivity_cases,
)

DRAG_EPSILON_PATHS = [
    "geometry.epsilon_in",
    "geometry.epsilon_body",
    "geometry.epsilon_solar",
    "geometry.epsilon_rad",
    "geometry.epsilon_in_norm",
]

PARAMETERS: list[tuple[str, str | list[str]]] = [
    ("eta_solar", "solar.eta_solar"),
    ("eta_prop", "thruster.eff"),
    ("eta_coll", "refueling.coll_eff"),
    ("eta_ref", "refueling.eta_refuel"),
    ("eta_elec", "solar.eta_power"),
    ("epsilon", DRAG_EPSILON_PATHS),
    ("P_prop", "thruster.power"),
    ("I_sp", "thruster.specific_impulse"),
    ("chi", "geometry.intake_area_ratio"),
    ("AR_in", "geometry.AR_in"),
    ("AR_solar", "geometry.AR_solar"),
    ("T_des", "thermal.T_des"),
]

OUTPUT_H = "orbit.altitude"
OUTPUT_TREF = "refueling.t_refuel"

DEFAULT_CASES: list[str] = []


def _to_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric and numeric not in (float("inf"), float("-inf")) else None


def _resolve_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _central_derivative(x_minus: float | None, x_plus: float | None, y_minus: float | None, y_plus: float | None) -> float | None:
    if x_minus is None or x_plus is None or y_minus is None or y_plus is None:
        return None
    denominator = x_plus - x_minus
    if abs(denominator) <= 1.0e-20:
        return None
    return (y_plus - y_minus) / denominator


def _central_normalized(y0: float | None, y_minus: float | None, y_plus: float | None, perturbation: float) -> float | None:
    if y0 is None or y_minus is None or y_plus is None:
        return None
    if abs(y0) <= 1.0e-20 or perturbation <= 0.0:
        return None
    value = ((y_plus - y_minus) / (2.0 * y0)) / perturbation
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6e}"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_core_base_results(csv_path: Path) -> None:
    print("\nCore Base Sensitivity")
    print("input | S_h | S_t_ref")
    print("---------------------")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            print(f"{row['input']} | {row['S_h']} | {row['S_t_ref']}")


def _save_plot(path: Path, fig) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _evaluate_outputs(
    sc: Any,
    output_paths: list[str],
    *,
    max_iterations: int,
    mode: str,
) -> dict[str, Any]:
    try:
        if mode == "refuel_search":
            final_sc, ok, error = find_min_refuel_time(sc, max_iterations=max_iterations)
        else:
            final_sc, ok, _ = run_sizing_loop(sc, max_iterations=max_iterations)
            error = None
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "outputs": {path: None for path in output_paths},
        }

    if not ok or final_sc is None:
        return {
            "ok": False,
            "error": error or "Sizing loop did not converge.",
            "outputs": {path: None for path in output_paths},
        }

    outputs: dict[str, Any] = {}
    for path in output_paths:
        value = get_path(final_sc, path)
        if path == OUTPUT_TREF and value is not None:
            outputs[path] = float(value) / (30.4375 * 24.0 * 3600.0)
        else:
            outputs[path] = value

    return {
        "ok": True,
        "error": None,
        "outputs": outputs,
    }


def _compute_ranking_rows(
    case_path: Path | None,
    *,
    perturbation: float,
    max_iterations: int,
) -> list[dict[str, Any]]:
    base_sc = load_spacecraft_from_base_config(case_path=case_path)

    base_h_run = _evaluate_outputs(
        deepcopy(base_sc),
        [OUTPUT_H],
        max_iterations=max_iterations,
        mode="direct",
    )
    base_t_run = _evaluate_outputs(
        deepcopy(base_sc),
        [OUTPUT_TREF],
        max_iterations=max_iterations,
        mode="refuel_search",
    )
    y0_h = _to_finite_float(base_h_run["outputs"][OUTPUT_H])
    y0_t = _to_finite_float(base_t_run["outputs"][OUTPUT_TREF])

    rows: list[dict[str, Any]] = []
    for label, path_spec in PARAMETERS:
        paths = [path_spec] if isinstance(path_spec, str) else [str(path) for path in path_spec]
        path_label = ", ".join(paths)

        try:
            base_values = [_to_finite_float(get_path(base_sc, path)) for path in paths]
        except Exception as exc:
            rows.append(
                {
                    "input": label,
                    "path": path_label,
                    "base_value": None,
                    "minus_value": None,
                    "plus_value": None,
                    "S_h": "N/A",
                    "S_t_ref": "N/A",
                    "S_h_normalized": "N/A",
                    "S_t_ref_normalized": "N/A",
                    "error": f"{exc}",
                }
            )
            continue

        if any(value is None or abs(float(value)) <= 1.0e-20 for value in base_values):
            rows.append(
                {
                    "input": label,
                    "path": path_label,
                    "base_value": "; ".join(f"{path}={value}" for path, value in zip(paths, base_values)),
                    "minus_value": None,
                    "plus_value": None,
                    "S_h": "N/A",
                    "S_t_ref": "N/A",
                    "S_h_normalized": "N/A",
                    "S_t_ref_normalized": "N/A",
                    "error": "Base value non-finite or zero.",
                }
            )
            continue

        base_values_float = [float(value) for value in base_values]
        minus_values = [value * (1.0 - perturbation) for value in base_values_float]
        plus_values = [value * (1.0 + perturbation) for value in base_values_float]

        sc_minus = deepcopy(base_sc)
        sc_plus = deepcopy(base_sc)
        for path, minus_value in zip(paths, minus_values):
            set_path(sc_minus, path, minus_value)
        for path, plus_value in zip(paths, plus_values):
            set_path(sc_plus, path, plus_value)

        minus_h = _evaluate_outputs(sc_minus, [OUTPUT_H], max_iterations=max_iterations, mode="direct")
        plus_h = _evaluate_outputs(sc_plus, [OUTPUT_H], max_iterations=max_iterations, mode="direct")
        minus_t = _evaluate_outputs(sc_minus, [OUTPUT_TREF], max_iterations=max_iterations, mode="refuel_search")
        plus_t = _evaluate_outputs(sc_plus, [OUTPUT_TREF], max_iterations=max_iterations, mode="refuel_search")

        y_minus_h = _to_finite_float(minus_h["outputs"][OUTPUT_H])
        y_plus_h = _to_finite_float(plus_h["outputs"][OUTPUT_H])
        y_minus_t = _to_finite_float(minus_t["outputs"][OUTPUT_TREF])
        y_plus_t = _to_finite_float(plus_t["outputs"][OUTPUT_TREF])

        s_h = _central_derivative(minus_values[0], plus_values[0], y_minus_h, y_plus_h)
        s_t = _central_derivative(minus_values[0], plus_values[0], y_minus_t, y_plus_t)
        s_h_norm = _central_normalized(y0_h, y_minus_h, y_plus_h, perturbation)
        s_t_norm = _central_normalized(y0_t, y_minus_t, y_plus_t, perturbation)

        error_parts = [msg for msg in [minus_h["error"], plus_h["error"], minus_t["error"], plus_t["error"]] if msg]
        rows.append(
            {
                "input": label,
                "path": path_label,
                "base_value": "; ".join(f"{path}={value:.6g}" for path, value in zip(paths, base_values_float)),
                "minus_value": "; ".join(f"{path}={value:.6g}" for path, value in zip(paths, minus_values)),
                "plus_value": "; ".join(f"{path}={value:.6g}" for path, value in zip(paths, plus_values)),
                "S_h": _fmt(s_h),
                "S_t_ref": _fmt(s_t),
                "S_h_normalized": _fmt(s_h_norm),
                "S_t_ref_normalized": _fmt(s_t_norm),
                "error": " | ".join(error_parts),
            }
        )

    return rows


def run_case(
    *,
    case_path: Path | None,
    case_name: str,
    output_dir: Path,
    perturbation: float,
    max_iterations: int,
    show_plots: bool,
) -> tuple[Path, Path, Path]:
    title_prefix = "" if case_name == "core_base_sensitivity" else f"{case_name} - "

    left_data, right_data = run_original_sensitivity_cases(
        case_path=case_path,
        max_iterations=max_iterations,
    )
    geometry_data = run_geometry_sensitivity_cases(
        case_path=case_path,
        max_iterations=max_iterations,
    )

    fig_side, _ = plot_by_index(
        PLOT_MULTI_SENSITIVITY_SIDE_BY_SIDE,
        (left_data, right_data),
        left_output=OUTPUT_H,
        right_output=OUTPUT_TREF,
        left_log_y=False,
        right_log_y=True,
        left_title=f"{title_prefix}Orbit altitude",
        right_title=f"{title_prefix}Refueling time",
        x_label="Parameter value [-]",
        paper_style=True,
        show=show_plots,
    )
    side_plot_path = output_dir / f"{case_name}_sensitivity_curves.png"
    _save_plot(side_plot_path, fig_side)

    fig_geom, _ = plot_by_index(
        PLOT_GEOMETRY_ASPECT_RATIO_BARS,
        geometry_data,
        title_left=f"{title_prefix}No refueling",
        title_right=f"{title_prefix}With refueling",
        show=show_plots,
    )
    geom_plot_path = output_dir / f"{case_name}_geometry_sensitivity.png"
    _save_plot(geom_plot_path, fig_geom)

    rows = _compute_ranking_rows(
        case_path,
        perturbation=perturbation,
        max_iterations=max_iterations,
    )
    ranking_name = (
        "core_base_sensitivity.csv"
        if case_name == "core_base_sensitivity"
        else f"{case_name}_sensitivity.csv"
    )
    ranking_csv_path = output_dir / ranking_name
    _write_csv(rows, ranking_csv_path)
    return ranking_csv_path, side_plot_path, geom_plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all sensitivity cases and plot via ariss.utils.ploting helpers.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=DEFAULT_CASES,
        help="Optional extra case TOML paths (core-base runs by default).",
    )
    parser.add_argument(
        "--skip-core-base",
        action="store_true",
        help="Skip the core baseline run.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/sensitivity/results",
        help="Output directory for CSV and PNG files.",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.10,
        help="Relative perturbation for ranking (0.10 means +/-10%%).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=220,
        help="Maximum iterations per solve.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display generated figures while running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("ariss").setLevel(logging.WARNING)
    logging.getLogger("ariss.core.simulation").setLevel(logging.WARNING)

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[tuple[str, Path | None]] = []
    if not bool(args.skip_core_base):
        scenarios.append(("core_base_sensitivity", None))

    for raw_case in args.cases:
        case_path = _resolve_path(raw_case)
        if not case_path.exists():
            raise FileNotFoundError(f"Case file not found: {case_path}")
        scenarios.append((case_path.stem, case_path))

    generated: list[tuple[str, Path, Path, Path]] = []
    for case_name, case_path in scenarios:
        ranking_csv, side_plot, geom_plot = run_case(
            case_path=case_path,
            case_name=case_name,
            output_dir=output_dir,
            perturbation=float(args.perturbation),
            max_iterations=int(args.max_iterations),
            show_plots=bool(args.show_plots),
        )
        generated.append((case_name, ranking_csv, side_plot, geom_plot))
        print(f"[sensitivity] wrote {ranking_csv}")
        print(f"[sensitivity] wrote {side_plot}")
        print(f"[sensitivity] wrote {geom_plot}")

    core_row = next((row for row in generated if row[0] == "core_base_sensitivity"), None)
    if core_row is not None:
        print(f"\nCSV: {core_row[1]}")
        print(f"Plot (curves): {core_row[2]}")
        print(f"Plot (geometry): {core_row[3]}")
        _print_core_base_results(core_row[1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
