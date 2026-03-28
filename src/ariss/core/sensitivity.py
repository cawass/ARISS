from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from ariss.core.simulation import load_spacecraft_from_base_config, run_sizing_loop

SECONDS_PER_MONTH = 30.4375 * 24.0 * 3600.0

SensitivityResult = dict[str, Any]
SensitivityRankingResult = dict[str, Any]


# --------------------------------------------------------------------------------------
# Core path access
# --------------------------------------------------------------------------------------
def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    object.__setattr__(obj, parts[-1], value)


def as_list(value: str | Sequence[Any] | None) -> list[Any]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def convert_output(path: str, value: Any) -> Any:
    if path == "refueling.t_refuel" and value is not None:
        return float(value) / SECONDS_PER_MONTH
    return value


def infer_mode(output_paths: Sequence[str], mode: str | None = None) -> str:
    if mode is not None:
        return mode
    return "refuel_search" if "refueling.t_refuel" in output_paths else "direct"


# --------------------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------------------
def find_min_refuel_time(
    sc: Any,
    *,
    max_iterations: int,
    min_value: float = 1.0,
    shrink: float = 0.9,
    expand: float = 1.25,
    coarse_steps: int = 80,
    refine_steps: int = 20,
) -> tuple[Any | None, bool, str | None]:
    sc = deepcopy(sc)
    set_path(sc, "mission_profile.active_refueling", True)

    first_error = None
    t = max(float(get_path(sc, "refueling.t_refuel")), min_value)
    best_sc = None
    best_t = None

    def solve(value: float) -> tuple[Any | None, bool]:
        nonlocal first_error
        trial = deepcopy(sc)
        set_path(trial, "refueling.t_refuel", value)
        try:
            final_sc, ok, _ = run_sizing_loop(trial, max_iterations=max_iterations)
            return final_sc, ok
        except Exception as exc:
            if first_error is None:
                first_error = str(exc)
            return None, False

    # Expand until success.
    for _ in range(coarse_steps):
        final_sc, ok = solve(t)
        if ok:
            best_sc, best_t = final_sc, t
            break
        t *= expand

    if best_sc is None:
        return None, False, first_error or "No converged refueling point found."

    # Shrink until first failure.
    low = best_t
    high = None
    t = best_t

    for _ in range(coarse_steps):
        next_t = max(t * shrink, min_value)
        if next_t >= t:
            break

        final_sc, ok = solve(next_t)
        if ok:
            low = next_t
            best_t = next_t
            best_sc = final_sc
            t = next_t
            if next_t <= min_value:
                break
        else:
            high = next_t
            break

    # Refine with bisection.
    if high is not None:
        for _ in range(refine_steps):
            if abs(high - low) <= max(1e-6, 1e-4 * max(low, 1.0)):
                break

            mid = 0.5 * (low + high)
            final_sc, ok = solve(mid)
            if ok:
                low = mid
                best_t = mid
                best_sc = final_sc
            else:
                high = mid

    if best_sc is None or best_t is None:
        return None, False, first_error or "Failed to determine minimum refueling time."

    set_path(best_sc, "refueling.t_refuel", best_t)
    return best_sc, True, None


def evaluate(
    sc: Any,
    output_paths: Sequence[str],
    *,
    max_iterations: int,
    mode: str = "direct",
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

    return {
        "ok": True,
        "error": None,
        "outputs": {path: convert_output(path, get_path(final_sc, path)) for path in output_paths},
    }


# --------------------------------------------------------------------------------------
# Generic sweep engine
# --------------------------------------------------------------------------------------
def sweep(
    cases: Sequence[Mapping[str, Any]],
    output_paths: str | Sequence[str],
    *,
    case_path=None,
    base_config_path=None,
    base_sc: Any | None = None,
    max_iterations: int = 200,
    mode: str | None = None,
) -> dict[str, Any]:
    output_paths = as_list(output_paths)
    mode = infer_mode(output_paths, mode)

    base_sc = deepcopy(base_sc) if base_sc is not None else load_spacecraft_from_base_config(
        case_path=case_path,
        base_config_path=base_config_path,
    )

    result_cases: dict[str, Any] = {}

    for i, case in enumerate(cases):
        label = str(case.get("label", f"Case {i + 1}"))
        values = list(case.get("values", [None]))
        x_label = str(case.get("x_label", "x"))
        assign = case["assign"]

        outputs = {path: [] for path in output_paths}
        converged: list[bool] = []
        errors: list[str | None] = []

        for x in values:
            sc = deepcopy(base_sc)
            assignments = assign(x) if callable(assign) else dict(assign)

            for path, value in assignments.items():
                set_path(sc, path, value)

            run = evaluate(
                sc,
                output_paths,
                max_iterations=max_iterations,
                mode=mode,
            )

            for path in output_paths:
                outputs[path].append(run["outputs"][path])

            converged.append(run["ok"])
            errors.append(run["error"])

        result_cases[label] = {
            "label": label,
            "x_label": x_label,
            "values": values,
            "outputs": outputs,
            "converged": converged,
            "errors": errors,
        }

    return {
        "output_paths": output_paths,
        "cases": result_cases,
    }


# --------------------------------------------------------------------------------------
# Multi-variable sensitivity wrapper
# --------------------------------------------------------------------------------------
def _normalize_sweep_inputs(
    variable_paths: str | Sequence[str],
    variable_values: Sequence[Any] | Sequence[Sequence[Any]],
) -> tuple[list[str], list[list[Any]]]:
    paths = [variable_paths] if isinstance(variable_paths, str) else [str(p) for p in variable_paths]
    if not paths:
        raise ValueError("variable_paths cannot be empty.")

    # Single-variable form:
    #   variable_paths="orbit.altitude"
    #   variable_values=[180, 200, 220]
    if isinstance(variable_paths, str):
        values = list(variable_values)  # type: ignore[arg-type]
        if not values:
            raise ValueError("variable_values cannot be empty.")
        return paths, [values]

    # Multi-variable form:
    #   variable_paths=["a", "b"]
    #   variable_values=[[1, 2, 3], [10, 20, 30]]
    if len(variable_values) != len(paths):  # type: ignore[arg-type]
        raise ValueError("variable_paths and variable_values must have the same length.")

    value_lists: list[list[Any]] = []
    for values in variable_values:  # type: ignore[assignment]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            current = [values]
        else:
            current = list(values)

        if not current:
            raise ValueError("Each entry in variable_values must contain at least one value.")
        value_lists.append(current)

    return paths, value_lists


def _build_sweep_points(
    variable_paths: Sequence[str],
    variable_values: Sequence[Sequence[Any]],
    *,
    combine: str = "zip",
) -> list[dict[str, Any]]:
    if combine == "zip":
        lengths = {len(v) for v in variable_values if len(v) > 1}
        n = max(lengths, default=1)

        if any(len(v) not in (1, n) for v in variable_values):
            raise ValueError(
                "For combine='zip', each value list must have length 1 or the same common length."
            )

        return [
            {
                path: values[0] if len(values) == 1 else values[i]
                for path, values in zip(variable_paths, variable_values)
            }
            for i in range(n)
        ]

    if combine == "product":
        return [
            {path: value for path, value in zip(variable_paths, combo)}
            for combo in product(*variable_values)
        ]

    raise ValueError("combine must be 'zip' or 'product'.")


def _format_point_label(point: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in point.items())


def run_sensitivity_analysis(
    variable_paths: str | Sequence[str],
    variable_values: Sequence[Any] | Sequence[Sequence[Any]],
    output_paths: str | Sequence[str],
    *,
    combine: str = "zip",
    case_path=None,
    base_config_path=None,
    base_sc: Any | None = None,
    max_iterations: int = 200,
    mode: str | None = None,
) -> SensitivityResult:
    paths, value_lists = _normalize_sweep_inputs(variable_paths, variable_values)
    points = _build_sweep_points(paths, value_lists, combine=combine)

    result = sweep(
        cases=[
            {
                "label": "Sensitivity",
                "values": points,
                "x_label": paths[0] if len(paths) == 1 else "case",
                "assign": lambda point: dict(point),
            }
        ],
        output_paths=output_paths,
        case_path=case_path,
        base_config_path=base_config_path,
        base_sc=base_sc,
        max_iterations=max_iterations,
        mode=mode,
    )

    case = result["cases"]["Sensitivity"]
    case["variable_paths"] = paths
    case["points"] = points
    case["point_labels"] = [_format_point_label(point) for point in points]
    return case


