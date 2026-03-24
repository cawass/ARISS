from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, NullFormatter

from ariss.core.simulation import load_spacecraft_from_base_config, run_sizing_loop

SECONDS_PER_MONTH = 30.4375 * 24.0 * 3600.0


@dataclass
class SensitivityResult:
    variable_path: str
    variable_values: list[Any]
    output_paths: list[str]
    outputs: dict[str, list[Any]]
    converged: list[bool]
    errors: list[str | None]


@dataclass
class SensitivityCase:
    label: str
    variable_path: str
    variable_values: list[Any]


@dataclass
class MultiSensitivityResult:
    output_paths: list[str]
    cases: dict[str, SensitivityResult]


def get_attr(obj: Any, path: str) -> Any:
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def set_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    object.__setattr__(obj, parts[-1], value)


def _run_single_loop(sc, max_iterations: int):
    return run_sizing_loop(
        sc,
        max_iterations=max_iterations,
    )


def _to_output_units(path: str, value: Any) -> Any:
    # Keep internal solving in SI units; only convert displayed/output values.
    if path == "refueling.t_refuel" and value is not None:
        return float(value) / SECONDS_PER_MONTH
    return value


def _find_min_refueling_time(
    base_sc,
    *,
    max_iterations: int,
    min_t_refuel: float = 1.0,
    shrink_factor: float = 0.9,
    expand_factor: float = 1.25,
    coarse_steps: int = 80,
    refine_steps: int = 20,
):
    # Inputs:
    #   base_sc: spacecraft state for one sweep point (already includes swept variable).
    #   max_iterations: iterations passed to run_sizing_loop.
    #
    # Output:
    #   (best_sc, converged, error_text)
    #
    # Method:
    #   1) Ensure refueling branch is active.
    #   2) Increase t_refuel until a converged point is found (if needed).
    #   3) Decrease t_refuel until the first non-converged point appears.
    #   4) Refine the minimum converged t_refuel by bisection.

    set_attr(base_sc, "mission_profile.active_refueling", True)
    initial_t = float(get_attr(base_sc, "refueling.t_refuel"))
    trial_t = max(initial_t, min_t_refuel)

    best_sc = None
    best_t = None
    first_error = None

    # Step 1: find a converged starting point (expand t_refuel if required).
    for _ in range(coarse_steps):
        sc_trial = deepcopy(base_sc)
        set_attr(sc_trial, "refueling.t_refuel", trial_t)
        try:
            final_sc, converged, _ = _run_single_loop(sc_trial, max_iterations=max_iterations)
        except Exception as exc:
            converged = False
            if first_error is None:
                first_error = str(exc)

        if converged:
            best_sc = final_sc
            best_t = trial_t
            break

        trial_t *= expand_factor

    if best_sc is None or best_t is None:
        error = first_error or "No converged refueling point found in search range."
        return None, False, error

    # Step 2: decrease t_refuel until first non-converged point (divergence).
    low = best_t  # converged
    high = None   # non-converged

    trial_t = best_t
    for _ in range(coarse_steps):
        next_t = max(trial_t * shrink_factor, min_t_refuel)
        if next_t >= trial_t:
            break

        sc_trial = deepcopy(base_sc)
        set_attr(sc_trial, "refueling.t_refuel", next_t)
        try:
            final_sc, converged, _ = _run_single_loop(sc_trial, max_iterations=max_iterations)
        except Exception as exc:
            converged = False
            if first_error is None:
                first_error = str(exc)

        if converged:
            low = next_t
            best_t = next_t
            best_sc = final_sc
            trial_t = next_t
            if next_t <= min_t_refuel:
                break
        else:
            high = next_t
            break

    # Step 3: if divergence found, refine minimum converged point by bisection.
    if high is not None:
        for _ in range(refine_steps):
            if abs(high - low) <= max(1e-6, 1e-4 * max(low, 1.0)):
                break
            mid = 0.5 * (low + high)

            sc_trial = deepcopy(base_sc)
            set_attr(sc_trial, "refueling.t_refuel", mid)
            try:
                final_sc, converged, _ = _run_single_loop(sc_trial, max_iterations=max_iterations)
            except Exception as exc:
                converged = False
                if first_error is None:
                    first_error = str(exc)

            if converged:
                low = mid
                best_t = mid
                best_sc = final_sc
            else:
                high = mid

    if best_sc is not None and best_t is not None:
        set_attr(best_sc, "refueling.t_refuel", best_t)
        return best_sc, True, None

    error = first_error or "Failed to determine minimum converged refueling time."
    return None, False, error


def run_sensitivity_analysis(
    variable_path: str,
    variable_values: Sequence[Any],
    output_paths: str | Sequence[str],
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
):
    if isinstance(output_paths, str):
        output_paths = [output_paths]

    base_sc = load_spacecraft_from_base_config(
        case_path=case_path,
        base_config_path=base_config_path,
    )

    outputs = {p: [] for p in output_paths}
    converged = []
    errors = []
    find_min_refuel_time = "refueling.t_refuel" in output_paths

    for value in variable_values:
        sc = deepcopy(base_sc)
        set_attr(sc, variable_path, value)

        try:
            if find_min_refuel_time:
                final_sc, conv, err_msg = _find_min_refueling_time(
                    sc,
                    max_iterations=max_iterations,
                )
                if final_sc is None:
                    raise RuntimeError(err_msg or "Refueling-time search failed.")
            else:
                final_sc, conv, _ = _run_single_loop(
                    sc,
                    max_iterations=max_iterations,
                )
                err_msg = None

            if not conv:
                for p in output_paths:
                    outputs[p].append(None)

                converged.append(False)
                if err_msg is None:
                    try:
                        final_alt = float(get_attr(final_sc, "orbit.altitude"))
                        err_msg = f"Non-converged sizing loop (final altitude: {final_alt:.2f} km)"
                    except Exception:
                        err_msg = "Non-converged sizing loop"
                errors.append(err_msg)
                continue

            for p in output_paths:
                raw_value = get_attr(final_sc, p)
                outputs[p].append(_to_output_units(p, raw_value))

            converged.append(conv)
            errors.append(err_msg)

        except Exception as e:
            for p in output_paths:
                outputs[p].append(None)

            converged.append(False)
            errors.append(str(e))

    return SensitivityResult(
        variable_path=variable_path,
        variable_values=list(variable_values),
        output_paths=list(output_paths),
        outputs=outputs,
        converged=converged,
        errors=errors,
    )


def run_sensitivity(*args: Any, **kwargs: Any) -> SensitivityResult:
    return run_sensitivity_analysis(*args, **kwargs)


def _normalize_sensitivity_case(case: SensitivityCase | Mapping[str, Any]) -> SensitivityCase:
    if isinstance(case, SensitivityCase):
        return case

    label = str(case["label"])
    variable_path = str(case["variable_path"])
    variable_values = list(case["variable_values"])
    return SensitivityCase(
        label=label,
        variable_path=variable_path,
        variable_values=variable_values,
    )


def run_multi_sensitivity(
    cases: Sequence[SensitivityCase | Mapping[str, Any]],
    output_paths: str | Sequence[str],
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
) -> MultiSensitivityResult:
    # Run multiple independent sensitivity sweeps and collect them together.
    if isinstance(output_paths, str):
        normalized_output_paths = [output_paths]
    else:
        normalized_output_paths = list(output_paths)

    if not normalized_output_paths:
        raise ValueError("output_paths cannot be empty.")

    if not cases:
        raise ValueError("cases cannot be empty.")

    results: dict[str, SensitivityResult] = {}
    for raw_case in cases:
        case = _normalize_sensitivity_case(raw_case)
        case_result = run_sensitivity_analysis(
            variable_path=case.variable_path,
            variable_values=case.variable_values,
            output_paths=normalized_output_paths,
            case_path=case_path,
            base_config_path=base_config_path,
            max_iterations=max_iterations,
        )
        results[case.label] = case_result

    return MultiSensitivityResult(
        output_paths=normalized_output_paths,
        cases=results,
    )


def run_efficiency_sensitivities(
    values: Sequence[float],
    output_paths: str | Sequence[str],
    *,
    epsilon_path: str = "geometry.epsilon_body",
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
) -> MultiSensitivityResult:
    # Convenience bundle for the four common sweeps requested in validation:
    # thruster efficiency, collection efficiency, accommodation coefficient,
    # and solar-cell efficiency.
    sweep_values = list(values)
    cases = [
        SensitivityCase(
            label="Thruster efficiency",
            variable_path="thruster.eff",
            variable_values=sweep_values,
        ),
        SensitivityCase(
            label="Collection efficiency",
            variable_path="refueling.coll_eff",
            variable_values=sweep_values,
        ),
        SensitivityCase(
            label="Accommodation coefficient",
            variable_path=epsilon_path,
            variable_values=sweep_values,
        ),
        SensitivityCase(
            label="Solar-cell efficiency",
            variable_path="solar.eta_solar",
            variable_values=sweep_values,
        ),
    ]

    return run_multi_sensitivity(
        cases=cases,
        output_paths=output_paths,
        case_path=case_path,
        base_config_path=base_config_path,
        max_iterations=max_iterations,
    )


def _default_geometry_configs() -> list[dict[str, Any]]:
    # Four requested geometric shape configurations.
    return [
        {
            "label": "Sq intake / Sq body",
            "overrides": {"geometry.S_in": "s", "geometry.S_body": "s"},
        },
        {
            "label": "Sq intake / Circ body",
            "overrides": {"geometry.S_in": "s", "geometry.S_body": "c"},
        },
        {
            "label": "Circ intake / Circ body",
            "overrides": {"geometry.S_in": "c", "geometry.S_body": "c"},
        },
        {
            "label": "Circ intake / Sq body",
            "overrides": {"geometry.S_in": "c", "geometry.S_body": "s"},
        },
    ]


def run_geometry_aspect_ratio_sensitivity(
    aspect_ratios: Sequence[float] | None = None,
    geometry_configs: Sequence[Mapping[str, Any]] | None = None,
    *,
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
) -> dict[str, dict[str, Any]]:
    # Inputs:
    #   aspect_ratios: AR values applied to geometry.AR_in and geometry.AR_body.
    #   geometry_configs:
    #       Optional list of dicts with:
    #         - label: display name
    #         - overrides: dotted-path overrides for geometry definition
    #       If omitted, uses four default shape configurations.
    #   case_path/base_config_path/max_iterations:
    #       forwarded to spacecraft loading / sizing loop.
    #
    # Output:
    #   Dictionary with two modes:
    #     - "without_refueling"
    #     - "with_refueling"
    #   Each mode stores altitude/refueling-time grids and convergence flags by
    #   [geometry_config, aspect_ratio].

    ars = [0.25, 0.5, 1.0, 1.5, 2.0] if aspect_ratios is None else [float(value) for value in aspect_ratios]
    if not ars:
        raise ValueError("aspect_ratios cannot be empty.")

    configs = list(geometry_configs) if geometry_configs is not None else _default_geometry_configs()
    if not configs:
        raise ValueError("geometry_configs cannot be empty.")

    base_sc = load_spacecraft_from_base_config(
        case_path=case_path,
        base_config_path=base_config_path,
    )

    modes = [
        ("without_refueling", False),
        ("with_refueling", True),
    ]
    results: dict[str, dict[str, Any]] = {}

    for mode_key, active_refueling in modes:
        labels: list[str] = []
        value_grid: list[list[float | None]] = []
        converged_grid: list[list[bool]] = []
        error_grid: list[list[str | None]] = []

        for config in configs:
            label = str(config.get("label", "Geometry"))
            overrides = dict(config.get("overrides", {}))
            labels.append(label)

            values_for_config: list[float | None] = []
            converged_for_config: list[bool] = []
            errors_for_config: list[str | None] = []

            for ar_value in ars:
                sc = deepcopy(base_sc)
                set_attr(sc, "mission_profile.active_refueling", bool(active_refueling))

                for path, override_value in overrides.items():
                    set_attr(sc, str(path), override_value)

                set_attr(sc, "geometry.AR_in", float(ar_value))
                set_attr(sc, "geometry.AR_body", float(ar_value))

                try:
                    if active_refueling:
                        final_sc, converged, err_msg = _find_min_refueling_time(
                            sc,
                            max_iterations=max_iterations,
                        )
                        if final_sc is not None and converged:
                            refuel_seconds = float(get_attr(final_sc, "refueling.t_refuel"))
                            refuel_months = float(_to_output_units("refueling.t_refuel", refuel_seconds))
                            values_for_config.append(refuel_months)
                            converged_for_config.append(True)
                            errors_for_config.append(None)
                        else:
                            values_for_config.append(None)
                            converged_for_config.append(False)
                            errors_for_config.append(err_msg or "Non-converged refueling-time search")
                    else:
                        final_sc, converged, _ = _run_single_loop(
                            sc,
                            max_iterations=max_iterations,
                        )
                        if converged:
                            altitude_value = float(get_attr(final_sc, "orbit.altitude"))
                            values_for_config.append(altitude_value)
                            converged_for_config.append(True)
                            errors_for_config.append(None)
                        else:
                            values_for_config.append(None)
                            converged_for_config.append(False)
                            try:
                                final_alt = float(get_attr(final_sc, "orbit.altitude"))
                                errors_for_config.append(
                                    f"Non-converged sizing loop (final altitude: {final_alt:.2f} km)"
                                )
                            except Exception:
                                errors_for_config.append("Non-converged sizing loop")

                except Exception as exc:
                    values_for_config.append(None)
                    converged_for_config.append(False)
                    errors_for_config.append(str(exc))

            value_grid.append(values_for_config)
            converged_grid.append(converged_for_config)
            error_grid.append(errors_for_config)

        results[mode_key] = {
            "active_refueling": bool(active_refueling),
            "geometry_labels": labels,
            "aspect_ratios": ars,
            "values": value_grid,
            "value_label": (
                "Minimum refueling time [months]"
                if active_refueling
                else "Orbit altitude [km]"
            ),
            "converged": converged_grid,
            "errors": error_grid,
        }

    return results


def _load_plot_style():
    # Try to use the shared validation style used across validation plots.
    repo_root = Path(__file__).resolve().parents[3]
    validation_dir = repo_root / "tests" / "Validation"
    if validation_dir.exists():
        style_path = str(validation_dir)
        if style_path not in sys.path:
            sys.path.insert(0, style_path)

    try:
        from plot_style import PALETTE, apply_validation_style, style_axis, style_legend
    except Exception:
        PALETTE = {
            "secondary_text": "#4A4A4A",
            "l1_teal": "#5BC8D0",
            "sernn_pink": "#F08FA7",
            "choice_mid": "#C59A4A",
            "goal_dark": "#1E7F78",
            "cat_purple": "#9A5CB8",
            "cat_green": "#76C56E",
            "cat_red": "#E85C62",
            "cat_yellow": "#DCCB4F",
        }

        def apply_validation_style():
            pass

        def style_axis(axis):
            axis.grid(True, linewidth=0.6, alpha=0.6)

        def style_legend(legend):
            return legend

    return PALETTE, apply_validation_style, style_axis, style_legend


def _format_plain_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:g}"


def _apply_horizontal_grid_and_y_ticks(axis, *, log_y: bool) -> None:
    # Horizontal grid emphasis requested by user.
    axis.grid(False)

    if log_y:
        axis.yaxis.set_major_locator(LogLocator(base=10.0))
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        axis.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _pos: (
                    "" if (not np.isfinite(value) or value <= 0.0) else _format_plain_number(value)
                )
            )
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: _format_plain_number(value))
        )

    axis.grid(which="major", axis="y", color="0.86", linewidth=0.8)
    axis.grid(which="minor", axis="y", color="0.92", linewidth=0.55)


def _compute_boundary_converged_point(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    converged_flags: Sequence[bool],
    *,
    log_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return one real converged point adjacent to the first failed block.

    Behavior:
    - If failures occur after convergence, return the last converged point
      immediately before the first failed block.
    - If failures are leading, return the first converged point immediately
      after that failed block.
    - Never fabricate y-values.
    """
    finite_x = np.isfinite(x_arr)
    finite_y = np.isfinite(y_arr)
    valid_y = finite_y & ((y_arr > 0.0) if log_y else True)

    converged = np.asarray(list(converged_flags), dtype=bool)
    failed = (~converged) & finite_x

    if not np.any(failed):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    failed_indices = np.where(failed)[0]

    # First contiguous failed block
    block_start = int(failed_indices[0])
    block_end = block_start
    for idx in failed_indices[1:]:
        if idx == block_end + 1:
            block_end = int(idx)
        else:
            break

    converged_valid = converged & finite_x & valid_y
    idx_all = np.arange(x_arr.size)

    # Prefer the last converged point before the failed block.
    left_candidates = np.where(converged_valid & (idx_all < block_start))[0]
    if left_candidates.size > 0:
        idx = int(left_candidates[-1])
        return (
            np.asarray([x_arr[idx]], dtype=float),
            np.asarray([y_arr[idx]], dtype=float),
        )

    # For leading failures, fall back to the first converged point after the block.
    right_candidates = np.where(converged_valid & (idx_all > block_end))[0]
    if right_candidates.size > 0:
        idx = int(right_candidates[0])
        return (
            np.asarray([x_arr[idx]], dtype=float),
            np.asarray([y_arr[idx]], dtype=float),
        )

    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def plot_geometry_aspect_ratio_bars(
    sensitivity_data: Mapping[str, Mapping[str, Any]],
    *,
    show: bool = True,
    title_left: str = "No Refueling",
    title_right: str = "With Refueling",
):
    # Inputs:
    #   sensitivity_data: output of run_geometry_aspect_ratio_sensitivity.
    #   show: if True display figure.
    #   title_left/title_right: subplot titles.
    #
    # Output:
    #   (figure, axes) matplotlib objects.

    required_keys = {"without_refueling", "with_refueling"}
    if not required_keys.issubset(set(sensitivity_data.keys())):
        missing = required_keys - set(sensitivity_data.keys())
        raise KeyError(f"sensitivity_data missing required modes: {sorted(missing)}")

    PALETTE, apply_validation_style, style_axis, style_legend = _load_plot_style()
    apply_validation_style()

    left = sensitivity_data["without_refueling"]
    right = sensitivity_data["with_refueling"]

    geometry_labels = list(left["geometry_labels"])
    aspect_ratios = list(left["aspect_ratios"])
    if geometry_labels != list(right["geometry_labels"]):
        raise ValueError("Geometry labels differ between refueling modes.")
    if aspect_ratios != list(right["aspect_ratios"]):
        raise ValueError("Aspect-ratio lists differ between refueling modes.")

    n_groups = len(geometry_labels)
    n_ar = len(aspect_ratios)
    if n_groups == 0 or n_ar == 0:
        raise ValueError("No geometry/aspect-ratio data available for plotting.")

    x_centers = np.arange(n_groups, dtype=float)
    group_width = 0.82
    bar_width = group_width / max(n_ar, 1)

    fig = plt.figure(figsize=(11.2, 4.8), dpi=150)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.45))
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    legend_axis = fig.add_subplot(grid[0, 2])
    legend_axis.axis("off")

    bar_colors = [
        PALETTE.get("l1_teal", "#5BC8D0"),
        PALETTE.get("sernn_pink", "#F08FA7"),
        PALETTE.get("choice_mid", "#C59A4A"),
        PALETTE.get("goal_dark", "#1E7F78"),
        PALETTE.get("cat_purple", "#9A5CB8"),
        PALETTE.get("cat_green", "#76C56E"),
        PALETTE.get("cat_yellow", "#DCCB4F"),
        PALETTE.get("cat_red", "#E85C62"),
    ]

    def draw_mode(axis, mode_data: Mapping[str, Any], subplot_title: str) -> bool:
        value_grid = mode_data["values"]
        converged_grid = mode_data["converged"]

        non_converged_present = False
        finite_values: list[float] = []
        for row in value_grid:
            for value in row:
                if value is not None and np.isfinite(value):
                    finite_values.append(float(value))

        for ar_idx, ar_value in enumerate(aspect_ratios):
            x_offset = (ar_idx - 0.5 * (n_ar - 1)) * bar_width
            x_positions = x_centers + x_offset

            heights: list[float] = []
            for group_idx in range(n_groups):
                value = value_grid[group_idx][ar_idx]
                heights.append(np.nan if value is None else float(value))

            axis.bar(
                x_positions,
                np.asarray(heights, dtype=float),
                width=bar_width * 0.92,
                color=bar_colors[ar_idx % len(bar_colors)],
                edgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidth=0.6,
                zorder=2,
            )

            for group_idx, x_pos in enumerate(x_positions):
                if not bool(converged_grid[group_idx][ar_idx]):
                    non_converged_present = True
                    if finite_values:
                        marker_y = float(min(finite_values)) - 0.04 * max(
                            float(max(finite_values) - min(finite_values)),
                            1.0,
                        )
                    else:
                        marker_y = 0.0
                    axis.scatter(
                        [x_pos],
                        [marker_y],
                        marker="s",
                        s=28,
                        facecolors=PALETTE.get("cat_red", "#E85C62"),
                        edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                        linewidths=0.7,
                        zorder=4,
                    )

        axis.set_xticks(x_centers)
        two_line_labels = [label.replace(" / ", "\n") for label in geometry_labels]
        axis.set_xticklabels(two_line_labels)
        axis.set_xlabel("Geometry configuration")
        axis.set_title(subplot_title)
        style_axis(axis)
        _apply_horizontal_grid_and_y_ticks(axis, log_y=False)

        if finite_values:
            y_min = float(min(finite_values))
            y_max = float(max(finite_values))
            spread = max(y_max - y_min, 1.0)
            low_pad = 0.10 * spread
            high_pad = 0.08 * spread
            axis.set_ylim(y_min - low_pad, y_max + high_pad)

        return non_converged_present

    left_has_non_converged = draw_mode(ax_left, left, title_left)
    right_has_non_converged = draw_mode(ax_right, right, title_right)
    ax_left.set_ylabel(str(left["value_label"]))
    ax_right.set_ylabel(str(right["value_label"]))

    legend_handles = [
        Patch(
            facecolor=bar_colors[idx % len(bar_colors)],
            edgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            linewidth=0.6,
            label=f"AR = {float(ar_value):g}",
        )
        for idx, ar_value in enumerate(aspect_ratios)
    ]

    if left_has_non_converged or right_has_non_converged:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=7,
                markerfacecolor=PALETTE.get("cat_red", "#E85C62"),
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=0.8,
                label="Non-converged",
            )
        )

    legend = legend_axis.legend(
        handles=legend_handles,
        title="Aspect Ratio",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=1.8,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.92, wspace=0.20)

    if show:
        plt.show(block=True)

    return fig, (ax_left, ax_right)


def plot_sensitivity(
    result: SensitivityResult,
    output_path: str | None = None,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    log_y: bool = True,
    show: bool = True,
):
    # Inputs:
    #   result: output from run_sensitivity/run_sensitivity_analysis.
    #   output_path: output variable path to plot (defaults to first output path).
    #   x_label, y_label, title: optional custom labels/title.
    #   show: if True, display the figure.
    #
    # Output:
    #   (fig, ax) matplotlib objects.

    PALETTE, apply_validation_style, style_axis, style_legend = _load_plot_style()
    apply_validation_style()

    selected_output = output_path if output_path is not None else result.output_paths[0]
    if selected_output not in result.outputs:
        raise KeyError(
            f"Unknown output_path '{selected_output}'. Available: {result.output_paths}"
        )

    x_vals = list(result.variable_values)
    y_vals = list(result.outputs[selected_output])
    y_numeric = [float(v) if v is not None else float("nan") for v in y_vals]

    fig = plt.figure(figsize=(8.4, 4.2), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.34))
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_numeric, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if log_y:
        valid &= y_arr > 0.0

    ax.plot(
        x_arr[valid],
        y_arr[valid],
        color=PALETTE.get("secondary_text", "#4A4A4A"),
        linewidth=2.2,
        label="ARISS full loop",
    )
    ax.scatter(
        x_arr[valid],
        y_arr[valid],
        color=PALETTE.get("l1_teal", "#5BC8D0"),
        edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
        linewidths=0.8,
        s=30,
        zorder=3,
        label="Sweep points",
    )

    boundary_x, boundary_y = _compute_boundary_converged_point(
        x_arr,
        y_arr,
        result.converged,
        log_y=log_y,
    )
    if boundary_x.size > 0:
        ax.scatter(
            boundary_x,
            boundary_y,
            marker="s",
            s=42,
            facecolors=PALETTE.get("cat_red", "#E85C62"),
            edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
            linewidths=0.8,
            zorder=4,
            label="Last converged",
        )

    ax.set_xlabel(x_label or result.variable_path)
    if y_label is not None:
        ax.set_ylabel(y_label)
    elif selected_output == "refueling.t_refuel":
        ax.set_ylabel("refueling.t_refuel [months]")
    else:
        ax.set_ylabel(selected_output)
    if title is not None:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")

    style_axis(ax)
    _apply_horizontal_grid_and_y_ticks(ax, log_y=log_y)
    handles, labels = ax.get_legend_handles_labels()
    legend = legend_axis.legend(
        handles,
        labels,
        title="Curves",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94, wspace=0.05)

    if show:
        plt.show(block=True)

    return fig, ax


def plot_multi_sensitivity(
    multi_result: MultiSensitivityResult,
    output_path: str | None = None,
    *,
    x_label: str | None = "Parameter value [-]",
    y_label: str | None = None,
    title: str | None = None,
    log_y: bool = True,
    show: bool = True,
):
    # Overlay multiple sensitivity sweeps on one figure.
    PALETTE, apply_validation_style, style_axis, style_legend = _load_plot_style()
    apply_validation_style()

    selected_output = output_path if output_path is not None else multi_result.output_paths[0]
    if selected_output not in multi_result.output_paths:
        raise KeyError(
            f"Unknown output_path '{selected_output}'. Available: {multi_result.output_paths}"
        )

    fig = plt.figure(figsize=(8.8, 4.4), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.34))
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    color_cycle = [
        PALETTE.get("secondary_text", "#4A4A4A"),
        PALETTE.get("l1_teal", "#5BC8D0"),
        PALETTE.get("sernn_pink", "#F08FA7"),
        PALETTE.get("choice_mid", "#C59A4A"),
        PALETTE.get("goal_dark", "#1E7F78"),
        PALETTE.get("cat_purple", "#9A5CB8"),
        PALETTE.get("cat_green", "#76C56E"),
        PALETTE.get("cat_red", "#E85C62"),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<"]
    boundary_legend_added = False

    for idx, (label, result) in enumerate(multi_result.cases.items()):
        if selected_output not in result.outputs:
            raise KeyError(
                f"Case '{label}' does not contain output '{selected_output}'. "
                f"Available: {result.output_paths}"
            )

        x_vals = list(result.variable_values)
        y_vals = result.outputs[selected_output]
        y_numeric = [float(v) if v is not None else float("nan") for v in y_vals]
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_numeric, dtype=float)
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if log_y:
            valid &= y_arr > 0.0
        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]

        ax.plot(
            x_arr[valid],
            y_arr[valid],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=0.6,
            label=label,
        )

        boundary_x, boundary_y = _compute_boundary_converged_point(
            x_arr,
            y_arr,
            result.converged,
            log_y=log_y,
        )
        if boundary_x.size > 0:
            boundary_label = "Last converged" if not boundary_legend_added else "_nolegend_"
            boundary_legend_added = True
            ax.scatter(
                boundary_x,
                boundary_y,
                marker="s",
                s=36,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=0.75,
                zorder=5,
                label=boundary_label,
            )

    ax.set_xlabel(x_label or "Parameter value [-]")
    if y_label is not None:
        ax.set_ylabel(y_label)
    elif selected_output == "refueling.t_refuel":
        ax.set_ylabel("refueling.t_refuel [months]")
    else:
        ax.set_ylabel(selected_output)
    if title is not None:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")

    style_axis(ax)
    _apply_horizontal_grid_and_y_ticks(ax, log_y=log_y)
    handles, labels = ax.get_legend_handles_labels()
    legend = legend_axis.legend(
        handles,
        labels,
        title="Sensitivity Cases",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94, wspace=0.05)

    if show:
        plt.show(block=True)

    return fig, ax


def plot_multi_sensitivity_side_by_side(
    multi_result: MultiSensitivityResult,
    right_multi_result: MultiSensitivityResult | None = None,
    *,
    left_output: str = "orbit.altitude",
    right_output: str = "refueling.t_refuel",
    left_log_y: bool = False,
    right_log_y: bool = True,
    left_title: str = "Orbit altitude",
    right_title: str = "Refueling time",
    x_label: str = "Parameter value [-]",
    show: bool = True,
):
    # Plot two outputs side by side with one shared legend.
    PALETTE, apply_validation_style, style_axis, style_legend = _load_plot_style()
    apply_validation_style()

    left_result = multi_result
    right_result = multi_result if right_multi_result is None else right_multi_result

    if left_output not in left_result.output_paths:
        raise KeyError(f"Unknown left_output '{left_output}'. Available: {left_result.output_paths}")
    if right_output not in right_result.output_paths:
        raise KeyError(f"Unknown right_output '{right_output}'. Available: {right_result.output_paths}")

    left_labels = list(left_result.cases.keys())
    right_labels = list(right_result.cases.keys())
    if set(left_labels) != set(right_labels):
        raise ValueError(
            "Left and right sensitivity results must have identical case labels. "
            f"Left={left_labels}, Right={right_labels}"
        )

    fig = plt.figure(figsize=(11.0, 4.5), dpi=150)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.46))
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    legend_axis = fig.add_subplot(grid[0, 2])
    legend_axis.axis("off")

    color_cycle = [
        PALETTE.get("secondary_text", "#4A4A4A"),
        PALETTE.get("l1_teal", "#5BC8D0"),
        PALETTE.get("sernn_pink", "#F08FA7"),
        PALETTE.get("choice_mid", "#C59A4A"),
        PALETTE.get("goal_dark", "#1E7F78"),
        PALETTE.get("cat_purple", "#9A5CB8"),
        PALETTE.get("cat_green", "#76C56E"),
        PALETTE.get("cat_red", "#E85C62"),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<"]

    boundary_present = False
    case_handles: list[Line2D] = []

    for idx, label in enumerate(left_labels):
        left_case = left_result.cases[label]
        right_case = right_result.cases[label]
        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]
        case_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.0,
                marker=marker,
                markersize=5.0,
                markerfacecolor=color,
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=0.6,
                label=label,
            )
        )

        left_x = np.asarray(left_case.variable_values, dtype=float)
        left_y_vals = left_case.outputs[left_output]
        left_y = np.asarray([float(v) if v is not None else float("nan") for v in left_y_vals], dtype=float)
        left_valid = np.isfinite(left_x) & np.isfinite(left_y)
        if left_log_y:
            left_valid &= left_y > 0.0

        ax_left.plot(
            left_x[left_valid],
            left_y[left_valid],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=0.6,
            zorder=2,
        )

        left_boundary_x, left_boundary_y = _compute_boundary_converged_point(
            left_x,
            left_y,
            left_case.converged,
            log_y=left_log_y,
        )
        if left_boundary_x.size > 0:
            boundary_present = True
            ax_left.scatter(
                left_boundary_x,
                left_boundary_y,
                marker="s",
                s=36,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=0.75,
                zorder=5,
            )

        right_x = np.asarray(right_case.variable_values, dtype=float)
        right_y_vals = right_case.outputs[right_output]
        right_y = np.asarray([float(v) if v is not None else float("nan") for v in right_y_vals], dtype=float)
        right_valid = np.isfinite(right_x) & np.isfinite(right_y)
        if right_log_y:
            right_valid &= right_y > 0.0

        ax_right.plot(
            right_x[right_valid],
            right_y[right_valid],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=0.6,
            zorder=2,
        )

        right_boundary_x, right_boundary_y = _compute_boundary_converged_point(
            right_x,
            right_y,
            right_case.converged,
            log_y=right_log_y,
        )
        if right_boundary_x.size > 0:
            boundary_present = True
            ax_right.scatter(
                right_boundary_x,
                right_boundary_y,
                marker="s",
                s=36,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=0.75,
                zorder=5,
            )

    ax_left.set_xlabel(x_label)
    ax_right.set_xlabel(x_label)
    ax_left.set_ylabel("Orbit altitude [km]" if left_output == "orbit.altitude" else left_output)
    ax_right.set_ylabel("Refueling time [months]" if right_output == "refueling.t_refuel" else right_output)
    ax_left.set_title(left_title)
    ax_right.set_title(right_title)

    if left_log_y:
        ax_left.set_yscale("log")
    if right_log_y:
        ax_right.set_yscale("log")

    style_axis(ax_left)
    style_axis(ax_right)
    _apply_horizontal_grid_and_y_ticks(ax_left, log_y=left_log_y)
    _apply_horizontal_grid_and_y_ticks(ax_right, log_y=right_log_y)

    legend_handles: list[Any] = list(case_handles)
    if boundary_present:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=7,
                markerfacecolor=PALETTE.get("cat_red", "#E85C62"),
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=0.8,
                label="Last converged",
            )
        )

    legend = legend_axis.legend(
        handles=legend_handles,
        title="Sensitivity Cases",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.92, wspace=0.22)

    if show:
        plt.show(block=True)

    return fig, (ax_left, ax_right)


if __name__ == "__main__":
    geometry_bar_data = run_geometry_aspect_ratio_sensitivity(
        aspect_ratios=[0.25, 0.5, 1.0, 1.5, 2.0],
        max_iterations=200,
    )
    plot_geometry_aspect_ratio_bars(
        geometry_bar_data,
        show=True,
        title_left="No Refueling (Orbit Altitude)",
        title_right="With Refueling (Refueling Time)",
    )

    sweep_values = [
        0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
        0.95, 1.0,
    ]

    result_altitude = run_efficiency_sensitivities(
        values=sweep_values,
        output_paths=["orbit.altitude"],
    )
    result_refueling = run_efficiency_sensitivities(
        values=sweep_values,
        output_paths=["refueling.t_refuel"],
    )
    plot_multi_sensitivity_side_by_side(
        result_altitude,
        right_multi_result=result_refueling,
        left_output="orbit.altitude",
        right_output="refueling.t_refuel",
        left_log_y=False,
        right_log_y=True,
        left_title="Orbit altitude",
        right_title="Refueling time",
        show=True,
    )
