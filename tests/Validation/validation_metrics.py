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
#      Shared validation metrics for datapoint-based comparisons.
#
#  Project:        ARISS
#  Module:         validation_metrics.py
# ============================================================================== #

from __future__ import annotations

from typing import Sequence

import numpy as np


# (max_relative_error, mean_relative_error, line_of_max_relative_error, n_rel, pearson_r, n_corr)
DatapointStats = tuple[float, float, int, int, float, int]


def paired_model_reference_samples(
    model_x: np.ndarray | Sequence[float],
    model_y: np.ndarray | Sequence[float],
    ref_x: np.ndarray | Sequence[float],
    ref_y: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model_x_arr = np.asarray(model_x, dtype=float)
    model_y_arr = np.asarray(model_y, dtype=float)
    ref_x_arr = np.asarray(ref_x, dtype=float)
    ref_y_arr = np.asarray(ref_y, dtype=float)

    valid_model = np.isfinite(model_x_arr) & np.isfinite(model_y_arr)
    if np.count_nonzero(valid_model) < 2:
        return None

    model_x_arr = model_x_arr[valid_model]
    model_y_arr = model_y_arr[valid_model]
    model_order = np.argsort(model_x_arr)
    model_x_arr = model_x_arr[model_order]
    model_y_arr = model_y_arr[model_order]

    model_x_unique, unique_idx = np.unique(model_x_arr, return_index=True)
    model_y_unique = model_y_arr[unique_idx]
    if len(model_x_unique) < 2:
        return None

    line_ids = np.arange(1, len(ref_y_arr) + 1, dtype=int)
    valid_ref = np.isfinite(ref_x_arr) & np.isfinite(ref_y_arr)
    if not np.any(valid_ref):
        return None

    ref_x_arr = ref_x_arr[valid_ref]
    ref_y_arr = ref_y_arr[valid_ref]
    line_ids = line_ids[valid_ref]

    x_low = float(np.min(model_x_unique))
    x_high = float(np.max(model_x_unique))
    in_range = (ref_x_arr >= x_low) & (ref_x_arr <= x_high)
    if not np.any(in_range):
        return None

    ref_x_arr = ref_x_arr[in_range]
    ref_y_arr = ref_y_arr[in_range]
    line_ids = line_ids[in_range]

    model_at_ref = np.interp(ref_x_arr, model_x_unique, model_y_unique)
    return model_at_ref, ref_y_arr, line_ids


def datapoint_relative_and_corr_stats(
    model_x: np.ndarray | Sequence[float],
    model_y: np.ndarray | Sequence[float],
    ref_x: np.ndarray | Sequence[float],
    ref_y: np.ndarray | Sequence[float],
) -> DatapointStats | None:
    paired = paired_model_reference_samples(model_x, model_y, ref_x, ref_y)
    if paired is None:
        return None

    model_at_ref, ref_y_used, line_ids = paired
    nonzero_ref = np.abs(ref_y_used) > 1.0e-12

    if np.any(nonzero_ref):
        relative_error = np.abs(model_at_ref[nonzero_ref] - ref_y_used[nonzero_ref]) / np.abs(ref_y_used[nonzero_ref])
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

    if len(model_at_ref) >= 2:
        pearson_r = float(np.corrcoef(model_at_ref, ref_y_used)[0, 1])
    else:
        pearson_r = float("nan")

    return (
        max_relative_error,
        mean_relative_error,
        max_rel_line,
        n_rel,
        pearson_r,
        int(len(model_at_ref)),
    )


def datapoint_relative_and_corr_stats_xy(
    model_xy: tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]],
    ref_xy: tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]],
) -> DatapointStats | None:
    return datapoint_relative_and_corr_stats(model_xy[0], model_xy[1], ref_xy[0], ref_xy[1])


def minimum_finite(values: Sequence[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    return float(min(finite))


def mse_summary(values: np.ndarray | Sequence[float] | float) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.min(finite)), float(np.max(finite)), float(np.mean(finite)), int(finite.size)
