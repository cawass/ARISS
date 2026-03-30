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
#      Shared CSV extraction helpers for validation datasets. Supports the
#      wide XY format used across GOCEE, Crandall-Wirz, and Mansur cases.
#
#  Project:        ARISS
#  Module:         csv_helper.py
#  Author:         Carlos Carrasco Requejo, Lucas Calderon del Rio
# ============================================================================ #
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

WideXYDict = dict[Any, tuple[np.ndarray, np.ndarray]]


def extract_first_number(text: str | None) -> float | None:
    if text is None:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    return float(matches[0]) if matches else None


def extract_value_after_token(text: str | None, token: str = "=") -> float | None:
    if text is None:
        return None
    content = str(text)
    if token not in content:
        return None
    try:
        return float(content.split(token, 1)[1].strip())
    except (TypeError, ValueError):
        return extract_first_number(content)


def load_wide_xy_csv(
    path: str | Path,
    *,
    label_parser: Callable[[str], Any | None] | None = None,
    label_transform: Callable[[Any], Any] | None = None,
    data_start_row: int = 2,
    min_rows: int = 3,
    sort_by: str = "x",
    encoding: str = "utf-8-sig",
) -> WideXYDict:
    # Inputs:
    #   path: CSV file path.
    #   label_parser: optional parser applied to each header label.
    #   label_transform: optional transform applied after parsing.
    #   data_start_row: row index where numeric data begins.
    #   min_rows: minimum row count required.
    #   sort_by: "x", "y", or "none".
    #   encoding: file encoding.
    #
    # Output:
    #   Mapping from parsed label -> (x_array, y_array).
    csv_path = Path(path)
    with csv_path.open("r", encoding=encoding, newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < int(min_rows):
        raise ValueError(f"Dataset {csv_path} has insufficient rows.")

    if sort_by not in {"x", "y", "none"}:
        raise ValueError("sort_by must be 'x', 'y', or 'none'.")

    header = rows[0]
    curves: WideXYDict = {}

    for column in range(0, len(header), 2):
        raw_label = header[column].strip() if column < len(header) else ""
        if not raw_label:
            continue

        key: Any = raw_label
        if label_parser is not None:
            key = label_parser(raw_label)
        if key is None:
            continue
        if label_transform is not None:
            key = label_transform(key)

        x_vals: list[float] = []
        y_vals: list[float] = []

        for row in rows[data_start_row:]:
            if column + 1 >= len(row):
                continue

            x_text = str(row[column]).strip()
            y_text = str(row[column + 1]).strip()
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

        if not x_vals:
            continue

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)

        if sort_by == "x":
            order = np.argsort(x_arr)
            x_arr = x_arr[order]
            y_arr = y_arr[order]
        elif sort_by == "y":
            order = np.argsort(y_arr)
            x_arr = x_arr[order]
            y_arr = y_arr[order]

        curves[key] = (x_arr, y_arr)

    return curves


def split_labeled_contours(
    series: Mapping[Any, tuple[np.ndarray, np.ndarray]],
    *,
    contour_prefix: str = "h",
    solution_token: str = "solution",
    contour_parser: Callable[[str], float | None] | None = None,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    # Inputs:
    #   series: label -> (x, y) data.
    #   contour_prefix: labels starting with this token are treated as contours.
    #   solution_token: labels containing this token are treated as solution.
    #   contour_parser: parser to extract contour level from label.
    #
    # Output:
    #   (contour_dict, solution_curve).
    parser = extract_first_number if contour_parser is None else contour_parser
    contours: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    solution = (np.asarray([], dtype=float), np.asarray([], dtype=float))

    contour_prefix_l = contour_prefix.lower()
    solution_token_l = solution_token.lower()

    for key, (x_vals, y_vals) in series.items():
        label = str(key).strip()
        label_l = label.lower()
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)

        if label_l.startswith(contour_prefix_l):
            level = parser(label)
            if level is not None:
                contours[float(level)] = (x_arr, y_arr)
        elif solution_token_l in label_l:
            solution = (x_arr, y_arr)

    return contours, solution


__all__ = [
    "WideXYDict",
    "extract_first_number",
    "extract_value_after_token",
    "load_wide_xy_csv",
    "split_labeled_contours",
]

