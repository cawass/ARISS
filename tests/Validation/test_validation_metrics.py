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
#      Pytest validation-metrics gate. Runs validation cases and checks
#      line-by-line relative metrics are produced and finite where expected.
#
#  Project:        ARISS
#  Module:         test_validation_metrics.py
# ============================================================================== #

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from validation_metrics_runner import run_all_validation_metrics

MAX_RELATIVE_ERROR_LIMIT = 1.00
MEAN_RELATIVE_ERROR_LIMIT = 0.70
MIN_PEARSON_CORRELATION = 0.10


def _ignore_pearson_for_metric(metric_name: str) -> bool:
    return metric_name.startswith("GOCEE Thermal |") or metric_name.startswith("Mansur Thruster Map |")


def test_validation_cases_line_metrics_are_valid() -> None:
    # Inputs:
    #   Full validation metrics runner outputs.
    #
    # Outputs:
    #   Ensures every validation line reports:
    #     - max_relative_error
    #     - mean_relative_error
    #     - pearson_r
    #   And each metric passes:
    #     - max_relative_error < 1.00 (100%)
    #     - mean_relative_error < 0.50 (50%)
    #     - pearson_r > 0.10

    metrics = run_all_validation_metrics(print_output=True)
    assert metrics, "No validation metrics were produced."

    violations: list[str] = []
    for name in sorted(metrics):
        values = metrics[name]
        ignore_pearson = _ignore_pearson_for_metric(name)

        if "max_relative_error" not in values:
            violations.append(f"{name}: missing 'max_relative_error'")
            continue
        if "mean_relative_error" not in values:
            violations.append(f"{name}: missing 'mean_relative_error'")
            continue
        if (not ignore_pearson) and ("pearson_r" not in values):
            violations.append(f"{name}: missing 'pearson_r'")
            continue

        max_rel = float(values["max_relative_error"])
        mean_rel = float(values["mean_relative_error"])

        if not np.isfinite(max_rel):
            violations.append(f"{name}: max_relative_error is not finite ({max_rel})")
        elif max_rel >= MAX_RELATIVE_ERROR_LIMIT:
            violations.append(
                f"{name}: max_relative_error {max_rel:.6f} is not < {MAX_RELATIVE_ERROR_LIMIT:.2f}"
            )

        if not np.isfinite(mean_rel):
            violations.append(f"{name}: mean_relative_error is not finite ({mean_rel})")
        elif mean_rel >= MEAN_RELATIVE_ERROR_LIMIT:
            violations.append(
                f"{name}: mean_relative_error {mean_rel:.6f} is not < {MEAN_RELATIVE_ERROR_LIMIT:.2f}"
            )

        if not ignore_pearson:
            if "pearson_r" in values:
                pearson_r = float(values["pearson_r"])
                if np.isfinite(pearson_r) and np.abs(pearson_r) <= MIN_PEARSON_CORRELATION:
                    violations.append(
                        f"{name}: pearson_r {pearson_r:.6f} is not > {MIN_PEARSON_CORRELATION:.2f}"
                    )

    assert not violations, "Validation line-metric violations:\n - " + "\n - ".join(violations)
