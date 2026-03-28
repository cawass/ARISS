from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.sensitivity import SensitivityRankingItem, run_sensitivity_ranking

PARAMETERS: list[tuple[str, str]] = [
    ("eta_solar", "solar.eta_solar"),
    ("eta_prop", "thruster.eff"),
    ("eta_coll", "refueling.coll_eff"),
    ("eta_ref", "refueling.eta_refuel"),
    ("eta_elec", "solar.eta_power"),
    ("epsilon", "geometry.epsilon_body"),
    ("P_prop", "thruster.power"),
    ("I_sp", "thruster.specific_impulse"),
    ("chi", "geometry.intake_area_ratio"),
    ("AR_in", "geometry.AR_in"),
    ("AR_solar", "geometry.AR_solar"),
    ("T_des", "thermal.T_des"),
]

OUTPUT_H = "orbit.altitude"
OUTPUT_TREF = "refueling.t_refuel"

DEFAULT_CASES = [
    "tests/Validation/CrandallWirz2022-Drag_Simplified_Trust/CrandallWirz2022_3U.toml",
    "tests/Validation/CrandallWirz2022-Drag_Simplified_Trust/CrandallWirz2022_6U.toml",
]


def _to_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric and numeric not in (float("inf"), float("-inf")) else None


def _central_derivative(item: SensitivityRankingItem, output_path: str) -> float | None:
    x_minus = _to_finite_float(item.minus_value)
    x_plus = _to_finite_float(item.plus_value)
    y_minus = _to_finite_float(item.outputs_minus.get(output_path))
    y_plus = _to_finite_float(item.outputs_plus.get(output_path))

    if x_minus is None or x_plus is None or y_minus is None or y_plus is None:
        return None

    denominator = x_plus - x_minus
    if abs(denominator) <= 1.0e-20:
        return None

    return (y_plus - y_minus) / denominator


def _format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6e}"


def _escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def _sanitize_latex_label(text: str) -> str:
    sanitized = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in text.lower())
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "sensitivity_case"


def _resolve_case_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(rows: list[dict[str, Any]], case_name: str, output_path: Path) -> None:
    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \begin{tabular}{lcc}")
    lines.append(r"        \toprule")
    lines.append(r"        Input & $S_h$ & $S_{t_{ref}}$ \\")
    lines.append(r"        \midrule")
    for row in rows:
        label = _escape_latex(str(row["input"]))
        sh = str(row["S_h"])
        st = str(row["S_t_ref"])
        lines.append(f"        {label} & {sh} & {st} \\\\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    label_key = _sanitize_latex_label(case_name)
    lines.append(f"    \\caption{{10\\% perturbation sensitivities for {_escape_latex(case_name)}.}}")
    lines.append(rf"    \label{{tab:sensitivity_{label_key}}}")
    lines.append(r"\end{table}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(
    case_path: Path | None,
    output_dir: Path,
    perturbation: float,
    max_iterations: int,
    *,
    name_override: str | None = None,
) -> tuple[Path, Path]:
    parameter_defs = [{"label": label, "variable_path": path} for label, path in PARAMETERS]

    with redirect_stdout(io.StringIO()):
        ranking = run_sensitivity_ranking(
            parameters=parameter_defs,
            output_paths=[OUTPUT_H, OUTPUT_TREF],
            perturbation=perturbation,
            case_path=case_path,
            max_iterations=max_iterations,
        )

    rows: list[dict[str, Any]] = []
    for item in ranking.items:
        sh_raw = _central_derivative(item, OUTPUT_H)
        st_raw = _central_derivative(item, OUTPUT_TREF)

        rows.append(
            {
                "input": item.label,
                "path": item.variable_path,
                "base_value": item.base_value,
                "minus_value": item.minus_value,
                "plus_value": item.plus_value,
                "S_h": _format_value(sh_raw),
                "S_t_ref": _format_value(st_raw),
                "S_h_normalized": _format_value(_to_finite_float(item.s_central.get(OUTPUT_H))),
                "S_t_ref_normalized": _format_value(_to_finite_float(item.s_central.get(OUTPUT_TREF))),
                "converged_minus": item.converged_minus,
                "converged_plus": item.converged_plus,
                "error_minus": item.error_minus or "",
                "error_plus": item.error_plus or "",
            }
        )

    if name_override is not None and name_override.strip():
        stem = name_override.strip()
    elif case_path is None:
        stem = "core_base"
    else:
        stem = case_path.stem
    csv_path = output_dir / f"{stem}_10pct_sensitivity.csv"
    tex_path = output_dir / f"{stem}_10pct_sensitivity.tex"
    _write_csv(rows, csv_path)
    _write_latex(rows, stem, tex_path)
    return csv_path, tex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 10 percent sensitivity table using ariss.core.sensitivity."
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=DEFAULT_CASES,
        help="Case TOML paths (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--core-base",
        action="store_true",
        help="Run from core baseline (no case_path override), matching sensitivity.py defaults.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/sensitivity/results",
        help="Output directory for CSV and LaTeX tables.",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.10,
        help="Relative perturbation (0.10 means +/-10 percent).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=260,
        help="Maximum iterations for each sizing loop solve.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("ariss").setLevel(logging.WARNING)
    logging.getLogger("ariss.core.simulation").setLevel(logging.WARNING)

    output_dir = _resolve_case_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[Path, Path]] = []
    if bool(args.core_base):
        generated.append(
            run_case(
                case_path=None,
                output_dir=output_dir,
                perturbation=float(args.perturbation),
                max_iterations=int(args.max_iterations),
                name_override="core_base",
            )
        )
    else:
        for raw_case in args.cases:
            case_path = _resolve_case_path(raw_case)
            if not case_path.exists():
                raise FileNotFoundError(f"Case file not found: {case_path}")
            generated.append(
                run_case(
                    case_path=case_path,
                    output_dir=output_dir,
                    perturbation=float(args.perturbation),
                    max_iterations=int(args.max_iterations),
                )
            )

    for csv_path, tex_path in generated:
        print(f"[sensitivity] wrote {csv_path}")
        print(f"[sensitivity] wrote {tex_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
