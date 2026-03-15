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
#      Run all verification sweep cases and export results directly to Excel.
#
#  Project:        ARISS
#  Module:         create_sweep_excel.py
# ============================================================================== #

from __future__ import annotations

import argparse
import io
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = (
    Path(__file__).parent / "configs"
    if (Path(__file__).parent / "configs").exists()
    else ROOT / "tests" / "Verification" / "configs"
)
XLSX_PATH = Path(__file__).parent / "sweep_results.xlsx"


def run_sweep_cases(config_dir: Path, max_iterations: int, mass_tolerance: float):

    src = ROOT / "src"
    sys.path[:0] = [str(ROOT), str(src)]

    from ariss.core.simulation import logger as simulation_logger
    from ariss.core.simulation import run_sizing_loop
    from ariss.core.spacecraft import SpacecraftState

    configs = sorted(config_dir.glob("*.toml"))
    if not configs:
        raise FileNotFoundError(f"No TOML configs found in: {config_dir}")

    rows = []

    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)

    try:
        for config in configs:

            sc = SpacecraftState.from_toml(ROOT / "src" / "ariss" / "core" / "base_config.toml")
            sc.update_from_toml(config)

            with redirect_stdout(io.StringIO()):
                final_sc, converged, history = run_sizing_loop(
                    sc,
                    max_iterations=max_iterations,
                    mass_tolerance=mass_tolerance,
                )

            rho = final_sc.orbit.density
            v = final_sc.orbit.velocity
            drag = final_sc.drag.drag_total
            rows.append(
                dict(
                    config_file=config.name,
                    case_name=str(getattr(final_sc, "name", "")).strip(),
                    converged=converged,
                    solution_status="Converged" if converged else "Not converged",
                    iterations=max(len(history) - 1, 0),
                    power_total_w=final_sc.power.Power_total,
                    mass_total_kg=final_sc.mass.Mass_total,
                    orbital_altitude_km=final_sc.orbit.altitude,
                    drag=final_sc.drag.drag_total,
                    thrust_n=final_sc.thruster.thrust,
                    thruster_mass_flow_kg_s=final_sc.thruster.m_flow,
                    refueling_mass_flow_kg_s=final_sc.refueling.m_flow,
                )
            )

    finally:
        simulation_logger.setLevel(previous_level)

    return rows


def write_excel(rows: list[dict[str, object]], xlsx_path: Path, sheet_name: str = "sweep_results"):

    from openpyxl import Workbook

    if not rows:
        raise ValueError("No results to write.")

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name

    headers = list(rows[0].keys())
    worksheet.append(headers)

    for row in rows:
        worksheet.append([row[h] for h in headers])

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    workbook.save(xlsx_path)


def main():

    parser = argparse.ArgumentParser(
        description="Run sweep cases and export results directly to Excel."
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Directory containing sweep TOML configs",
    )

    parser.add_argument(
        "--xlsx",
        type=Path,
        default=XLSX_PATH,
        help="Output Excel path",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=120,
        help="Max iterations for sizing loop",
    )

    parser.add_argument(
        "--mass-tolerance",
        type=float,
        default=1e-3,
        help="Convergence tolerance",
    )

    args = parser.parse_args()

    rows = run_sweep_cases(
        args.config_dir,
        args.max_iterations,
        args.mass_tolerance,
    )

    write_excel(rows, args.xlsx)

    converged_count = sum(r["converged"] for r in rows)

    print(f"Wrote Excel file: {args.xlsx}")
    print(f"Rows: {len(rows)} | Converged: {converged_count}")


if __name__ == "__main__":
    main()
