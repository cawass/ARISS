from __future__ import annotations

import argparse
import io
import logging
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CORE_CONFIG_DIR = ROOT / "tests" / "Verification" / "configs"
DEFAULT_XLSX = ROOT / "tests" / "Verification" / "core_case_ranges.xlsx"
BASE_CONFIG_PATH = ROOT / "src" / "ariss" / "core" / "base_config.toml"

AR_VALUES = [2.0, 1.0, 0.5]
INTAKE_AREA_RATIO_VALUES = [0.5, 1.0, 2.0]
SHAPES = {
    "CC": ("c", "c"),
    "CR": ("c", "r"),
    "RR": ("r", "r"),
    "RC": ("r", "c"),
}
ATMOS_MODES = {
    "av_atmos": True,
    "fixed_point_00": False,
}
REFUELING_MODES = {
    "off": {"active_refueling": False, "active_and_bypass": False},
    "active_tank_only": {"active_refueling": True, "active_and_bypass": False},
    "active_with_bypass": {"active_refueling": True, "active_and_bypass": True},
}
GEOMETRY_MODES = {
    "free_inlet_fixed_body": {"use_intake_area_ratio": False, "fixed_body": True},
    "fixed_inlet_fixed_body": {"use_intake_area_ratio": True, "fixed_body": True},
    "fixed_inlet_free_body": {"use_intake_area_ratio": True, "fixed_body": False},
}


@dataclass
class CoreCaseSpec:
    filename: str
    name: str
    core_case: str
    shape: str
    ar: float
    intake_area_ratio: float
    atmosphere: str
    refueling: str
    geometry_mode: str
    overrides: dict[str, dict[str, Any]]


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


def _ar_tag(ar: float) -> str:
    mapping = {2.0: "2", 1.0: "1", 0.5: "05"}
    if ar in mapping:
        return mapping[ar]
    return str(ar).replace(".", "p")


def _build_core_case_modes() -> list[dict[str, Any]]:
    modes: list[dict[str, Any]] = []
    core_idx = 1
    for atmos_key, use_average in ATMOS_MODES.items():
        for refuel_key, refuel_mode in REFUELING_MODES.items():
            for geom_key, geom_mode in GEOMETRY_MODES.items():
                core_case = f"core_{core_idx:02d}"
                modes.append(
                    {
                        "core_case": core_case,
                        "atmosphere": atmos_key,
                        "use_average": bool(use_average),
                        "refueling": refuel_key,
                        "refuel_active": bool(refuel_mode["active_refueling"]),
                        "refuel_bypass": bool(refuel_mode["active_and_bypass"]),
                        "geometry_mode": geom_key,
                        "geom_mode": geom_mode,
                    }
                )
                core_idx += 1
    return modes


def _build_core_case_specs() -> list[CoreCaseSpec]:
    specs: list[CoreCaseSpec] = []
    core_modes = _build_core_case_modes()
    for mode in core_modes:
        for ar in AR_VALUES:
            for intake_area_ratio in INTAKE_AREA_RATIO_VALUES:
                for shape_key, (s_in, s_body) in SHAPES.items():
                    ar_key = _ar_tag(ar)
                    iar_key = _ar_tag(intake_area_ratio)
                    filename = (
                        f"case_{mode['core_case']}_{shape_key.lower()}_ar{ar_key}_iar{iar_key}.toml"
                    )
                    name = (
                        f"{mode['core_case'].upper()} {shape_key} AR={ar:g} IAR={intake_area_ratio:g} "
                        f"{mode['atmosphere']} {mode['refueling']} {mode['geometry_mode']}"
                    )
                    overrides = {
                        "orbit": {
                            "use_average": bool(mode["use_average"]),
                            "latitude": 0.0,
                            "longitude": 0.0,
                        },
                        "geometry": {
                            "S_in": s_in,
                            "S_body": s_body,
                            "AR_in": float(ar),
                            "AR_body": float(ar),
                            "intake_area_ratio": float(intake_area_ratio),
                            "use_intake_area_ratio": bool(mode["geom_mode"]["use_intake_area_ratio"]),
                            "fixed_body": bool(mode["geom_mode"]["fixed_body"]),
                        },
                        "mission_profile": {
                            "active_refueling": bool(mode["refuel_active"]),
                            "active_and_bypass": bool(mode["refuel_bypass"]),
                        },
                    }
                    specs.append(
                        CoreCaseSpec(
                            filename=filename,
                            name=name,
                            core_case=str(mode["core_case"]),
                            shape=shape_key,
                            ar=float(ar),
                            intake_area_ratio=float(intake_area_ratio),
                            atmosphere=str(mode["atmosphere"]),
                            refueling=str(mode["refueling"]),
                            geometry_mode=str(mode["geometry_mode"]),
                            overrides=overrides,
                        )
                    )
    return specs


def _render_case_toml(spec: CoreCaseSpec) -> str:
    lines: list[str] = []
    lines.append(f'name = "{spec.name}"')
    lines.append("")
    section_order = ["orbit", "geometry", "mission_profile"]
    for section in section_order:
        payload = spec.overrides[section]
        lines.append(f"[{section}]")
        for key, value in payload.items():
            lines.append(f"{key} = {_toml_literal(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_core_case_files(output_dir: Path) -> list[CoreCaseSpec]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("case_core_*.toml"):
        old_path.unlink()
    specs = _build_core_case_specs()
    for spec in specs:
        case_path = output_dir / spec.filename
        case_path.write_text(_render_case_toml(spec), encoding="utf-8")
    return specs


def _safe_range(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(min(values)), float(max(values))


def _format_range(min_value: float | None, max_value: float | None, *, digits: int = 6) -> str | None:
    if min_value is None or max_value is None:
        return None
    return f"{min_value:.{digits}g} to {max_value:.{digits}g}"


def run_core_cases(
    specs: list[CoreCaseSpec],
    *,
    config_dir: Path,
    max_iterations: int,
    mass_tolerance: float,
) -> list[dict[str, Any]]:
    from ariss.core.simulation import logger as simulation_logger
    from ariss.core.simulation import run_sizing_loop
    from ariss.core.spacecraft import SpacecraftState

    rows: list[dict[str, Any]] = []

    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)

    try:
        for spec in specs:
            case_path = config_dir / spec.filename
            sc = SpacecraftState.from_toml(BASE_CONFIG_PATH)
            sc.update_from_toml(case_path)

            try:
                with redirect_stdout(io.StringIO()):
                    final_sc, converged, history = run_sizing_loop(
                        sc,
                        max_iterations=max_iterations,
                        mass_tolerance=mass_tolerance,
                    )

                if converged:
                    # Use only final converged values per case (no initial/history values).
                    altitude_val = float(final_sc.orbit.altitude)
                    mass_val = float(final_sc.mass.Mass_total)
                    isp_val = float(final_sc.thruster.specific_impulse)
                    thrust_val = float(final_sc.thruster.thrust)

                    altitude_min = altitude_max = altitude_val
                    mass_min = mass_max = mass_val
                    isp_min = isp_max = isp_val
                    thrust_min = thrust_max = thrust_val
                    error_text = ""
                else:
                    altitude_min = altitude_max = None
                    mass_min = mass_max = None
                    isp_min = isp_max = None
                    thrust_min = thrust_max = None
                    error_text = "Not converged"

                rows.append(
                    {
                        "config_file": spec.filename,
                        "case_name": spec.name,
                        "core_case": spec.core_case,
                        "shape": spec.shape,
                        "ar": spec.ar,
                        "intake_area_ratio": spec.intake_area_ratio,
                        "atmosphere": spec.atmosphere,
                        "refueling": spec.refueling,
                        "geometry_mode": spec.geometry_mode,
                        "converged": bool(converged),
                        "iterations": max(len(history) - 1, 0),
                        "altitude_min_km": altitude_min,
                        "altitude_max_km": altitude_max,
                        "mass_min_kg": mass_min,
                        "mass_max_kg": mass_max,
                        "isp_min_s": isp_min,
                        "isp_max_s": isp_max,
                        "thrust_min_n": thrust_min,
                        "thrust_max_n": thrust_max,
                        "error": error_text,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "config_file": spec.filename,
                        "case_name": spec.name,
                        "core_case": spec.core_case,
                        "shape": spec.shape,
                        "ar": spec.ar,
                        "intake_area_ratio": spec.intake_area_ratio,
                        "atmosphere": spec.atmosphere,
                        "refueling": spec.refueling,
                        "geometry_mode": spec.geometry_mode,
                        "converged": False,
                        "iterations": 0,
                        "altitude_min_km": None,
                        "altitude_max_km": None,
                        "mass_min_kg": None,
                        "mass_max_kg": None,
                        "isp_min_s": None,
                        "isp_max_s": None,
                        "thrust_min_n": None,
                        "thrust_max_n": None,
                        "error": str(exc),
                    }
                )
    finally:
        simulation_logger.setLevel(previous_level)

    return rows


def write_excel(rows: list[dict[str, Any]], xlsx_path: Path) -> None:
    from openpyxl import Workbook

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    ws_core = workbook.active
    ws_core.title = "core_case_ranges"
    core_headers = [
        "core_case",
        "atmosphere",
        "refueling",
        "geometry_mode",
        "num_cases",
        "num_converged",
        "altitude_range_km",
        "mass_range_kg",
        "isp_range_s",
        "thrust_range_n",
    ]
    ws_core.append(core_headers)

    mode_map: dict[str, dict[str, Any]] = {
        str(mode["core_case"]): mode for mode in _build_core_case_modes()
    }
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in mode_map}
    for row in rows:
        core_key = str(row.get("core_case", ""))
        if core_key in grouped:
            grouped[core_key].append(row)

    for core_key in sorted(mode_map.keys()):
        meta = mode_map[core_key]
        items = grouped.get(core_key, [])
        converged_items = [item for item in items if bool(item.get("converged"))]

        altitude_mins = [
            float(item["altitude_min_km"])
            for item in converged_items
            if item.get("altitude_min_km") is not None
        ]
        altitude_maxs = [
            float(item["altitude_max_km"])
            for item in converged_items
            if item.get("altitude_max_km") is not None
        ]
        mass_mins = [
            float(item["mass_min_kg"])
            for item in converged_items
            if item.get("mass_min_kg") is not None
        ]
        mass_maxs = [
            float(item["mass_max_kg"])
            for item in converged_items
            if item.get("mass_max_kg") is not None
        ]
        isp_mins = [
            float(item["isp_min_s"])
            for item in converged_items
            if item.get("isp_min_s") is not None
        ]
        isp_maxs = [
            float(item["isp_max_s"])
            for item in converged_items
            if item.get("isp_max_s") is not None
        ]
        thrust_mins = [
            float(item["thrust_min_n"])
            for item in converged_items
            if item.get("thrust_min_n") is not None
        ]
        thrust_maxs = [
            float(item["thrust_max_n"])
            for item in converged_items
            if item.get("thrust_max_n") is not None
        ]

        ws_core.append(
            [
                core_key,
                meta.get("atmosphere"),
                meta.get("refueling"),
                meta.get("geometry_mode"),
                len(items),
                len(converged_items),
                _format_range(
                    min(altitude_mins) if altitude_mins else None,
                    max(altitude_maxs) if altitude_maxs else None,
                ),
                _format_range(
                    min(mass_mins) if mass_mins else None,
                    max(mass_maxs) if mass_maxs else None,
                ),
                _format_range(
                    min(isp_mins) if isp_mins else None,
                    max(isp_maxs) if isp_maxs else None,
                ),
                _format_range(
                    min(thrust_mins) if thrust_mins else None,
                    max(thrust_maxs) if thrust_maxs else None,
                ),
            ]
        )

    ws_core.freeze_panes = "A2"
    ws_core.auto_filter.ref = ws_core.dimensions

    workbook.save(xlsx_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate core verification cases (AR, intake/body ratio, shape, atmosphere, refueling, geometry) "
            "and export altitude/mass/Isp ranges to Excel."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CORE_CONFIG_DIR,
        help="Directory where core TOML cases are written.",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=DEFAULT_XLSX,
        help="Output Excel path.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Max iterations for sizing loop.",
    )
    parser.add_argument(
        "--mass-tolerance",
        type=float,
        default=1e-3,
        help="Mass convergence tolerance for sizing loop.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate TOML cases, skip simulation and Excel export.",
    )
    args = parser.parse_args()

    specs = generate_core_case_files(args.config_dir)
    print(f"Generated {len(specs)} core TOML cases in: {args.config_dir}")

    if args.generate_only:
        return

    rows = run_core_cases(
        specs,
        config_dir=args.config_dir,
        max_iterations=args.max_iterations,
        mass_tolerance=args.mass_tolerance,
    )
    write_excel(rows, args.xlsx)
    print(f"Wrote Excel: {args.xlsx}")
    print(f"Rows: {len(rows)} | Converged: {sum(bool(r['converged']) for r in rows)}")


if __name__ == "__main__":
    main()
