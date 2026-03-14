from __future__ import annotations

from os import PathLike
from pathlib import Path

from ariss.core.spacecraft import SpacecraftState


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = ROOT / "src" / "ariss" / "core" / "base_config.toml"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def verification_case_paths(
    config_dir: Path | None = None,
    *,
    include_no_modifications: bool = False,
) -> list[Path]:
    # Inputs:
    #   config_dir: optional directory containing verification TOML files.
    #   include_no_modifications: include explicit no-modification probe cases.
    #
    # Output:
    #   Sorted list of verification case TOMLs.

    case_dir = CONFIG_DIR if config_dir is None else Path(config_dir)
    candidates: list[Path] = []
    for pattern in ("drag_*.toml", "case_*.toml"):
        candidates.extend(case_dir.glob(pattern))
    if not candidates:
        candidates.extend(case_dir.glob("*.toml"))

    if not include_no_modifications:
        candidates = [path for path in candidates if "no_modifications" not in path.name.lower()]

    unique: dict[str, Path] = {str(path.resolve()): path for path in candidates}
    return sorted(unique.values(), key=lambda path: path.name.lower())


def build_spacecraft_from_case(
    case_path: str | PathLike[str],
    *,
    base_config_path: str | PathLike[str] = BASE_CONFIG_PATH,
) -> SpacecraftState:
    # Inputs:
    #   case_path: case override TOML path.
    #   base_config_path: base spacecraft TOML path.
    #
    # Output:
    #   Spacecraft loaded from base config then updated with case overrides.

    sc = SpacecraftState.from_toml(base_config_path)
    sc.update_from_toml(case_path)
    return sc


__all__ = [
    "BASE_CONFIG_PATH",
    "build_spacecraft_from_case",
    "verification_case_paths",
]

