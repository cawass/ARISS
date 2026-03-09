from __future__ import annotations

import argparse

from ariss.visualization.history_ui import plot_simulation_history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ariss.visualization")
    parser.add_argument("spacecraft", nargs="?", help="Path to a spacecraft TOML or JSON file")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--mass-tolerance", type=float, default=1.0e-3)
    args = parser.parse_args(argv)

    plot_simulation_history(
        sc=args.spacecraft,
        max_iterations=args.max_iterations,
        mass_tolerance=args.mass_tolerance,
        show=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
