from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from ariss.core.spacecraft import load_spacecraft


def _simulation_summary(spacecraft, converged: bool, history) -> dict[str, object]:
    return {
        "converged": converged,
        "iterations": len(history),
        "mass_total": spacecraft.mass.Mass_total,
        "altitude": spacecraft.orbit.altitude,
        "density": spacecraft.orbit.density,
        "velocity": spacecraft.orbit.velocity,
        "drag_total": spacecraft.drag.drag_total,
        "power_total": spacecraft.power.Power_total,
        "thrust": spacecraft.thruster.thrust,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ariss")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ui_parser = subparsers.add_parser("ui", help="Launch the history UI")
    ui_parser.add_argument("spacecraft", nargs="?", help="Path to a spacecraft TOML or JSON file")
    ui_parser.add_argument("--max-iterations", type=int, default=200)
    ui_parser.add_argument("--mass-tolerance", type=float, default=1.0e-3)

    sim_parser = subparsers.add_parser("sim", help="Run the sizing simulation")
    sim_parser.add_argument("spacecraft", nargs="?", help="Path to a spacecraft TOML or JSON file")
    sim_parser.add_argument("--max-iterations", type=int, default=200)
    sim_parser.add_argument("--mass-tolerance", type=float, default=1.0e-3)
    sim_parser.add_argument("--json", action="store_true", help="Print the simulation summary as JSON")

    spacecraft_parser = subparsers.add_parser("spacecraft", help="Load and print a spacecraft file")
    spacecraft_parser.add_argument("spacecraft", help="Path to a spacecraft TOML or JSON file")

    args = parser.parse_args(argv)

    if args.command == "ui":
        from ariss.visualization.history_ui import plot_simulation_history

        plot_simulation_history(
            sc=args.spacecraft,
            max_iterations=args.max_iterations,
            mass_tolerance=args.mass_tolerance,
            show=True,
        )
        return 0

    if args.command == "sim":
        from ariss import run_simulation

        spacecraft, converged, history = run_simulation(
            sc=args.spacecraft,
            max_iterations=args.max_iterations,
            mass_tolerance=args.mass_tolerance,
        )
        summary = _simulation_summary(spacecraft, converged, history)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(
                "Simulation complete | "
                f"converged={summary['converged']} | "
                f"iterations={summary['iterations']} | "
                f"mass_total={summary['mass_total']:.6f} kg | "
                f"altitude={summary['altitude']:.6f} km | "
                f"density={summary['density']:.6e} kg/m^3 | "
                f"drag_total={summary['drag_total']:.6e} N | "
                f"power_total={summary['power_total']:.6f} W"
            )
        return 0

    if args.command == "spacecraft":
        spacecraft = load_spacecraft(args.spacecraft)
        print(json.dumps(asdict(spacecraft), indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
