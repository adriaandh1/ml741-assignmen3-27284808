"""Command-line entry point for the Phase 0 project scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml741_assignment3 import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml741-a3",
        description="Run reproducible ML741 Assignment 3 experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    smoke = subparsers.add_parser("smoke", help="run the synthetic end-to-end smoke test")
    smoke.add_argument(
        "--output",
        type=Path,
        default=Path("results-smoke"),
        help="artifact directory (default: results-smoke)",
    )
    smoke.add_argument("--seed", type=int, default=None, help="override the configuration seed")
    smoke.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke.yaml"),
        help="smoke configuration (default: configs/smoke.yaml)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.command == "smoke":
        from ml741_assignment3.smoke import run_smoke

        results = run_smoke(
            arguments.output,
            seed=arguments.seed,
            config_path=arguments.config,
        )
        for result in results:
            score = float(result.metrics["macro_f1"])
            print(
                f"{result.method}: macro_f1={score:.4f}, "
                f"hidden_units={result.final_hidden_units}"
            )
        print(f"artifacts: {arguments.output.resolve()}")
        return 0
    if arguments.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
