"""Sanity checks for the Phase 0 package scaffold."""

from ml741_assignment3 import __version__
from ml741_assignment3.cli import build_parser


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_name() -> None:
    assert build_parser().prog == "ml741-a3"
