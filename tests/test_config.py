from pathlib import Path

import pytest

from ml741_assignment3.config import load_yaml, require_keys


def test_load_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 741\nmethod: adaptive\n", encoding="utf-8")
    assert load_yaml(path) == {"seed": 741, "method": "adaptive"}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)


def test_require_keys_reports_missing() -> None:
    with pytest.raises(ValueError, match="method"):
        require_keys({"seed": 741}, "seed", "method")
