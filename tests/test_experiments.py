"""Cost accounting and protocol guard tests."""

import json

import pytest

from ml741_assignment3.artifacts import file_sha256
from ml741_assignment3.experiments import training_work, verify_lock


def test_work_counts_short_minibatches_and_rehearsal():
    result = training_work(
        [{"seen_classes": [0, 1]}, {"seen_classes": [0, 1, 2]}], [7, 10, 12], 8
    )
    assert result == {
        "gradient_updates": 7,
        "training_example_presentations": 46,
        "training_epochs": 2,
    }


def test_changed_protocol_rejected_before_data_loading(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    target = folder / "final.yaml"
    target.write_text("status: locked\ndatasets: {}\n")
    lock = {"sha256": {"configs/final.yaml": file_sha256(target)}}
    (folder / "protocol_lock.json").write_text(json.dumps(lock))
    config, _ = verify_lock(tmp_path)
    assert config["status"] == "locked"
    target.write_text("status: unlocked\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_lock(tmp_path)
