import json
from pathlib import Path

import numpy as np
import torch

from ml741_assignment3.artifacts import (
    file_sha256,
    load_model_checkpoint,
    save_run_bundle,
)
from ml741_assignment3.model import AdditiveGrowingNetwork
from ml741_assignment3.training import RunResult


def fake_result() -> RunResult:
    return RunResult(
        method="test_method",
        seed=741,
        class_order=[1, 0],
        metrics={"macro_f1": 0.5, "per_class": [], "confusion_matrix": [[1, 0], [1, 0]]},
        epoch_records=[{"epoch": 1, "seen_classes": [1, 0]}],
        stage_records=[],
        events=[{"event": "test"}],
        final_hidden_units=0,
        parameter_count=4,
        runtime_seconds=0.1,
        targets=np.asarray([0, 1], dtype=np.int64),
        predictions=np.asarray([0, 0], dtype=np.int64),
        model_state={"weight": torch.ones(2, 2)},
        model_shape={
            "input_features": 1,
            "output_classes": 2,
            "hidden_units": 0,
            "activation": "tanh",
        },
    )


def test_save_run_bundle_is_complete(tmp_path: Path) -> None:
    destination = save_run_bundle(fake_result(), tmp_path, configuration={"seed": 741})
    expected = {
        "summary.json",
        "configuration.json",
        "environment.json",
        "events.json",
        "epoch_history.csv",
        "stage_history.csv",
        "predictions.csv",
        "checksums.json",
    }
    assert expected <= {path.name for path in destination.iterdir()}
    checkpoint = destination / "checkpoints" / "final.pt"
    assert checkpoint.exists()
    checksums = json.loads((destination / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["checkpoints/final.pt"] == file_sha256(checkpoint)


def test_checkpoint_reconstruction_preserves_predictions(tmp_path: Path) -> None:
    torch.manual_seed(741)
    original = AdditiveGrowingNetwork(3, 2)
    original.add_hidden_units(2, preserve_function=False)
    features = torch.randn(6, 3)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": original.state_dict(),
            "shape": {
                "input_features": 3,
                "output_classes": 2,
                "hidden_units": 2,
                "activation": "tanh",
            },
            "class_order": [4, 2],
        },
        checkpoint,
    )
    restored, class_order = load_model_checkpoint(checkpoint)
    torch.testing.assert_close(original(features), restored(features))
    assert class_order == [4, 2]
