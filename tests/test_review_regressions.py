"""Regression coverage for issues found before the pilot study."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from ml741_assignment3.controller import ControllerConfig
from ml741_assignment3.data import (
    controlled_long_tail_indices,
    make_smoke_dataset,
    stratified_splits,
)
from ml741_assignment3.model import AdditiveGrowingNetwork
from ml741_assignment3.smoke import run_smoke
from ml741_assignment3.training import (
    TrainingConfig,
    _fit_adaptive_stage,
    train_adaptive_incremental,
    train_all_class_baseline,
    train_fixed_incremental,
)


@pytest.mark.parametrize("method", ["baseline", "fixed", "adaptive"])
def test_pilot_does_not_touch_test_arrays(method):
    x, y = make_smoke_dataset(samples=200)
    splits = stratified_splits(x, y, seed=741)
    # Deliberately unusable held-out arrays prove evaluation is never attempted.
    splits = replace(splits, x_test=None, y_test=None)
    config = TrainingConfig(max_epochs=2, patience=2)
    if method == "baseline":
        result = train_all_class_baseline(splits, hidden_units=2, config=config)
    elif method == "fixed":
        result = train_fixed_incremental(splits, hidden_units=2, config=config)
    else:
        result = train_adaptive_incremental(
            splits,
            config=config,
            controller_config=ControllerConfig(min_epochs=1, max_epochs_per_stage=2),
        )
    assert result.metrics["evaluation_partition"] == "validation"
    np.testing.assert_array_equal(result.targets, splits.y_validation)


def test_duplicate_downsampling_order_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        controlled_long_tail_indices(
            np.repeat([0, 1], 20),
            maximum_to_minimum_ratio=2,
            seed=741,
            class_order=[0, 1, 1],
        )


def test_smoke_refuses_to_overwrite(tmp_path):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        run_smoke(tmp_path)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_numeric_configuration_rejected(value):
    with pytest.raises(ValueError):
        TrainingConfig(weight_decay=value)
    with pytest.raises(ValueError):
        ControllerConfig(min_delta=value)


def test_growth_keeps_best_score_and_respects_total_stage_budget(monkeypatch):
    import ml741_assignment3.training as training

    model = AdditiveGrowingNetwork(2, 2)
    # Initial score, then one good epoch, plateau/growth, then worse epochs.
    scores = iter([0.2, 0.9, 0.5, 0.6, 0.6])
    epoch = 0

    def fake_epoch(model, *args, **kwargs):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            model.direct.bias.fill_(epoch)

    def fake_evaluation(*args, **kwargs):
        score = next(scores)
        return 1.0, {"macro_f1": 0.7}, 1.0, {"macro_f1": score}

    monkeypatch.setattr(training, "_train_epoch", fake_epoch)
    monkeypatch.setattr(training, "_stage_evaluation", fake_evaluation)
    x = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    records, events, reason, count = _fit_adaptive_stage(
        model=model,
        train_features=x,
        train_targets=y,
        validation_features=x,
        validation_targets=y,
        classes=[0, 1],
        config=TrainingConfig(),
        stage=0,
        global_epoch_start=0,
        controller_config=ControllerConfig(
            min_epochs=1,
            patience=1,
            min_delta=0.01,
            underfit_train_f1=0.8,
            underfit_max_gap=0.3,
            max_hidden_units=4,
            max_epochs_per_stage=4,
        ),
    )
    assert count == 4
    assert reason == "budget_stop"
    assert records[-1]["controller_action"] == "budget_stop"
    assert len(events) == 1
    # The state with score 0.9 must survive subsequent lower-scoring epochs.
    torch.testing.assert_close(model.direct.bias, torch.ones(2))
