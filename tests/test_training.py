import numpy as np

from ml741_assignment3.controller import ControllerConfig
from ml741_assignment3.data import make_smoke_dataset, stratified_splits
from ml741_assignment3.training import (
    TrainingConfig,
    train_adaptive_incremental,
    train_all_class_baseline,
    train_fixed_incremental,
)


def small_problem():
    features, targets = make_smoke_dataset(samples=200)
    return stratified_splits(features, targets, seed=741)


def short_training() -> TrainingConfig:
    return TrainingConfig(
        learning_rate=0.01,
        batch_size=32,
        max_epochs=3,
        patience=2,
        min_delta=0.001,
        seed=741,
        consolidation_epochs=1,
    )


def test_baseline_completes_and_is_reproducible() -> None:
    splits = small_problem()
    first = train_all_class_baseline(splits, hidden_units=2, config=short_training())
    second = train_all_class_baseline(splits, hidden_units=2, config=short_training())
    assert first.method == "all_class_baseline"
    np.testing.assert_array_equal(first.targets, splits.y_validation)
    np.testing.assert_array_equal(first.predictions, second.predictions)


def test_fixed_incremental_adds_every_remaining_class() -> None:
    splits = small_problem()
    result = train_fixed_incremental(splits, hidden_units=2, config=short_training())
    class_events = [event for event in result.events if event["event"] == "class_added"]
    assert len(class_events) == len(np.unique(splits.y_train)) - 2
    assert set(result.class_order) == set(np.unique(splits.y_train).tolist())
    assert any(event["event"] == "consolidation_completed" for event in result.events)


def test_adaptive_incremental_grows_and_terminates() -> None:
    splits = small_problem()
    training = TrainingConfig(
        learning_rate=1e-6,
        batch_size=32,
        max_epochs=3,
        patience=2,
        min_delta=0.001,
        seed=741,
    )
    controller = ControllerConfig(
        min_epochs=2,
        patience=1,
        min_delta=1.0,
        underfit_train_f1=1.01,
        underfit_max_gap=1.0,
        overfit_gap=2.0,
        max_hidden_units=1,
        max_epochs_per_stage=6,
    )
    result = train_adaptive_incremental(
        splits,
        config=training,
        controller_config=controller,
    )
    growth = [event for event in result.events if event["event"] == "hidden_unit_added"]
    assert growth
    assert result.final_hidden_units == 1
    np.testing.assert_array_equal(result.targets, splits.y_validation)
