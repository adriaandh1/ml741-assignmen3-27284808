import pytest

from ml741_assignment3.metrics import average_forgetting, classification_metrics


def test_classification_metrics_include_absent_prediction_class() -> None:
    metrics = classification_metrics(
        [0, 0, 1, 1, 2, 2],
        [0, 0, 0, 1, 1, 1],
        labels=[0, 1, 2],
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    per_class = metrics["per_class"]
    assert isinstance(per_class, list)
    assert per_class[2]["recall"] == 0.0


def test_average_forgetting_uses_best_earlier_recall() -> None:
    history = [
        {0: 0.8, 1: 0.7},
        {0: 0.9, 1: 0.6, 2: 0.5},
        {0: 0.7, 1: 0.65, 2: 0.6},
    ]
    assert average_forgetting(history) == pytest.approx((0.2 + 0.05 + 0.0) / 3)


def test_average_forgetting_is_zero_for_one_stage() -> None:
    assert average_forgetting([{0: 0.8, 1: 0.7}]) == 0.0
