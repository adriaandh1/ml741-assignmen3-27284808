import numpy as np
import pandas as pd
import pytest

from ml741_assignment3.data import (
    controlled_long_tail_indices,
    encode_labels,
    load_numeric_csv,
    make_smoke_dataset,
    minority_first_order,
    stratified_splits,
)


def test_encode_labels_is_deterministic() -> None:
    encoded, values = encode_labels(["zebra", "ant", "zebra", "bird"])
    assert values == ("ant", "bird", "zebra")
    assert encoded.tolist() == [2, 0, 2, 1]


def test_minority_first_order_has_stable_ties() -> None:
    labels = np.asarray([3, 3, 2, 2, 2, 1, 1, 1, 1])
    assert minority_first_order(labels) == [3, 2, 1]


def test_stratified_splits_are_disjoint_and_training_scaled() -> None:
    features, labels = make_smoke_dataset(samples=240)
    splits = stratified_splits(features, labels, seed=741)
    train = set(splits.train_indices.tolist())
    validation = set(splits.validation_indices.tolist())
    test = set(splits.test_indices.tolist())
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(range(len(labels)))
    np.testing.assert_allclose(splits.x_train.mean(axis=0), 0.0, atol=1e-6)
    reconstructed = (features[splits.train_indices] - splits.scaler_mean) / splits.scaler_scale
    np.testing.assert_allclose(splits.x_train, reconstructed, atol=1e-6)


def test_split_scaler_does_not_use_validation_outlier() -> None:
    features, labels = make_smoke_dataset(samples=240)
    splits = stratified_splits(features, labels, seed=741)
    expected = features[splits.train_indices].mean(axis=0)
    np.testing.assert_allclose(splits.scaler_mean, expected)


def test_controlled_long_tail_is_reproducible() -> None:
    labels = np.repeat(np.arange(4), 100)
    first = controlled_long_tail_indices(
        labels,
        maximum_to_minimum_ratio=10,
        seed=741,
    )
    second = controlled_long_tail_indices(
        labels,
        maximum_to_minimum_ratio=10,
        seed=741,
    )
    np.testing.assert_array_equal(first, second)
    counts = [int(np.sum(labels[first] == label)) for label in range(4)]
    assert counts[0] == 100
    assert counts[-1] == 10
    assert counts == sorted(counts, reverse=True)


def test_controlled_long_tail_rejects_bad_class_order() -> None:
    with pytest.raises(ValueError, match="every observed class"):
        controlled_long_tail_indices(
            np.repeat(np.arange(3), 10),
            maximum_to_minimum_ratio=5,
            seed=1,
            class_order=[0, 1],
        )


def test_load_numeric_csv_validates_and_drops_identifier(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "id": [10, 11, 12],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4, 5, 6],
            "class": ["rare", "common", "common"],
        }
    ).to_csv(path, index=False)
    dataset = load_numeric_csv(path, target_column="class", drop_columns=["id"])
    assert dataset.features.shape == (3, 2)
    assert dataset.feature_names == ("feature_a", "feature_b")
    assert dataset.targets.tolist() == ["rare", "common", "common"]


def test_load_numeric_csv_rejects_implicit_missing_value_policy(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"feature": [1.0, np.nan], "class": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing values"):
        load_numeric_csv(path, target_column="class")
