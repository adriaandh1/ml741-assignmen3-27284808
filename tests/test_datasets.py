"""Regression checks for real-data partitioning and pilot isolation."""

from dataclasses import asdict

import numpy as np
import pytest

from ml741_assignment3.datasets import load_prepared, prepare


def fixture_arrays():
    rng = np.random.default_rng(12)
    return rng.normal(size=(900, 4)), np.tile(np.arange(3), 300)


def test_duplicates_removed_before_random_split_and_deterministic():
    x, y = fixture_arrays()
    x = np.vstack([x, x[:12]])
    y = np.r_[y, y[:12]]
    a, audit = prepare(x, y)
    b, _ = prepare(x, y)
    assert audit["duplicate_rows_removed"] == 12
    assert np.array_equal(a.train_indices, b.train_indices)
    sets = [
        set(map(tuple, x[idx]))
        for idx in (a.train_indices, a.validation_indices, a.test_indices)
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_official_test_keeps_priority_and_never_enters_development():
    x, y = fixture_arrays()
    x[600], y[600] = x[0], y[0]
    splits, audit = prepare(x, y, official_boundary=600)
    assert audit["cross_official_duplicate_groups"] == 1
    assert 600 in splits.test_indices
    assert 0 not in splits.train_indices and 0 not in splits.validation_indices
    assert max(splits.train_indices) < 600
    assert max(splits.validation_indices) < 600
    assert min(splits.test_indices) >= 600


def test_long_tail_changes_training_only_and_scaler_fits_retained_rows():
    x, y = fixture_arrays()
    natural, _ = prepare(x, y)
    tail, _ = prepare(x, y, long_tail_ratio=10)
    assert np.array_equal(natural.validation_indices, tail.validation_indices)
    assert np.array_equal(natural.test_indices, tail.test_indices)
    assert np.allclose(tail.scaler_mean, x[tail.train_indices].mean(axis=0))
    assert np.allclose(tail.x_train.mean(axis=0), 0, atol=1e-6)
    assert len(tail.y_train) < len(natural.y_train)


def test_conflicting_duplicate_labels_rejected():
    x, y = fixture_arrays()
    x[1] = x[0]
    with pytest.raises(ValueError, match="conflicting"):
        prepare(x, y)


def test_pilot_loader_does_not_load_test_arrays(tmp_path):
    x, y = fixture_arrays()
    splits, _ = prepare(x, y)
    values = asdict(splits)
    # Object arrays cannot load with allow_pickle=False: this demonstrates
    # the pilot genuinely does not deserialize held-out features or labels.
    values["x_test"] = np.asarray([object()])
    values["y_test"] = np.asarray([object()])
    path = tmp_path / "data.npz"
    np.savez_compressed(path, **values)
    pilot = load_prepared(path)
    assert pilot.x_test is None and pilot.y_test is None
    with pytest.raises(ValueError, match="Object arrays"):
        load_prepared(path, include_test=True)
