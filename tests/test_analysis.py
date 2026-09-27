"""Final analysis regression tests, including adversarial inputs."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import logsumexp

from ml741_assignment3.analysis import (
    bootstrap_mean,
    frequency_groups,
    holm,
    paired_values,
    representative_seed,
    require,
    sign_flip,
    summary_statistics,
    validate_matrix,
)
from ml741_assignment3.metrics import average_forgetting


def fixture_frame():
    return pd.DataFrame(
        [
            {"dataset": "d", "method": method, "seed": seed, "macro_f1": value}
            for method, seed, value in [
                ("a", 2, 0.9),
                ("b", 1, 0.5),
                ("a", 1, 0.7),
                ("b", 2, 0.8),
            ]
        ]
    )


def test_matrix_complete():
    validate_matrix(fixture_frame(), ["d"], ["a", "b"], [1, 2])


def test_matrix_missing():
    with pytest.raises(ValueError, match="missing"):
        validate_matrix(fixture_frame().iloc[:-1], ["d"], ["a", "b"], [1, 2])


def test_matrix_duplicate():
    frame = fixture_frame()
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix(pd.concat([frame, frame.iloc[:1]]), ["d"], ["a", "b"], [1, 2])


def test_pairs_align_seed_not_row_position():
    assert np.allclose(paired_values(fixture_frame(), "a", "b", "macro_f1"), [0.2, 0.1])


def test_unpaired_rejected():
    with pytest.raises(ValueError, match="unpaired"):
        paired_values(fixture_frame().iloc[:-1], "a", "b", "macro_f1")


def test_duplicate_pair_rejected():
    frame = fixture_frame()
    with pytest.raises(ValueError, match="duplicate"):
        paired_values(pd.concat([frame, frame]), "a", "b", "macro_f1")


def test_bootstrap_reproducible_and_constant():
    assert bootstrap_mean([0.2] * 10) == pytest.approx((0.2, 0.2))
    assert bootstrap_mean(np.arange(10)) == bootstrap_mean(np.arange(10))


@pytest.mark.parametrize("sample", [[], [1], [1, float("nan")]])
def test_bootstrap_invalid(sample):
    with pytest.raises(ValueError):
        bootstrap_mean(sample)


def test_exact_sign_flip_extremes():
    assert sign_flip([0] * 10) == 1
    assert sign_flip([0.1] * 10) == 2 / 1024
    assert sign_flip([-0.1] * 10) == 2 / 1024


def test_holm_unsorted_and_monotone():
    assert holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert holm([1, 1, 1]) == pytest.approx([1, 1, 1])


def test_holm_invalid():
    with pytest.raises(ValueError):
        holm([-0.1])


def test_groups_training_frequency_and_ties():
    low, high = frequency_groups([20, 10, 10, 50, 80, 30, 20])
    assert low == [1, 2, 0]
    assert high == [4, 3, 5]


def test_summary_sample_sd_and_quartiles():
    summary = summary_statistics([1, 2, 3, 4])
    assert summary["mean"] == 2.5
    assert summary["sd"] == pytest.approx(np.sqrt(5 / 3))
    assert summary["q1"] == 1.75
    assert summary["q3"] == 3.25


def test_representative_lower_median_with_seed_tie():
    frame = pd.DataFrame({"seed": [4, 3, 1, 2], "macro_f1": [0.8, 0.8, 0.8, 0.8]})
    assert representative_seed(frame) == 2


def test_missing_stage_class_is_not_zero_recall():
    recalls = [{0: 0.8}, {0: 0.7, 1: 0.9}, {0: 0.75, 1: 0.85, 2: 0.2}]
    # Newly introduced class 2 has no earlier recall and is excluded.
    assert average_forgetting(recalls) == pytest.approx(0.05)
    assert 1 not in recalls[0]


def test_stable_cross_entropy_for_extreme_logits():
    logits = np.asarray([[1000.0, -1000.0], [2.0, 1.0]])
    targets = np.asarray([1, 0])
    stable = np.mean(logsumexp(logits, axis=1) - logits[np.arange(2), targets])
    expected = torch.nn.functional.cross_entropy(
        torch.from_numpy(logits), torch.from_numpy(targets)
    )
    assert stable == pytest.approx(expected.item())
    assert stable > 1000  # Probability clipping must not hide the confident error.


def test_require_not_disabled_by_python_optimisation():
    with pytest.raises(ValueError, match="invalid"):
        require(False, "invalid")
