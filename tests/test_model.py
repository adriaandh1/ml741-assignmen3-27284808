import torch

from ml741_assignment3.model import AdditiveGrowingNetwork


def test_zero_hidden_units_equal_direct_head() -> None:
    model = AdditiveGrowingNetwork(3, 2)
    features = torch.randn(5, 3)
    torch.testing.assert_close(model(features), model.direct(features))


def test_function_preserving_hidden_growth() -> None:
    torch.manual_seed(741)
    model = AdditiveGrowingNetwork(4, 3)
    features = torch.randn(8, 4)
    before = model(features).detach().clone()
    model.add_hidden_unit(preserve_function=True)
    after = model(features).detach().clone()
    assert model.hidden_units == 1
    torch.testing.assert_close(before, after)


def test_non_preserving_initial_growth_changes_logits() -> None:
    torch.manual_seed(741)
    model = AdditiveGrowingNetwork(4, 3)
    features = torch.randn(8, 4)
    before = model(features).detach().clone()
    model.add_hidden_unit(preserve_function=False)
    after = model(features).detach().clone()
    assert not torch.equal(before, after)


def test_output_growth_preserves_existing_logits() -> None:
    torch.manual_seed(741)
    model = AdditiveGrowingNetwork(4, 2)
    model.add_hidden_units(2, preserve_function=False)
    features = torch.randn(8, 4)
    before = model(features).detach().clone()
    model.expand_output()
    after = model(features).detach().clone()
    assert model.output_classes == 3
    torch.testing.assert_close(before, after[:, :2])


def test_shape_and_parameter_count_track_growth() -> None:
    model = AdditiveGrowingNetwork(4, 2, activation="relu")
    initial_count = model.parameter_count()
    model.add_hidden_unit(preserve_function=True)
    assert model.shape.hidden_units == 1
    assert model.shape.activation == "relu"
    assert model.parameter_count() > initial_count
