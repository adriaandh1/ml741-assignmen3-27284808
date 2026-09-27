from ml741_assignment3.controller import (
    AdaptiveController,
    ControllerAction,
    ControllerConfig,
)


def controller() -> AdaptiveController:
    return AdaptiveController(
        ControllerConfig(
            min_epochs=2,
            patience=2,
            min_delta=0.01,
            underfit_train_f1=0.8,
            underfit_max_gap=0.1,
            overfit_gap=0.2,
            max_hidden_units=2,
            max_epochs_per_stage=10,
        )
    )


def test_controller_continues_before_minimum_epochs() -> None:
    action = controller().decide(
        epoch=1,
        train_macro_f1=[0.4],
        validation_macro_f1=[0.4],
        validation_loss=[1.0],
        hidden_units=0,
    )
    assert action == ControllerAction.CONTINUE


def test_controller_grows_after_underfit_plateau() -> None:
    action = controller().decide(
        epoch=4,
        train_macro_f1=[0.40, 0.50, 0.50, 0.50],
        validation_macro_f1=[0.40, 0.49, 0.49, 0.49],
        validation_loss=[1.2, 1.0, 1.0, 1.0],
        hidden_units=0,
    )
    assert action == ControllerAction.GROW


def test_controller_advances_after_well_fit_plateau() -> None:
    action = controller().decide(
        epoch=4,
        train_macro_f1=[0.85, 0.90, 0.90, 0.90],
        validation_macro_f1=[0.84, 0.89, 0.89, 0.89],
        validation_loss=[0.5, 0.4, 0.4, 0.4],
        hidden_units=1,
    )
    assert action == ControllerAction.ADVANCE


def test_controller_stops_on_overfitting() -> None:
    action = controller().decide(
        epoch=3,
        train_macro_f1=[0.7, 0.9, 0.95],
        validation_macro_f1=[0.65, 0.68, 0.60],
        validation_loss=[0.8, 0.7, 0.9],
        hidden_units=1,
    )
    assert action == ControllerAction.OVERFIT_STOP


def test_controller_stops_at_budget() -> None:
    action = controller().decide(
        epoch=10,
        train_macro_f1=[0.5],
        validation_macro_f1=[0.5],
        validation_loss=[1.0],
        hidden_units=0,
    )
    assert action == ControllerAction.BUDGET_STOP
