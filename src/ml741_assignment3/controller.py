"""Objective stage and capacity decisions for adaptive incremental learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class ControllerAction(StrEnum):
    CONTINUE = "continue"
    GROW = "grow"
    ADVANCE = "advance"
    OVERFIT_STOP = "overfit_stop"
    BUDGET_STOP = "budget_stop"


@dataclass(frozen=True)
class ControllerConfig:
    min_epochs: int = 5
    patience: int = 5
    min_delta: float = 0.002
    underfit_train_f1: float = 0.80
    underfit_max_gap: float = 0.08
    overfit_gap: float = 0.15
    max_hidden_units: int = 16
    max_epochs_per_stage: int = 50

    def __post_init__(self) -> None:
        if self.min_epochs < 1 or self.patience < 1:
            raise ValueError("epoch controls must be positive")
        if self.max_epochs_per_stage < self.min_epochs:
            raise ValueError("maximum stage epochs must cover minimum epochs")
        if self.max_hidden_units < 0:
            raise ValueError("maximum hidden units must be non-negative")
        for value in (
            self.min_delta,
            self.underfit_train_f1,
            self.underfit_max_gap,
            self.overfit_gap,
        ):
            if not isfinite(value) or value < 0:
                raise ValueError("controller thresholds must be non-negative")


class AdaptiveController:
    """Translate validation evidence into reproducible structural decisions."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def decide(
        self,
        *,
        epoch: int,
        train_macro_f1: list[float],
        validation_macro_f1: list[float],
        validation_loss: list[float],
        hidden_units: int,
    ) -> ControllerAction:
        if not train_macro_f1 or len(train_macro_f1) != len(validation_macro_f1):
            raise ValueError("controller histories must be non-empty and aligned")
        if len(validation_loss) != len(validation_macro_f1):
            raise ValueError("validation histories must be aligned")
        if epoch >= self.config.max_epochs_per_stage:
            return ControllerAction.BUDGET_STOP
        if epoch < self.config.min_epochs:
            return ControllerAction.CONTINUE

        gap = train_macro_f1[-1] - validation_macro_f1[-1]
        earlier_losses = validation_loss[:-1]
        loss_worsened = bool(
            earlier_losses and validation_loss[-1] > min(earlier_losses) + self.config.min_delta
        )
        if gap >= self.config.overfit_gap and loss_worsened:
            return ControllerAction.OVERFIT_STOP

        required = self.config.patience + 1
        if len(validation_macro_f1) < required:
            return ControllerAction.CONTINUE
        prior = validation_macro_f1[: -self.config.patience]
        recent = validation_macro_f1[-self.config.patience :]
        plateau = max(recent) <= max(prior) + self.config.min_delta
        if not plateau:
            return ControllerAction.CONTINUE

        underfit = (
            train_macro_f1[-1] < self.config.underfit_train_f1
            and gap <= self.config.underfit_max_gap
        )
        if underfit and hidden_units < self.config.max_hidden_units:
            return ControllerAction.GROW
        return ControllerAction.ADVANCE
