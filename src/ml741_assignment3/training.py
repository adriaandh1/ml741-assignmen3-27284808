"""Baseline, fixed incremental, and adaptive incremental training workflows."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ml741_assignment3.controller import (
    AdaptiveController,
    ControllerAction,
    ControllerConfig,
)
from ml741_assignment3.data import DatasetSplits, minority_first_order
from ml741_assignment3.metrics import average_forgetting, classification_metrics
from ml741_assignment3.model import AdditiveGrowingNetwork
from ml741_assignment3.reproducibility import seed_everything


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 6
    min_delta: float = 0.002
    activation: str = "tanh"
    seed: int = 741
    device: str = "cpu"
    consolidation_epochs: int = 0
    evaluate_test: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.batch_size < 1:
            raise ValueError("learning rate and batch size must be positive")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("epoch controls must be positive")
        if self.consolidation_epochs < 0:
            raise ValueError("consolidation epochs must be non-negative")
        if any(
            not np.isfinite(v) or v < 0
            for v in (self.learning_rate, self.weight_decay, self.min_delta)
        ):
            raise ValueError("training thresholds must be finite and non-negative")


@dataclass
class RunResult:
    method: str
    seed: int
    class_order: list[int]
    metrics: dict[str, Any]
    epoch_records: list[dict[str, Any]]
    stage_records: list[dict[str, Any]]
    events: list[dict[str, Any]]
    final_hidden_units: int
    parameter_count: int
    runtime_seconds: float
    targets: NDArray[np.int64] = field(repr=False)
    predictions: NDArray[np.int64] = field(repr=False)
    model_state: dict[str, Tensor] = field(repr=False)
    model_shape: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "class_order": self.class_order,
            "metrics": self.metrics,
            "final_hidden_units": self.final_hidden_units,
            "parameter_count": self.parameter_count,
            "runtime_seconds": self.runtime_seconds,
            "model_shape": self.model_shape,
        }


def _tensor(array: NDArray[np.floating] | NDArray[np.integer], device: str) -> Tensor:
    tensor = torch.from_numpy(np.asarray(array))
    return tensor.to(device=device)


def _mapping(classes: list[int]) -> dict[int, int]:
    return {label: index for index, label in enumerate(classes)}


def _local_targets(targets: NDArray[np.int64], classes: list[int]) -> NDArray[np.int64]:
    mapping = _mapping(classes)
    return np.asarray([mapping[int(value)] for value in targets], dtype=np.int64)


def _subset(
    features: NDArray[np.float32],
    targets: NDArray[np.int64],
    classes: list[int],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    mask = np.isin(targets, np.asarray(classes, dtype=np.int64))
    return features[mask], targets[mask]


def _predict(
    model: AdditiveGrowingNetwork,
    features: NDArray[np.float32],
    classes: list[int],
    device: str,
) -> NDArray[np.int64]:
    model.eval()
    with torch.no_grad():
        positions = model(_tensor(features, device).float()).argmax(dim=1).cpu().numpy()
    class_array = np.asarray(classes, dtype=np.int64)
    return class_array[positions]


def _loss_and_predictions(
    model: AdditiveGrowingNetwork,
    features: NDArray[np.float32],
    targets: NDArray[np.int64],
    classes: list[int],
    device: str,
) -> tuple[float, NDArray[np.int64]]:
    model.eval()
    local = _local_targets(targets, classes)
    with torch.no_grad():
        logits = model(_tensor(features, device).float())
        loss = nn.functional.cross_entropy(logits, _tensor(local, device).long())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite evaluation loss")
        positions = logits.argmax(dim=1).cpu().numpy()
    predictions = np.asarray(classes, dtype=np.int64)[positions]
    return float(loss.item()), predictions


def _train_epoch(
    model: AdditiveGrowingNetwork,
    optimizer: torch.optim.Optimizer,
    features: NDArray[np.float32],
    targets: NDArray[np.int64],
    classes: list[int],
    config: TrainingConfig,
    *,
    epoch_seed: int,
) -> None:
    local = _local_targets(targets, classes)
    dataset = TensorDataset(
        _tensor(features, config.device).float(),
        _tensor(local, config.device).long(),
    )
    generator = torch.Generator(device="cpu").manual_seed(epoch_seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    model.train()
    for batch_features, batch_targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(batch_features), batch_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise FloatingPointError("non-finite training gradient")
        optimizer.step()


def _optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _stage_evaluation(
    model: AdditiveGrowingNetwork,
    train_features: NDArray[np.float32],
    train_targets: NDArray[np.int64],
    validation_features: NDArray[np.float32],
    validation_targets: NDArray[np.int64],
    classes: list[int],
    device: str,
) -> tuple[float, dict[str, Any], float, dict[str, Any]]:
    train_loss, train_predictions = _loss_and_predictions(
        model, train_features, train_targets, classes, device
    )
    validation_loss, validation_predictions = _loss_and_predictions(
        model,
        validation_features,
        validation_targets,
        classes,
        device,
    )
    train_metrics = classification_metrics(train_targets, train_predictions, labels=classes)
    validation_metrics = classification_metrics(
        validation_targets,
        validation_predictions,
        labels=classes,
    )
    return train_loss, train_metrics, validation_loss, validation_metrics


def _fit_static_stage(
    *,
    model: AdditiveGrowingNetwork,
    train_features: NDArray[np.float32],
    train_targets: NDArray[np.int64],
    validation_features: NDArray[np.float32],
    validation_targets: NDArray[np.int64],
    classes: list[int],
    config: TrainingConfig,
    method: str,
    stage: int,
    global_epoch_start: int,
) -> tuple[list[dict[str, Any]], str, int]:
    optimizer = _optimizer(model, config)
    _, _, _, initial_validation_metrics = _stage_evaluation(
        model,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        classes,
        config.device,
    )
    best_score = float(initial_validation_metrics["macro_f1"])
    best_state = _snapshot(model)
    stale_epochs = 0
    records: list[dict[str, Any]] = []
    termination = "maximum_epochs"
    for epoch in range(1, config.max_epochs + 1):
        _train_epoch(
            model,
            optimizer,
            train_features,
            train_targets,
            classes,
            config,
            epoch_seed=config.seed + global_epoch_start + epoch,
        )
        train_loss, train_metrics, validation_loss, validation_metrics = _stage_evaluation(
            model,
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            classes,
            config.device,
        )
        score = float(validation_metrics["macro_f1"])
        records.append(
            {
                "method": method,
                "stage": stage,
                "epoch": epoch,
                "global_epoch": global_epoch_start + epoch,
                "seen_classes": classes.copy(),
                "hidden_units": model.hidden_units,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "validation_macro_f1": score,
            }
        )
        significant_improvement = score > best_score + config.min_delta
        if score > best_score:
            best_score = score
            best_state = _snapshot(model)
        if significant_improvement:
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            termination = "validation_stagnation"
            break
    model.load_state_dict(best_state)
    return records, termination, len(records)


def _fit_adaptive_stage(
    *,
    model: AdditiveGrowingNetwork,
    train_features: NDArray[np.float32],
    train_targets: NDArray[np.int64],
    validation_features: NDArray[np.float32],
    validation_targets: NDArray[np.int64],
    classes: list[int],
    config: TrainingConfig,
    controller_config: ControllerConfig,
    stage: int,
    global_epoch_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, int]:
    controller = AdaptiveController(controller_config)
    optimizer = _optimizer(model, config)
    _, _, _, initial_metrics = _stage_evaluation(
        model,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        classes,
        config.device,
    )
    best_score = float(initial_metrics["macro_f1"])
    best_state = _snapshot(model)
    decision_train: list[float] = []
    decision_validation: list[float] = []
    decision_losses: list[float] = []
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    termination = ControllerAction.BUDGET_STOP.value

    for epoch in range(1, controller_config.max_epochs_per_stage + 1):
        _train_epoch(
            model,
            optimizer,
            train_features,
            train_targets,
            classes,
            config,
            epoch_seed=config.seed + global_epoch_start + epoch,
        )
        train_loss, train_metrics, validation_loss, validation_metrics = _stage_evaluation(
            model,
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            classes,
            config.device,
        )
        train_score = float(train_metrics["macro_f1"])
        validation_score = float(validation_metrics["macro_f1"])
        decision_train.append(train_score)
        decision_validation.append(validation_score)
        decision_losses.append(validation_loss)
        if validation_score > best_score:
            best_score = validation_score
            best_state = _snapshot(model)

        action = controller.decide(
            epoch=len(decision_train),
            train_macro_f1=decision_train,
            validation_macro_f1=decision_validation,
            validation_loss=decision_losses,
            hidden_units=model.hidden_units,
        )
        if epoch >= controller_config.max_epochs_per_stage:
            action = ControllerAction.BUDGET_STOP
        records.append(
            {
                "method": "adaptive_capacity_incremental",
                "stage": stage,
                "epoch": epoch,
                "global_epoch": global_epoch_start + epoch,
                "seen_classes": classes.copy(),
                "hidden_units": model.hidden_units,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_macro_f1": train_score,
                "validation_macro_f1": validation_score,
                "controller_action": action.value,
            }
        )
        if action == ControllerAction.CONTINUE:
            continue
        if action == ControllerAction.GROW:
            model.load_state_dict(best_state)
            before = model.hidden_units
            model.add_hidden_unit(preserve_function=True)
            events.append(
                {
                    "event": "hidden_unit_added",
                    "stage": stage,
                    "epoch": epoch,
                    "global_epoch": global_epoch_start + epoch,
                    "hidden_units_before": before,
                    "hidden_units_after": model.hidden_units,
                }
            )
            optimizer = _optimizer(model, config)
            best_state = _snapshot(model)
            decision_train.clear()
            decision_validation.clear()
            decision_losses.clear()
            continue
        termination = action.value
        model.load_state_dict(best_state)
        break
    model.load_state_dict(best_state)
    return records, events, termination, len(records)


def _consolidate(
    *,
    model: AdditiveGrowingNetwork,
    splits: DatasetSplits,
    classes: list[int],
    config: TrainingConfig,
    method: str,
    global_epoch_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    consolidation_config = replace(
        config,
        max_epochs=config.consolidation_epochs,
        patience=config.consolidation_epochs + 1,
    )
    records, termination, epochs = _fit_static_stage(
        model=model,
        train_features=splits.x_train,
        train_targets=splits.y_train,
        validation_features=splits.x_validation,
        validation_targets=splits.y_validation,
        classes=classes,
        config=consolidation_config,
        method=method,
        stage=len(classes) - 1,
        global_epoch_start=global_epoch_start,
    )
    predictions = _predict(model, splits.x_validation, classes, config.device)
    metrics = classification_metrics(
        splits.y_validation,
        predictions,
        labels=classes,
    )
    metrics.update(
        {
            "stage": len(classes) - 1,
            "phase": "consolidation",
            "epochs": epochs,
            "termination": termination,
        }
    )
    return records, metrics


def _finalise(
    *,
    model: AdditiveGrowingNetwork,
    splits: DatasetSplits,
    classes: list[int],
    method: str,
    config: TrainingConfig,
    epoch_records: list[dict[str, Any]],
    stage_records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    started: float,
) -> RunResult:
    evaluation_x = splits.x_test if config.evaluate_test else splits.x_validation
    evaluation_y = splits.y_test if config.evaluate_test else splits.y_validation
    predictions = _predict(model, evaluation_x, classes, config.device)
    labels = sorted(classes)
    metrics = classification_metrics(evaluation_y, predictions, labels=labels)
    metrics["evaluation_partition"] = "test" if config.evaluate_test else "validation"
    stage_recalls = [
        {int(item["label"]): float(item["recall"]) for item in stage["per_class"]}
        for stage in stage_records
    ]
    metrics["average_forgetting"] = average_forgetting(stage_recalls)
    metrics["forgetting_partition"] = "validation"
    return RunResult(
        method=method,
        seed=config.seed,
        class_order=classes,
        metrics=metrics,
        epoch_records=epoch_records,
        stage_records=stage_records,
        events=events,
        final_hidden_units=model.hidden_units,
        parameter_count=model.parameter_count(),
        runtime_seconds=time.perf_counter() - started,
        targets=evaluation_y.copy(),
        predictions=predictions,
        model_state=_snapshot(model),
        model_shape=asdict(model.shape),
    )


def train_all_class_baseline(
    splits: DatasetSplits,
    *,
    hidden_units: int,
    config: TrainingConfig,
) -> RunResult:
    """Train a validation-selected fixed architecture on every class at once."""
    started = time.perf_counter()
    seed_everything(config.seed)
    classes = sorted(np.unique(splits.y_train).tolist())
    model = AdditiveGrowingNetwork(
        splits.x_train.shape[1],
        len(classes),
        activation=config.activation,
    ).to(config.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model.add_hidden_units(hidden_units, preserve_function=False, generator=generator)
    records, termination, epochs = _fit_static_stage(
        model=model,
        train_features=splits.x_train,
        train_targets=splits.y_train,
        validation_features=splits.x_validation,
        validation_targets=splits.y_validation,
        classes=classes,
        config=config,
        method="all_class_baseline",
        stage=0,
        global_epoch_start=0,
    )
    validation_predictions = _predict(model, splits.x_validation, classes, config.device)
    stage_metrics = classification_metrics(
        splits.y_validation,
        validation_predictions,
        labels=classes,
    )
    stage_metrics.update({"stage": 0, "epochs": epochs, "termination": termination})
    return _finalise(
        model=model,
        splits=splits,
        classes=classes,
        method="all_class_baseline",
        config=config,
        epoch_records=records,
        stage_records=[stage_metrics],
        events=[],
        started=started,
    )


def train_fixed_incremental(
    splits: DatasetSplits,
    *,
    hidden_units: int,
    config: TrainingConfig,
) -> RunResult:
    """Train a fixed-capacity network through a minority-first class schedule."""
    started = time.perf_counter()
    seed_everything(config.seed)
    order = minority_first_order(splits.y_train)
    seen = order[:2]
    model = AdditiveGrowingNetwork(
        splits.x_train.shape[1],
        2,
        activation=config.activation,
    ).to(config.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model.add_hidden_units(hidden_units, preserve_function=False, generator=generator)
    epoch_records: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    global_epoch = 0
    for stage in range(len(order) - 1):
        if stage > 0:
            new_class = order[stage + 1]
            model.expand_output(generator=generator)
            seen.append(new_class)
            events.append(
                {
                    "event": "class_added",
                    "stage": stage,
                    "class": new_class,
                    "seen_classes": seen.copy(),
                }
            )
        train_x, train_y = _subset(splits.x_train, splits.y_train, seen)
        validation_x, validation_y = _subset(
            splits.x_validation,
            splits.y_validation,
            seen,
        )
        records, termination, epochs = _fit_static_stage(
            model=model,
            train_features=train_x,
            train_targets=train_y,
            validation_features=validation_x,
            validation_targets=validation_y,
            classes=seen,
            config=config,
            method="fixed_capacity_incremental",
            stage=stage,
            global_epoch_start=global_epoch,
        )
        epoch_records.extend(records)
        global_epoch += epochs
        validation_predictions = _predict(model, validation_x, seen, config.device)
        stage_metrics = classification_metrics(
            validation_y,
            validation_predictions,
            labels=seen,
        )
        stage_metrics.update({"stage": stage, "epochs": epochs, "termination": termination})
        stage_records.append(stage_metrics)
    if config.consolidation_epochs:
        records, consolidation_metrics = _consolidate(
            model=model,
            splits=splits,
            classes=seen,
            config=config,
            method="fixed_capacity_incremental",
            global_epoch_start=global_epoch,
        )
        epoch_records.extend(records)
        stage_records.append(consolidation_metrics)
        events.append(
            {
                "event": "consolidation_completed",
                "epochs": len(records),
                "seen_classes": seen.copy(),
            }
        )
    return _finalise(
        model=model,
        splits=splits,
        classes=seen,
        method="fixed_capacity_incremental",
        config=config,
        epoch_records=epoch_records,
        stage_records=stage_records,
        events=events,
        started=started,
    )


def train_adaptive_incremental(
    splits: DatasetSplits,
    *,
    config: TrainingConfig,
    controller_config: ControllerConfig,
) -> RunResult:
    """Train the proposed minority-first network with validation-driven growth."""
    started = time.perf_counter()
    seed_everything(config.seed)
    order = minority_first_order(splits.y_train)
    seen = order[:2]
    model = AdditiveGrowingNetwork(
        splits.x_train.shape[1],
        2,
        activation=config.activation,
    ).to(config.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    epoch_records: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    global_epoch = 0
    for stage in range(len(order) - 1):
        if stage > 0:
            new_class = order[stage + 1]
            model.expand_output(generator=generator)
            seen.append(new_class)
            events.append(
                {
                    "event": "class_added",
                    "stage": stage,
                    "class": new_class,
                    "seen_classes": seen.copy(),
                }
            )
        train_x, train_y = _subset(splits.x_train, splits.y_train, seen)
        validation_x, validation_y = _subset(
            splits.x_validation,
            splits.y_validation,
            seen,
        )
        records, stage_events, termination, epochs = _fit_adaptive_stage(
            model=model,
            train_features=train_x,
            train_targets=train_y,
            validation_features=validation_x,
            validation_targets=validation_y,
            classes=seen,
            config=config,
            controller_config=controller_config,
            stage=stage,
            global_epoch_start=global_epoch,
        )
        epoch_records.extend(records)
        events.extend(stage_events)
        global_epoch += epochs
        validation_predictions = _predict(model, validation_x, seen, config.device)
        stage_metrics = classification_metrics(
            validation_y,
            validation_predictions,
            labels=seen,
        )
        stage_metrics.update(
            {
                "stage": stage,
                "epochs": epochs,
                "termination": termination,
                "hidden_units": model.hidden_units,
            }
        )
        stage_records.append(stage_metrics)
    if config.consolidation_epochs:
        records, consolidation_metrics = _consolidate(
            model=model,
            splits=splits,
            classes=seen,
            config=config,
            method="adaptive_capacity_incremental",
            global_epoch_start=global_epoch,
        )
        epoch_records.extend(records)
        stage_records.append(consolidation_metrics)
        events.append(
            {
                "event": "consolidation_completed",
                "epochs": len(records),
                "seen_classes": seen.copy(),
            }
        )
    return _finalise(
        model=model,
        splits=splits,
        classes=seen,
        method="adaptive_capacity_incremental",
        config=config,
        epoch_records=epoch_records,
        stage_records=stage_records,
        events=events,
        started=started,
    )
