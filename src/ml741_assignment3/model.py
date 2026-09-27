"""Additive single-hidden-layer neural network with safe structural growth."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelShape:
    input_features: int
    output_classes: int
    hidden_units: int
    activation: str


class AdditiveGrowingNetwork(nn.Module):
    """A linear class head plus additive nonlinear hidden-unit contributions."""

    def __init__(
        self,
        input_features: int,
        output_classes: int,
        *,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be positive")
        if output_classes < 2:
            raise ValueError("output_classes must be at least two")
        if activation not in {"relu", "tanh"}:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.input_features = input_features
        self.output_classes = output_classes
        self.activation_name = activation
        self.direct = nn.Linear(input_features, output_classes)
        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()
        self.hidden_outputs = nn.ParameterList()

    @property
    def hidden_units(self) -> int:
        return len(self.hidden_weights)

    @property
    def shape(self) -> ModelShape:
        return ModelShape(
            input_features=self.input_features,
            output_classes=self.output_classes,
            hidden_units=self.hidden_units,
            activation=self.activation_name,
        )

    def _activate(self, values: Tensor) -> Tensor:
        if self.activation_name == "relu":
            return torch.relu(values)
        return torch.tanh(values)

    def forward(self, features: Tensor) -> Tensor:
        logits = self.direct(features)
        for weight, bias, output in zip(
            self.hidden_weights,
            self.hidden_biases,
            self.hidden_outputs,
            strict=True,
        ):
            hidden = self._activate(features @ weight + bias)
            logits = logits + hidden.unsqueeze(1) * output.unsqueeze(0)
        return logits

    def add_hidden_unit(
        self,
        *,
        preserve_function: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        """Append one hidden unit, optionally preserving all current logits."""
        bound = 1.0 / math.sqrt(self.input_features)
        weight = torch.empty(
            self.input_features,
            device=self.direct.weight.device,
            dtype=self.direct.weight.dtype,
        )
        weight.uniform_(-bound, bound, generator=generator)
        bias = torch.zeros((), device=weight.device, dtype=weight.dtype)
        output = torch.zeros(
            self.output_classes,
            device=weight.device,
            dtype=weight.dtype,
        )
        if not preserve_function:
            output.uniform_(-bound, bound, generator=generator)
        self.hidden_weights.append(nn.Parameter(weight))
        self.hidden_biases.append(nn.Parameter(bias))
        self.hidden_outputs.append(nn.Parameter(output))

    def add_hidden_units(
        self,
        count: int,
        *,
        preserve_function: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        if count < 0:
            raise ValueError("hidden-unit count must be non-negative")
        for _ in range(count):
            self.add_hidden_unit(
                preserve_function=preserve_function,
                generator=generator,
            )

    def expand_output(
        self,
        additional_classes: int = 1,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Append output classes while preserving logits for existing classes."""
        if additional_classes < 1:
            raise ValueError("additional_classes must be positive")
        old_classes = self.output_classes
        new_classes = old_classes + additional_classes
        old_direct = self.direct
        replacement = nn.Linear(self.input_features, new_classes).to(
            device=old_direct.weight.device,
            dtype=old_direct.weight.dtype,
        )
        bound = 1.0 / math.sqrt(self.input_features)
        with torch.no_grad():
            replacement.weight[:old_classes].copy_(old_direct.weight)
            replacement.bias[:old_classes].copy_(old_direct.bias)
            replacement.weight[old_classes:].uniform_(-bound, bound, generator=generator)
            replacement.bias[old_classes:].zero_()
        self.direct = replacement

        expanded_outputs = nn.ParameterList()
        for old_output in self.hidden_outputs:
            expanded = torch.zeros(
                new_classes,
                device=old_output.device,
                dtype=old_output.dtype,
            )
            with torch.no_grad():
                expanded[:old_classes].copy_(old_output)
            expanded_outputs.append(nn.Parameter(expanded))
        self.hidden_outputs = expanded_outputs
        self.output_classes = new_classes

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
