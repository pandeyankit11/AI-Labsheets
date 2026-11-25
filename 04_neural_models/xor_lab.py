from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

X_XOR = torch.tensor(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]],
    dtype=torch.float32,
)
Y_XOR = torch.tensor(
    [[0.0],
     [1.0],
     [1.0],
     [0.0]],
    dtype=torch.float32,
)

X_3CLASS = X_XOR.clone()
Y_3CLASS = torch.tensor([0, 1, 1, 2], dtype=torch.long)

SEED = 2
LR = 0.05
STEPS = 1000


@dataclass
class BinaryResult:
    model_type: str
    activation: str
    initial_loss: float
    final_loss: float
    probabilities: list[float]
    predictions: list[int]
    early_first_layer_grad_norm: float
    steps: int
    learning_rate: float
    seed: int


class XORNet(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.activation = activation
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class LinearBinaryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ThreeClassNet(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.activation = activation
        self.fc2 = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def classify_binary(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).to(torch.int64)
    return probabilities, predictions


def train_binary(
    activation: nn.Module,
    activation_name: str,
    *,
    seed: int = SEED,
    lr: float = LR,
    steps: int = STEPS,
) -> tuple[XORNet, BinaryResult]:
    set_seed(seed)
    model = XORNet(activation)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.no_grad():
        initial_loss = float(loss_fn(model(X_XOR), Y_XOR).item())

    early_grad_norm: float | None = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_XOR)
        loss = loss_fn(logits, Y_XOR)
        loss.backward()
        if step == 0:
            early_grad_norm = float(model.fc1.weight.grad.norm().item())
        optimizer.step()

    with torch.no_grad():
        final_logits = model(X_XOR)
        final_loss = float(loss_fn(final_logits, Y_XOR).item())
        probabilities, predictions = classify_binary(final_logits)

    assert early_grad_norm is not None
    result = BinaryResult(
        model_type="2-2-1",
        activation=activation_name,
        initial_loss=initial_loss,
        final_loss=final_loss,
        probabilities=[float(v) for v in probabilities.flatten()],
        predictions=[int(v) for v in predictions.flatten()],
        early_first_layer_grad_norm=early_grad_norm,
        steps=steps,
        learning_rate=lr,
        seed=seed,
    )
    return model, result


def train_linear_baseline(*, seed: int = SEED, lr: float = LR, steps: int = STEPS) -> dict:
    set_seed(seed)
    model = LinearBinaryNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.no_grad():
        initial_loss = float(loss_fn(model(X_XOR), Y_XOR).item())

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X_XOR), Y_XOR)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X_XOR)
        final_loss = float(loss_fn(logits, Y_XOR).item())
        probabilities, predictions = classify_binary(logits)

    return {
        "model_type": "single affine + sigmoid output",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "probabilities": [float(v) for v in probabilities.flatten()],
        "predictions": [int(v) for v in predictions.flatten()],
        "correct": int((predictions.flatten() == Y_XOR.flatten().to(torch.int64)).sum().item()),
        "steps": steps,
        "learning_rate": lr,
        "seed": seed,
    }


def train_symmetry_experiment(*, steps: int = 6, lr: float = LR) -> dict:
    set_seed(SEED)
    model = XORNet(nn.Tanh())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X_XOR), Y_XOR)
        loss.backward()
        with torch.no_grad():
            row_diff = float((model.fc1.weight[0] - model.fc1.weight[1]).norm().item())
            grad_diff = float((model.fc1.weight.grad[0] - model.fc1.weight.grad[1]).norm().item())
            grad_norm = float(model.fc1.weight.grad.norm().item())
            rows.append({
                "step": step,
                "loss": float(loss.item()),
                "row_difference": row_diff,
                "gradient_row_difference": grad_diff,
                "gradient_norm": grad_norm,
                "weights": model.fc1.weight.detach().cpu().numpy().round(8).tolist(),
            })
        optimizer.step()

    return {
        "architecture": "2-2-1 with tanh",
        "initialisation": "all weights and biases = 0",
        "rows": rows,
        "conclusion": "The hidden-unit rows remain identical and the first-layer gradient is zero because the identical hidden units receive identical inputs/outputs and the output-layer weights start at zero, so no symmetry-breaking signal reaches W1."
    }


def activation_experiments() -> list[BinaryResult]:
    configs: list[tuple[str, Callable[[], nn.Module]]] = [
        ("sigmoid", nn.Sigmoid),
        ("tanh", nn.Tanh),
        ("ReLU", nn.ReLU),
    ]
    results = []
    for name, ctor in configs:
        _, result = train_binary(ctor(), name)
        results.append(result)
    return results


def train_three_class(*, seed: int = SEED, lr: float = LR, steps: int = STEPS) -> dict:
    set_seed(seed)
    model = ThreeClassNet(nn.Tanh())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.no_grad():
        initial_loss = float(loss_fn(model(X_3CLASS), Y_3CLASS).item())

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X_3CLASS), Y_3CLASS)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X_3CLASS)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        final_loss = float(loss_fn(logits, Y_3CLASS).item())
        probability_sums = probabilities.sum(dim=1)

        shifted_logits = logits + 100.0
        shifted_probabilities = torch.softmax(shifted_logits, dim=1)
        stability_error = float((shifted_probabilities - probabilities).abs().max().item())

    return {
        "architecture": "2-2-3",
        "hidden_activation": "tanh",
        "output": "3 logits + CrossEntropyLoss",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "logits": logits.cpu().numpy().round(6).tolist(),
        "probabilities": probabilities.cpu().numpy().round(6).tolist(),
        "probability_sums": [float(v) for v in probability_sums],
        "predictions": [int(v) for v in predictions],
        "accuracy": float((predictions == Y_3CLASS).float().mean().item()),
        "max_shift_invariance_error": stability_error,
        "final_weight_shape": list(model.fc2.weight.shape),
        "seed": seed,
        "steps": steps,
        "learning_rate": lr,
    }


def save_json(name: str, payload: object) -> None:
    (RESULTS / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    linear = train_linear_baseline()
    save_json("linear_baseline.json", linear)

    binary_results = activation_experiments()
    save_json("activation_results.json", [asdict(result) for result in binary_results])

    with (RESULTS / "activation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(binary_results[0]).keys()))
        writer.writeheader()
        for result in binary_results:
            writer.writerow(asdict(result))

    symmetry = train_symmetry_experiment()
    save_json("symmetry_results.json", symmetry)

    three_class = train_three_class()
    save_json("three_class_results.json", three_class)

    print("=== LINEAR XOR BASELINE ===")
    print(json.dumps(linear, indent=2))
    print("\n=== ACTIVATION EXPERIMENT ===")
    for result in binary_results:
        print(json.dumps(asdict(result), indent=2))
    print("\n=== SYMMETRY EXPERIMENT ===")
    for row in symmetry["rows"]:
        print(row)
    print("\n=== THREE-CLASS EXTENSION ===")
    print(json.dumps(three_class, indent=2))


if __name__ == "__main__":
    main()
