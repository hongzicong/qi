from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from .swap import should_quantize


@dataclass
class _RunningAbsMean:
    total: torch.Tensor | None = None
    count: int = 0

    def update(self, x: torch.Tensor) -> None:
        values = x.detach().float().abs()
        if values.dim() == 1:
            values = values.view(1, -1)
        values = values.reshape(-1, values.shape[-1])
        batch_sum = values.sum(dim=0).cpu()
        if self.total is None:
            self.total = batch_sum
        else:
            self.total += batch_sum
        self.count += values.shape[0]

    def mean(self) -> torch.Tensor:
        if self.total is None or self.count == 0:
            raise RuntimeError("No calibration activations were collected.")
        return self.total / self.count


def _call_model(model: nn.Module, batch, forward_fn: Callable | None):
    if forward_fn is not None:
        return forward_fn(model, batch)
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


@torch.no_grad()
def collect_awq_activation_stats(
    model: nn.Module,
    calibration_data: Iterable,
    cfg,
    forward_fn: Callable[[nn.Module, object], object] | None = None,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    collectors: dict[str, _RunningAbsMean] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            collectors[name].update(inputs[0])

        return hook

    for name, module in model.named_modules():
        if should_quantize(name, module, cfg):
            collectors[name] = _RunningAbsMean()
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    if not collectors:
        return {}

    was_training = model.training
    model.eval()
    try:
        for batch_idx, batch in enumerate(calibration_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            _call_model(model, batch, forward_fn)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {name: collector.mean() for name, collector in collectors.items()}


@dataclass
class _RunningAWQCache:
    max_samples: int
    total: torch.Tensor | None = None
    count: int = 0
    samples: list[torch.Tensor] | None = None

    def update(self, x: torch.Tensor) -> None:
        values = x.detach().float()
        if values.dim() == 1:
            values = values.view(1, -1)
        values = values.reshape(-1, values.shape[-1])
        abs_values = values.abs()
        batch_sum = abs_values.sum(dim=0).cpu()
        if self.total is None:
            self.total = batch_sum
        else:
            self.total += batch_sum
        self.count += values.shape[0]

        if self.samples is None:
            self.samples = []
        current = sum(sample.shape[0] for sample in self.samples)
        remaining = self.max_samples - current
        if remaining > 0:
            self.samples.append(values[:remaining].cpu())

    def result(self) -> dict[str, torch.Tensor]:
        if self.total is None or self.count == 0:
            raise RuntimeError("No calibration activations were collected.")
        if not self.samples:
            raise RuntimeError("No AWQ calibration samples were collected.")
        return {
            "abs_mean": self.total / self.count,
            "inputs": torch.cat(self.samples, dim=0),
        }


@torch.no_grad()
def collect_awq_calibration_cache(
    model: nn.Module,
    calibration_data: Iterable,
    cfg,
    forward_fn: Callable[[nn.Module, object], object] | None = None,
    max_batches: int | None = None,
    max_samples_per_module: int = 256,
) -> dict[str, dict[str, torch.Tensor]]:
    collectors: dict[str, _RunningAWQCache] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            collectors[name].update(inputs[0])

        return hook

    for name, module in model.named_modules():
        if should_quantize(name, module, cfg):
            collectors[name] = _RunningAWQCache(max_samples=int(max_samples_per_module))
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    if not collectors:
        return {}

    was_training = model.training
    model.eval()
    try:
        for batch_idx, batch in enumerate(calibration_data):
            if max_batches is not None and batch_idx >= max_batches:
                break
            _call_model(model, batch, forward_fn)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {name: collector.result() for name, collector in collectors.items()}
