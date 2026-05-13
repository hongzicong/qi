from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FixedStepCacheConfig:
    reuse_steps: int = 1
    warmup_steps: int = 1
    cooldown_steps: int = 1

    def __post_init__(self) -> None:
        if self.reuse_steps < 0:
            raise ValueError(f"`reuse_steps` must be non-negative, got {self.reuse_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"`warmup_steps` must be non-negative, got {self.warmup_steps}")
        if self.cooldown_steps < 0:
            raise ValueError(f"`cooldown_steps` must be non-negative, got {self.cooldown_steps}")


@dataclass(frozen=True)
class FixedStepCacheDecision:
    should_compute: bool
    should_reuse: bool


class FixedStepCacheState:
    """Fixed-step residual cache for a single DiT body during inference."""

    def __init__(self, config: FixedStepCacheConfig, num_steps: int):
        if num_steps <= 0:
            raise ValueError(f"`num_steps` must be positive, got {num_steps}")
        self.config = config
        self.num_steps = int(num_steps)
        self.previous_residual: torch.Tensor | None = None
        self.last_compute_step: int | None = None
        self.pending_compute_step: int | None = None
        self.compute_count = 0
        self.reuse_count = 0

    @classmethod
    def from_values(
        cls,
        *,
        num_steps: int,
        reuse_steps: int = 1,
        warmup_steps: int = 1,
        cooldown_steps: int = 1,
    ) -> "FixedStepCacheState":
        return cls(
            FixedStepCacheConfig(
                reuse_steps=int(reuse_steps),
                warmup_steps=int(warmup_steps),
                cooldown_steps=int(cooldown_steps),
            ),
            num_steps=int(num_steps),
        )

    def begin_step(self, step_index: int) -> FixedStepCacheDecision:
        if step_index < 0 or step_index >= self.num_steps:
            raise ValueError(
                f"`step_index` must be in [0, {self.num_steps}), got {step_index}"
            )

        force_compute = (
            self.previous_residual is None
            or self.last_compute_step is None
            or step_index < self.config.warmup_steps
            or step_index >= self.num_steps - self.config.cooldown_steps
        )
        if force_compute:
            self.pending_compute_step = step_index
            return FixedStepCacheDecision(should_compute=True, should_reuse=False)

        steps_since_compute = step_index - self.last_compute_step
        should_reuse = 0 < steps_since_compute <= self.config.reuse_steps
        if should_reuse:
            self.reuse_count += 1
            self.pending_compute_step = None
            return FixedStepCacheDecision(should_compute=False, should_reuse=True)

        self.pending_compute_step = step_index
        return FixedStepCacheDecision(should_compute=True, should_reuse=False)

    def reuse(self, body_input: torch.Tensor) -> torch.Tensor:
        if self.previous_residual is None:
            raise RuntimeError("Cannot reuse fixed-step residual before a body computation.")
        return body_input + self.previous_residual.to(device=body_input.device, dtype=body_input.dtype)

    def update(self, body_input: torch.Tensor, body_output: torch.Tensor) -> None:
        if self.pending_compute_step is None:
            raise RuntimeError("`update` called without a pending fixed-step cache computation.")
        if body_input.shape != body_output.shape:
            raise ValueError(
                "`body_input` and `body_output` shape mismatch: "
                f"{tuple(body_input.shape)} vs {tuple(body_output.shape)}"
            )
        self.previous_residual = (body_output - body_input).detach().clone()
        self.last_compute_step = self.pending_compute_step
        self.pending_compute_step = None
        self.compute_count += 1

    def update_residual(self, residual: torch.Tensor) -> None:
        if self.pending_compute_step is None:
            raise RuntimeError("`update_residual` called without a pending fixed-step cache computation.")
        self.previous_residual = residual.detach()
        self.last_compute_step = self.pending_compute_step
        self.pending_compute_step = None
        self.compute_count += 1
