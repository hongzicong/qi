from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class QuantizedLinearWeight:
    qweight: torch.Tensor | None
    scales: torch.Tensor | None
    zeros: torch.Tensor | None
    input_scale: torch.Tensor | None = None
    qweight_packed: torch.Tensor | None = None
    scale_factors: torch.Tensor | None = None
    weight_global_scale: torch.Tensor | None = None
    weight_fp8: torch.Tensor | None = None
    weight_scale: torch.Tensor | None = None
    activation_scale: torch.Tensor | None = None


def effective_group_size(group_size: int, in_features: int) -> int:
    if group_size <= 0 or group_size > in_features:
        return in_features
    return int(group_size)


def quantize_weight_rtn(
    weight: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D linear weight, got shape {tuple(weight.shape)}.")
    if bits not in (4, 8):
        raise ValueError(f"`bits` must be 4 or 8, got {bits}.")

    out_features, in_features = weight.shape
    group_size = effective_group_size(group_size, in_features)
    num_groups = math.ceil(in_features / group_size)

    qweight = torch.empty((out_features, in_features), dtype=torch.int8, device=weight.device)
    scales = torch.empty((out_features, num_groups), dtype=torch.float32, device=weight.device)
    zeros = None
    if not symmetric:
        zeros = torch.empty((out_features, num_groups), dtype=torch.int16, device=weight.device)

    weight_f = weight.detach().float()
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        block = weight_f[:, start:end]

        if symmetric:
            qmax = 2 ** (bits - 1) - 1
            scale = block.abs().amax(dim=1).clamp(min=1e-8) / qmax
            q = torch.round(block / scale[:, None]).clamp(-qmax, qmax)
            qweight[:, start:end] = q.to(torch.int8)
            scales[:, group_idx] = scale
        else:
            qmin = 0
            qmax = 2**bits - 1
            block_min = block.amin(dim=1)
            block_max = block.amax(dim=1)
            scale = ((block_max - block_min) / max(qmax - qmin, 1)).clamp(min=1e-8)
            zero = torch.round(qmin - block_min / scale).clamp(qmin, qmax)
            q = torch.round(block / scale[:, None] + zero[:, None]).clamp(qmin, qmax)
            qweight[:, start:end] = q.to(torch.int8)
            scales[:, group_idx] = scale
            assert zeros is not None
            zeros[:, group_idx] = zero.to(torch.int16)

    return qweight, scales, zeros


def dequantize_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    bits: int,
    group_size: int,
    symmetric: bool = True,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight.dim() != 2:
        raise ValueError(f"Expected a 2D qweight, got shape {tuple(qweight.shape)}.")

    out_features, in_features = qweight.shape
    group_size = effective_group_size(group_size, in_features)
    num_groups = math.ceil(in_features / group_size)
    if scales.shape != (out_features, num_groups):
        raise ValueError(
            f"Expected scales shape {(out_features, num_groups)}, got {tuple(scales.shape)}."
        )
    if not symmetric and zeros is None:
        raise ValueError("Asymmetric dequantization requires zeros.")

    out_dtype = dtype or torch.float16
    weight = torch.empty((out_features, in_features), dtype=out_dtype, device=qweight.device)
    qweight_f = qweight.float()

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        scale = scales[:, group_idx].float()[:, None]
        q = qweight_f[:, start:end]
        if symmetric:
            block = q * scale
        else:
            assert zeros is not None
            block = (q - zeros[:, group_idx].float()[:, None]) * scale
        weight[:, start:end] = block.to(out_dtype)

    return weight
