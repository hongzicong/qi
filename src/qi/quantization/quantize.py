from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from .backends.flashrt_nvfp4_ops import load_flashrt_nvfp4_ops


@dataclass
class QuantizedLinearWeight:
    qweight: torch.Tensor | None
    scales: torch.Tensor | None
    zeros: torch.Tensor | None
    input_scale: torch.Tensor | None = None
    qweight_packed: torch.Tensor | None = None
    scale_factors: torch.Tensor | None = None
    weight_global_scale: torch.Tensor | None = None


def _effective_group_size(group_size: int, in_features: int) -> int:
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
    group_size = _effective_group_size(group_size, in_features)
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
    group_size = _effective_group_size(group_size, in_features)
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


def _resolve_act_stats(act_stats, in_features: int, device: torch.device) -> torch.Tensor:
    if act_stats is None:
        raise ValueError("AWQ quantization requires calibration activation statistics.")
    if isinstance(act_stats, Mapping):
        if "abs_mean" in act_stats:
            act_stats = act_stats["abs_mean"]
        elif "mean_abs" in act_stats:
            act_stats = act_stats["mean_abs"]
        else:
            raise ValueError("AWQ act_stats mapping must contain `abs_mean` or `mean_abs`.")
    act = torch.as_tensor(act_stats, dtype=torch.float32, device=device).flatten()
    if act.numel() != in_features:
        raise ValueError(f"Expected AWQ act_stats with {in_features} values, got {act.numel()}.")
    return act


def _awq_input_scale(weight: torch.Tensor, act_stats, alpha: float, eps: float) -> torch.Tensor:
    _, in_features = weight.shape
    act = _resolve_act_stats(act_stats, in_features, weight.device).clamp(min=eps)
    weight_col = weight.detach().float().abs().amax(dim=0).clamp(min=eps)
    scale = act.pow(float(alpha)) / weight_col.pow(1.0 - float(alpha))
    scale = scale / torch.sqrt(scale.amax() * scale.amin()).clamp(min=eps)
    return scale.clamp(min=eps).to(dtype=torch.float32)


def quantize_weight_awq(
    weight: torch.Tensor,
    act_stats,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-4,
) -> QuantizedLinearWeight:
    input_scale = _awq_input_scale(weight, act_stats, alpha=alpha, eps=eps)
    scaled_weight = weight.detach().float() * input_scale.to(device=weight.device)[None, :]
    qweight, scales, zeros = quantize_weight_rtn(
        scaled_weight,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    return QuantizedLinearWeight(qweight=qweight, scales=scales, zeros=zeros, input_scale=input_scale)


def quantize_weight_nvfp4_rtn(weight: torch.Tensor) -> QuantizedLinearWeight:
    if not weight.is_cuda:
        raise ValueError("FlashRT NVFP4 quantization requires CUDA weights.")
    if weight.shape[1] % 16 != 0:
        raise ValueError(f"FlashRT NVFP4 requires in_features to be divisible by 16, got {weight.shape[1]}.")
    ops = load_flashrt_nvfp4_ops(required=True)
    qweight_packed, scale_factors, weight_global_scale = ops.quantize_weight_dynamic(
        weight.detach().to(dtype=torch.bfloat16).contiguous()
    )
    return QuantizedLinearWeight(
        qweight=None,
        scales=None,
        zeros=None,
        qweight_packed=qweight_packed,
        scale_factors=scale_factors,
        weight_global_scale=weight_global_scale,
    )


def quantize_weight_nvfp4_awq(
    weight: torch.Tensor,
    act_stats,
    alpha: float = 0.5,
    eps: float = 1e-4,
) -> QuantizedLinearWeight:
    if not weight.is_cuda:
        raise ValueError("FlashRT NVFP4 quantization requires CUDA weights.")
    if weight.shape[1] % 16 != 0:
        raise ValueError(f"FlashRT NVFP4 requires in_features to be divisible by 16, got {weight.shape[1]}.")
    input_scale = _awq_input_scale(weight, act_stats, alpha=alpha, eps=eps)
    scaled_weight = weight.detach().float() * input_scale.to(device=weight.device)[None, :]
    ops = load_flashrt_nvfp4_ops(required=True)
    qweight_packed, scale_factors, weight_global_scale = ops.quantize_weight_dynamic(
        scaled_weight.to(dtype=torch.bfloat16).contiguous()
    )
    return QuantizedLinearWeight(
        qweight=None,
        scales=None,
        zeros=None,
        input_scale=input_scale,
        qweight_packed=qweight_packed,
        scale_factors=scale_factors,
        weight_global_scale=weight_global_scale,
    )


def quantize_linear_weight(weight: torch.Tensor, cfg, act_stats=None) -> QuantizedLinearWeight:
    if hasattr(cfg, "validate"):
        cfg.validate()

    if getattr(cfg, "quant_dtype", "int") == "nvfp4":
        if cfg.algo == "rtn":
            return quantize_weight_nvfp4_rtn(weight)
        if cfg.algo == "awq":
            return quantize_weight_nvfp4_awq(
                weight,
                act_stats=act_stats,
                alpha=getattr(cfg, "awq_alpha", 0.5),
                eps=getattr(cfg, "awq_scale_eps", 1e-4),
            )
        raise ValueError(f"Unknown quantization algo: {cfg.algo}")

    if cfg.algo == "rtn":
        qweight, scales, zeros = quantize_weight_rtn(
            weight,
            bits=cfg.bits,
            group_size=cfg.group_size,
            symmetric=cfg.symmetric,
        )
        return QuantizedLinearWeight(qweight=qweight, scales=scales, zeros=zeros)
    if cfg.algo == "awq":
        return quantize_weight_awq(
            weight,
            act_stats=act_stats,
            bits=cfg.bits,
            group_size=cfg.group_size,
            symmetric=cfg.symmetric,
            alpha=getattr(cfg, "awq_alpha", 0.5),
            eps=getattr(cfg, "awq_scale_eps", 1e-4),
        )
    raise ValueError(f"Unknown quantization algo: {cfg.algo}")
