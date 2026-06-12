from __future__ import annotations

from collections.abc import Mapping

import torch

from ..backends.flashrt_fp8_ops import load_flashrt_fp8_ops, quantize_to_fp8_e4m3
from .rtn import QuantizedLinearWeight, quantize_weight_rtn


def _resolve_act_stats(act_stats, in_features: int, device: torch.device) -> torch.Tensor:
    if act_stats is None:
        raise ValueError("SmoothQuant requires calibration activation statistics.")
    if isinstance(act_stats, Mapping):
        for key in ("abs_mean", "mean_abs", "abs_max", "amax", "max_abs"):
            if key in act_stats:
                act_stats = act_stats[key]
                break
        else:
            raise ValueError(
                "act_stats mapping must contain one of abs_mean, mean_abs, abs_max, amax, or max_abs."
            )
    act = torch.as_tensor(act_stats, dtype=torch.float32, device=device).flatten()
    if act.numel() != in_features:
        raise ValueError(f"Expected activation stats with {in_features} values, got {act.numel()}.")
    return act


def compute_smoothquant_input_scale(
    weight: torch.Tensor,
    act_stats,
    alpha: float,
    eps: float,
) -> torch.Tensor:
    _, in_features = weight.shape
    act = _resolve_act_stats(act_stats, in_features, weight.device).clamp(min=eps)
    weight_col = weight.detach().float().abs().amax(dim=0).clamp(min=eps)
    scale = act.pow(float(alpha)) / weight_col.pow(1.0 - float(alpha))
    scale = scale / torch.sqrt(scale.amax() * scale.amin()).clamp(min=eps)
    return scale.clamp(min=eps).to(dtype=torch.float32)


def _activation_scale_from_stats(act_stats, device: torch.device, fallback: float = 1.0) -> torch.Tensor:
    if act_stats is None:
        scale = float(fallback)
    elif isinstance(act_stats, Mapping):
        source = None
        for key in ("global_abs_max", "abs_max", "amax", "max_abs", "abs_mean", "mean_abs"):
            if key in act_stats:
                source = act_stats[key]
                break
        if source is None:
            scale = float(fallback)
        else:
            scale = float(torch.as_tensor(source, dtype=torch.float32).abs().max().item()) / 448.0
    else:
        scale = float(torch.as_tensor(act_stats, dtype=torch.float32).abs().max().item()) / 448.0
    return torch.tensor([max(scale, 1e-12)], dtype=torch.float32, device=device)


def quantize_weight_int8_smoothquant(
    weight: torch.Tensor,
    act_stats,
    alpha: float = 0.5,
    eps: float = 1e-4,
) -> QuantizedLinearWeight:
    input_scale = compute_smoothquant_input_scale(weight, act_stats, alpha=alpha, eps=eps)
    scaled_weight = weight.detach().float() * input_scale.to(device=weight.device)[None, :]
    qweight, scales, zeros = quantize_weight_rtn(
        scaled_weight, bits=8, group_size=-1, symmetric=True
    )
    return QuantizedLinearWeight(qweight=qweight, scales=scales, zeros=zeros, input_scale=input_scale)


def quantize_weight_fp8_rtn(weight: torch.Tensor, activation_scale: float = 1.0) -> QuantizedLinearWeight:
    ops = load_flashrt_fp8_ops(required=False)
    if ops is not None and weight.is_cuda:
        weight_fp8, weight_scale = ops.quantize_weight(weight.detach())
    else:
        weight_fp8, weight_scale = quantize_to_fp8_e4m3(weight.detach().contiguous())
    act_scale = torch.tensor([float(activation_scale)], dtype=torch.float32, device=weight.device).clamp(min=1e-12)
    return QuantizedLinearWeight(
        qweight=None,
        scales=None,
        zeros=None,
        weight_fp8=weight_fp8,
        weight_scale=weight_scale,
        activation_scale=act_scale,
    )


def quantize_weight_fp8_smoothquant(
    weight: torch.Tensor,
    act_stats,
    alpha: float = 0.5,
    eps: float = 1e-4,
) -> QuantizedLinearWeight:
    input_scale = compute_smoothquant_input_scale(weight, act_stats, alpha=alpha, eps=eps)
    scaled_weight = weight.detach().float() * input_scale.to(device=weight.device)[None, :]
    quantized = quantize_weight_fp8_rtn(scaled_weight, activation_scale=1.0)
    quantized.input_scale = input_scale
    quantized.activation_scale = _activation_scale_from_stats(act_stats, weight.device)
    return quantized
