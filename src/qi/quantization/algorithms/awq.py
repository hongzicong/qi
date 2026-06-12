from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .rtn import QuantizedLinearWeight, dequantize_weight, quantize_weight_rtn


def _calibration_inputs_from_stats(
    act_stats,
    in_features: int,
    device: torch.device,
    max_samples: int | None = None,
) -> torch.Tensor:
    if isinstance(act_stats, Mapping):
        source = None
        for key in ("inputs", "input", "samples", "x"):
            if key in act_stats:
                source = act_stats[key]
                break
        if source is None:
            raise ValueError(
                "AWQ requires calibration input samples. Use collect_awq_calibration_cache() "
                "or provide a mapping with one of: inputs, input, samples, x."
            )
    elif isinstance(act_stats, torch.Tensor) and act_stats.dim() > 1:
        source = act_stats
    elif isinstance(act_stats, Sequence) and not isinstance(act_stats, (str, bytes)):
        source = act_stats
    else:
        raise ValueError(
            "AWQ requires calibration input samples, not only channel activation stats. "
            "SmoothQuant is the stats-only activation-smoothing path."
        )

    if isinstance(source, Sequence) and not isinstance(source, torch.Tensor):
        tensors = [torch.as_tensor(item) for item in source]
        if not tensors:
            raise ValueError("AWQ calibration sample sequence is empty.")
        source = torch.cat([tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors], dim=0)

    samples = torch.as_tensor(source, dtype=torch.float32, device=device)
    if samples.dim() == 1:
        raise ValueError(
            "AWQ calibration samples must include a sample dimension. "
            f"Expected shape (*, {in_features}), got {tuple(samples.shape)}."
        )
    samples = samples.reshape(-1, samples.shape[-1])
    if samples.shape[-1] != in_features:
        raise ValueError(f"Expected AWQ calibration samples with K={in_features}, got {samples.shape[-1]}.")
    if max_samples is not None and samples.shape[0] > max_samples:
        samples = samples[:max_samples]
    return samples.contiguous()


def _candidate_awq_scale(
    weight: torch.Tensor,
    samples: torch.Tensor,
    ratio: float,
    eps: float,
) -> torch.Tensor:
    act = samples.detach().float().abs().mean(dim=0).clamp(min=eps)
    weight_col = weight.detach().float().abs().amax(dim=0).clamp(min=eps)
    scale = act.pow(float(ratio)) / weight_col.pow(1.0 - float(ratio))
    scale = scale / torch.sqrt(scale.amax() * scale.amin()).clamp(min=eps)
    return scale.clamp(min=eps).to(dtype=torch.float32)


@torch.no_grad()
def search_awq_input_scale(
    weight: torch.Tensor,
    act_stats,
    *,
    bias: torch.Tensor | None = None,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    n_grid: int = 20,
    eps: float = 1e-4,
    max_samples: int | None = 256,
) -> torch.Tensor:
    if n_grid <= 0:
        raise ValueError(f"`n_grid` must be positive, got {n_grid}.")

    in_features = weight.shape[1]
    samples = _calibration_inputs_from_stats(act_stats, in_features, weight.device, max_samples=max_samples)
    samples = samples.to(device=weight.device, dtype=torch.float32)
    weight_f = weight.detach().float()
    bias_f = None if bias is None else bias.detach().to(device=weight.device, dtype=torch.float32)
    target = F.linear(samples, weight_f, bias_f)

    best_scale: torch.Tensor | None = None
    best_loss: torch.Tensor | None = None
    denominator = max(n_grid - 1, 1)
    for grid_idx in range(n_grid):
        ratio = grid_idx / float(denominator)
        scale = _candidate_awq_scale(weight_f, samples, ratio=ratio, eps=eps)
        scaled_weight = weight_f * scale[None, :]
        qweight, scales, zeros = quantize_weight_rtn(
            scaled_weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
        )
        qdq_weight = dequantize_weight(
            qweight=qweight,
            scales=scales,
            zeros=zeros,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            dtype=torch.float32,
        )
        out = F.linear(samples / scale[None, :], qdq_weight, bias_f)
        loss = (out - target).float().pow(2).mean()
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_scale = scale

    assert best_scale is not None
    return best_scale.detach().to(dtype=torch.float32)


@torch.no_grad()
def search_awq_clip(
    scaled_weight: torch.Tensor,
    scaled_samples: torch.Tensor,
    target: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_sample_token: int = 512,
) -> torch.Tensor:
    if n_grid <= 0:
        raise ValueError(f"`n_grid` must be positive, got {n_grid}.")
    if not 0.0 <= float(max_shrink) < 1.0:
        raise ValueError(f"`max_shrink` must be in [0, 1), got {max_shrink}.")
    if n_sample_token <= 0:
        raise ValueError(f"`n_sample_token` must be positive, got {n_sample_token}.")

    weight_f = scaled_weight.detach().float()
    samples = scaled_samples.detach().float()
    if samples.shape[0] > n_sample_token:
        samples = samples[:n_sample_token]
        target = target[:n_sample_token]
    target = target.detach().float()
    bias_f = None if bias is None else bias.detach().to(device=weight_f.device, dtype=torch.float32)

    out_features = weight_f.shape[0]
    original_max = weight_f.abs().amax(dim=1).clamp(min=1e-8)
    best_weight = weight_f.clone()
    best_loss = torch.full((out_features,), float("inf"), dtype=torch.float32, device=weight_f.device)

    denominator = max(n_grid - 1, 1)
    for grid_idx in range(n_grid):
        shrink = float(max_shrink) * grid_idx / float(denominator)
        clip_max = original_max * (1.0 - shrink)
        clipped = torch.minimum(torch.maximum(weight_f, -clip_max[:, None]), clip_max[:, None])
        qweight, scales, zeros = quantize_weight_rtn(
            clipped,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
        )
        qdq_weight = dequantize_weight(
            qweight=qweight,
            scales=scales,
            zeros=zeros,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            dtype=torch.float32,
        )
        out = F.linear(samples, qdq_weight, bias_f)
        loss = (out - target).float().pow(2).mean(dim=0)
        improved = loss < best_loss
        if improved.any():
            best_loss = torch.where(improved, loss, best_loss)
            best_weight[improved] = clipped[improved]

    return best_weight


def quantize_weight_awq(
    weight: torch.Tensor,
    act_stats,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    eps: float = 1e-4,
    n_grid: int = 20,
    max_calib_samples: int | None = 256,
    bias: torch.Tensor | None = None,
    enable_clip: bool = True,
    clip_n_grid: int = 20,
    clip_max_shrink: float = 0.5,
    clip_n_sample_token: int = 512,
    pack_backend: str | None = None,
) -> QuantizedLinearWeight:
    input_scale = search_awq_input_scale(
        weight,
        act_stats,
        bias=bias,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        n_grid=n_grid,
        eps=eps,
        max_samples=max_calib_samples,
    )
    samples = _calibration_inputs_from_stats(
        act_stats,
        weight.shape[1],
        weight.device,
        max_samples=max(max_calib_samples or 0, clip_n_sample_token) if enable_clip else max_calib_samples,
    )
    weight_f = weight.detach().float()
    bias_f = None if bias is None else bias.detach().to(device=weight.device, dtype=torch.float32)
    scaled_weight = weight_f * input_scale.to(device=weight.device)[None, :]
    if enable_clip:
        scaled_samples = samples.to(device=weight.device, dtype=torch.float32) / input_scale.to(
            device=weight.device, dtype=torch.float32
        )[None, :]
        target = F.linear(samples.to(device=weight.device, dtype=torch.float32), weight_f, bias_f)
        scaled_weight = search_awq_clip(
            scaled_weight,
            scaled_samples,
            target,
            bias=bias_f,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            n_grid=clip_n_grid,
            max_shrink=clip_max_shrink,
            n_sample_token=clip_n_sample_token,
        )
    qweight, scales, zeros = quantize_weight_rtn(
        scaled_weight,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    qweight_packed = None
    scale_factors = None
    if pack_backend == "int4_w4a16":
        from ..backends.int4_w4a16 import pack_int4_w4a16_weight

        qweight_packed, scale_factors = pack_int4_w4a16_weight(
            qweight,
            scales,
            zeros,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
        )
    elif pack_backend not in (None, "reference", "cuda_ext"):
        raise ValueError(f"Unsupported AWQ pack backend: {pack_backend}")
    return QuantizedLinearWeight(
        qweight=qweight,
        scales=scales,
        zeros=zeros,
        input_scale=input_scale,
        qweight_packed=qweight_packed,
        scale_factors=scale_factors,
    )
