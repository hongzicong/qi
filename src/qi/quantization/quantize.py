from __future__ import annotations

import torch

from .algorithms import (
    QuantizedLinearWeight,
    dequantize_weight,
    quantize_weight_awq,
    quantize_weight_fp8_rtn,
    quantize_weight_fp8_smoothquant,
    quantize_weight_int8_smoothquant,
    quantize_weight_rtn,
)


def quantize_linear_weight(
    weight: torch.Tensor,
    cfg,
    act_stats=None,
    bias: torch.Tensor | None = None,
) -> QuantizedLinearWeight:
    if hasattr(cfg, "validate"):
        cfg.validate()

    if getattr(cfg, "quant_dtype", "int") == "int" and hasattr(cfg, "activation_granularity"):
        if cfg.algo == "rtn":
            qweight, scales, zeros = quantize_weight_rtn(
                weight, bits=8, group_size=-1, symmetric=True
            )
            return QuantizedLinearWeight(qweight=qweight, scales=scales, zeros=zeros)
        if cfg.algo == "smoothquant":
            return quantize_weight_int8_smoothquant(
                weight,
                act_stats=act_stats,
                alpha=getattr(cfg, "smoothquant_alpha", 0.5),
                eps=getattr(cfg, "smoothquant_scale_eps", 1e-4),
            )
        raise ValueError(f"Unknown INT8 W8A8 quantization algo: {cfg.algo}")

    if getattr(cfg, "quant_dtype", "int") == "fp8":
        if cfg.algo == "rtn":
            return quantize_weight_fp8_rtn(
                weight,
                activation_scale=getattr(cfg, "activation_scale", 1.0),
            )
        if cfg.algo == "smoothquant":
            return quantize_weight_fp8_smoothquant(
                weight,
                act_stats=act_stats,
                alpha=getattr(cfg, "smoothquant_alpha", 0.5),
                eps=getattr(cfg, "smoothquant_scale_eps", 1e-4),
            )
        raise ValueError(f"Unknown W8A8 quantization algo: {cfg.algo}")

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
            eps=getattr(cfg, "awq_scale_eps", 1e-4),
            n_grid=getattr(cfg, "awq_n_grid", 20),
            max_calib_samples=getattr(cfg, "awq_max_calib_samples", 256),
            bias=bias,
            enable_clip=getattr(cfg, "awq_enable_clip", True),
            clip_n_grid=getattr(cfg, "awq_clip_n_grid", 20),
            clip_max_shrink=getattr(cfg, "awq_clip_max_shrink", 0.5),
            clip_n_sample_token=getattr(cfg, "awq_clip_n_sample_token", 512),
            pack_backend=getattr(cfg, "backend", None),
        )
    raise ValueError(f"Unknown quantization algo: {cfg.algo}")
