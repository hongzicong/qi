from __future__ import annotations

import torch
import torch.nn as nn


class WeightOnlyLinear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor | None,
        scales: torch.Tensor | None,
        zeros: torch.Tensor | None,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        bits: int,
        group_size: int,
        symmetric: bool,
        backend,
        input_scale: torch.Tensor | None = None,
        qweight_packed: torch.Tensor | None = None,
        scale_factors: torch.Tensor | None = None,
        weight_global_scale: torch.Tensor | None = None,
        activation_scale: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        weight_fp8: torch.Tensor | None = None,
        activation_granularity: str | None = None,
        weight_granularity: str | None = None,
        keep_output_dtype: str = "input",
    ):
        super().__init__()
        if qweight is None:
            self.qweight = None
        else:
            self.register_buffer("qweight", qweight.contiguous())
        if scales is None:
            self.scales = None
        else:
            self.register_buffer("scales", scales.contiguous())
        if qweight_packed is None:
            self.qweight_packed = None
        else:
            self.register_buffer("qweight_packed", qweight_packed.contiguous())
        if scale_factors is None:
            self.scale_factors = None
        else:
            self.register_buffer("scale_factors", scale_factors.contiguous())
        if weight_global_scale is None:
            self.weight_global_scale = None
        else:
            self.register_buffer("weight_global_scale", weight_global_scale.contiguous().float())
        if activation_scale is None:
            self.activation_scale = None
        else:
            self.register_buffer("activation_scale", activation_scale.contiguous().float())
        if weight_scale is None:
            self.weight_scale = None
        else:
            self.register_buffer("weight_scale", weight_scale.contiguous().float())
        if weight_fp8 is None:
            self.weight_fp8 = None
        else:
            self.register_buffer("weight_fp8", weight_fp8.contiguous())
        if zeros is None:
            self.zeros = None
        else:
            self.register_buffer("zeros", zeros.contiguous())
        if input_scale is None:
            self.input_scale = None
        else:
            self.register_buffer("input_scale", input_scale.contiguous().float())

        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.symmetric = bool(symmetric)
        self.backend = backend
        self.activation_granularity = activation_granularity
        self.weight_granularity = weight_granularity
        self.keep_output_dtype = keep_output_dtype

    @classmethod
    def from_linear(cls, linear: nn.Linear, quantized, cfg, backend) -> "WeightOnlyLinear":
        return cls(
            qweight=quantized.qweight,
            scales=quantized.scales,
            zeros=quantized.zeros,
            input_scale=quantized.input_scale,
            qweight_packed=getattr(quantized, "qweight_packed", None),
            scale_factors=getattr(quantized, "scale_factors", None),
            weight_global_scale=getattr(quantized, "weight_global_scale", None),
            activation_scale=getattr(quantized, "activation_scale", None),
            weight_scale=getattr(quantized, "weight_scale", None),
            weight_fp8=getattr(quantized, "weight_fp8", None),
            bias=linear.bias.detach() if linear.bias is not None else None,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=cfg.bits,
            group_size=int(getattr(cfg, "group_size", -1)),
            symmetric=bool(getattr(cfg, "symmetric", True)),
            backend=backend,
            activation_granularity=getattr(cfg, "activation_granularity", None),
            weight_granularity=getattr(cfg, "weight_granularity", None),
            keep_output_dtype=getattr(cfg, "keep_output_dtype", "input"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.linear(
            x=x,
            qweight=self.qweight,
            scales=self.scales,
            zeros=self.zeros,
            bias=self.bias,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            input_scale=self.input_scale,
            qweight_packed=self.qweight_packed,
            scale_factors=self.scale_factors,
            weight_global_scale=self.weight_global_scale,
            activation_scale=self.activation_scale,
            weight_scale=self.weight_scale,
            weight_fp8=self.weight_fp8,
            activation_granularity=self.activation_granularity,
            weight_granularity=self.weight_granularity,
            keep_output_dtype=self.keep_output_dtype,
        )
