from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from ..quantize import dequantize_weight


class QuantBackend(Protocol):
    def linear(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        bias: torch.Tensor | None,
        bits: int,
        group_size: int,
        symmetric: bool,
        input_scale: torch.Tensor | None,
        keep_output_dtype: str,
    ) -> torch.Tensor:
        ...


def _output_dtype(x: torch.Tensor, keep_output_dtype: str) -> torch.dtype:
    if keep_output_dtype == "input":
        return x.dtype
    if keep_output_dtype == "fp16":
        return torch.float16
    if keep_output_dtype == "bf16":
        return torch.bfloat16
    if keep_output_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported keep_output_dtype: {keep_output_dtype}")


class ReferenceBackend:
    def linear(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        bias: torch.Tensor | None,
        bits: int,
        group_size: int,
        symmetric: bool,
        input_scale: torch.Tensor | None,
        keep_output_dtype: str,
    ) -> torch.Tensor:
        out_dtype = _output_dtype(x, keep_output_dtype)
        compute_dtype = torch.float32 if x.dtype == torch.float32 else out_dtype
        x_compute = x.to(compute_dtype)
        if input_scale is not None:
            x_compute = x_compute / input_scale.to(device=x.device, dtype=compute_dtype)
        weight = dequantize_weight(
            qweight=qweight,
            scales=scales,
            zeros=zeros,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            dtype=compute_dtype,
        )
        bias_compute = None if bias is None else bias.to(device=x.device, dtype=compute_dtype)
        return F.linear(x_compute, weight, bias_compute).to(out_dtype)


class CudaExtBackend(ReferenceBackend):
    def linear(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("cuda_ext backend is reserved for the in-house fused kernel.")
