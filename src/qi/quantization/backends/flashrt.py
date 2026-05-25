from __future__ import annotations

import importlib

import torch

from .reference import ReferenceBackend


def _load_flashrt_op():
    candidates = (
        "flash_rt",
        "flash_rt.ops",
        "flash_rt.gemm",
        "flashrt",
        "flashrt.ops",
        "flashrt.gemm",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in ("w4a16_gemm", "fused_w4a16_gemm", "flashrt_fused_w4a16_gemm"):
            op = getattr(module, attr, None)
            if op is not None:
                return op
    return None


class FlashRTBackend(ReferenceBackend):
    def __init__(self) -> None:
        self._op = _load_flashrt_op()

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
        if bits != 4:
            raise ValueError("FlashRT backend currently supports 4-bit weight-only GEMM only.")
        if not symmetric or zeros is not None:
            raise ValueError("FlashRT backend currently expects symmetric quantization.")
        x_kernel = x
        if input_scale is not None:
            x_kernel = x / input_scale.to(device=x.device, dtype=x.dtype)
        if self._op is None:
            return super().linear(
                x=x,
                qweight=qweight,
                scales=scales,
                zeros=zeros,
                bias=bias,
                bits=bits,
                group_size=group_size,
                symmetric=symmetric,
                input_scale=input_scale,
                keep_output_dtype=keep_output_dtype,
            )
        return self._op(
            x_kernel,
            qweight.contiguous(),
            scales.contiguous(),
            bias,
            group_size,
        )
