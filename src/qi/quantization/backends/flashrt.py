from __future__ import annotations

import importlib

import torch

from .flashrt_nvfp4_ops import FlashRTNVFP4Unavailable, load_flashrt_nvfp4_ops
from .reference import ReferenceBackend, _output_dtype


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
        qweight_packed: torch.Tensor | None = None,
        scale_factors: torch.Tensor | None = None,
        weight_global_scale: torch.Tensor | None = None,
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


class FlashRTNVFP4Backend(ReferenceBackend):
    def __init__(self) -> None:
        self._ops = load_flashrt_nvfp4_ops(required=False)

    def linear(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor | None,
        scales: torch.Tensor | None,
        zeros: torch.Tensor | None,
        bias: torch.Tensor | None,
        bits: int,
        group_size: int,
        symmetric: bool,
        input_scale: torch.Tensor | None,
        keep_output_dtype: str,
        qweight_packed: torch.Tensor | None = None,
        scale_factors: torch.Tensor | None = None,
        weight_global_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del qweight, scales, weight_global_scale
        if self._ops is None:
            raise FlashRTNVFP4Unavailable(
                "FlashRT NVFP4 kernels are not available. Build/install FlashRT so "
                "`flash_rt.flash_rt_kernels` exposes GemmRunner.fp4_nn_dev and "
                "NVFP4 quantize functions."
            )
        if bits != 4 or group_size != 16:
            raise ValueError("FlashRT NVFP4 backend requires bits=4 and group_size=16.")
        if not symmetric or zeros is not None:
            raise ValueError("FlashRT NVFP4 backend currently expects symmetric quantization.")
        if qweight_packed is None or scale_factors is None:
            raise ValueError("FlashRT NVFP4 backend requires qweight_packed and scale_factors buffers.")
        if not x.is_cuda or not qweight_packed.is_cuda or not scale_factors.is_cuda:
            raise ValueError("FlashRT NVFP4 backend requires CUDA tensors.")
        if x.shape[-1] % 16 != 0:
            raise ValueError(f"FlashRT NVFP4 requires K divisible by 16, got {x.shape[-1]}.")

        out_dtype = _output_dtype(x, keep_output_dtype)
        x_kernel = x
        if input_scale is not None:
            x_kernel = x_kernel / input_scale.to(device=x.device, dtype=x.dtype)

        in_features = int(x_kernel.shape[-1])
        out_features = int(qweight_packed.shape[0])
        x_2d = x_kernel.reshape(-1, in_features).to(dtype=torch.bfloat16).contiguous()
        m = int(x_2d.shape[0])

        a_packed, sfa, _ = self._ops.quantize_weight_dynamic(x_2d)
        out = torch.empty((m, out_features), dtype=torch.bfloat16, device=x.device)
        self._ops.fp4_nn_dev(
            a_packed,
            sfa,
            qweight_packed.contiguous(),
            scale_factors.contiguous(),
            out,
            m,
            out_features,
            in_features,
        )
        if bias is not None:
            out = out + bias.to(device=x.device, dtype=out.dtype)
        return out.reshape(*x.shape[:-1], out_features).to(out_dtype)
