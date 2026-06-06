from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

import torch


_KERNEL_MODULE_CANDIDATES = (
    "flash_rt.flash_rt_kernels",
    "flash_rt_kernels",
)

_REQUIRED_KERNEL_FUNCTIONS = ("quantize_fp8_static",)
_FP8_E4M3_MAX = 448.0
_FP8_SCALE_FLOOR = 1e-12


class FlashRTFP8Unavailable(RuntimeError):
    pass



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


def _fp8_scale(tensor: torch.Tensor) -> float:
    amax = float(tensor.detach().float().abs().max().item())
    return max(amax / _FP8_E4M3_MAX, _FP8_SCALE_FLOOR)


def quantize_to_fp8_e4m3(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _fp8_scale(tensor)
    scaled = tensor.detach().float().div(scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    qbytes = scaled.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device=tensor.device)
    return qbytes, scale_tensor


def dequantize_fp8_e4m3(qbytes: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    scale_f = float(scale.item() if isinstance(scale, torch.Tensor) else scale)
    return qbytes.view(torch.float8_e4m3fn).float().mul(scale_f)


@dataclass(frozen=True)
class FlashRTFP8Ops:
    kernels: ModuleType
    runner: object
    quantize_activation_static: Callable

    def quantize_weight(self, weight_nk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if weight_nk.dim() != 2:
            raise ValueError(f"Expected a 2D linear weight, got shape {tuple(weight_nk.shape)}.")
        # nn.Linear and fp8_run_dev both consume B as (N, K).
        return quantize_to_fp8_e4m3(weight_nk.detach().contiguous())

    def fp8_run_dev(
        self,
        a_fp8: torch.Tensor,
        b_fp8: torch.Tensor,
        out: torch.Tensor,
        m: int,
        n: int,
        k: int,
        act_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        self.runner.fp8_run_dev(
            a_fp8.data_ptr(),
            b_fp8.data_ptr(),
            out.data_ptr(),
            int(m),
            int(n),
            int(k),
            act_scale.data_ptr(),
            weight_scale.data_ptr(),
            torch.cuda.current_stream(out.device).cuda_stream,
        )



def _hardware_note() -> str:
    if not torch.cuda.is_available():
        return " CUDA is not available."
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if major < 8:
        return f" Current GPU is {name} sm_{major}{minor}; FlashRT FP8 expects NVIDIA FP8-capable CUDA hardware."
    return f" Current GPU is {name} sm_{major}{minor}."


def _import_kernel_module() -> ModuleType:
    errors: list[str] = []
    for module_name in _KERNEL_MODULE_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
    raise FlashRTFP8Unavailable(
        "Could not import FlashRT kernel module. Tried: "
        + "; ".join(errors)
        + "."
        + _hardware_note()
    )


def load_flashrt_fp8_ops(required: bool = True) -> FlashRTFP8Ops | None:
    try:
        kernels = _import_kernel_module()
        runner_cls = getattr(kernels, "GemmRunner", None)
        if runner_cls is None:
            raise FlashRTFP8Unavailable("FlashRT kernel module is missing GemmRunner.")

        missing = [name for name in _REQUIRED_KERNEL_FUNCTIONS if getattr(kernels, name, None) is None]
        if missing:
            raise FlashRTFP8Unavailable(
                "FlashRT kernel module is missing FP8 functions: "
                + ", ".join(missing)
                + "."
                + _hardware_note()
            )

        runner = runner_cls()
        if getattr(runner, "fp8_run_dev", None) is None:
            raise FlashRTFP8Unavailable("FlashRT GemmRunner is missing fp8_run_dev.")

        return FlashRTFP8Ops(
            kernels=kernels,
            runner=runner,
            quantize_activation_static=getattr(kernels, "quantize_fp8_static"),
        )
    except FlashRTFP8Unavailable:
        if required:
            raise
        return None


class FlashRTFP8Backend:
    def __init__(self) -> None:
        self._ops = load_flashrt_fp8_ops(required=False)

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
        activation_scale: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        weight_fp8: torch.Tensor | None = None,
        activation_granularity: str | None = None,
        weight_granularity: str | None = None,
    ) -> torch.Tensor:
        del qweight, scales, zeros, group_size, symmetric, activation_granularity, weight_granularity
        if self._ops is None:
            raise FlashRTFP8Unavailable(
                "FlashRT FP8 kernels are not available. Build/install FlashRT so "
                "`flash_rt.flash_rt_kernels` exposes quantize_fp8_static and "
                "GemmRunner.fp8_run_dev."
                + _hardware_note()
            )
        if bits != 8:
            raise ValueError("FlashRT FP8 backend requires W8A8, so bits must be 8.")

        weight_fp8 = weight_fp8 if weight_fp8 is not None else qweight_packed
        weight_scale = weight_scale if weight_scale is not None else weight_global_scale
        activation_scale = activation_scale if activation_scale is not None else scale_factors
        if weight_fp8 is None or weight_scale is None or activation_scale is None:
            raise ValueError("FlashRT FP8 backend requires weight_fp8, weight_scale, and activation_scale buffers.")
        if not x.is_cuda or not weight_fp8.is_cuda or not weight_scale.is_cuda or not activation_scale.is_cuda:
            raise ValueError("FlashRT FP8 backend requires CUDA tensors.")

        out_dtype = _output_dtype(x, keep_output_dtype)
        x_kernel = x
        if input_scale is not None:
            x_kernel = x_kernel / input_scale.to(device=x.device, dtype=x.dtype)

        in_features = int(x_kernel.shape[-1])
        if weight_fp8.dim() != 2:
            raise ValueError(f"Expected weight_fp8 to be 2D (N, K), got {tuple(weight_fp8.shape)}.")
        if int(weight_fp8.shape[1]) != in_features:
            raise ValueError(
                f"FlashRT FP8 K mismatch: activation K={in_features}, weight K={int(weight_fp8.shape[1])}."
            )
        out_features = int(weight_fp8.shape[0])

        x_2d = x_kernel.reshape(-1, in_features).to(dtype=torch.bfloat16).contiguous()
        m = int(x_2d.shape[0])
        x_fp8 = torch.empty((m, in_features), dtype=torch.uint8, device=x.device)
        self._ops.quantize_activation_static(
            x_2d.data_ptr(),
            x_fp8.data_ptr(),
            activation_scale.data_ptr(),
            m * in_features,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        out = torch.empty((m, out_features), dtype=torch.bfloat16, device=x.device)
        self._ops.fp8_run_dev(
            x_fp8,
            weight_fp8.contiguous(),
            out,
            m,
            out_features,
            in_features,
            activation_scale,
            weight_scale,
        )
        if bias is not None:
            out = out + bias.to(device=x.device, dtype=out.dtype)
        return out.reshape(*x.shape[:-1], out_features).to(out_dtype)
