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

_REQUIRED_KERNEL_FUNCTIONS = (
    "quantize_bf16_to_nvfp4_swizzled",
    "quantize_bf16_to_nvfp4_swizzled_static_groups",
    "awq_quant_bf16_to_nvfp4_swizzled",
)


class FlashRTNVFP4Unavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FlashRTNVFP4Ops:
    kernels: ModuleType
    runner: object
    quantize_dynamic: Callable
    quantize_static_groups: Callable
    awq_quantize: Callable

    @staticmethod
    def packed_shape(rows: int, cols: int) -> tuple[int, int]:
        if cols % 2 != 0:
            raise ValueError(f"NVFP4 packing requires an even K dimension, got {cols}.")
        return int(rows), int(cols) // 2

    @staticmethod
    def scale_factors_numel(rows: int, cols: int) -> int:
        if cols % 16 != 0:
            raise ValueError(f"NVFP4 scale factors require K to be a multiple of 16, got {cols}.")
        n_blocks = int(cols) // 16
        n_row_super = (int(rows) + 127) // 128
        n_col_super = (n_blocks + 3) // 4
        return n_row_super * n_col_super * 512

    def empty_quantized_tensors(
        self,
        rows: int,
        cols: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qweight_packed = torch.empty(self.packed_shape(rows, cols), dtype=torch.uint8, device=device)
        scale_factors = torch.zeros(self.scale_factors_numel(rows, cols), dtype=torch.uint8, device=device)
        weight_global_scale = torch.ones((1,), dtype=torch.float32, device=device)
        return qweight_packed, scale_factors, weight_global_scale

    def quantize_weight_dynamic(
        self,
        weight_bf16: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weight_bf16.dtype != torch.bfloat16:
            raise ValueError(f"FlashRT NVFP4 quantize expects bf16 input, got {weight_bf16.dtype}.")
        if weight_bf16.dim() != 2:
            raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(weight_bf16.shape)}.")
        rows, cols = weight_bf16.shape
        qweight_packed, scale_factors, weight_global_scale = self.empty_quantized_tensors(
            rows,
            cols,
            weight_bf16.device,
        )
        self.quantize_dynamic(
            weight_bf16.data_ptr(),
            qweight_packed.data_ptr(),
            scale_factors.data_ptr(),
            int(rows),
            int(cols),
            torch.cuda.current_stream(weight_bf16.device).cuda_stream,
        )
        return qweight_packed, scale_factors, weight_global_scale

    def quantize_weight_awq(
        self,
        weight_bf16: torch.Tensor,
        inv_scale_bf16: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weight_bf16.dtype != torch.bfloat16:
            raise ValueError(f"FlashRT NVFP4 AWQ quantize expects bf16 input, got {weight_bf16.dtype}.")
        if inv_scale_bf16.dtype != torch.bfloat16:
            raise ValueError(f"FlashRT NVFP4 AWQ inverse scale expects bf16, got {inv_scale_bf16.dtype}.")
        if weight_bf16.dim() != 2:
            raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(weight_bf16.shape)}.")
        if inv_scale_bf16.numel() != weight_bf16.shape[1]:
            raise ValueError(
                f"Expected AWQ inverse scale with {weight_bf16.shape[1]} values, "
                f"got {inv_scale_bf16.numel()}."
            )
        rows, cols = weight_bf16.shape
        qweight_packed, scale_factors, weight_global_scale = self.empty_quantized_tensors(
            rows,
            cols,
            weight_bf16.device,
        )
        self.awq_quantize(
            weight_bf16.data_ptr(),
            inv_scale_bf16.contiguous().data_ptr(),
            qweight_packed.data_ptr(),
            scale_factors.data_ptr(),
            int(rows),
            int(cols),
            torch.cuda.current_stream(weight_bf16.device).cuda_stream,
        )
        return qweight_packed, scale_factors, weight_global_scale

    def fp4_nn_dev(
        self,
        a_fp4: torch.Tensor,
        sfa: torch.Tensor,
        b_fp4: torch.Tensor,
        sfb: torch.Tensor,
        out: torch.Tensor,
        m: int,
        n: int,
        k: int,
    ) -> None:
        self.runner.fp4_nn_dev(
            a_fp4.data_ptr(),
            sfa.data_ptr(),
            b_fp4.data_ptr(),
            sfb.data_ptr(),
            out.data_ptr(),
            int(m),
            int(n),
            int(k),
            torch.cuda.current_stream(out.device).cuda_stream,
        )


def _hardware_note() -> str:
    if not torch.cuda.is_available():
        return " CUDA is not available."
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if major < 12:
        return f" Current GPU is {name} sm_{major}{minor}; FlashRT NVFP4 requires SM120/Blackwell."
    return f" Current GPU is {name} sm_{major}{minor}."


def _import_kernel_module() -> ModuleType:
    errors: list[str] = []
    for module_name in _KERNEL_MODULE_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
    raise FlashRTNVFP4Unavailable(
        "Could not import FlashRT kernel module. Tried: "
        + "; ".join(errors)
        + "."
        + _hardware_note()
    )


def load_flashrt_nvfp4_ops(required: bool = True) -> FlashRTNVFP4Ops | None:
    try:
        kernels = _import_kernel_module()
        runner_cls = getattr(kernels, "GemmRunner", None)
        if runner_cls is None:
            raise FlashRTNVFP4Unavailable("FlashRT kernel module is missing GemmRunner.")

        missing = [name for name in _REQUIRED_KERNEL_FUNCTIONS if getattr(kernels, name, None) is None]
        if missing:
            raise FlashRTNVFP4Unavailable(
                "FlashRT kernel module is missing NVFP4 functions: "
                + ", ".join(missing)
                + "."
                + _hardware_note()
            )

        runner = runner_cls()
        if getattr(runner, "fp4_nn_dev", None) is None:
            raise FlashRTNVFP4Unavailable("FlashRT GemmRunner is missing fp4_nn_dev.")

        return FlashRTNVFP4Ops(
            kernels=kernels,
            runner=runner,
            quantize_dynamic=getattr(kernels, "quantize_bf16_to_nvfp4_swizzled"),
            quantize_static_groups=getattr(kernels, "quantize_bf16_to_nvfp4_swizzled_static_groups"),
            awq_quantize=getattr(kernels, "awq_quant_bf16_to_nvfp4_swizzled"),
        )
    except FlashRTNVFP4Unavailable:
        if required:
            raise
        return None
