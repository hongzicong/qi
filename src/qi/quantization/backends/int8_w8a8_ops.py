from __future__ import annotations

import importlib
from types import ModuleType

import torch


_REQUIRED_FUNCTIONS = (
    "quantize_int8_rowwise",
    "cutlass_int8_rowwise_bf16out",
)


class Int8W8A8Unavailable(RuntimeError):
    pass


def _hardware_note() -> str:
    if not torch.cuda.is_available():
        return " CUDA is not available."
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    return f" Current GPU is {name} sm_{major}{minor}."


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


def load_int8_w8a8_ops(required: bool = True) -> ModuleType | None:
    try:
        ops = importlib.import_module("qi.quantization._int8_w8a8_ops")
        missing = [name for name in _REQUIRED_FUNCTIONS if getattr(ops, name, None) is None]
        if missing:
            raise Int8W8A8Unavailable(
                "Qi INT8 W8A8 extension is missing functions: "
                + ", ".join(missing)
                + "."
                + _hardware_note()
            )
        return ops
    except (ImportError, Int8W8A8Unavailable):
        if required:
            raise Int8W8A8Unavailable(
                "Qi INT8 W8A8 kernels are not available. Build/install with "
                "`QI_BUILD_INT8_W8A8=1` and CUTLASS available at "
                "`third_party/cutlass` or `QI_CUTLASS_DIR`."
                + _hardware_note()
            )
        return None


class Int8W8A8Backend:
    def __init__(self) -> None:
        self._ops = load_int8_w8a8_ops(required=False)
        self._custom_linear = getattr(torch.ops.qi_int8_w8a8, "linear", None)
        self._quant_workspace: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def clear_workspace_cache(self) -> None:
        self._quant_workspace.clear()

    def _get_quant_workspace(
        self,
        *,
        device: torch.device,
        m: int,
        k: int,
        stream: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (int(device_index), int(m), int(k), int(stream))
        workspace = self._quant_workspace.get(key)
        if workspace is None:
            workspace = (
                torch.empty((m, k), dtype=torch.int8, device=device),
                torch.empty((m,), dtype=torch.float32, device=device),
            )
            self._quant_workspace[key] = workspace
        return workspace

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
        del zeros, qweight_packed, scale_factors, weight_global_scale, activation_scale, weight_scale, weight_fp8
        if self._ops is None:
            raise Int8W8A8Unavailable(
                "Qi INT8 W8A8 kernels are not available. Build/install with "
                "`QI_BUILD_INT8_W8A8=1` and CUTLASS available at "
                "`third_party/cutlass` or `QI_CUTLASS_DIR`."
                + _hardware_note()
            )
        if self._custom_linear is None:
            raise Int8W8A8Unavailable("Qi INT8 W8A8 custom op qi_int8_w8a8::linear is not registered.")
        if bits != 8:
            raise ValueError("INT8 W8A8 backend requires bits=8.")
        if group_size != -1:
            raise ValueError("INT8 W8A8 backend requires per-channel weights with group_size=-1.")
        if not symmetric:
            raise ValueError("INT8 W8A8 backend requires symmetric quantization.")
        if activation_granularity != "per_token" or weight_granularity != "per_channel":
            raise ValueError("INT8 W8A8 backend requires per-token activations and per-channel weights.")
        if qweight is None or scales is None:
            raise ValueError("INT8 W8A8 backend requires qweight and per-channel scales.")
        if not x.is_cuda or not qweight.is_cuda or not scales.is_cuda:
            raise ValueError("INT8 W8A8 backend requires CUDA tensors.")

        out_dtype = _output_dtype(x, keep_output_dtype)
        x_kernel = x
        if input_scale is not None:
            x_kernel = x_kernel / input_scale.to(device=x.device, dtype=x.dtype)

        in_features = int(x_kernel.shape[-1])
        if qweight.dim() != 2:
            raise ValueError(f"Expected qweight to be 2D (N, K), got {tuple(qweight.shape)}.")
        out_features = int(qweight.shape[0])
        if int(qweight.shape[1]) != in_features:
            raise ValueError(
                f"INT8 W8A8 K mismatch: activation K={in_features}, weight K={int(qweight.shape[1])}."
            )

        weight_scales = scales
        if weight_scales.dim() == 2:
            if weight_scales.shape != (out_features, 1):
                raise ValueError(
                    f"Expected per-channel scales shape {(out_features, 1)}, got {tuple(weight_scales.shape)}."
                )
            weight_scales = weight_scales[:, 0]
        if weight_scales.shape != (out_features,):
            raise ValueError(f"Expected weight scales shape {(out_features,)}, got {tuple(weight_scales.shape)}.")

        x_2d = x_kernel.reshape(-1, in_features).to(dtype=torch.bfloat16).contiguous()
        qweight_2d = qweight.contiguous()
        weight_scales = weight_scales.contiguous().float()
        bias_kernel = None
        if bias is not None:
            if bias.numel() != out_features:
                raise ValueError(f"Expected bias with {out_features} values, got {bias.numel()}.")
            bias_kernel = bias.reshape(-1)
            if (
                not bias_kernel.is_cuda
                or bias_kernel.device != x.device
                or bias_kernel.dtype != torch.bfloat16
                or not bias_kernel.is_contiguous()
            ):
                bias_kernel = bias_kernel.to(device=x.device, dtype=torch.bfloat16).contiguous()

        out = self._custom_linear(x_2d, qweight_2d, weight_scales, bias_kernel)

        return out.reshape(*x.shape[:-1], out_features).to(out_dtype)
