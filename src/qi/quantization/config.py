from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_TARGET_MODULES = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


@dataclass
class WeightOnlyQuantConfig:
    enabled: bool = False
    algo: Literal["awq", "rtn"] = "awq"
    quant_dtype: Literal["int"] = "int"
    bits: int = 8
    group_size: int = 128
    symmetric: bool = True
    backend: Literal["reference", "cuda_ext", "int4_w4a16"] = "reference"

    target_expert: Literal["all", "both", "action", "video"] = "all"
    action_expert_markers: tuple[str, ...] = ("action_expert", "action")
    video_expert_markers: tuple[str, ...] = ("video_expert", "video", "dit")
    target_modules: tuple[str, ...] = _TARGET_MODULES
    skip_modules: tuple[str, ...] = ()
    quantize_ffn: bool = True
    quantize_attn_proj: bool = True

    keep_output_dtype: Literal["input", "fp16", "bf16", "fp32"] = "input"

    awq_scale_eps: float = 1e-4
    awq_n_grid: int = 20
    awq_max_calib_samples: int = 256
    awq_enable_clip: bool = True
    awq_clip_n_grid: int = 20
    awq_clip_max_shrink: float = 0.5
    awq_clip_n_sample_token: int = 512

    def validate(self) -> None:
        if self.quant_dtype != "int":
            raise ValueError(
                "WeightOnlyQuantConfig only supports quant_dtype=\"int\". "
                "Use WeightActivationQuantConfig for W8A8 quantization."
            )
        if self.bits not in (4, 8):
            raise ValueError(f"`bits` must be 4 or 8, got {self.bits}.")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError(f"`group_size` must be positive or -1, got {self.group_size}.")
        if self.backend == "int4_w4a16":
            if self.bits != 4:
                raise ValueError(f"INT4 W4A16 backend requires bits=4, got {self.bits}.")
            if not self.symmetric:
                raise ValueError("INT4 W4A16 backend requires symmetric quantization.")
            if self.group_size not in (-1, 128):
                raise ValueError(f"INT4 W4A16 backend requires group_size=-1 or 128, got {self.group_size}.")
        if int(self.awq_n_grid) <= 0:
            raise ValueError(f"`awq_n_grid` must be positive, got {self.awq_n_grid}.")
        if int(self.awq_max_calib_samples) <= 0:
            raise ValueError(
                f"`awq_max_calib_samples` must be positive, got {self.awq_max_calib_samples}."
            )
        if int(self.awq_clip_n_grid) <= 0:
            raise ValueError(f"`awq_clip_n_grid` must be positive, got {self.awq_clip_n_grid}.")
        if not 0.0 <= float(self.awq_clip_max_shrink) < 1.0:
            raise ValueError(
                f"`awq_clip_max_shrink` must be in [0, 1), got {self.awq_clip_max_shrink}."
            )
        if int(self.awq_clip_n_sample_token) <= 0:
            raise ValueError(
                f"`awq_clip_n_sample_token` must be positive, got {self.awq_clip_n_sample_token}."
            )


@dataclass
class WeightActivationQuantConfig:
    enabled: bool = False
    algo: Literal["rtn", "smoothquant"] = "rtn"
    quant_dtype: Literal["fp8", "int"] = "fp8"
    bits: int = 8
    activation_bits: int = 8
    backend: Literal["flashrt_fp8", "reference", "int8_w8a8"] = "flashrt_fp8"
    activation_granularity: Literal["per_tensor", "per_token"] = "per_tensor"
    weight_granularity: Literal["per_tensor", "per_channel"] = "per_tensor"
    group_size: int = -1
    symmetric: bool = True

    target_expert: Literal["all", "both", "action", "video"] = "all"
    action_expert_markers: tuple[str, ...] = ("action_expert", "action")
    video_expert_markers: tuple[str, ...] = ("video_expert", "video", "dit")
    target_modules: tuple[str, ...] = _TARGET_MODULES
    skip_modules: tuple[str, ...] = ()
    quantize_ffn: bool = True
    quantize_attn_proj: bool = True

    keep_output_dtype: Literal["input", "fp16", "bf16", "fp32"] = "input"

    smoothquant_alpha: float = 0.5
    smoothquant_scale_eps: float = 1e-4
    activation_scale: float = 1.0

    def validate(self) -> None:
        if self.bits != 8:
            raise ValueError(f"W8A8 requires `bits=8`, got {self.bits}.")
        if self.activation_bits != 8:
            raise ValueError(f"W8A8 requires `activation_bits=8`, got {self.activation_bits}.")
        if self.algo not in ("rtn", "smoothquant"):
            raise ValueError(f"Unsupported W8A8 algo: {self.algo}.")
        if self.quant_dtype == "fp8":
            if self.backend != "flashrt_fp8":
                raise ValueError(f"FP8 W8A8 requires backend=\"flashrt_fp8\", got {self.backend}.")
            if self.activation_granularity != "per_tensor" or self.weight_granularity != "per_tensor":
                raise ValueError("FP8 W8A8 currently supports per_tensor activation and weight scales only.")
        elif self.quant_dtype == "int":
            if self.backend not in ("reference", "int8_w8a8"):
                raise ValueError(f"INT8 W8A8 requires backend=\"reference\" or \"int8_w8a8\", got {self.backend}.")
            if self.activation_granularity != "per_token":
                raise ValueError("INT8 W8A8 currently requires activation_granularity=\"per_token\".")
            if self.weight_granularity != "per_channel":
                raise ValueError("INT8 W8A8 currently requires weight_granularity=\"per_channel\".")
            if self.group_size != -1:
                raise ValueError("INT8 W8A8 per-channel weights require group_size=-1.")
            if not self.symmetric:
                raise ValueError("INT8 W8A8 currently requires symmetric quantization.")
        else:
            raise ValueError(f"Unsupported W8A8 quant_dtype: {self.quant_dtype}.")
        if not 0.0 <= float(self.smoothquant_alpha) <= 1.0:
            raise ValueError(f"`smoothquant_alpha` must be in [0, 1], got {self.smoothquant_alpha}.")
        if float(self.smoothquant_scale_eps) <= 0:
            raise ValueError("`smoothquant_scale_eps` must be positive.")
        if float(self.activation_scale) <= 0:
            raise ValueError("`activation_scale` must be positive.")
