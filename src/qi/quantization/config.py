from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class WeightOnlyQuantConfig:
    enabled: bool = False
    algo: Literal["awq", "rtn"] = "awq"
    bits: int = 8
    group_size: int = 128
    symmetric: bool = True
    backend: Literal["reference", "flashrt", "cuda_ext"] = "reference"

    target_expert: Literal["all", "both", "action", "video"] = "all"
    action_expert_markers: tuple[str, ...] = ("action_expert", "action")
    video_expert_markers: tuple[str, ...] = ("video_expert", "video", "dit")
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2")
    skip_modules: tuple[str, ...] = ()
    quantize_ffn: bool = True
    quantize_attn_proj: bool = True

    keep_output_dtype: Literal["input", "fp16", "bf16", "fp32"] = "input"

    awq_alpha: float = 0.5
    awq_scale_eps: float = 1e-4

    def validate(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError(f"`bits` must be 4 or 8, got {self.bits}.")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError(f"`group_size` must be positive or -1, got {self.group_size}.")
        if self.backend == "flashrt" and self.bits != 4:
            raise ValueError("FlashRT backend currently expects 4-bit weights.")
        if not 0.0 <= float(self.awq_alpha) <= 1.0:
            raise ValueError(f"`awq_alpha` must be in [0, 1], got {self.awq_alpha}.")
