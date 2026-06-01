from __future__ import annotations

from .flashrt_fp8_ops import FlashRTFP8Backend
from .int8_w8a8_ops import Int8W8A8Backend
from .reference import CudaExtBackend, QuantBackend, ReferenceBackend


def get_backend(name: str) -> QuantBackend:
    if name == "reference":
        return ReferenceBackend()
    if name == "cuda_ext":
        return CudaExtBackend()
    if name == "flashrt_fp8":
        return FlashRTFP8Backend()
    if name == "int8_w8a8":
        return Int8W8A8Backend()
    raise ValueError(f"Unknown quantization backend: {name}")


__all__ = [
    "CudaExtBackend",
    "FlashRTFP8Backend",
    "Int8W8A8Backend",
    "QuantBackend",
    "ReferenceBackend",
    "get_backend",
]
