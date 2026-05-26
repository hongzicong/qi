from __future__ import annotations

from .flashrt_nvfp4_ops import FlashRTNVFP4Backend
from .reference import CudaExtBackend, QuantBackend, ReferenceBackend


def get_backend(name: str) -> QuantBackend:
    if name == "reference":
        return ReferenceBackend()
    if name == "cuda_ext":
        return CudaExtBackend()
    if name == "flashrt_nvfp4":
        return FlashRTNVFP4Backend()
    raise ValueError(f"Unknown quantization backend: {name}")


__all__ = [
    "CudaExtBackend",
    "FlashRTNVFP4Backend",
    "QuantBackend",
    "ReferenceBackend",
    "get_backend",
]
