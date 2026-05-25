from __future__ import annotations

from .flashrt import FlashRTBackend
from .reference import CudaExtBackend, QuantBackend, ReferenceBackend


def get_backend(name: str) -> QuantBackend:
    if name == "reference":
        return ReferenceBackend()
    if name == "flashrt":
        return FlashRTBackend()
    if name == "cuda_ext":
        return CudaExtBackend()
    raise ValueError(f"Unknown quantization backend: {name}")


__all__ = [
    "CudaExtBackend",
    "FlashRTBackend",
    "QuantBackend",
    "ReferenceBackend",
    "get_backend",
]
