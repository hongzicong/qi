from __future__ import annotations

# INT4 W4A16 packing logic follows the vendored CUDA kernel layout.

import numpy as np
import torch

from ..algorithms.rtn import dequantize_weight
from .int4_w4a16_ops import Int4W4A16OpsUnavailable, load_int4_w4a16_ops


class Int4W4A16Unavailable(Int4W4A16OpsUnavailable):
    pass


def _get_perms() -> tuple[torch.Tensor, list[int], list[int]]:
    perm: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm_np = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm_np = perm_np.reshape((-1, 8))[:, interleave].ravel()
    perm_tensor = torch.from_numpy(perm_np)

    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])

    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])

    return perm_tensor, scale_perm, scale_perm_single


_PERM, _SCALE_PERM, _SCALE_PERM_SINGLE = _get_perms()


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


def _effective_int4_w4a16_group_size(group_size: int, in_features: int) -> int:
    return in_features if group_size == -1 else int(group_size)


def validate_int4_w4a16_shape(in_features: int, out_features: int, group_size: int) -> int:
    effective_group_size = _effective_int4_w4a16_group_size(group_size, in_features)
    if group_size not in (-1, 128):
        raise ValueError(f"INT4 W4A16 backend only supports group_size=-1 or 128, got {group_size}.")
    if in_features % 128 != 0 or out_features % 256 != 0:
        raise ValueError(
            "INT4 W4A16 backend requires in_features divisible by 128 and out_features divisible by 256, "
            f"got in_features={in_features}, out_features={out_features}."
        )
    if in_features % effective_group_size != 0:
        raise ValueError(
            f"INT4 W4A16 backend requires in_features divisible by group_size, got {in_features} and {effective_group_size}."
        )
    return effective_group_size


def int4_w4a16_packed_shapes(in_features: int, out_features: int, group_size: int) -> tuple[tuple[int, int], tuple[int, int]]:
    effective_group_size = validate_int4_w4a16_shape(in_features, out_features, group_size)
    return (in_features // 16, out_features * 16 // 8), (in_features // effective_group_size, out_features)


def _pack_fake_quantized_weight(weight: torch.Tensor, scales: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.dtype != torch.float16:
        raise ValueError("INT4 W4A16 packing requires float16 fake-quantized weights.")

    out_features, in_features = weight.shape
    effective_group_size = validate_int4_w4a16_shape(in_features, out_features, group_size)
    tile = 16
    maxq = 2**4 - 1

    scale_perm = torch.tensor(_SCALE_PERM, device=scales.device, dtype=torch.long)
    scale_perm_single = torch.tensor(_SCALE_PERM_SINGLE, device=scales.device, dtype=torch.long)
    perm = _PERM.to(device=weight.device, dtype=torch.long)

    s = scales.t()
    w = weight.t()
    if effective_group_size != in_features:
        w = w.reshape((-1, effective_group_size, out_features))
        w = w.permute(1, 0, 2)
        w = w.reshape((effective_group_size, -1))
        s = s.reshape((1, -1))

    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)

    if effective_group_size != in_features:
        w = w.reshape((effective_group_size, -1, out_features))
        w = w.permute(1, 0, 2)
        w = w.reshape((in_features, out_features)).contiguous()
        s = s.reshape((-1, len(_SCALE_PERM)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(_SCALE_PERM_SINGLE)))[:, scale_perm_single]
    s = s.reshape((-1, out_features)).contiguous()

    w = w.reshape((in_features // tile, tile, out_features // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((in_features // tile, out_features * tile))
    res = w.reshape((-1, _PERM.numel()))[:, perm].reshape(w.shape)

    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res_np = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res_np[:, i::8] << (4 * i)
    qweight_packed = torch.from_numpy(q.astype(np.int32)).to(device=weight.device)
    return qweight_packed.contiguous(), s.to(dtype=torch.float16).contiguous()


def pack_int4_w4a16_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits != 4:
        raise ValueError(f"INT4 W4A16 backend requires 4-bit weights, got bits={bits}.")
    if not symmetric:
        raise ValueError("INT4 W4A16 backend requires symmetric quantization.")
    if zeros is not None:
        raise ValueError("INT4 W4A16 backend does not support asymmetric zero points.")
    if qweight is None or scales is None:
        raise ValueError("INT4 W4A16 packing requires qweight and scales.")

    out_features, in_features = qweight.shape
    validate_int4_w4a16_shape(in_features, out_features, group_size)
    dequantized = dequantize_weight(
        qweight=qweight,
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        dtype=torch.float16,
    )
    return _pack_fake_quantized_weight(dequantized, scales.to(device=dequantized.device, dtype=torch.float16), group_size)


class Int4W4A16Backend:
    def __init__(self) -> None:
        self._ops = load_int4_w4a16_ops(required=False)

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
        del qweight, scales, zeros, weight_global_scale, activation_scale, weight_scale, weight_fp8
        del activation_granularity, weight_granularity
        if self._ops is None:
            raise Int4W4A16Unavailable(
                "Qi INT4 W4A16 kernels are not available. Build/install with QI_BUILD_INT4_W4A16=1 and sources "
                "available at src/qi/quantization/csrc/int4_w4a16."
            )
        if bits != 4:
            raise ValueError(f"INT4 W4A16 backend requires 4-bit weights, got bits={bits}.")
        if not symmetric:
            raise ValueError("INT4 W4A16 backend requires symmetric quantization.")
        if qweight_packed is None or scale_factors is None:
            raise ValueError("INT4 W4A16 backend requires qweight_packed and scale_factors buffers.")
        if not x.is_cuda or not qweight_packed.is_cuda or not scale_factors.is_cuda:
            raise ValueError("INT4 W4A16 backend requires CUDA tensors.")

        out_dtype = _output_dtype(x, keep_output_dtype)
        x_kernel = x
        if input_scale is not None:
            x_kernel = x_kernel / input_scale.to(device=x.device, dtype=x.dtype)
        x_2d = x_kernel.reshape(-1, x_kernel.shape[-1]).to(dtype=torch.float16).contiguous()

        out_features = int(scale_factors.shape[1])
        validate_int4_w4a16_shape(int(x_2d.shape[1]), out_features, group_size)
        out_2d = torch.empty((x_2d.shape[0], out_features), dtype=torch.float16, device=x.device)
        workspace = torch.zeros(out_features // 128 * 16, dtype=torch.int32, device=x.device)
        self._ops.mul(x_2d, qweight_packed, out_2d, scale_factors, workspace, -1, -1, -1, 16)
        if bias is not None:
            out_2d = out_2d + bias.to(device=x.device, dtype=out_2d.dtype)
        return out_2d.reshape(*x.shape[:-1], out_features).to(out_dtype)
