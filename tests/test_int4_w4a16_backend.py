from __future__ import annotations

import importlib.util

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="INT4 W4A16 backend requires CUDA")
def test_awq_int4_w4a16_backend_matches_reference_on_supported_shape():
    if importlib.util.find_spec("qi.quantization._int4_w4a16_ops") is None:
        pytest.skip("Qi INT4 W4A16 extension is not installed")

    from qi.quantization.backends import get_backend
    from qi.quantization.config import WeightOnlyQuantConfig
    from qi.quantization.modules import WeightOnlyLinear
    from qi.quantization.quantize import quantize_linear_weight

    torch.manual_seed(0)
    device = torch.device("cuda")
    in_features = 128
    out_features = 256
    linear = torch.nn.Linear(in_features, out_features, bias=True, device=device, dtype=torch.float16)
    calib = torch.randn(64, in_features, device=device, dtype=torch.float16)
    x = torch.randn(16, in_features, device=device, dtype=torch.float16)

    common = dict(
        enabled=True,
        algo="awq",
        bits=4,
        group_size=128,
        symmetric=True,
        awq_n_grid=4,
        awq_enable_clip=True,
        awq_clip_n_grid=4,
        awq_clip_n_sample_token=32,
        keep_output_dtype="fp16",
    )
    cfg_ref = WeightOnlyQuantConfig(**common, backend="reference")
    cfg_int4 = WeightOnlyQuantConfig(**common, backend="int4_w4a16")

    q_ref = quantize_linear_weight(
        linear.weight.detach(),
        cfg_ref,
        act_stats={"inputs": calib},
        bias=linear.bias.detach(),
    )
    q_int4 = quantize_linear_weight(
        linear.weight.detach(),
        cfg_int4,
        act_stats={"inputs": calib},
        bias=linear.bias.detach(),
    )

    assert q_int4.qweight_packed is not None
    assert q_int4.scale_factors is not None

    mod_ref = WeightOnlyLinear.from_linear(linear, q_ref, cfg_ref, get_backend("reference")).to(device)
    mod_int4 = WeightOnlyLinear.from_linear(linear, q_int4, cfg_int4, get_backend("int4_w4a16")).to(device)

    with torch.no_grad():
        y_ref = mod_ref(x)
        y_int4 = mod_int4(x)
        torch.cuda.synchronize()

    rel_error = (y_int4 - y_ref).abs().mean() / y_ref.abs().mean().clamp_min(1e-6)
    assert torch.isfinite(y_int4).all()
    assert float(rel_error) < 2e-3
