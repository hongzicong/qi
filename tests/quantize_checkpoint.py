#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate

from qi.api import dtype_from_mixed_precision, load_config
from qi.quantization import (
    WeightOnlyQuantConfig,
    load_checkpoint_payload,
    quantize_loaded_model,
    save_quantized_checkpoint,
)
from qi.utils.config_resolvers import register_default_resolvers
from qi.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


def _csv_tuple(value: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Qi weight-only quantized checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Source fp checkpoint.")
    parser.add_argument("--output-ckpt", required=True, help="Output quantized checkpoint.")
    parser.add_argument("--dataset-stats", required=True, help="Dataset stats path used to compose the model config.")
    parser.add_argument("--task", default="real_cleaning_uncond_3cam_384_1e-4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")

    parser.add_argument("--quant-algo", choices=["rtn", "awq"], default="rtn")
    parser.add_argument("--quant-dtype", choices=["int", "nvfp4"], default="int")
    parser.add_argument(
        "--quant-backend",
        choices=["reference", "cuda_ext", "flashrt_nvfp4"],
        default="reference",
    )
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--quant-asymmetric", action="store_true")
    parser.add_argument("--target-expert", choices=["all", "both", "action", "video"], default="all")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,fc1,fc2")
    parser.add_argument("--skip-modules", default="")
    parser.add_argument("--no-quantize-ffn", action="store_true")
    parser.add_argument("--no-quantize-attn-proj", action="store_true")
    parser.add_argument("--keep-output-dtype", choices=["input", "fp16", "bf16", "fp32"], default="input")
    parser.add_argument("--awq-alpha", type=float, default=0.5)
    parser.add_argument("--awq-scale-eps", type=float, default=1e-4)
    parser.add_argument(
        "--awq-stats",
        default=None,
        help="Path to torch-saved AWQ activation stats. Required for --quant-algo awq.",
    )
    return parser.parse_args()


def build_quant_config(args: argparse.Namespace) -> WeightOnlyQuantConfig:
    return WeightOnlyQuantConfig(
        enabled=True,
        algo=args.quant_algo,
        quant_dtype=args.quant_dtype,
        bits=args.quant_bits,
        group_size=args.quant_group_size,
        symmetric=not args.quant_asymmetric,
        backend=args.quant_backend,
        target_expert=args.target_expert,
        target_modules=_csv_tuple(args.target_modules),
        skip_modules=_csv_tuple(args.skip_modules),
        quantize_ffn=not args.no_quantize_ffn,
        quantize_attn_proj=not args.no_quantize_attn_proj,
        keep_output_dtype=args.keep_output_dtype,
        awq_alpha=args.awq_alpha,
        awq_scale_eps=args.awq_scale_eps,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args)
    model_dtype = dtype_from_mixed_precision(args.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    source_payload = model.load_checkpoint(args.ckpt)
    source_payload = dict(source_payload)

    quant_cfg = build_quant_config(args)
    quant_cfg.validate()
    act_stats = None
    if quant_cfg.algo == "awq":
        if args.awq_stats is None:
            raise ValueError("--awq-stats is required when --quant-algo awq.")
        act_stats = torch.load(args.awq_stats, map_location="cpu")

    logger.info(
        "Quantizing checkpoint: algo=%s quant_dtype=%s bits=%s backend=%s target_expert=%s.",
        quant_cfg.algo,
        quant_cfg.quant_dtype,
        quant_cfg.bits,
        quant_cfg.backend,
        quant_cfg.target_expert,
    )
    quantize_loaded_model(model, quant_cfg, act_stats=act_stats)
    save_quantized_checkpoint(
        model,
        output_path=args.output_ckpt,
        cfg=quant_cfg,
        source_payload=source_payload,
        source_checkpoint=args.ckpt,
    )

    payload = load_checkpoint_payload(args.output_ckpt)
    logger.info("Saved quantized checkpoint to %s with keys %s.", args.output_ckpt, sorted(payload.keys()))


if __name__ == "__main__":
    main()
