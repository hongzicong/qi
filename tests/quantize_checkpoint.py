#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from hydra.utils import instantiate

from qi.api import dtype_from_mixed_precision, load_config
from qi.quantization import (
    WeightActivationQuantConfig,
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
    parser.add_argument(
        "--model-base-path",
        default=None,
        help=(
            "Local DiffSynth/Wan checkpoint root containing subdirectories like "
            "Wan-AI/Wan2.2-TI2V-5B. Sets DIFFSYNTH_MODEL_BASE_PATH before model loading."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Set DIFFSYNTH_SKIP_DOWNLOAD=true so missing pretrained files fail instead of downloading.",
    )

    parser.add_argument("--quant-algo", choices=["rtn", "awq", "smoothquant"], default="rtn")
    parser.add_argument("--quant-dtype", choices=["int", "fp8"], default="int")
    parser.add_argument(
        "--quant-backend",
        choices=["reference", "cuda_ext", "flashrt_fp8", "int8_w8a8", "int4_w4a16"],
        default="reference",
    )
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--quant-asymmetric", action="store_true")
    parser.add_argument("--target-expert", choices=["all", "both", "action", "video"], default="all")
    parser.add_argument("--target-modules", default="self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2")
    parser.add_argument("--skip-modules", default="")
    parser.add_argument("--no-quantize-ffn", action="store_true")
    parser.add_argument("--no-quantize-attn-proj", action="store_true")
    parser.add_argument("--keep-output-dtype", choices=["input", "fp16", "bf16", "fp32"], default="input")
    parser.add_argument("--awq-scale-eps", type=float, default=1e-4)
    parser.add_argument("--awq-n-grid", type=int, default=20)
    parser.add_argument("--awq-max-calib-samples", type=int, default=256)
    parser.add_argument("--no-awq-clip", action="store_true")
    parser.add_argument("--awq-clip-n-grid", type=int, default=20)
    parser.add_argument("--awq-clip-max-shrink", type=float, default=0.5)
    parser.add_argument("--awq-clip-n-sample-token", type=int, default=512)
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--smoothquant-scale-eps", type=float, default=1e-4)
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--activation-granularity", choices=["per_tensor", "per_token"], default=None)
    parser.add_argument("--weight-granularity", choices=["per_tensor", "per_channel"], default=None)
    parser.add_argument(
        "--awq-stats",
        default=None,
        help="Path to torch-saved calibration data. AWQ requires collect_awq_calibration_cache() output with input samples; SmoothQuant accepts activation stats.",
    )
    return parser.parse_args()


def configure_local_model_loading(args: argparse.Namespace) -> None:
    if args.model_base_path:
        model_base_path = Path(args.model_base_path).expanduser().resolve()
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base_path)
        logger.info("Using local model base path: %s", model_base_path)
    if args.skip_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        logger.info("DIFFSYNTH_SKIP_DOWNLOAD=true; missing pretrained files will not be downloaded.")


def configure_checkpoint_model_loading(cfg) -> None:
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    logger.info("Skipping pretrained DiT backbones; weights will be loaded from --ckpt.")


def build_quant_config(args: argparse.Namespace) -> WeightOnlyQuantConfig | WeightActivationQuantConfig:
    common = dict(
        enabled=True,
        target_expert=args.target_expert,
        target_modules=_csv_tuple(args.target_modules),
        skip_modules=_csv_tuple(args.skip_modules),
        quantize_ffn=not args.no_quantize_ffn,
        quantize_attn_proj=not args.no_quantize_attn_proj,
        keep_output_dtype=args.keep_output_dtype,
    )
    if args.quant_dtype == "fp8":
        backend = "flashrt_fp8" if args.quant_backend == "reference" else args.quant_backend
        return WeightActivationQuantConfig(
            **common,
            algo=args.quant_algo,
            quant_dtype="fp8",
            bits=8,
            activation_bits=8,
            backend=backend,
            activation_granularity=args.activation_granularity or "per_tensor",
            weight_granularity=args.weight_granularity or "per_tensor",
            smoothquant_alpha=args.smoothquant_alpha,
            smoothquant_scale_eps=args.smoothquant_scale_eps,
            activation_scale=args.activation_scale,
        )
    if (
        args.quant_backend == "int8_w8a8"
        or args.quant_algo == "smoothquant"
        or args.activation_granularity is not None
        or args.weight_granularity is not None
    ):
        return WeightActivationQuantConfig(
            **common,
            algo=args.quant_algo,
            quant_dtype="int",
            bits=8,
            activation_bits=8,
            backend=args.quant_backend,
            activation_granularity=args.activation_granularity or "per_token",
            weight_granularity=args.weight_granularity or "per_channel",
            smoothquant_alpha=args.smoothquant_alpha,
            smoothquant_scale_eps=args.smoothquant_scale_eps,
        )
    return WeightOnlyQuantConfig(
        **common,
        algo=args.quant_algo,
        quant_dtype=args.quant_dtype,
        bits=args.quant_bits,
        group_size=args.quant_group_size,
        symmetric=not args.quant_asymmetric,
        backend=args.quant_backend,
        awq_scale_eps=args.awq_scale_eps,
        awq_n_grid=args.awq_n_grid,
        awq_max_calib_samples=args.awq_max_calib_samples,
        awq_enable_clip=not args.no_awq_clip,
        awq_clip_n_grid=args.awq_clip_n_grid,
        awq_clip_max_shrink=args.awq_clip_max_shrink,
        awq_clip_n_sample_token=args.awq_clip_n_sample_token,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    configure_local_model_loading(args)
    cfg = load_config(args)
    configure_checkpoint_model_loading(cfg)
    model_dtype = dtype_from_mixed_precision(args.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    source_payload = model.load_checkpoint(args.ckpt)
    source_payload = dict(source_payload)

    quant_cfg = build_quant_config(args)
    quant_cfg.validate()
    act_stats = None
    if quant_cfg.algo in ("awq", "smoothquant"):
        if args.awq_stats is None:
            raise ValueError("--awq-stats is required when --quant-algo awq/smoothquant.")
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
