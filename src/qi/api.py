
from typing import Any
from pathlib import Path

import argparse
import logging
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from qi.quantization import (
    checkpoint_is_quantized,
    load_checkpoint_payload,
    prepare_model_for_quantized_checkpoint,
)
from qi.utils.logging_config import get_logger

logger = get_logger(__name__)


def dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def load_config(
    config_dir: str | argparse.Namespace | None = None,
    task: str | None = None,
    mixed_precision: str | None = None,
    dataset_stats: str | None = None,
) -> DictConfig:
    if isinstance(config_dir, argparse.Namespace):
        args = config_dir
        config_dir = args.config_dir
        task = args.task
        mixed_precision = args.mixed_precision
        dataset_stats = args.dataset_stats

    config_dir = str(Path(config_dir).resolve())
    overrides = [f"task={task}", f"mixed_precision={mixed_precision}"]
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)

    cfg.model.load_text_encoder = True

    # Pin dataset splits to the provided stats path instead of the one saved in the config.
    if "data" in cfg:
        for split in ("train", "val"):
            if split in cfg.data:
                cfg.data[split].pretrained_norm_stats = dataset_stats
    return cfg


def load_model(
    ckpt: str,
    device: str,
    mixed_precision: str,
    config_dir: str,
    task: str,
    dataset_stats: str,
    torch_compile: bool = False,
    no_cuda_graph: bool = False,
) -> Any:
    cfg = load_config(config_dir, task, mixed_precision, dataset_stats)
    model_dtype = dtype_from_mixed_precision(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.cfg = cfg

    payload = load_checkpoint_payload(ckpt)
    if checkpoint_is_quantized(payload):
        quant_cfg = prepare_model_for_quantized_checkpoint(model, payload)
        logger.info(
            "Detected quantized checkpoint: algo=%s bits=%s backend=%s.",
            quant_cfg.algo,
            quant_cfg.bits,
            quant_cfg.backend,
        )

    payload = model.load_checkpoint(ckpt)
    model.eval()
    logger.info("Loaded checkpoint %s with keys %s.", ckpt, sorted(payload.keys()))

    if torch_compile:
        logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
        logger.info("torch.compile enabled on MoT body forward (Inductor, default mode)")

    if not no_cuda_graph or torch_compile:
        model._setup_graph_mgr(torch_compile=torch_compile)
        if not no_cuda_graph:
            logger.info("CUDA Graph enabled (wraps VAE encode, prefill, and MoT body)")
        else:
            logger.info("CUDA Graph disabled, torch.compile only")

    return model
