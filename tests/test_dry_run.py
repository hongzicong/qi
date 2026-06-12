#!/usr/bin/env python3
"""Open-loop WAM video dry-run for real-data checkpoints.

This entrypoint intentionally does not import the real-robot deployment scripts:
it should be safe to run on a training/evaluation server without ROS, CAN, or
robot SDK dependencies. It can either sample an existing LeRobot dataset item
and save predicted-vs-GT videos, or use one RGB snapshot from each camera plus a
state JSON file and save the predicted future video.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

from qi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from qi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from qi.utils.config_resolvers import register_default_resolvers
from qi.utils.image_utils import load_rgb_image, load_state_vector, pil_frame_to_input_image, preprocess_real_images, save_image
from qi.utils.logging_config import get_logger, setup_logging
from qi.utils.video_utils import save_mp4
from qi.api import load_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

register_default_resolvers()
logger = get_logger(__name__)

_PY_NVTX: Any = None
_PY_NVTX_READY = False
_NVTX_WARNED = False


def _load_py_nvtx() -> Any:
    global _PY_NVTX, _PY_NVTX_READY
    if not _PY_NVTX_READY:
        try:
            _PY_NVTX = __import__("nvtx")
        except Exception:
            _PY_NVTX = None
        _PY_NVTX_READY = True
    return _PY_NVTX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WAM open-loop video inference without touching robot IO."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoints/weights/step_xxxxxx.pt.")
    parser.add_argument("--dataset-stats", required=True, help="Path to dataset_stats.json from the same run/data.")
    parser.add_argument("--task", default="real_cleaning_uncond_3cam_384_1e-4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--output-dir", default="open_loop_video_dry_run")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--expert-cache",
        action="store_true",
        help="Enable fixed-step action expert DiT body residual caching during inference.",
    )
    parser.add_argument(
        "--expert-cache-reuse-steps",
        type=int,
        default=1,
        help="Number of adjacent denoising steps that reuse the previous computed action expert body residual.",
    )
    parser.add_argument(
        "--expert-cache-warmup-steps",
        type=int,
        default=1,
        help="Number of initial action denoising steps that always compute the DiT body.",
    )
    parser.add_argument(
        "--expert-cache-cooldown-steps",
        type=int,
        default=1,
        help="Number of final action denoising steps that always compute the DiT body.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help=(
            "Files mode only: repeatedly generate this many chunks. "
            "Each chunk after the first uses the previous predicted last frame as the new condition image."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA Graph for action denoising (for baseline comparison).")
    parser.add_argument(
        "--infer-action-only",
        action="store_true",
        help=(
            "Run action-only inference path (model.infer_action) for profiling. "
            "In this mode, chunk conditioning image is not rolled forward."
        ),
    )
    parser.add_argument(
        "--nsys-cuda-profiler-per-chunk",
        action="store_true",
        help=(
            "Wrap the whole multi-chunk rollout with a single cudaProfilerStart/Stop "
            "so nsys --capture-range=cudaProfilerApi produces one report while "
            "keeping per-chunk NVTX labels."
        ),
    )
    parser.add_argument(
        "--nsys-skip-first-chunk-capture",
        action="store_true",
        help=(
            "When used with --nsys-cuda-profiler-per-chunk, delay cudaProfilerStart "
            "until after chunk0, so report keeps chunks 1..N-1 while execution still "
            "runs all chunks."
        ),
    )
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile on action expert inference (Inductor backend, default mode).")
    parser.add_argument(
        "--async-save",
        action="store_true",
        help=(
            "Pipeline chunks: drop per-chunk torch.cuda.synchronize() calls, return action "
            "tensors on-device, and offload save_image/save_actions to a background thread. "
            "Eliminates the inter-chunk GPU bubble. infer_action_only mode only."
        ),
    )
    parser.add_argument(
        "--per-chunk-timing",
        action="store_true",
        help=(
            "Force a torch.cuda.synchronize() around every chunk to log its wall-clock time. "
            "Default: off (the host-side bubble it introduces is what we are trying to hide)."
        ),
    )

    parser.add_argument("--cam-high", default=None)
    parser.add_argument("--cam-left-wrist", default=None)
    parser.add_argument("--cam-right-wrist", default=None)
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--prompt", default="clean the table")
    return parser.parse_args()


@contextmanager
def maybe_nvtx_range(name: str, enabled: bool):
    global _NVTX_WARNED
    if not enabled:
        yield
        return
    backend = None
    token: Any = None
    py_nvtx = _load_py_nvtx()
    try:
        if py_nvtx is not None:
            token = py_nvtx.start_range(message=name)
            backend = "nvtx"
        else:
            torch.cuda.nvtx.range_push(name)
            backend = "torch"
    except Exception as exc:
        if not _NVTX_WARNED:
            logger.warning("NVTX range emit failed once; continuing without NVTX ranges: %s", exc)
            _NVTX_WARNED = True
        backend = None
    try:
        yield
    finally:
        if backend == "nvtx":
            try:
                py_nvtx.end_range(token)
            except Exception:
                pass
        elif backend == "torch":
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


def default_num_video_frames(cfg: DictConfig) -> int:
    num_frames = int(cfg.data.train.num_frames)
    ratio = int(cfg.data.train.get("action_video_freq_ratio", 1))
    return ((num_frames - 1) // ratio) + 1


def default_action_horizon(cfg: DictConfig) -> int:
    return int(cfg.data.train.num_frames) - 1


def save_actions(
    action: torch.Tensor | np.ndarray | None,
    output_dir: Path,
    prefix: str,
    processor: Any | None = None,
    proprio: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if action is None:
        return None
    action_tensor = action.detach().cpu().to(torch.float32) if isinstance(action, torch.Tensor) else torch.as_tensor(action)
    action_denorm = denormalize_merged_action(processor, action_tensor, proprio) if processor is not None else action_tensor
    action_np = action_denorm.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / f"{prefix}_pred_actions.npy", action_np)
    with open(output_dir / f"{prefix}_pred_actions.json", "w", encoding="utf-8") as file:
        json.dump({"action": action_np.tolist()}, file, ensure_ascii=False, indent=2)
    return action_denorm


def denormalize_merged_action(
    processor: Any,
    action: torch.Tensor,
    proprio: torch.Tensor | None,
) -> torch.Tensor:
    if action.ndim == 2:
        action_btd = action.unsqueeze(0)
    elif action.ndim == 3 and action.shape[0] == 1:
        action_btd = action
    else:
        raise ValueError(f"Expected action [T,D] or [1,T,D], got {tuple(action.shape)}")

    action_btd = action_btd.detach().cpu().to(torch.float32)
    if proprio is None:
        state_btd = torch.zeros(
            action_btd.shape[0],
            action_btd.shape[1],
            int(processor.proprio_output_dim),
            dtype=torch.float32,
        )
    else:
        state = proprio.detach().cpu().to(torch.float32)
        if state.ndim == 1:
            state_btd = state.view(1, 1, -1).expand(action_btd.shape[0], action_btd.shape[1], -1).clone()
        elif state.ndim == 2:
            state_btd = state.unsqueeze(0)
        elif state.ndim == 3 and state.shape[0] == 1:
            state_btd = state
        else:
            raise ValueError(f"Expected proprio [D], [T,D], or [1,T,D], got {tuple(state.shape)}")
        if state_btd.shape[1] != action_btd.shape[1]:
            state_btd = state_btd[:, :1, :].expand(action_btd.shape[0], action_btd.shape[1], -1).clone()

    batch = {"action": action_btd, "state": state_btd}
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged = {
        "action": {
            meta["key"]: batch["action"][meta["key"]].squeeze(0)
            for meta in processor.shape_meta["action"]
        },
        "state": {
            meta["key"]: batch["state"][meta["key"]].squeeze(0)
            for meta in processor.shape_meta["state"]
        },
    }
    merged = processor.action_state_merger.forward(merged)
    return merged["action"].detach().cpu().to(torch.float32)


def normalize_proprio(processor: Any, state: np.ndarray) -> torch.Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    expected_dim = int(processor.proprio_output_dim)
    if state_tensor.numel() != expected_dim:
        raise ValueError(f"Expected proprio/state dim {expected_dim}, got {state_tensor.numel()}")
    batch = {"state": {"default": state_tensor.view(1, -1)}}
    batch = processor.normalizer.forward(batch)
    return batch["state"]["default"].squeeze(0)


def build_processor(cfg: DictConfig, dataset_stats_path: str | Path) -> Any:
    processor = instantiate(cfg.data.train.processor)
    stats = load_dataset_stats_from_json(dataset_stats_path)
    processor.set_normalizer_from_stats(stats)
    processor.eval()
    return processor


def run_file_source(args: argparse.Namespace, cfg: DictConfig, model: Any, output_dir: Path, prefix: str) -> None:
    missing = [
        name
        for name, value in {
            "--cam-high": args.cam_high,
            "--cam-left-wrist": args.cam_left_wrist,
            "--cam-right-wrist": args.cam_right_wrist,
            "--state-json": args.state_json,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError("File source requires: " + ", ".join(missing))

    processor = build_processor(cfg, args.dataset_stats)
    input_image = preprocess_real_images(
        cam_high=load_rgb_image(args.cam_high),
        cam_left_wrist=load_rgb_image(args.cam_left_wrist),
        cam_right_wrist=load_rgb_image(args.cam_right_wrist),
        device=model.device,
        dtype=model.torch_dtype,
    )
    proprio = normalize_proprio(processor, load_state_vector(args.state_json))
    num_video_frames = default_num_video_frames(cfg)
    action_horizon = default_action_horizon(cfg)
    num_chunks = int(args.num_chunks)
    if num_chunks < 1:
        raise ValueError(f"--num-chunks must be >= 1, got {num_chunks}")

    formatted_prompt = DEFAULT_PROMPT.format(task=args.prompt)
    prompt: str | None = formatted_prompt
    context = None
    context_mask = None

    rollout_frames: list[Image.Image] = []
    rollout_actions: list[torch.Tensor] = []
    chunk_metrics: list[dict[str, Any]] = []
    current_input = input_image
    cuda_device = isinstance(model.device, torch.device) and model.device.type == "cuda"
    enable_chunk_nvtx = bool(cuda_device)
    enable_chunk_profiler_api = bool(args.nsys_cuda_profiler_per_chunk and cuda_device)
    skip_first_capture = bool(args.nsys_skip_first_chunk_capture)

    if enable_chunk_profiler_api and not skip_first_capture:
        logger.info("NSYS capture start: before chunk loop (includes chunk 00).")
        torch.cuda.cudart().cudaProfilerStart()
    profiler_started = bool(enable_chunk_profiler_api and not skip_first_capture)

    # --- Async-save / pipeline mode -------------------------------------------------
    # When `--async-save` is set we (1) drop the per-chunk torch.cuda.synchronize()
    # calls so the host can keep launching kernels for chunk N+1 while chunk N is
    # still running on the GPU, (2) ask `infer_action` to return its prediction on
    # the GPU instead of doing a blocking D2H, and (3) hand the actual file IO
    # (PNG / NPY / JSON) off to a background thread so `save_image` / `save_actions`
    # never stall the main loop.  The background thread is also responsible for the
    # final D2H copy (cheap once the chunk's denoise has finished).  This collapses
    # the inter-chunk bubble that nsys shows between chunks.
    async_save = bool(args.async_save and cuda_device)
    if args.async_save and not args.infer_action_only:
        # `infer` returns a video tensor (and pil_frame_to_input_image consumes the
        # last frame to roll the next chunk's conditioning forward), which has a
        # data dependency we cannot pipeline trivially. Disable the optimisation
        # rather than silently produce wrong outputs.
        logger.warning("--async-save is currently only supported with --infer-action-only; ignoring.")
        async_save = False
    per_chunk_timing = bool(args.per_chunk_timing or not async_save)

    save_executor: ThreadPoolExecutor | None = None
    pending_saves: list[Future[Any]] = []
    # Dedicated CUDA stream for D2H copies so they don't block the default
    # stream used by CUDA Graph replay in the next chunk.
    d2h_stream: torch.cuda.Stream | None = None
    if async_save:
        save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-save")
        d2h_stream = torch.cuda.Stream(device=model.device)

    def _save_chunk_outputs(
        chunk_idx_local: int,
        chunk_prefix_local: str,
        input_image_cpu: torch.Tensor,
        action_tensor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Runs in the background thread. `input_image_cpu` is already on CPU.
        # When async_save=True the action tensor is still on GPU; we perform
        # the D2H copy on a *separate* CUDA stream (``d2h_stream``) that has
        # already been ordered after the chunk's GPU work via an event.  This
        # keeps the default stream free for the next chunk's CUDA Graph replay.
        save_image(input_image_cpu, output_dir / f"{chunk_prefix_local}_input.png")
        action_cpu: torch.Tensor | np.ndarray | None = action_tensor
        if action_tensor is not None and action_tensor.is_cuda and d2h_stream is not None:
            with torch.cuda.stream(d2h_stream):
                action_cpu = action_tensor.detach().to(device="cpu", dtype=torch.float32)
        return save_actions(
            action_cpu,
            output_dir=output_dir,
            prefix=chunk_prefix_local,
            processor=processor,
            proprio=proprio,
        )

    # In infer_action_only the input image is fixed across chunks, so cache the CPU
    # copy once to keep `_save_chunk_outputs` free of GPU sync.
    cached_input_cpu: torch.Tensor | None = None
    if async_save:
        cached_input_cpu = current_input.detach().to("cpu")
    # ---------------------------------------------------------------------------

    try:
        for chunk_idx in range(num_chunks):
            if enable_chunk_profiler_api and skip_first_capture and (not profiler_started) and chunk_idx == 1:
                logger.info("NSYS capture start: at chunk %02d (chunk 00 excluded).", chunk_idx)
                torch.cuda.cudart().cudaProfilerStart()
                profiler_started = True
            chunk_prefix = prefix if num_chunks == 1 else f"{prefix}_chunk{chunk_idx:02d}"
            chunk_seed = None if args.seed is None else int(args.seed) + chunk_idx
            if per_chunk_timing:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk_label = f"chunk {chunk_idx:02d}"
            with maybe_nvtx_range(chunk_label, enabled=enable_chunk_nvtx):
                if args.infer_action_only:
                    pred = model.infer_action(
                        prompt=prompt,
                        input_image=current_input,
                        action_horizon=action_horizon,
                        proprio=proprio,
                        context=context,
                        context_mask=context_mask,
                        text_cfg_scale=1.0,
                        num_inference_steps=args.num_inference_steps,
                        seed=chunk_seed,
                        rand_device=args.rand_device,
                        tiled=args.tiled,
                        expert_cache=args.expert_cache,
                        cuda_graph=not args.no_cuda_graph,
                        expert_cache_reuse_steps=args.expert_cache_reuse_steps,
                        expert_cache_warmup_steps=args.expert_cache_warmup_steps,
                        expert_cache_cooldown_steps=args.expert_cache_cooldown_steps,
                        return_on_device=async_save,
                    )
                else:
                    pred = model.infer(
                        prompt=prompt,
                        input_image=current_input,
                        num_frames=num_video_frames,
                        action=None,
                        action_horizon=action_horizon,
                        proprio=proprio,
                        context=context,
                        context_mask=context_mask,
                        text_cfg_scale=1.0,
                        action_cfg_scale=1.0,
                        num_inference_steps=args.num_inference_steps,
                        seed=chunk_seed,
                        rand_device=args.rand_device,
                        tiled=args.tiled,
                        expert_cache=args.expert_cache,
                        cuda_graph=not args.no_cuda_graph,
                        expert_cache_reuse_steps=args.expert_cache_reuse_steps,
                        expert_cache_warmup_steps=args.expert_cache_warmup_steps,
                        expert_cache_cooldown_steps=args.expert_cache_cooldown_steps,
                        time_inference=True,
                    )
                if per_chunk_timing:
                    torch.cuda.synchronize()
            if per_chunk_timing:
                logger.info("Inference time (chunk %d): %.3f s", chunk_idx, time.perf_counter() - t0)

            if async_save and save_executor is not None:
                assert cached_input_cpu is not None
                # Record an event on the default stream *after* this chunk's
                # GPU work finishes, then order the D2H stream to wait on it.
                # This guarantees the D2H copy sees correct data while running
                # concurrently with the next chunk's GPU kernels.
                assert d2h_stream is not None
                d2h_event = torch.cuda.Event()
                d2h_event.record(torch.cuda.default_stream(model.device))
                d2h_stream.wait_event(d2h_event)
                pending_saves.append(
                    save_executor.submit(
                        _save_chunk_outputs,
                        chunk_idx,
                        chunk_prefix,
                        cached_input_cpu,
                        pred.get("action"),
                    )
                )
                pred_action_denorm = None
            else:
                save_image(current_input, output_dir / f"{chunk_prefix}_input.png")
                pred_action_denorm = save_actions(
                    pred.get("action"),
                    output_dir=output_dir,
                    prefix=chunk_prefix,
                    processor=processor,
                    proprio=proprio,
                )
            if pred_action_denorm is not None:
                rollout_actions.append(pred_action_denorm)

            if not args.infer_action_only:
                pred_frames = pred["video"]
                save_mp4(pred_frames, str(output_dir / f"{chunk_prefix}_pred.mp4"), fps=args.fps)
                rollout_frames.extend(pred_frames if chunk_idx == 0 else pred_frames[1:])
                current_input = pil_frame_to_input_image(pred_frames[-1], device=model.device, dtype=model.torch_dtype)
            chunk_metrics.append(
                {
                    "chunk_index": chunk_idx,
                    "seed": chunk_seed,
                    "video_path": None if args.infer_action_only else f"{chunk_prefix}_pred.mp4",
                    "input_path": f"{chunk_prefix}_input.png",
                    "pred_actions_path": f"{chunk_prefix}_pred_actions.json",
                }
            )
    finally:
        if profiler_started:
            logger.info("NSYS capture stop: after chunk loop.")
            torch.cuda.cudart().cudaProfilerStop()
        if save_executor is not None:
            for fut in pending_saves:
                result = fut.result()
                if result is not None:
                    rollout_actions.append(result)
            save_executor.shutdown(wait=True)

    if num_chunks > 1 and not args.infer_action_only:
        save_mp4(rollout_frames, str(output_dir / f"{prefix}_rollout_pred.mp4"), fps=args.fps)
    if num_chunks > 1 and rollout_actions:
        action_rollout = torch.cat(rollout_actions, dim=0)
        action_np = action_rollout.detach().cpu().numpy().astype(np.float32)
        np.save(output_dir / f"{prefix}_rollout_pred_actions.npy", action_np)
        with open(output_dir / f"{prefix}_rollout_pred_actions.json", "w", encoding="utf-8") as file:
            json.dump({"action": action_np.tolist()}, file, ensure_ascii=False, indent=2)

    metrics = {
        "source": "files",
        "num_video_frames": num_video_frames,
        "action_horizon": action_horizon,
        "num_chunks": num_chunks,
        "rollout_conditioning": "previous chunk last predicted frame",
        "num_inference_steps": int(args.num_inference_steps),
        "expert_cache": bool(args.expert_cache),
        "expert_cache_reuse_steps": int(args.expert_cache_reuse_steps),
        "expert_cache_warmup_steps": int(args.expert_cache_warmup_steps),
        "expert_cache_cooldown_steps": int(args.expert_cache_cooldown_steps),
        "seed": int(args.seed),
        "prompt": formatted_prompt,
        "state_json": args.state_json,
        "used_zero_state": args.state_json is None,
        "chunks": chunk_metrics,
    }
    with open(output_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info("Saved file dry-run outputs to %s.", output_dir)


def run() -> None:
    args = parse_args()
    setup_logging(log_level=logging.INFO)

    model = load_model(
        ckpt=args.ckpt,
        device=args.device,
        mixed_precision=args.mixed_precision,
        config_dir=args.config_dir,
        task=args.task,
        dataset_stats=args.dataset_stats,
        torch_compile=args.torch_compile,
        no_cuda_graph=args.no_cuda_graph,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or "snapshot"

    run_file_source(args, model.cfg, model, output_dir, prefix)


if __name__ == "__main__":
    run()
