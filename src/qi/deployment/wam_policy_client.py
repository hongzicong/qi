"""WAM policy adapter with an RTC-compatible client interface."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from qi.datasets.dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving
from qi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from qi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from qi.api import load_model_from_config
from qi.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True)
class WAMPolicyConfig:
    """Configuration needed to load WAM as a policy backend."""

    ckpt: str
    dataset_stats: str
    task: str = "real_cleaning_uncond_3cam_384_1e-4"
    config_dir: str = "configs"
    train_config: str | None = None
    device: str = "cuda:0"
    mixed_precision: str = "bf16"
    prompt: str = "clean the table"
    context_cache_dir: str | None = None
    use_text_encoder: bool = False
    action_horizon: int = 32
    num_inference_steps: int = 10
    seed: int = 42
    rand_device: str = "cpu"
    expert_cache: bool = False
    expert_cache_reuse_steps: int = 1
    expert_cache_warmup_steps: int = 1
    expert_cache_cooldown_steps: int = 1
    cuda_graph: bool = False
    torch_compile: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], base_dir: str | Path | None = None) -> "WAMPolicyConfig":
        """Build config from a deployment YAML mapping.

        The mapping may either contain these keys directly or under a
        top-level ``wam`` section.
        """
        raw = payload.get("wam", payload)
        if not isinstance(raw, Mapping):
            raise TypeError("Expected a mapping or a top-level `wam` mapping.")

        def get(name: str, default: Any = None) -> Any:
            return raw[name] if name in raw else default

        def bool_value(name: str, default: bool = False) -> bool:
            value = get(name, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        def path_value(name: str, required: bool = False) -> str | None:
            value = get(name)
            if value is None:
                if required:
                    raise ValueError(f"Missing required WAM policy config key: {name}")
                return None
            path = Path(str(value)).expanduser()
            if not path.is_absolute() and base_dir is not None:
                path = Path(base_dir).expanduser() / path
            return str(path)

        return cls(
            ckpt=path_value("ckpt", required=True) or "",
            dataset_stats=path_value("dataset_stats", required=True) or "",
            task=str(get("task", cls.task)),
            config_dir=path_value("config_dir") or cls.config_dir,
            train_config=path_value("train_config"),
            device=str(get("device", cls.device)),
            mixed_precision=str(get("mixed_precision", cls.mixed_precision)),
            prompt=str(get("prompt", cls.prompt)),
            context_cache_dir=path_value("context_cache_dir"),
            use_text_encoder=bool_value("use_text_encoder", cls.use_text_encoder),
            action_horizon=int(get("action_horizon", cls.action_horizon)),
            num_inference_steps=int(get("num_inference_steps", cls.num_inference_steps)),
            seed=int(get("seed", cls.seed)),
            rand_device=str(get("rand_device", cls.rand_device)),
            expert_cache=bool_value("expert_cache", cls.expert_cache),
            expert_cache_reuse_steps=int(get("expert_cache_reuse_steps", cls.expert_cache_reuse_steps)),
            expert_cache_warmup_steps=int(get("expert_cache_warmup_steps", cls.expert_cache_warmup_steps)),
            expert_cache_cooldown_steps=int(get("expert_cache_cooldown_steps", cls.expert_cache_cooldown_steps)),
            cuda_graph=bool_value("cuda_graph", cls.cuda_graph),
            torch_compile=bool_value("torch_compile", cls.torch_compile),
        )


@dataclass
class WAMObservation:
    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    state: np.ndarray
    prompt: str


class WAMPolicyClient:
    """Local WAM policy client compatible with RTC-style deployment code."""

    def __init__(self, config: WAMPolicyConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, WAMPolicyConfig) else WAMPolicyConfig.from_mapping(config)
        self.cfg = self._load_config()
        model_dtype = dtype_from_mixed_precision(self.config.mixed_precision)
        self.model = load_model_from_config(
            cfg=self.cfg,
            ckpt=self.config.ckpt,
            device=self.config.device,
            model_dtype=model_dtype,
            torch_compile=self.config.torch_compile,
            cuda_graph=self.config.cuda_graph,
        )

        self.processor = instantiate(self.cfg.data.train.processor)
        stats = load_dataset_stats_from_json(self.config.dataset_stats)
        self.processor.set_normalizer_from_stats(stats)
        self.processor.eval()

        self.observation: WAMObservation | None = None
        self._context_cache: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        """Store the latest RTC-style observation for the next ``get_action`` call."""
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an `images` mapping.")

        missing = [key for key in IMAGE_KEYS if key not in images or images[key] is None]
        if missing:
            raise ValueError(f"Observation is missing required image keys: {missing}")

        if "state" not in obs:
            raise ValueError("Observation must contain a `state` vector.")

        prompt = str(obs.get("prompt", self.config.prompt))
        self.observation = WAMObservation(
            cam_high=np.asarray(images["cam_high"]),
            cam_left_wrist=np.asarray(images["cam_left_wrist"]),
            cam_right_wrist=np.asarray(images["cam_right_wrist"]),
            state=np.asarray(obs["state"], dtype=np.float32),
            prompt=prompt,
        )

    def get_action(self) -> np.ndarray:
        """Run WAM inference and return a denormalized ``[T,D]`` action chunk.

        ``D`` is the checkpoint's action dimension (14 for dual-arm, 7 for a
        single arm); the policy stack is action-dim agnostic.
        """
        if self.observation is None:
            raise RuntimeError("Policy observation is empty. Call update_observation(obs) before get_action().")

        prompt, context, context_mask = self._resolve_text_condition(self.observation.prompt)
        image = preprocess_real_images(self.observation, self.model.device, self.model.torch_dtype)
        proprio = normalize_proprio(self.processor, self.observation.state)

        infer_kwargs = {
            "prompt": prompt,
            "input_image": image,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": self.config.action_horizon,
            "num_inference_steps": self.config.num_inference_steps,
            "seed": self.config.seed,
            "rand_device": self.config.rand_device,
        }
        infer_params = inspect.signature(self.model.infer_action).parameters
        if "num_video_frames" in infer_params:
            infer_kwargs["num_video_frames"] = default_num_video_frames(self.cfg)
        if "expert_cache" in infer_params:
            infer_kwargs.update(
                {
                    "expert_cache": self.config.expert_cache,
                    "expert_cache_reuse_steps": self.config.expert_cache_reuse_steps,
                    "expert_cache_warmup_steps": self.config.expert_cache_warmup_steps,
                    "expert_cache_cooldown_steps": self.config.expert_cache_cooldown_steps,
                }
            )
        if "cuda_graph" in infer_params:
            infer_kwargs["cuda_graph"] = self.config.cuda_graph

        with torch.no_grad():
            output = self.model.infer_action(**infer_kwargs)

        actions = denormalize_actions(self.processor, output["action"])
        return actions.detach().cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        """Clear per-episode observation state."""
        self.observation = None

    def _load_config(self) -> DictConfig:
        if self.config.train_config:
            cfg = OmegaConf.load(self.config.train_config)
            if not isinstance(cfg, DictConfig):
                raise ValueError(f"Expected DictConfig in {self.config.train_config}, got {type(cfg)}")
            if self.config.use_text_encoder:
                cfg.model.load_text_encoder = True
            return cfg

        config_dir = str(Path(self.config.config_dir).expanduser().resolve())
        overrides = [f"task={self.config.task}", f"mixed_precision={self.config.mixed_precision}"]
        if self.config.use_text_encoder:
            overrides.append("model.load_text_encoder=true")
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            return compose(config_name="train", overrides=overrides)

    def _resolve_text_condition(self, task_prompt: str) -> tuple[str | None, torch.Tensor | None, torch.Tensor | None]:
        formatted_prompt = format_prompt(task_prompt)
        if self.config.use_text_encoder:
            return formatted_prompt, None, None

        if formatted_prompt not in self._context_cache:
            cache_dir = self.config.context_cache_dir or self.cfg.data.train.get("text_embedding_cache_dir")
            if cache_dir is None:
                raise ValueError("No context cache dir found. Set context_cache_dir or use_text_encoder=true.")
            context_len = int(self.cfg.data.train.get("context_len", 128))
            self._context_cache[formatted_prompt] = load_cached_context(cache_dir, formatted_prompt, context_len)

        context, context_mask = self._context_cache[formatted_prompt]
        return None, context, context_mask


def dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def format_prompt(task_prompt: str) -> str:
    return DEFAULT_PROMPT.format(task=task_prompt)


def default_num_video_frames(cfg: DictConfig) -> int:
    num_frames = int(cfg.data.train.num_frames)
    ratio = int(cfg.data.train.get("action_video_freq_ratio", 1))
    return ((num_frames - 1) // ratio) + 1


def load_cached_context(cache_dir: str | Path, formatted_prompt: str, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir)
    cache_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache: {cache_path}. "
            "Run scripts/precompute_text_embeds.py for this prompt, or set use_text_encoder=true."
        )
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.ndim != 2 or context_mask.ndim != 1:
        raise ValueError(f"Invalid cached context shapes in {cache_path}: {context.shape}, {context_mask.shape}")
    if context.shape[0] != context_len or context_mask.shape[0] != context_len:
        raise ValueError(
            f"Cached context length mismatch in {cache_path}: "
            f"context={context.shape[0]}, mask={context_mask.shape[0]}, expected={context_len}"
        )
    context = context.clone()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask, dtype=torch.bool)
    return context, context_mask


def rgb_to_float_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image as [H,W,3], got {rgb_uint8.shape}")
    rgb_uint8 = np.ascontiguousarray(rgb_uint8).copy()
    return torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(torch.float32) / 255.0


def preprocess_real_images(observation: WAMObservation, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cam_high = transforms_F.resize(
        rgb_to_float_chw(observation.cam_high),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        rgb_to_float_chw(observation.cam_left_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        rgb_to_float_chw(observation.cam_right_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_high = transforms_F.resize(
        cam_high,
        [256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        cam_left,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        cam_right,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    image = torch.cat([cam_high, bottom], dim=-2)
    resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 384})
    crop = CenterCrop(args={"img_w": 320, "img_h": 384})
    image = crop(resize(image))
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def normalize_proprio(processor: Any, state: np.ndarray) -> torch.Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    expected_dim = int(processor.proprio_output_dim)
    if state_tensor.numel() != expected_dim:
        raise ValueError(f"Expected proprio/state dim {expected_dim}, got {state_tensor.numel()}")
    batch = {"state": {"default": state_tensor.view(1, -1)}}
    batch = processor.normalizer.forward(batch)
    return batch["state"]["default"].squeeze(0)


def denormalize_actions(processor: Any, action_norm: torch.Tensor) -> torch.Tensor:
    if action_norm.ndim == 3 and action_norm.shape[0] == 1:
        action_norm = action_norm.squeeze(0)
    if action_norm.ndim == 1:
        action_norm = action_norm.view(1, -1)
    if action_norm.ndim != 2:
        raise ValueError(f"Expected normalized action [T,D], got {tuple(action_norm.shape)}")
    dummy_state = torch.zeros(
        1,
        action_norm.shape[0],
        int(processor.proprio_output_dim),
        dtype=torch.float32,
        device=action_norm.device,
    )
    batch = {
        "action": {"default": action_norm.unsqueeze(0).to(torch.float32)},
        "state": {"default": dummy_state},
    }
    batch = processor.normalizer.backward(batch)
    return batch["action"]["default"].squeeze(0)


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_state_vector(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        if "state" not in payload:
            raise ValueError(f"{path} must contain a `state` key or be a raw list.")
        payload = payload["state"]
    state = np.asarray(payload, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"State must be a 1D vector, got shape {state.shape}")
    return state