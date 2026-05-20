from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from qi.datasets.dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving


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


def rgb_to_float_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image as [H,W,3], got {rgb_uint8.shape}")
    rgb_uint8 = np.ascontiguousarray(rgb_uint8).copy()
    return torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(torch.float32) / 255.0


def pil_frame_to_input_image(frame: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
    image = torch.from_numpy(arr).permute(2, 0, 1).contiguous() / 255.0
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def pil_frames_to_video_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("`frames` must be non-empty.")
    frame_tensors = []
    for frame in frames:
        arr = np.array(frame.convert("RGB"), dtype=np.float32) / 255.0
        frame_tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())  # [3, H, W]
    return torch.stack(frame_tensors, dim=1)  # [3, T, H, W]


def save_image(input_image: torch.Tensor, path: str | Path) -> None:
    frame = input_image.detach().float().cpu().squeeze(0)
    tensor_video_to_pil_frames(frame.unsqueeze(1), "minus_one_one")[0].save(path)


def preprocess_real_images(
    cam_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cam_high_t = transforms_F.resize(
        rgb_to_float_chw(cam_high),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left_t = transforms_F.resize(
        rgb_to_float_chw(cam_left_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right_t = transforms_F.resize(
        rgb_to_float_chw(cam_right_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_high_t = transforms_F.resize(
        cam_high_t,
        [256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left_t = transforms_F.resize(
        cam_left_t,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right_t = transforms_F.resize(
        cam_right_t,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left_t, cam_right_t], dim=-1)
    image = torch.cat([cam_high_t, bottom], dim=-2)
    resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 384})
    crop = CenterCrop(args={"img_w": 320, "img_h": 384})
    image = crop(resize(image))
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def tensor_video_to_pil_frames(video: torch.Tensor, value_range: str) -> list[Image.Image]:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"Expected video [3,T,H,W], got {tuple(video.shape)}")
    video = video.detach().float().cpu()
    if value_range == "minus_one_one":
        video = (video.clamp(-1.0, 1.0) + 1.0) * 0.5
    elif value_range == "zero_one":
        video = video.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported value_range: {value_range}")
    frames = []
    for t in range(video.shape[1]):
        arr = (video[:, t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames
