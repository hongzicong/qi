import time
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from qi.dit_cache import FixedStepCacheState
from qi.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        if model.text_encoder is not None:
            model.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return model

    def to(self, *args, **kwargs):
        # text_encoder is managed separately (CPU-only except during encode_prompt)
        te = self._modules.pop("text_encoder", None)
        super().to(*args, **kwargs)
        if te is not None:
            self._modules["text_encoder"] = te
        self.mot.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.inference_mode()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        cache_key = prompt if isinstance(prompt, str) else tuple(prompt)
        if not hasattr(self, "_encode_prompt_cache"):
            self._encode_prompt_cache: dict = {}
        if cache_key in self._encode_prompt_cache:
            return self._encode_prompt_cache[cache_key]
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        self.mot.to("cpu")
        self.vae.to("cpu")
        self.text_encoder.to(self.device)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        self.text_encoder.to("cpu")
        self.mot.to(self.device)
        self.vae.to(self.device)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        result = (prompt_emb.to(device=self.device), mask)
        self._encode_prompt_cache[cache_key] = result
        return result

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.inference_mode()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.inference_mode()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.inference_mode()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _make_action_expert_cache(
        self,
        *,
        enabled: bool,
        num_inference_steps: int,
        reuse_steps: int,
        warmup_steps: int,
        cooldown_steps: int,
    ) -> Optional[FixedStepCacheState]:
        if not enabled:
            return None
        return FixedStepCacheState.from_values(
            num_steps=num_inference_steps,
            reuse_steps=reuse_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
        )

    @staticmethod
    def _log_action_expert_cache(cache: Optional[FixedStepCacheState]) -> None:
        if cache is None:
            return
        logger.info(
            "Action expert fixed-step cache reused %d/%d denoising steps (body computes=%d).",
            cache.reuse_count,
            cache.num_steps,
            cache.compute_count,
        )

    def _run_action_body_with_video_cache(
        self,
        *,
        action_pre: dict[str, Any],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        expert_cache_state: Optional[FixedStepCacheState] = None,
        expert_cache_step: Optional[int] = None,
    ) -> torch.Tensor:
        body_input = action_pre["tokens"]
        if expert_cache_state is not None:
            if expert_cache_step is None:
                raise ValueError("`expert_cache_step` is required when `expert_cache_state` is enabled.")
            decision = expert_cache_state.begin_step(expert_cache_step)
            if decision.should_reuse:
                return expert_cache_state.reuse(body_input)

        body_output = self.mot.forward_action_with_video_cache(
            action_tokens=body_input,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        if expert_cache_state is not None:
            expert_cache_state.update(body_input, body_output)
        return body_output

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.inference_mode()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        expert_cache_state: Optional[FixedStepCacheState] = None,
        expert_cache_step: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        if expert_cache_state is not None:
            video_seq_len = int(video_pre["tokens"].shape[1])
            video_kv_cache, video_tokens = self.mot.prefill_video_cache_with_tokens(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
            action_tokens = self._run_action_body_with_video_cache(
                action_pre=action_pre,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                expert_cache_state=expert_cache_state,
                expert_cache_step=expert_cache_step,
            )
            pred_video = self.video_expert.post_dit(video_tokens, video_pre)
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            return pred_video, pred_action

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.inference_mode()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.inference_mode()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        expert_cache_state: Optional[FixedStepCacheState] = None,
        expert_cache_step: Optional[int] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self._run_action_body_with_video_cache(
            action_pre=action_pre,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            expert_cache_state=expert_cache_state,
            expert_cache_step=expert_cache_step,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.inference_mode()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        expert_cache: bool = False,
        expert_cache_reuse_steps: int = 1,
        expert_cache_warmup_steps: int = 1,
        expert_cache_cooldown_steps: int = 1,
        test_action_with_infer_action: bool = True,
        time_inference: bool = False,
        profile_components: bool = False,
        cuda_graph_infer_action: bool = False,
        cuda_graph_warmup_steps: int = 3,
        torch_compile_infer_action: bool = False,
        torch_compile_fullgraph: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        component_profile: dict[str, Any] = {}
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            if time_inference:
                torch.cuda.synchronize()
                _t_action = time.perf_counter()
            action_only_result = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                expert_cache=expert_cache,
                expert_cache_reuse_steps=expert_cache_reuse_steps,
                expert_cache_warmup_steps=expert_cache_warmup_steps,
                expert_cache_cooldown_steps=expert_cache_cooldown_steps,
                cuda_graph=cuda_graph_infer_action,
                cuda_graph_warmup_steps=cuda_graph_warmup_steps,
                profile_components=profile_components,
                torch_compile_infer_action=torch_compile_infer_action,
                torch_compile_fullgraph=torch_compile_fullgraph,
            )
            action_only_out = action_only_result["action"]
            if profile_components and "component_profile" in action_only_result:
                component_profile["infer_action"] = action_only_result["component_profile"]
            if time_inference:
                torch.cuda.synchronize()
                logger.info("infer_action time: %.2f s", time.perf_counter() - _t_action)
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        action_expert_cache = self._make_action_expert_cache(
            enabled=expert_cache,
            num_inference_steps=num_inference_steps,
            reuse_steps=expert_cache_reuse_steps,
            warmup_steps=expert_cache_warmup_steps,
            cooldown_steps=expert_cache_cooldown_steps,
        )
        for step_idx, (step_t_video, step_delta_video, step_t_action, step_delta_action) in enumerate(
            zip(
                infer_timesteps_video,
                infer_deltas_video,
                infer_timesteps_action,
                infer_deltas_action,
            )
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                expert_cache_state=action_expert_cache,
                expert_cache_step=step_idx,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        self._log_action_expert_cache(action_expert_cache)
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        output = {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }
        if profile_components:
            output["component_profile"] = component_profile
        return output

    def _infer_action_compute_step(
        self,
        step_latents_action: torch.Tensor,
        step_timestep_action: torch.Tensor,
        step_delta_action: torch.Tensor,
        step_context: torch.Tensor,
        step_context_mask: torch.Tensor,
        step_video_kv_cache: list[dict[str, torch.Tensor]],
        step_attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=step_latents_action,
            timestep=step_timestep_action,
            context=step_context,
            context_mask=step_context_mask,
        )
        body_input = action_pre["tokens"]
        body_output = self.mot.forward_action_with_video_cache(
            action_tokens=body_input,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=step_video_kv_cache,
            attention_mask=step_attention_mask,
            video_seq_len=video_seq_len,
        )
        pred_action = self.action_expert.post_dit(body_output, action_pre)
        next_latents_action = self.infer_action_scheduler.step(
            pred_action,
            step_delta_action,
            step_latents_action,
        )
        return next_latents_action, body_output - body_input

    def _infer_action_reuse_step(
        self,
        step_latents_action: torch.Tensor,
        step_timestep_action: torch.Tensor,
        step_delta_action: torch.Tensor,
        cached_residual: torch.Tensor,
        step_context: torch.Tensor,
        step_context_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        del video_seq_len
        action_pre = self.action_expert.pre_dit(
            action_tokens=step_latents_action,
            timestep=step_timestep_action,
            context=step_context,
            context_mask=step_context_mask,
        )
        body_input = action_pre["tokens"]
        body_output = body_input + cached_residual.to(device=body_input.device, dtype=body_input.dtype)
        pred_action = self.action_expert.post_dit(body_output, action_pre)
        return self.infer_action_scheduler.step(
            pred_action,
            step_delta_action,
            step_latents_action,
        )

    def _get_infer_action_torch_compile_steps(
        self,
        *,
        compile_reuse: bool,
        fullgraph: bool,
    ):
        if not hasattr(torch, "compile"):
            raise ValueError("`torch_compile_infer_action=True` requires torch.compile support.")
        if not hasattr(self, "_infer_action_torch_compile_cache"):
            self._infer_action_torch_compile_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

        compile_mode = "max-autotune-no-cudagraphs"
        cache_key = (compile_mode, bool(fullgraph))
        cached = self._infer_action_torch_compile_cache.setdefault(cache_key, {})
        compile_kwargs = {
            "mode": compile_mode,
            "fullgraph": bool(fullgraph),
            "dynamic": False,
        }
        if "compute" not in cached:
            cached["compute"] = torch.compile(self._infer_action_compute_step, **compile_kwargs)
            logger.info(
                "Compiled infer_action compute path with torch.compile(mode=%s, fullgraph=%s, inductor_cuda_graphs=False).",
                compile_mode,
                bool(fullgraph),
            )
        if compile_reuse and "reuse" not in cached:
            cached["reuse"] = torch.compile(self._infer_action_reuse_step, **compile_kwargs)
            logger.info(
                "Compiled infer_action reuse path with torch.compile(mode=%s, fullgraph=%s, inductor_cuda_graphs=False).",
                compile_mode,
                bool(fullgraph),
            )
        return cached["compute"], cached.get("reuse", self._infer_action_reuse_step)

    @torch.inference_mode()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        expert_cache: bool = False,
        expert_cache_reuse_steps: int = 1,
        expert_cache_warmup_steps: int = 1,
        expert_cache_cooldown_steps: int = 1,
        cuda_graph: bool = False,
        cuda_graph_warmup_steps: int = 3,
        profile_components: bool = False,
        torch_compile_infer_action: bool = False,
        torch_compile_fullgraph: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        component_profile: dict[str, Any] = {}
        if cuda_graph:
            if self.device.type != "cuda":
                raise ValueError("`cuda_graph=True` for infer_action requires a CUDA device.")
            if not torch.cuda.is_available():
                raise ValueError("`cuda_graph=True` for infer_action requires torch.cuda.is_available().")
            if cuda_graph_warmup_steps < 0:
                raise ValueError(f"`cuda_graph_warmup_steps` must be non-negative, got {cuda_graph_warmup_steps}.")
            if isinstance(getattr(self.action_expert, "freqs", None), torch.Tensor):
                self.action_expert.freqs = self.action_expert.freqs.to(device=self.device)
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)

        if profile_components:
            encoder_start = torch.cuda.Event(enable_timing=True)
            encoder_end = torch.cuda.Event(enable_timing=True)
            encoder_start.record()

        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)

        if profile_components:
            encoder_end.record()
            torch.cuda.synchronize()
            component_profile["encoder_ms"] = float(encoder_start.elapsed_time(encoder_end))

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        if profile_components:
            video_prefill_start = torch.cuda.Event(enable_timing=True)
            video_prefill_end = torch.cuda.Event(enable_timing=True)
            video_prefill_start.record()

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        if profile_components:
            video_prefill_end.record()
            torch.cuda.synchronize()
            component_profile["video_dit_prefill_ms"] = float(
                video_prefill_start.elapsed_time(video_prefill_end)
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        action_expert_cache = self._make_action_expert_cache(
            enabled=expert_cache,
            num_inference_steps=num_inference_steps,
            reuse_steps=expert_cache_reuse_steps,
            warmup_steps=expert_cache_warmup_steps,
            cooldown_steps=expert_cache_cooldown_steps,
        )

        graph_state = None
        compute_graph_replays = 0
        reuse_graph_replays = 0
        compute_action_step = self._infer_action_compute_step
        reuse_action_step = self._infer_action_reuse_step
        if torch_compile_infer_action:
            compute_action_step, reuse_action_step = self._get_infer_action_torch_compile_steps(
                compile_reuse=action_expert_cache is not None,
                fullgraph=bool(torch_compile_fullgraph),
            )
        capture_warmup_steps = int(cuda_graph_warmup_steps)

        def _infer_action_graph_key() -> tuple[Any, ...]:
            video_cache_spec = tuple(
                (
                    tuple(layer["k"].shape),
                    str(layer["k"].dtype),
                    tuple(layer["v"].shape),
                    str(layer["v"].dtype),
                )
                for layer in video_kv_cache
            )
            return (
                "infer_action",
                str(self.device),
                str(latents_action.dtype),
                tuple(latents_action.shape),
                tuple(context.shape),
                str(context.dtype),
                tuple(context_mask.shape),
                str(context_mask.dtype),
                tuple(attention_mask.shape),
                str(attention_mask.dtype),
                int(video_seq_len),
                bool(action_expert_cache is not None),
                bool(torch_compile_infer_action),
                bool(torch_compile_fullgraph),
                video_cache_spec,
            )

        def _copy_infer_action_graph_inputs(state: dict[str, Any]) -> None:
            state["static_context"].copy_(context)
            state["static_context_mask"].copy_(context_mask)
            state["static_attention_mask"].copy_(attention_mask)
            for static_layer, layer in zip(state["static_video_kv_cache"], video_kv_cache):
                static_layer["k"].copy_(layer["k"])
                static_layer["v"].copy_(layer["v"])

        def _get_infer_action_graph_state() -> dict[str, Any]:
            if not hasattr(self, "_infer_action_cuda_graph_cache"):
                self._infer_action_cuda_graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
            cache_key = _infer_action_graph_key()
            cached_state = self._infer_action_cuda_graph_cache.get(cache_key)
            if cached_state is not None:
                _copy_infer_action_graph_inputs(cached_state)
                return cached_state

            state: dict[str, Any] = {
                "static_context": context.detach().clone(),
                "static_context_mask": context_mask.detach().clone(),
                "static_attention_mask": attention_mask.detach().clone(),
                "static_video_kv_cache": [
                    {
                        "k": layer["k"].detach().clone(),
                        "v": layer["v"].detach().clone(),
                    }
                    for layer in video_kv_cache
                ],
                "static_compute_latents": latents_action.detach().clone(),
                "static_compute_timestep": infer_timesteps_action[0].reshape(1).to(
                    dtype=latents_action.dtype,
                    device=self.device,
                ).detach().clone(),
                "static_compute_delta": infer_deltas_action[0].to(
                    dtype=latents_action.dtype,
                    device=self.device,
                ).detach().clone(),
            }

            with torch.cuda.device(self.device):
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(capture_warmup_steps):
                        compute_action_step(
                            state["static_compute_latents"],
                            state["static_compute_timestep"],
                            state["static_compute_delta"],
                            state["static_context"],
                            state["static_context_mask"],
                            state["static_video_kv_cache"],
                            state["static_attention_mask"],
                            video_seq_len,
                        )
                torch.cuda.current_stream().wait_stream(warmup_stream)

                state["compute_graph"] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(state["compute_graph"]):
                    (
                        state["static_compute_next"],
                        state["static_compute_residual"],
                    ) = compute_action_step(
                        state["static_compute_latents"],
                        state["static_compute_timestep"],
                        state["static_compute_delta"],
                        state["static_context"],
                        state["static_context_mask"],
                        state["static_video_kv_cache"],
                        state["static_attention_mask"],
                        video_seq_len,
                    )

                if action_expert_cache is not None:
                    state["static_reuse_latents"] = latents_action.detach().clone()
                    state["static_reuse_timestep"] = state["static_compute_timestep"].detach().clone()
                    state["static_reuse_delta"] = state["static_compute_delta"].detach().clone()
                    state["static_reuse_residual"] = torch.zeros_like(state["static_compute_residual"])

                    warmup_stream = torch.cuda.Stream()
                    warmup_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(capture_warmup_steps):
                            reuse_action_step(
                                state["static_reuse_latents"],
                                state["static_reuse_timestep"],
                                state["static_reuse_delta"],
                                state["static_reuse_residual"],
                                state["static_context"],
                                state["static_context_mask"],
                                video_seq_len,
                            )
                    torch.cuda.current_stream().wait_stream(warmup_stream)

                    state["reuse_graph"] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(state["reuse_graph"]):
                        state["static_reuse_next"] = reuse_action_step(
                            state["static_reuse_latents"],
                            state["static_reuse_timestep"],
                            state["static_reuse_delta"],
                            state["static_reuse_residual"],
                            state["static_context"],
                            state["static_context_mask"],
                            video_seq_len,
                        )
                else:
                    state["reuse_graph"] = None

            self._infer_action_cuda_graph_cache[cache_key] = state
            logger.info(
                "Captured infer_action CUDA graph%s.",
                "s (compute + reuse)" if action_expert_cache is not None else " (compute)",
            )
            return state

        def _replay_compute_action_step(
            state: dict[str, Any],
            step_latents_action: torch.Tensor,
            step_timestep_action: torch.Tensor,
            step_delta_action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state["static_compute_latents"].copy_(step_latents_action)
            state["static_compute_timestep"].copy_(step_timestep_action)
            state["static_compute_delta"].copy_(step_delta_action)
            state["compute_graph"].replay()
            return state["static_compute_next"], state["static_compute_residual"]

        def _replay_reuse_action_step(
            state: dict[str, Any],
            step_latents_action: torch.Tensor,
            step_timestep_action: torch.Tensor,
            step_delta_action: torch.Tensor,
            cached_residual: torch.Tensor,
        ) -> torch.Tensor:
            state["static_reuse_latents"].copy_(step_latents_action)
            state["static_reuse_timestep"].copy_(step_timestep_action)
            state["static_reuse_delta"].copy_(step_delta_action)
            state["static_reuse_residual"].copy_(cached_residual)
            state["reuse_graph"].replay()
            return state["static_reuse_next"]

        if cuda_graph:
            graph_state = _get_infer_action_graph_state()

        if profile_components:
            action_dit_start = torch.cuda.Event(enable_timing=True)
            action_dit_end = torch.cuda.Event(enable_timing=True)
            action_dit_start.record()

        for step_idx, (step_t_action, step_delta_action) in enumerate(
            zip(infer_timesteps_action, infer_deltas_action)
        ):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if cuda_graph:
                if action_expert_cache is None:
                    latents_action, _ = _replay_compute_action_step(
                        graph_state,
                        latents_action,
                        timestep_action,
                        step_delta_action,
                    )
                    compute_graph_replays += 1
                    continue

                decision = action_expert_cache.begin_step(step_idx)
                if decision.should_reuse:
                    if action_expert_cache.previous_residual is None:
                        raise RuntimeError("Cannot replay infer_action reuse CUDA graph without a cached residual.")
                    latents_action = _replay_reuse_action_step(
                        graph_state,
                        latents_action,
                        timestep_action,
                        step_delta_action,
                        action_expert_cache.previous_residual,
                    )
                    reuse_graph_replays += 1
                    continue

                latents_action, residual = _replay_compute_action_step(
                    graph_state,
                    latents_action,
                    timestep_action,
                    step_delta_action,
                )
                action_expert_cache.update_residual(residual)
                compute_graph_replays += 1
                continue

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                expert_cache_state=action_expert_cache,
                expert_cache_step=step_idx,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        if profile_components:
            action_dit_end.record()
            torch.cuda.synchronize()
            action_dit_ms = float(action_dit_start.elapsed_time(action_dit_end))
            component_profile["action_dit_diffusion_ms"] = action_dit_ms
            component_profile["action_dit_avg_step_ms"] = action_dit_ms / max(int(num_inference_steps), 1)
            component_profile["num_inference_steps"] = int(num_inference_steps)
            component_profile["expert_cache"] = int(expert_cache)
            component_profile["cuda_graph"] = int(cuda_graph)
            component_profile["torch_compile"] = int(torch_compile_infer_action)
            component_profile["torch_compile_fullgraph"] = bool(torch_compile_fullgraph)
            component_profile["torch_compile_inductor_cudagraphs"] = 0
            component_profile["cuda_graph_compute_replays"] = int(compute_graph_replays)
            component_profile["cuda_graph_reuse_replays"] = int(reuse_graph_replays)
            if action_expert_cache is not None:
                component_profile["expert_cache_compute_count"] = int(action_expert_cache.compute_count)
                component_profile["expert_cache_reuse_count"] = int(action_expert_cache.reuse_count)

        if cuda_graph:
            logger.info(
                "infer_action CUDA graph replays: compute=%d, reuse=%d.",
                compute_graph_replays,
                reuse_graph_replays,
            )
        self._log_action_expert_cache(action_expert_cache)
        output = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if profile_components:
            output["component_profile"] = component_profile
        return output

    @torch.inference_mode()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        expert_cache: bool = False,
        expert_cache_reuse_steps: int = 1,
        expert_cache_warmup_steps: int = 1,
        expert_cache_cooldown_steps: int = 1,
        time_inference: bool = False,
        profile_components: bool = False,
        cuda_graph_infer_action: bool = False,
        cuda_graph_warmup_steps: int = 3,
        torch_compile_infer_action: bool = False,
        torch_compile_fullgraph: bool = True,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            expert_cache=expert_cache,
            expert_cache_reuse_steps=expert_cache_reuse_steps,
            expert_cache_warmup_steps=expert_cache_warmup_steps,
            expert_cache_cooldown_steps=expert_cache_cooldown_steps,
            time_inference=time_inference,
            profile_components=profile_components,
            cuda_graph_infer_action=cuda_graph_infer_action,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            torch_compile_infer_action=torch_compile_infer_action,
            torch_compile_fullgraph=torch_compile_fullgraph,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
