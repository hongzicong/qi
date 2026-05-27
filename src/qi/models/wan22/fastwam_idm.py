from typing import Any, Optional

import torch
import torch.nn.functional as F

from qi.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)


class FastWAMIDM(FastWAMJoint):
    """IDM variant with teacher-forcing video conditioning for action denoising."""

    # Hardcoded probability: during training, cond-video is noised with this chance.
    video_cond_noise_prob = 0.5

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                "Teacher-forcing requires identical `tokens_per_frame` for noisy and cond video branches, "
                f"got {noisy_video_tokens_per_frame} and {cond_video_tokens_per_frame}."
            )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # noisy_video -> noisy_video
        mask[:noisy_end, :noisy_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=noisy_video_seq_len,
            video_tokens_per_frame=noisy_video_tokens_per_frame,
            device=device,
        )
        # cond_video -> cond_video
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[cond_end:, cond_end:] = True
        # action -> cond_video only
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
    ) -> dict[str, Any]:
        # Reuse infer_joint pipeline and keep infer_action output contract.
        out = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
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
            test_action_with_infer_action=False,
            expert_cache=expert_cache,
            expert_cache_reuse_steps=expert_cache_reuse_steps,
            expert_cache_warmup_steps=expert_cache_warmup_steps,
            expert_cache_cooldown_steps=expert_cache_cooldown_steps,
        )
        return {"action": out["action"]}

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()

        if action is not None:
            logger.warning(
                "`FastWAMIDM.infer_joint` ignores `action` input; "
                "video is denoised in a standalone first stage."
            )

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

        # Stage 1: denoise video only.
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            pred_video = self.video_expert(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        # Stage 2: freeze denoised video as cond and denoise action via video K/V cache.
        timestep_video_cond = torch.zeros(
            (latents_video.shape[0],), dtype=latents_video.dtype, device=self.device
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre_cond["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=video_pre_cond["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
            video_context_payload={
                "context": video_pre_cond["context"],
                "mask": video_pre_cond["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
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
        for step_idx, (step_t_action, step_delta_action) in enumerate(
            zip(infer_timesteps_action, infer_deltas_action)
        ):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
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
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        self._log_action_expert_cache(action_expert_cache)
        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
