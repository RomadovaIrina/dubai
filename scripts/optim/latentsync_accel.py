#!/usr/bin/env python3
"""Optimized LatentSync 1.6 execution wrapper: batched temporal windows, SDPA backend control, optional UNet compile.

What LatentSync does per call (third_party/latentsync/latentsync/pipelines/lipsync_pipeline.py::__call__):
  - whisper-tiny features -> one 50x384 audio chunk per video frame;
  - the frame list is cut into temporal WINDOWS of `num_frames` (=16) frames; the UNet3D sees one window as a
    (batch=1, C, F=16, H/8, W/8) tensor (batch=2 with classifier-free guidance: [unconditional, conditional]);
  - windows are denoised SEQUENTIALLY: window 0 -> 20 DDIM steps, window 1 -> 20 steps, ...
    `num_frames` is the temporal length of a window, NOT a batch size.

This wrapper keeps the model, weights, resolution, steps, guidance, seed, fp16 and DeepCache policy untouched and only
changes execution: independent full windows are stacked on the batch dimension and denoised together:
  latents          (B, 4, F, h, w)          window-major
  mask/masked/ref  (B, ., F, h, w)          per-window VAE encodes done in the ORIGINAL order (same RNG consumption)
  audio embeds     (B*F, 50, 384)           window-major, matching the `(b f)` row order Transformer3DModel uses
  CFG              cat([X]*2) -> [uncond w0..wB-1, cond w0..wB-1]; noise_pred.chunk(2) splits it back the same way
  scheduler        DDIM step is per-sample elementwise (eta=0): batch-agnostic
  last partial window (F < num_frames) is denoised alone so shapes stay regular and output length is unchanged.
With window_batch_size=1 the op sequence is identical to upstream.

Usage from Python:
    from resident_vram_probe import load_latentsync
    pipe = load_latentsync()                      # frozen project loader (fp16, stage2_512)
    acc = AccelLatentSync(pipe, window_batch_size=4, deepcache=True, compile_backend="none", sdpa_backend="auto")
    acc(video_path, audio_path, out_path, temp_dir)
"""
from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
from pilot_common import THIRD_PARTY  # noqa: E402

LS = THIRD_PARTY / "latentsync"
if str(LS) not in sys.path:
    sys.path.insert(0, str(LS))

SDPA_CHOICES = ("auto", "flash", "efficient", "math")
COMPILE_CHOICES = ("none", "inductor", "tensorrt")


def _sdpa_context(name: str):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if name == "auto":
        import contextlib
        return contextlib.nullcontext()
    be = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}[name]
    return sdpa_kernel(be)


class AccelLatentSync:
    def __init__(self, pipe, window_batch_size: int = 1, deepcache: bool = True, compile_backend: str = "none",
                 sdpa_backend: str = "auto", steps: int = 20, guidance: float = 1.5, seed: int = 1247,
                 compile_options: dict | None = None):
        from omegaconf import OmegaConf
        assert window_batch_size >= 1
        assert sdpa_backend in SDPA_CHOICES and compile_backend in COMPILE_CHOICES
        self.pipe = pipe
        self.B = window_batch_size
        self.sdpa = sdpa_backend
        self.steps, self.guidance, self.seed = steps, guidance, seed
        self.cfg = OmegaConf.load(LS / "configs/unet/stage2_512.yaml")
        self.num_frames = int(self.cfg.data.num_frames)
        self.resolution = int(self.cfg.data.resolution)
        self.mask_image_path = str(LS / self.cfg.data.mask_image_path)
        self.deepcache_helper = None
        if deepcache:
            from DeepCache import DeepCacheSDHelper
            h = DeepCacheSDHelper(pipe=pipe); h.set_params(cache_interval=3, cache_branch_id=0); h.enable()
            self.deepcache_helper = h
        self.compile_backend = compile_backend
        self.compile_s = 0.0
        self.unet = pipe.unet
        if compile_backend != "none":
            import torch._dynamo
            # DeepCache's per-step skip flag is a Python bool computed from the timestep index -> dynamo specializes
            # subgraphs per skip value; keep enough cache entries so it never silently falls back to eager.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        if compile_backend == "inductor":
            t0 = time.perf_counter()
            self.unet = torch.compile(pipe.unet, backend="inductor", **(compile_options or {}))
            self.compile_s = round(time.perf_counter() - t0, 3)   # lazy: the real compile happens at first call
        elif compile_backend == "tensorrt":
            import torch_tensorrt  # noqa: F401  raises ImportError if not installed
            t0 = time.perf_counter()
            opts = {"enabled_precisions": {torch.float16}, "truncate_long_and_double": True, "min_block_size": 1}
            opts.update(compile_options or {})
            self.unet = torch.compile(pipe.unet, backend="torch_tensorrt", dynamic=False, options=opts)
            self.compile_s = round(time.perf_counter() - t0, 3)
        self._image_processor = None
        self.last_stats: dict = {}

    # ---------------------------------------------------------------------------------------------------------
    def image_processor(self):
        """ImageProcessor (insightface detector + aligner) is created once and reused; upstream recreates it per call."""
        if self._image_processor is None:
            from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
            mask_image = load_fixed_mask(self.resolution, self.mask_image_path)
            self._image_processor = ImageProcessor(self.resolution, device="cuda", mask_image=mask_image)
        self.pipe.image_processor = self._image_processor
        return self._image_processor

    @torch.no_grad()
    def __call__(self, video_path, audio_path, video_out_path, temp_dir, weight_dtype=torch.float16,
                 video_fps: int = 25, audio_sample_rate: int = 16000, generator=None) -> dict:
        from accelerate.utils import set_seed
        from latentsync.utils.util import read_audio, read_video, write_video
        import soundfile as sf
        import shutil

        pipe = self.pipe
        set_seed(self.seed)
        old = os.getcwd(); os.chdir(LS)
        try:
            t_all = time.perf_counter()
            pipe.unet.eval()
            device = pipe._execution_device
            ip = self.image_processor()
            height = width = self.resolution
            nf = self.num_frames
            do_cfg = self.guidance > 1.0
            pipe.scheduler.set_timesteps(self.steps, device=device)
            timesteps = pipe.scheduler.timesteps
            extra = pipe.prepare_extra_step_kwargs(generator, 0.0)

            t0 = time.perf_counter()
            whisper_feature = pipe.audio_encoder.audio2feat(str(audio_path))
            whisper_chunks = pipe.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
            audio_samples = read_audio(str(audio_path))
            video_frames = read_video(str(video_path), use_decord=False)
            t_pre = time.perf_counter() - t0

            t0 = time.perf_counter()
            video_frames, faces, boxes, affine_matrices = pipe.loop_video(whisper_chunks, video_frames)
            t_affine = time.perf_counter() - t0

            n_chunks = len(whisper_chunks)
            all_latents = pipe.prepare_latents(n_chunks, pipe.vae.config.latent_channels, height, width, weight_dtype, device, generator)
            num_inf = math.ceil(n_chunks / nf)
            full = [i for i in range(num_inf) if (i + 1) * nf <= n_chunks]
            partial = [i for i in range(num_inf) if i not in full]
            groups = [full[k:k + self.B] for k in range(0, len(full), self.B)] + [[i] for i in partial]

            synced = [None] * num_inf
            t_unet = 0.0; t_vae = 0.0; unet_calls = 0
            for grp in groups:
                # ---- per-window conditioning, in upstream order (window-major; mask encode then ref encode) ----
                t0 = time.perf_counter()
                lat_l, mask_l, masked_l, ref_l, audio_l, ref_px_l, masks_l = [], [], [], [], [], [], []
                for i in grp:
                    sl = slice(i * nf, (i + 1) * nf)
                    inference_faces = faces[sl]
                    ref_px, masked_px, masks = ip.prepare_masks_and_masked_images(inference_faces, affine_transform=False)
                    m_lat, mi_lat = pipe.prepare_mask_latents(masks, masked_px, height, width, weight_dtype, device, generator, False)
                    r_lat = pipe.prepare_image_latents(ref_px, device, weight_dtype, generator, False)
                    lat_l.append(all_latents[:, :, sl]); mask_l.append(m_lat); masked_l.append(mi_lat); ref_l.append(r_lat)
                    audio_l.append(torch.stack(whisper_chunks[sl]))
                    ref_px_l.append(ref_px); masks_l.append(masks)
                latents = torch.cat(lat_l, dim=0)                       # (B, 4, F, h, w)
                mask_latents = torch.cat(mask_l, dim=0)
                masked_image_latents = torch.cat(masked_l, dim=0)
                ref_latents = torch.cat(ref_l, dim=0)
                audio_embeds = torch.cat(audio_l, dim=0).to(device, dtype=weight_dtype)   # (B*F, 50, 384)
                if do_cfg:
                    mask_latents = torch.cat([mask_latents] * 2)
                    masked_image_latents = torch.cat([masked_image_latents] * 2)
                    ref_latents = torch.cat([ref_latents] * 2)
                    audio_embeds = torch.cat([torch.zeros_like(audio_embeds), audio_embeds])
                torch.cuda.synchronize(); t_vae += time.perf_counter() - t0

                # ---- denoising loop for the whole group ----
                t0 = time.perf_counter()
                with _sdpa_context(self.sdpa):
                    for t in timesteps:
                        unet_input = torch.cat([latents] * 2) if do_cfg else latents
                        unet_input = pipe.scheduler.scale_model_input(unet_input, t)
                        unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                        unet_calls += 1
                        if do_cfg:
                            n_unc, n_aud = noise_pred.chunk(2)
                            noise_pred = n_unc + self.guidance * (n_aud - n_unc)
                        latents = pipe.scheduler.step(noise_pred, t, latents, **extra).prev_sample
                torch.cuda.synchronize(); t_unet += time.perf_counter() - t0

                # ---- decode per window, paste the untouched pixels back ----
                t0 = time.perf_counter()
                for k, i in enumerate(grp):
                    dec = pipe.decode_latents(latents[k:k + 1])
                    dec = pipe.paste_surrounding_pixels_back(dec, ref_px_l[k], 1 - masks_l[k], device, weight_dtype)
                    synced[i] = dec
                torch.cuda.synchronize(); t_vae += time.perf_counter() - t0

            t0 = time.perf_counter()
            synced_video_frames = pipe.restore_video(torch.cat(synced), video_frames, boxes, affine_matrices)
            t_restore = time.perf_counter() - t0
            remain = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
            audio_samples = audio_samples[:remain].cpu().numpy()

            t0 = time.perf_counter()
            temp_dir = str(temp_dir)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)
            sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-i", os.path.join(temp_dir, "video.mp4"),
                            "-i", os.path.join(temp_dir, "audio.wav"), "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
                            "-q:v", "0", "-q:a", "0", str(video_out_path)], check=True)
            t_write = time.perf_counter() - t0
            self.last_stats = {"frames": int(synced_video_frames.shape[0]), "windows": num_inf, "full_windows": len(full),
                               "partial_windows": len(partial), "groups": len(groups), "window_batch_size": self.B,
                               "unet_calls": unet_calls, "seconds": {"preprocess_audio_video": round(t_pre, 3), "affine": round(t_affine, 3),
                               "vae_and_cond": round(t_vae, 3), "unet_denoise": round(t_unet, 3), "restore": round(t_restore, 3),
                               "write": round(t_write, 3), "total": round(time.perf_counter() - t_all, 3)}}
            return self.last_stats
        finally:
            os.chdir(old)


def compare_videos(a: pathlib.Path, b: pathlib.Path, max_frames: int | None = None) -> dict:
    """Frame-wise PSNR between two videos (same frame count expected). Used to check batched output vs sequential."""
    import cv2
    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    psnrs = []; n = 0
    while True:
        oa, fa = ca.read(); ob, fb = cb.read()
        if not (oa and ob):
            break
        if fa.shape != fb.shape:
            break
        mse = float(np.mean((fa.astype(np.float32) - fb.astype(np.float32)) ** 2))
        psnrs.append(99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)); n += 1
        if max_frames and n >= max_frames:
            break
    na = int(ca.get(cv2.CAP_PROP_FRAME_COUNT)); nb = int(cb.get(cv2.CAP_PROP_FRAME_COUNT))
    ca.release(); cb.release()
    return {"frames_a": na, "frames_b": nb, "compared": n, "psnr_mean": round(float(np.mean(psnrs)), 2) if psnrs else None,
            "psnr_min": round(float(np.min(psnrs)), 2) if psnrs else None, "psnr_p05": round(float(np.percentile(psnrs, 5)), 2) if psnrs else None}
