#!/usr/bin/env python3
"""Training-free LatentSync 1.6 stabilisation variants (track A, 2026-10-16), built on scripts/optim/latentsync_accel.AccelLatentSync.

Upstream LatentSync (and AccelLatentSync) denoise an LS segment in independent, non-overlapping 16-frame windows and use the CURRENT
frames as the UNet reference image, although the model was trained with a reference window from a different time
(latentsync/data/unet_dataset.py: ref_start_idx outside +-num_frames of the target window). lipsync_jitter.py measured a x1.45-1.5 mouth
jump at every window start on 03. Variants (each one isolated; everything else = AccelLatentSync):

  stride S        overlapping windows: 16-frame windows every S frames (last one aligned to the segment end); at EVERY denoising step the
                  noise predictions of all windows are fused per frame with triangular weights (MultiDiffusion / MimicMotion-style progressive
                  latent fusion) before one DDIM step on the whole segment. DeepCache keeps working: each window group gets its own feature
                  cache (DeepCache's cache dict is swapped per group; its step index comes from the timestep value).
  ref_mode fixed  the UNet reference latents come from a fixed 16-frame reference clip (aligned crops of other frames of the same person,
                  chosen by the caller), tiled over every window; the paste-back of the unmasked upper crop still uses the CURRENT frame.
  mode wrapper    no generation: the aligned crop itself goes through restore (inverse affine) -> the caller's paste-back = jitter of
                  crop / affine / paste-back alone.
  mode vae        crop -> VAE encode / decode -> restore: adds the VAE's texture flicker.
  overlap continue  (memory-light, DeepCache-compatible) windows are denoised ONE AT A TIME with stride S; the frames a window shares
                  with the previous one are pinned to the previous result: at every step they are re-noised from it with the shared initial
                  noise (RePaint-style temporal continuation), so the new frames are generated consistent with the tail of the previous window.
  window_len L    temporal window length (16 = trained; the motion module's positional table allows up to 24).
  bridge Lb/k     (budget seam fix) after the normal independent windows, every seam gets a short BRIDGE window of Lb frames centred on it:
                  its latents are re-noised to the level of the last k DDIM steps (shared initial noise), denoised again for those k steps
                  (DeepCache per bridge window, one at a time), and blended back with a triangular weight peaking at the seam. The motion
                  module sees both sides of the seam in one window (what per-step fusion does at every step, here only at low noise).
With stride None, ref_mode 'same', mode 'ls' the class is AccelLatentSync unchanged.
"""
from __future__ import annotations
import math, os, pathlib, sys, time
import numpy as np
import torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "optim")); sys.path.insert(0, str(HERE.parent / "pilot"))
from latentsync_accel import LS, AccelLatentSync, _ReplayDetector, _sdpa_context   # noqa: E402


class StableLatentSync(AccelLatentSync):
    def __init__(self, pipe, *, stride: int | None = None, ref_mode: str = "same", mode: str = "ls", overlap: str = "fuse", window_len: int | None = None,
                 bridge_len: int = 0, bridge_steps: int = 0, **kw):
        super().__init__(pipe, **kw)
        assert mode in ("ls", "wrapper", "vae") and (ref_mode in ("same", "fixed") or ref_mode.startswith("shift")) and overlap in ("fuse", "continue")
        self.stride, self.ref_mode, self.mode, self.overlap = stride, ref_mode, mode, overlap
        self.bridge_len, self.bridge_steps = int(bridge_len), int(bridge_steps)
        if window_len:
            assert 1 <= window_len <= 24, "motion module positional table: <= 24 frames"
            self.num_frames = int(window_len)
        self.ref_frames = None; self.ref_dets = None          # set by the caller for ref_mode fixed: (16, H, W, 3) RGB uint8 + replayed detections

    def set_reference(self, frames_rgb: np.ndarray, detections: list):
        self.ref_frames, self.ref_dets = np.ascontiguousarray(frames_rgb), list(detections)

    # ---------------------------------------------------------------------------------------------------------
    def _ref_crops(self, ip):
        """aligned 512^2 crops of the reference clip, same affine code path (own p_bias state, reset afterwards)."""
        real = ip.face_detector; ip.face_detector = _ReplayDetector(real, self.ref_dets); ip.restorer.p_bias = None
        try:
            faces = [ip.affine_transform(f)[0] for f in self.ref_frames]
        finally:
            ip.face_detector = real; ip.restorer.p_bias = None
        return torch.stack(faces)

    @torch.no_grad()
    def _core(self, video_frames, audio_path, detections=None, weight_dtype=torch.float16, video_fps=25, generator=None):
        if self.stride is None and self.ref_mode == "same" and self.mode == "ls" and self.num_frames == 16 and not self.bridge_len:
            return super()._core(video_frames, audio_path, detections, weight_dtype, video_fps, generator)
        from accelerate.utils import set_seed
        pipe = self.pipe; set_seed(self.seed); old = os.getcwd(); os.chdir(LS)
        try:
            t_all = time.perf_counter(); pipe.unet.eval(); device = pipe._execution_device
            ip = self.image_processor(); ip.restorer.p_bias = None
            H = W = self.resolution; nf = self.num_frames; do_cfg = self.guidance > 1.0
            pipe.scheduler.set_timesteps(self.steps, device=device); timesteps = pipe.scheduler.timesteps
            extra = pipe.prepare_extra_step_kwargs(generator, 0.0)
            ref_faces = self._ref_crops(ip) if (self.ref_mode == "fixed" and self.mode == "ls") else None
            t0 = time.perf_counter()
            feat = pipe.audio_encoder.audio2feat(str(audio_path)); chunks = pipe.audio_encoder.feature2chunks(feature_array=feat, fps=video_fps)
            if self.chunk_hook is not None: chunks = self.chunk_hook(pipe.audio_encoder, chunks)
            t_pre = time.perf_counter() - t0; t0 = time.perf_counter()
            real = ip.face_detector; replay = None
            if detections is not None:
                replay = _ReplayDetector(real, detections); ip.face_detector = replay
            try:
                video_frames, faces, boxes, affine = pipe.loop_video(chunks, video_frames)
            finally:
                ip.face_detector = real
            t_aff = time.perf_counter() - t0; n = len(chunks)
            # current-frame pixels / masks (paste-back + masked image), per 16-frame chunk to bound memory
            cur_px, masked_px, masks = [], [], []
            for a in range(0, n, nf):
                p, mp, m = ip.prepare_masks_and_masked_images(faces[a:a + nf], affine_transform=False); cur_px.append(p); masked_px.append(mp); masks.append(m)
            cur_px, masked_px, masks = torch.cat(cur_px), torch.cat(masked_px), torch.cat(masks)
            t_vae = 0.0; t_unet = 0.0; unet_calls = 0; windows = []
            if self.mode == "wrapper":
                synced = cur_px.to(device, weight_dtype)
            elif self.mode == "vae":
                out = []; t0 = time.perf_counter()
                for a in range(0, n, nf):
                    lat = pipe.prepare_image_latents(cur_px[a:a + nf], device, weight_dtype, generator, False)
                    dec = pipe.decode_latents(lat)
                    out.append(pipe.paste_surrounding_pixels_back(dec, cur_px[a:a + nf], 1 - masks[a:a + nf], device, weight_dtype))
                synced = torch.cat(out); t_vae = time.perf_counter() - t0
            else:
                latents = pipe.prepare_latents(n, pipe.vae.config.latent_channels, H, W, weight_dtype, device, generator)   # (1, 4, n, h, w), same noise per frame
                eps_init = latents.clone()   # the exact initial noise of this trajectory (bridges re-noise with it)
                # per-frame conditioning latents, encoded once (chunks of 16, original order)
                t0 = time.perf_counter(); m_l, mi_l, r_l = [], [], []
                if ref_faces is not None:
                    rp, _, _ = ip.prepare_masks_and_masked_images(ref_faces, affine_transform=False)
                    ref_lat16 = pipe.prepare_image_latents(rp, device, weight_dtype, generator, False)          # (1, 4, R, h, w)
                for a in range(0, n, nf):
                    mk, mi = pipe.prepare_mask_latents(masks[a:a + nf], masked_px[a:a + nf], H, W, weight_dtype, device, generator, False)
                    m_l.append(mk); mi_l.append(mi)
                    if ref_faces is None:
                        r_l.append(pipe.prepare_image_latents(cur_px[a:a + nf], device, weight_dtype, generator, False))
                mask_lat, masked_lat = torch.cat(m_l, 2), torch.cat(mi_l, 2)
                if ref_faces is None:
                    ref_lat = torch.cat(r_l, 2)
                else:   # reference clip tiled so that every window sees the R reference frames in order
                    R = ref_lat16.shape[2]; ref_lat = ref_lat16[:, :, [k % R for k in range(n)]]
                if self.ref_mode.startswith("shift"):   # round 3: reference = the same segment S frames away (training-like: another time, same pose / light)
                    S = int(self.ref_mode[5:] or 16)
                    idx = [i + S if i + S < n else (i - S if i - S >= 0 else (i + n // 2) % n) for i in range(n)]
                    ref_lat = ref_lat[:, :, idx]
                t_vae += time.perf_counter() - t0
                audio_all = torch.stack(chunks).to(device, dtype=weight_dtype)                                    # (n, 50, 384)
                L = min(nf, n)
                if self.stride is None:
                    starts = list(range(0, n, nf))
                else:
                    starts = list(range(0, max(n - L, 0) + 1, self.stride))
                    if starts[-1] + L < n: starts.append(n - L)
                windows = [(s, min(s + L, n)) for s in starts]
                full = [w for w in windows if w[1] - w[0] == L]; part = [w for w in windows if w[1] - w[0] != L]
                groups = [full[k:k + self.B] for k in range(0, len(full), self.B)] + [[w] for w in part]
                def tri(Lw):
                    return torch.tensor([min(i + 1, Lw - i) for i in range(Lw)], device=device, dtype=torch.float32)
                caches = [{} for _ in groups]; dc = self.deepcache_helper
                t0 = time.perf_counter()
                if self.overlap == "continue" and self.stride is not None:
                    eps0 = latents.clone(); final = torch.empty_like(latents); done = 0
                    with _sdpa_context(self.sdpa):
                        for a_, b_ in windows:
                            o = max(0, done - a_); x = eps0[:, :, a_:b_].clone(); known = final[:, :, a_:a_ + o] if o > 0 else None
                            mk, mi, rl, au = mask_lat[:, :, a_:b_], masked_lat[:, :, a_:b_], ref_lat[:, :, a_:b_], audio_all[a_:b_]
                            if do_cfg:
                                mk, mi, rl, au = torch.cat([mk] * 2), torch.cat([mi] * 2), torch.cat([rl] * 2), torch.cat([torch.zeros_like(au), au])
                            if dc is not None:
                                dc.cached_output = {}
                            for t in timesteps:
                                if o > 0:   # pin the shared frames to the previous window's result at this noise level
                                    x[:, :, :o] = pipe.scheduler.add_noise(known, eps0[:, :, a_:a_ + o], t.reshape(1))
                                x2 = torch.cat([x] * 2) if do_cfg else x
                                npred = self.unet(torch.cat([pipe.scheduler.scale_model_input(x2, t), mk, mi, rl], dim=1), t, encoder_hidden_states=au).sample; unet_calls += 1
                                if do_cfg:
                                    nu, na = npred.chunk(2); npred = nu + self.guidance * (na - nu)
                                x = pipe.scheduler.step(npred, t, x, **extra).prev_sample
                            if o > 0:
                                x[:, :, :o] = known
                            final[:, :, a_:b_] = x; done = b_
                    latents = final
                    timesteps_done = True
                elif self.stride is None:   # plain (longer) windows: group by group with all steps inside, like upstream -> ONE DeepCache cache alive
                    with _sdpa_context(self.sdpa):
                        for gi, grp in enumerate(groups):
                            x = torch.cat([latents[:, :, a:b] for a, b in grp]); mk = torch.cat([mask_lat[:, :, a:b] for a, b in grp])
                            mi = torch.cat([masked_lat[:, :, a:b] for a, b in grp]); rl = torch.cat([ref_lat[:, :, a:b] for a, b in grp]); au = torch.cat([audio_all[a:b] for a, b in grp])
                            if do_cfg:
                                mk, mi, rl, au = torch.cat([mk] * 2), torch.cat([mi] * 2), torch.cat([rl] * 2), torch.cat([torch.zeros_like(au), au])
                            if dc is not None:
                                dc.cached_output = {}
                            for t in timesteps:
                                x2 = torch.cat([x] * 2) if do_cfg else x
                                npred = self.unet(torch.cat([pipe.scheduler.scale_model_input(x2, t), mk, mi, rl], dim=1), t, encoder_hidden_states=au).sample; unet_calls += 1
                                if do_cfg:
                                    nu, na = npred.chunk(2); npred = nu + self.guidance * (na - nu)
                                x = pipe.scheduler.step(npred, t, x, **extra).prev_sample
                            for k, (a, b) in enumerate(grp):
                                latents[:, :, a:b] = x[k:k + 1]
                    timesteps_done = True
                else:
                    if dc is not None:   # per-step fusion keeps every window group's DeepCache feature cache alive at once -> OOM on long segments
                        raise ValueError("overlap 'fuse' with a stride needs --no-deepcache (or use --overlap continue)")
                    timesteps_done = False
                with _sdpa_context(self.sdpa):
                    for t in ([] if timesteps_done else timesteps):
                        eps_sum = torch.zeros(latents.shape, device=device, dtype=torch.float32); w_sum = torch.zeros(n, device=device, dtype=torch.float32)
                        for gi, grp in enumerate(groups):
                            x = torch.cat([latents[:, :, a:b] for a, b in grp]); mk = torch.cat([mask_lat[:, :, a:b] for a, b in grp])
                            mi = torch.cat([masked_lat[:, :, a:b] for a, b in grp]); rl = torch.cat([ref_lat[:, :, a:b] for a, b in grp])
                            au = torch.cat([audio_all[a:b] for a, b in grp])
                            if do_cfg:
                                x2 = torch.cat([x] * 2); mk, mi, rl = torch.cat([mk] * 2), torch.cat([mi] * 2), torch.cat([rl] * 2); au = torch.cat([torch.zeros_like(au), au])
                            else:
                                x2 = x
                            inp = torch.cat([pipe.scheduler.scale_model_input(x2, t), mk, mi, rl], dim=1)
                            if dc is not None:
                                dc.cached_output = caches[gi]                                                    # per-group DeepCache state
                            npred = self.unet(inp, t, encoder_hidden_states=au).sample; unet_calls += 1
                            if do_cfg:
                                nu, na = npred.chunk(2); npred = nu + self.guidance * (na - nu)
                            for k, (a, b) in enumerate(grp):
                                w = tri(b - a) if self.stride is not None else torch.ones(b - a, device=device)
                                eps_sum[:, :, a:b] += npred[k:k + 1].float() * w.view(1, 1, -1, 1, 1); w_sum[a:b] += w
                        eps = (eps_sum / w_sum.view(1, 1, -1, 1, 1)).to(weight_dtype)
                        latents = pipe.scheduler.step(eps, t, latents, **extra).prev_sample
                torch.cuda.synchronize(); t_unet = time.perf_counter() - t0
                t_bridge = 0.0; bridges = 0
                if self.bridge_len and self.bridge_steps and self.stride is None:   # budget seam fix: low-noise bridge windows over every seam
                    t0 = time.perf_counter(); eps0 = eps_init
                    k0 = max(0, len(timesteps) - self.bridge_steps); half = self.bridge_len // 2
                    dc_start = getattr(dc, "start_timestep", None) if dc is not None else None
                    if dc is not None: dc.start_timestep = k0   # DeepCache counts its skip schedule from start_timestep: each bridge must open with a full step
                    with _sdpa_context(self.sdpa):
                        for seam in [w[0] for w in windows[1:]]:
                            a_, b_ = max(0, seam - half), min(n, seam + (self.bridge_len - half))
                            if b_ - a_ < 4: continue
                            x = pipe.scheduler.add_noise(latents[:, :, a_:b_], eps0[:, :, a_:b_], timesteps[k0].reshape(1))
                            mk, mi, rl, au = mask_lat[:, :, a_:b_], masked_lat[:, :, a_:b_], ref_lat[:, :, a_:b_], audio_all[a_:b_]
                            if do_cfg:
                                mk, mi, rl, au = torch.cat([mk] * 2), torch.cat([mi] * 2), torch.cat([rl] * 2), torch.cat([torch.zeros_like(au), au])
                            if dc is not None:
                                dc.cached_output = {}
                            for t in timesteps[k0:]:
                                x2 = torch.cat([x] * 2) if do_cfg else x
                                npred = self.unet(torch.cat([pipe.scheduler.scale_model_input(x2, t), mk, mi, rl], dim=1), t, encoder_hidden_states=au).sample; unet_calls += 1
                                if do_cfg:
                                    nu, na = npred.chunk(2); npred = nu + self.guidance * (na - nu)
                                x = pipe.scheduler.step(npred, t, x, **extra).prev_sample
                            L_ = b_ - a_; w = torch.tensor([min(i + 1, L_ - i) for i in range(L_)], device=device, dtype=torch.float32)
                            w = (w / w.max()).clamp(max=1.0).to(weight_dtype).view(1, 1, -1, 1, 1)
                            latents[:, :, a_:b_] = w * x + (1 - w) * latents[:, :, a_:b_]; bridges += 1
                    if dc is not None: dc.start_timestep = dc_start
                    torch.cuda.synchronize(); t_bridge = time.perf_counter() - t0
                if dc is not None:
                    dc.cached_output = {}
                t0 = time.perf_counter(); out = []
                for a in range(0, n, nf):
                    dec = pipe.decode_latents(latents[:, :, a:a + nf])
                    out.append(pipe.paste_surrounding_pixels_back(dec, cur_px[a:a + nf], 1 - masks[a:a + nf], device, weight_dtype))
                synced = torch.cat(out); torch.cuda.synchronize(); t_vae += time.perf_counter() - t0
            t0 = time.perf_counter(); frames = pipe.restore_video(synced, video_frames, boxes, affine); t_res = time.perf_counter() - t0
            # seams = first frame of NEW content after a previous window (segment-relative): window starts (none / fuse) or previous window ends (continue)
            if self.overlap == "continue" and self.stride is not None:
                seams = [windows[k - 1][1] for k in range(1, len(windows)) if windows[k - 1][1] < n]
            else:
                seams = [w[0] for w in windows[1:]] if windows else list(range(nf, n, nf))
            stats = {"frames": int(frames.shape[0]), "windows": len(windows), "seams": seams, "full_windows": len([w for w in windows if w[1] - w[0] == nf]),
                     "partial_windows": len([w for w in windows if w[1] - w[0] != nf]), "groups": unet_calls // max(len(timesteps), 1) if windows else 0,
                     "window_batch_size": self.B, "unet_calls": unet_calls, "stride": self.stride, "ref_mode": self.ref_mode, "mode": self.mode,
                     "overlap": self.overlap, "window_len": self.num_frames, "bridge": {"len": self.bridge_len, "steps": self.bridge_steps},
                     "detector": "replayed_verify_pass" if replay is not None else "latentsync_internal", "detector_fallback_calls": replay.fallbacks if replay else 0,
                     "seconds": {"preprocess_audio_video": round(t_pre, 3), "affine": round(t_aff, 3), "vae_and_cond": round(t_vae, 3), "unet_denoise": round(t_unet, 3),
                                 "restore": round(t_res, 3), "write": 0.0, "total": round(time.perf_counter() - t_all, 3)}}
            return frames, stats
        finally:
            os.chdir(old)


class StableRunner:
    """Same interface as face_aware_latentsync_accel.AccelRunner (run_frames / calls / config), executing StableLatentSync."""

    def __init__(self, window_batch_size=2, deepcache=True, steps=20, guidance=1.5, seed=1247, deepcache_interval=5, stride=None, ref_mode="same", mode="ls", pipe=None,
                 overlap="fuse", window_len=None, bridge_len=0, bridge_steps=0):
        from resident_vram_probe import load_latentsync
        t0 = time.perf_counter(); pipe = pipe or load_latentsync()
        self.acc = StableLatentSync(pipe, stride=stride, ref_mode=ref_mode, mode=mode, overlap=overlap, window_len=window_len, bridge_len=bridge_len, bridge_steps=bridge_steps,
                                    window_batch_size=window_batch_size,
                                    deepcache=deepcache, steps=steps, guidance=guidance, seed=seed, deepcache_interval=deepcache_interval)
        self.load_s = round(time.perf_counter() - t0, 2); self.steps, self.guidance, self.seed = steps, guidance, seed
        self.config = {"window_batch_size": window_batch_size, "deepcache": deepcache, "deepcache_interval": deepcache_interval if deepcache else None,
                       "steps": steps, "guidance": guidance, "seed": seed, "fp16": True, "stage2_512": True, "stride": stride, "ref_mode": ref_mode, "mode": mode,
                       "overlap": overlap, "window_len": window_len or 16, "bridge_len": bridge_len, "bridge_steps": bridge_steps}
        self.calls: list[dict] = []

    silence_mix = None   # round 3: r3_lipsync.SilenceMix (dub-silent mouth), hooked per segment

    def set_segment(self, start: int, n: int):
        self.acc.chunk_hook = self.silence_mix.hook(start, n) if self.silence_mix is not None else None

    def run_frames(self, frames_rgb, audio_wav, detections=None):
        t0 = time.perf_counter(); frames = self.acc.run_frames(frames_rgb, str(audio_wav), detections,
            video_fps=int(os.environ.get("DUB_LS_FPS", "25"))); dt = round(time.perf_counter() - t0, 2)   # r4 V: DUB_LS_FPS (a 30 fps run was windowed as 25 fps: SyncNet 0.6)
        st = self.acc.last_stats
        seams = st.get("seams", list(range(self.acc.num_frames, len(frames_rgb), self.acc.num_frames)))   # AccelLatentSync path: 16-frame window starts
        self.calls.append({"frames": st["frames"], "windows": st["windows"], "groups": st["groups"], "unet_calls": st["unet_calls"], "seconds": dt, "stages": st["seconds"], "seams": seams})
        return frames, dt
