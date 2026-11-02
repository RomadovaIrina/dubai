#!/usr/bin/env python3
"""r11 FACE: PER-SPEAKER LoRA fine-tune of LatentSync 1.6 on the input video itself (self-reconstruction with the SOURCE audio), so the
generated lips / teeth / colour become this person's. Then generate with the dub audio as usual (LS_LORA=<out> merges it at load).

Data = the production pieces, so training sees what inference sees: the 25 fps master frames of the run's work dir, LatentSync's own affine
512^2 face crops (detections replayed from the master MediaPipe landmarks), the fixed mask, the SOURCE audio's whisper chunks.
Objective = LatentSync's latent noise-prediction loss (mixed noise, epsilon), weighted x --mouth-w on the regenerated region; the UNet reference
comes from ANOTHER window (as in LatentSync training). The first --train-frac of the video trains, the rest validates (loss at fixed t).
    python scripts/r11/ft_speaker.py --work <video stage work dir> --out lora.pt [--steps 300 --rank 16 --lr 1e-4]
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys, time
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
for p in ("optim", "pilot", "quality", "r11"): sys.path.insert(0, str(HERE.parent / p))
LS_DIR = HERE.parents[1] / "third_party" / "latentsync"; sys.path.insert(0, str(LS_DIR))   # the latentsync package (and its relative config paths: chdir below)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the generator run's work dir (video/master_25fps.mp4, video/master_mediapipe.npz, video/own_audio_16k.wav)")
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--train-frac", type=float, default=0.8); ap.add_argument("--mouth-w", type=float, default=4.0)
    ap.add_argument("--val-every", type=int, default=100); ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--targets", choices=["all", "self"], default="all", help="all = attentions + motion modules; self = spatial self-attention only "
                    "(appearance; the audio cross-attention and the motion stay the base model's, so the dub articulation is untouched)")
    ap.add_argument("--open-bias", type=float, default=0.0, help="sample training windows with p ~ (the source mouth opening of the window)^k: her "
                    "high-articulation frames, not the closed-mouth majority (which teaches 'a small mouth'); 0 = uniform")
    ap.add_argument("--distill", type=float, default=0.0, help="weight of the shape-keeping term: on the same window with ANOTHER window's audio, the LoRA "
                    "UNet's low-passed prediction must match the frozen base UNet's (the base audio -> mouth-shape mapping is kept, the adapter is free only in "
                    "texture / colour); 0 = off")
    ap.add_argument("--distill-pool", type=int, default=4, help="low-pass for --distill: average-pool the latent prediction by this factor (64 -> 16)")
    a = ap.parse_args(); t_all = time.perf_counter(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    import av, clean_face_stack as cfs
    cfs.guard_nc()   # as production: LatentSync s InsightFace detector replaced by a stub (detections are replayed)
    from resident_vram_probe import load_latentsync
    from latentsync_accel import AccelLatentSync, _ReplayDetector
    import ls_lora
    work = pathlib.Path(a.work); vdir = work / "video"
    c = av.open(str(vdir / "master_25fps.mp4")); frames = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]); c.close()
    n, H, W = frames.shape[:3]
    mp = np.load(cfs.run_mediapipe(vdir / "master_25fps.mp4", vdir / "master_mediapipe.npz"))["lm"][:n]
    dets = cfs.ls_detections(mp, W, H, calib=True)
    import os; os.chdir(LS_DIR)   # as AccelLatentSync._core: LatentSync resolves configs / mask relative to its repo
    pipe = load_latentsync(); acc = AccelLatentSync(pipe, deepcache=False); dev = pipe._execution_device if hasattr(pipe, "_execution_device") else torch.device("cuda")
    dt = torch.float16; ip = acc.image_processor()
    # ---- data (no grad): whisper chunks of the SOURCE audio, aligned face crops, latents
    t0 = time.perf_counter()
    feat = pipe.audio_encoder.audio2feat(str(vdir / "own_audio_16k.wav")); chunks = pipe.audio_encoder.feature2chunks(feature_array=feat, fps=25)
    m = min(n, len(chunks)); frames = frames[:m]; chunks = chunks[:m]; dets = dets[:m]
    real = ip.face_detector; ip.face_detector = _ReplayDetector(real, dets); ip.restorer.p_bias = None
    try:
        _, faces, _, _ = pipe.loop_video(chunks, frames)
    finally:
        ip.face_detector = real
    with torch.no_grad():
        px, mpx, masks = ip.prepare_masks_and_masked_images(faces, affine_transform=False)
        lat, mlat, mk = [], [], []
        for s in range(0, m, 16):
            mk_, mi_ = pipe.prepare_mask_latents(masks[s:s + 16], mpx[s:s + 16], 512, 512, dt, dev, None, False)
            lat.append(pipe.prepare_image_latents(px[s:s + 16], dev, dt, None, False)); mlat.append(mi_); mk.append(mk_)
        lat, mlat, mk = torch.cat(lat, 2), torch.cat(mlat, 2), torch.cat(mk, 2)          # (1, 4 | 4 | 1, m, 64, 64)
    audio = torch.stack([torch.as_tensor(c_) for c_ in chunks]).to(dev, dt)             # (m, 50, 384)
    gen_w = (1 + (a.mouth_w - 1) * (1 - mk.float().clamp(0, 1))).to(dev)                 # mask latent 0 = regenerated region -> weight mouth_w
    t_data = time.perf_counter() - t0
    # ---- LoRA
    unet = pipe.unet; unet.requires_grad_(False)
    if hasattr(unet, "enable_gradient_checkpointing"): unet.enable_gradient_checkpointing()
    params = ls_lora.add_lora(unet, a.rank, float(a.rank), a.targets); unet.train()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = pipe.scheduler; T = int(sched.config.num_train_timesteps); ntr = int(m * a.train_frac); F = 16
    op = np.linalg.norm(mp[:m, 13, :2] - mp[:m, 14, :2], axis=1) / np.maximum(np.linalg.norm(mp[:m, 33, :2] - mp[:m, 263, :2], axis=1), 1e-6)
    op = np.nan_to_num(op); wop = np.array([op[s:s + F].mean() for s in range(max(1, ntr - F))]) + 1e-3
    p_win = (wop ** a.open_bias) / (wop ** a.open_bias).sum() if a.open_bias > 0 else None
    def batch(lo, hi, s=None, t=None):
        if s is None: s = int(rng.choice(len(p_win), p=p_win)) if (p_win is not None and lo == 0) else int(rng.integers(lo, hi - F))
        cand = [r for r in range(lo, hi - F) if abs(r - s) >= F] or [lo]; r = int(rng.choice(cand))   # the reference from ANOTHER window
        x0 = lat[:, :, s:s + F].float(); shared = torch.randn(1, 4, 1, 64, 64, device=dev); noise = (shared + torch.randn_like(x0)) / math.sqrt(2.0)   # mixed noise (alpha 1)
        tt = torch.tensor([int(rng.integers(0, T)) if t is None else t], device=dev)
        xt = sched.add_noise(x0, noise, tt)
        inp = torch.cat([xt.to(dt), mk[:, :, s:s + F], mlat[:, :, s:s + F], lat[:, :, r:r + F]], 1)
        return inp, tt, audio[s:s + F], noise, gen_w[:, :, s:s + F]
    def loss_of(inp, tt, au, noise, w):
        with torch.autocast("cuda", dtype=dt):
            pred = unet(inp, tt, encoder_hidden_states=au).sample
        return ((pred.float() - noise) ** 2 * w).mean()
    @torch.no_grad()
    def val():
        if m - ntr < F + 2: return None
        unet.eval(); ls = []
        for s in np.linspace(ntr, m - F - 1, 6).astype(int):
            for t in (100, 400, 700):
                ls.append(float(loss_of(*batch(ntr, m, int(s), t))))
        unet.train(); return round(float(np.mean(ls)), 5)
    hist = {"val_before": val(), "train": [], "val": []}; t0 = time.perf_counter()
    def distill_of(inp, tt):
        """shape-keeping term: the same noisy window with a FOREIGN audio window; LoRA vs frozen base on the regenerated region, low-passed"""
        q = int(rng.integers(0, m - F)); au2 = audio[q:q + F]
        lp = lambda z: torch.nn.functional.avg_pool3d(z.float(), (1, a.distill_pool, a.distill_pool))
        wlp = lp(1 - inp[:, 4:5].float().clamp(0, 1))                                    # the mask channel: regenerated region only
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            ls_lora.set_enabled(unet, False)
            try: ref = unet(inp, tt, encoder_hidden_states=au2).sample
            finally: ls_lora.set_enabled(unet, True)
        with torch.autocast("cuda", dtype=dt):
            pred = unet(inp, tt, encoder_hidden_states=au2).sample
        return ((lp(pred) - lp(ref)) ** 2 * wlp).sum() / (wlp.sum() * pred.shape[1] + 1e-6)
    for step in range(1, a.steps + 1):
        b_ = batch(0, ntr); loss = loss_of(*b_)
        if a.distill > 0:
            d = distill_of(b_[0], b_[1]); hist.setdefault("distill", []).append(round(float(d), 6)); loss = loss + a.distill * d
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        hist["train"].append(round(float(loss), 5))
        if step % a.val_every == 0 or step == a.steps:
            v = val(); hist["val"].append((step, v)); print(f"  step {step} train {np.mean(hist['train'][-a.val_every:]):.4f} val {v}" + (f" distill {np.mean(hist['distill'][-a.val_every:]):.5f}" if "distill" in hist else ""), flush=True)
    t_train = time.perf_counter() - t0
    k = ls_lora.save_lora(unet, a.out)
    dur_min = m / 25.0 / 60.0
    rep = {"frames": m, "train_frames": ntr, "adapters": k, "steps": a.steps, "rank": a.rank, "lr": a.lr, "targets": a.targets, "open_bias": a.open_bias, "distill": a.distill, "val_before": hist["val_before"], "val": hist["val"],
           "seconds": {"data": round(t_data, 1), "train": round(t_train, 1), "total": round(time.perf_counter() - t_all, 1)},
           "train_gpu_s_per_video_min": round(t_train / dur_min, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    pathlib.Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=1)); print("== ft_speaker", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
