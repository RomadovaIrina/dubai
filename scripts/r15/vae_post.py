#!/usr/bin/env python3
"""r15 audit: how large is the VAE posterior std that LatentSync SAMPLES for its masked / reference latents? (LS crops of the q2d dump)"""
import glob, pathlib, sys
import cv2, numpy as np, torch
R = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R / "scripts/pilot"))
from resident_vram_probe import MODELS
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(str(MODELS / "sd-vae-ft-mse"), torch_dtype=torch.float16).cuda().eval()
for v in sys.argv[1:]:
    cap = cv2.VideoCapture(v); fr = []
    while len(fr) < 16:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    x = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).float().div(127.5).sub(1).half().cuda()
    with torch.no_grad():
        d = vae.encode(x).latent_dist; mu, sd = d.mean.float() * 0.18215, d.std.float() * 0.18215
        dec_mu = vae.decode(d.mean).sample.float(); dec_s = vae.decode(d.sample(generator=torch.Generator("cuda").manual_seed(1247))).sample.float()
    px = (dec_s - dec_mu).abs().mul(127.5)
    print(f"{pathlib.Path(v).name} {tuple(x.shape)}: latent |mean| {mu.abs().mean():.4f}  posterior std mean {sd.mean():.5f} max {sd.max():.5f}"
          f"  std/|mean| {(sd.mean() / mu.abs().mean()):.4f}; decoded sample vs mean: mean {px.mean():.3f} / p99 {torch.quantile(px.flatten()[::97], 0.99):.3f} (0-255 levels)")
