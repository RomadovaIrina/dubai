#!/usr/bin/env python3
"""r9 speed L5 probe: LatentSync 1.6 UNet / VAE kernels at the generator's real shapes (window batch 2 x CFG = 4 x 16 frames, 512 px),
timed per variant, with the max |diff| of each variant vs the eager default (same weights, same inputs).
    python scripts/r9/ls_bench.py [--variants eager,cl,compile] [--iters 10]"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))


def timed(fn, iters):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): out = fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", default="eager,cl,compile"); ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--parts", default="vae_dec,vae_enc,unet"); ap.add_argument("--img", default=""); a = ap.parse_args()
    from resident_vram_probe import load_latentsync
    import os
    LS = HERE.parents[1] / "third_party" / "latentsync"; os.chdir(LS)
    pipe = load_latentsync(); dev = "cuda"; dt = torch.float16; g = torch.Generator(dev).manual_seed(0)
    px = (torch.rand(16, 3, 512, 512, device=dev, generator=g) * 2 - 1).to(dt)
    if a.img:   # real face frames -> real latents (the decode differences are meaningful only on real latents)
        import cv2, numpy as np
        cap = cv2.VideoCapture(a.img); fr = []
        for k in range(400):
            ok, f = cap.read()
            if not ok: break
            if k % 25 == 0: h_, w_ = f.shape[:2]; s_ = min(h_, w_); fr.append(cv2.resize(f[(h_ - s_) // 3:(h_ - s_) // 3 + s_, (w_ - s_) // 2:(w_ - s_) // 2 + s_], (512, 512))[..., ::-1].copy())
        px = torch.from_numpy(np.stack(fr[:16])).permute(0, 3, 1, 2).to(dev, dt) / 127.5 - 1
    with torch.no_grad():
        lat = ((pipe.vae.encode(px).latent_dist.mean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor).permute(1, 0, 2, 3).unsqueeze(0)
    uin = torch.randn(4, 13, 16, 64, 64, device=dev, dtype=dt, generator=g); aud = torch.randn(64, 50, 384, device=dev, dtype=dt, generator=g)
    t = pipe.scheduler.timesteps[0] if len(pipe.scheduler.timesteps) else torch.tensor(999, device=dev)
    res = {}; ref = {}
    vae, unet = pipe.vae, pipe.unet
    for v in a.variants.split(","):
        torch.backends.cudnn.benchmark = v != "eager"
        if v in ("cl", "compile"):
            vae.to(memory_format=torch.channels_last); unet.to(memory_format=torch.channels_last)
        dec = vae.decode; enc = vae.encode; uf = unet
        if v == "compile":
            dec = torch.compile(vae.decode, mode="max-autotune-no-cudagraphs"); enc = torch.compile(vae.encode, mode="max-autotune-no-cudagraphs")
            uf = torch.compile(unet, mode="max-autotune-no-cudagraphs")
        if v == "trt":   # NVIDIA TensorRT fp16 through torch_tensorrt (static shapes: one engine per shape)
            import torch_tensorrt  # noqa: F401
            opts = {"enabled_precisions": {torch.float16}, "truncate_long_and_double": True, "min_block_size": 1}
            uf = torch.compile(unet, backend="torch_tensorrt", dynamic=False, options=opts)
            dec = torch.compile(vae.decode, backend="torch_tensorrt", dynamic=False, options=opts); enc = vae.encode
        if v == "dec_rows":   # mouth-only decode: latent rows 15.. only (cost and max |diff| on the kept rows vs the full decode)
            dec = lambda z_: vae.decode(z_[:, :, 15:])
        r = {}
        with torch.no_grad():
            if "vae_dec" in a.parts:
                z = lat[0].permute(1, 0, 2, 3) / vae.config.scaling_factor + vae.config.shift_factor
                t0 = time.perf_counter(); s, o = timed(lambda: dec(z).sample, a.iters); r["vae_dec_16"] = round(s, 4); r["vae_dec_first_call"] = round(time.perf_counter() - t0, 1)
                if v == "eager": ref["dec"] = o.float()
                else:   # on the pixel rows the mask regenerates (190-485), in 0-255 units
                    rd = ref["dec"][:, :, 190:486]; od = o.float()[:, :, 190 - (120 if v == "dec_rows" else 0):486 - (120 if v == "dec_rows" else 0)]
                    r["vae_dec_maxdiff_255"] = round(float((od - rd).abs().max()) * 127.5, 2); r["vae_dec_meandiff_255"] = round(float((od - rd).abs().mean()) * 127.5, 3)
            if "vae_enc" in a.parts:
                s, o = timed(lambda: enc(px).latent_dist.mean, a.iters); r["vae_enc_16"] = round(s, 4)
                if v == "eager": ref["enc"] = o.float()
                else: r["vae_enc_maxdiff"] = round(float((o.float() - ref["enc"]).abs().max()), 4)
            if "unet" in a.parts:
                t0 = time.perf_counter(); s, o = timed(lambda: uf(uin, t, encoder_hidden_states=aud).sample, a.iters); r["unet_b4x16"] = round(s, 4); r["unet_first_call"] = round(time.perf_counter() - t0, 1)
                if v == "eager": ref["unet"] = o.float()
                else: r["unet_maxdiff"] = round(float((o.float() - ref["unet"]).abs().max()), 4); r["unet_meandiff"] = round(float((o.float() - ref["unet"]).abs().mean()), 5)
            if "shapes" in a.parts:   # per-frame UNet cost vs batch: (batch incl. CFG, frames): a bridge alone, bridges batched, B=2 / B=4 windows
                for bs, fr in ((2, 8), (4, 8), (8, 8), (4, 16)):   # 16 x 8 / 8 x 16 exceed a kernel launch limit in the motion-module attention
                    x = torch.randn(bs, 13, fr, 64, 64, device=dev, dtype=dt, generator=g); au = torch.randn(bs * fr, 50, 384, device=dev, dtype=dt, generator=g)
                    try: s, _ = timed(lambda: uf(x, t, encoder_hidden_states=au).sample, a.iters); r[f"unet_{bs}x{fr}_ms_per_frame"] = round(1000 * s / (bs * fr), 2)
                    except RuntimeError as e: r[f"unet_{bs}x{fr}_ms_per_frame"] = f"error: {str(e)[:60]}"; break
        res[v] = r; print(v, json.dumps(r), flush=True)
    print("== ls_bench", json.dumps(res))


if __name__ == "__main__":
    main()
