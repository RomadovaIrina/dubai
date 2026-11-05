#!/usr/bin/env python3
"""r14: a look at the student mouths (small, GPU): val windows -> rows of mouth halves
real | teacher | <ckpt>... (frames 0, 5, 10), saved as one jpg.
    python scripts/r14/viz_ckpts.py --out DIR(train dir with t/) --ckpts a.pt b.pt --jpg out.jpg [--n 3]"""
import argparse, glob, pathlib, random, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "r13" / "distill"))
import numpy as np, torch
import distill as D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--ckpts", nargs="+", required=True); ap.add_argument("--jpg", required=True)
    ap.add_argument("--n", type=int, default=3); a = ap.parse_args(); torch.set_num_threads(24)
    fs = sorted(glob.glob(str(pathlib.Path(a.out) / "t" / "*.pt"))); random.Random(1).shuffle(fs)
    pick = [f for f in fs if (lambda d: d["cond"].shape[1] == 16 and d.get("mel") is not None and not d.get("flip"))(torch.load(f))][:a.n]
    unet = D.load_unet(); D.add_lora(unet, 32, 32); vae = D.load_vae(); sS = D.sched(4); P = dict(unet.named_parameters())
    dec = lambda x: ((vae.decode(x[0].permute(1, 0, 2, 3)[[0, 5, 10]].cuda().half() / 0.18215).sample.clamp(-1, 1) + 1) * 127.5)
    rows = []
    with torch.no_grad():
        for f in pick:
            d = torch.load(f); cols = [dec(d["cond"][None, 5:9]), dec(d["x0_teacher"][None])]
            for ck in a.ckpts:
                sd = torch.load(ck, map_location="cpu")["lora"]
                for n, v in sd.items(): P[n].copy_(v.to(P[n].dtype))
                with torch.autocast("cuda", dtype=torch.float16): x = D.sample(unet, sS, d["z"][None].cuda(), d["cond"][None].cuda(), d["audio"][None].cuda(), cfg=1.0)
                cols.append(dec(x))
            # each column: 3 frames' mouth halves stacked vertically
            rows.append(np.concatenate([np.concatenate([c[i, :, 256:].permute(1, 2, 0).float().cpu().numpy() for i in range(3)], 0) for c in cols], 1)); print(f, flush=True)
    from PIL import Image
    im = Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)); im.resize((im.width // 3, im.height // 3)).save(a.jpg, quality=85)


if __name__ == "__main__":
    main()
