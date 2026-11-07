#!/usr/bin/env python3
"""r16 L (A): continue the r15 LCM-LoRA student of the KeySync INTERPOLATION UNet (ks_lcm.py) with a TEMPORAL JITTER loss on the
lower face OUTSIDE the lips (cheek / jaw bands), against the source's own rate. Resumable short chunks (the GPU is shared).
  CD loss   = ks_lcm.py unchanged (teacher = prod interp UNet, 3-way CFG, one Euler step on AYS-8; target = student stop-grad).
  jitter    = on the student's x0 prediction fs (14 frames): per band cell (latent avg-pooled 4x4), d1 = x_t - x_t-1, d2 = x_t+1 - 2x_t + x_t-1;
              J = mean_band relu(|dk fs| - |dk x0| - tau)^2 (k = 1, 2), x0 = the real latents (the source's own rate).
              bands = inside the generated mask box: the left / right 28 % columns (cheeks) + the bottom 22 % rows (jaw / chin);
              the lip zone (middle columns above the jaw band) is free, so the mouth (sync) is not damped.
  audio swap= with prob --swap the segment's audio conditions are replaced by another segment's (dub-like: mouth != source mouth),
              the teacher sees the same swapped audio, the jitter reference stays the source's own cheek rate.
    python ks_lcm_jit.py --dump DIR --out DIR --init lora_pick.pt [--resume] --minutes 12 --lam 1 --swap 0.5"""
import argparse, functools, glob, json, math, os, random, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/workspace/dub/keysync")
os.environ.pop("KS_DC", None)
import torch, torch.nn.functional as F
import torch.utils.checkpoint as tuc
import scripts.sampling.dubbing_pipeline_raw as d
from ks_lcm import LORA_ON, add_lora, lora_state, cm_coef, seg, load_segments


def band_weights(m):   # m: (14, 1|4, h, w) generated mask -> (1, 1, h, w) band weights (cheeks + jaw), lip zone 0
    mm = (m[:, :1] > 0.5).float().amax(0, keepdim=True)   # union over the segment
    ys, xs = torch.nonzero(mm[0, 0], as_tuple=True)
    B = torch.zeros_like(mm)
    if len(ys) < 16: return B
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1; h, w = r1 - r0, c1 - c0
    cw, jh = max(1, round(0.28 * w)), max(1, round(0.22 * h))
    B[..., r0:r1, c0:c0 + cw] = 1; B[..., r0:r1, c1 - cw:c1] = 1; B[..., r1 - jh:r1, c0:c1] = 1
    return B * mm


def jitter(fs, x0, B, tau, pool=4, top=0.0):
    """top > 0: CVaR = the mean of the top fraction of the (frame, channel, cell) excesses -> aims at the spikes, not the average."""
    P = lambda z: F.avg_pool2d(z * B, pool) / F.avg_pool2d(B, pool).clamp(min=1e-3)
    wb = (F.avg_pool2d(B, pool) > 0.5).float()   # cells mostly inside the band
    pf, pg = P(fs), P(x0); out = {}
    for k, op in ((1, lambda z: z[1:] - z[:-1]), (2, lambda z: z[2:] - 2 * z[1:-1] + z[:-2])):
        e = F.relu(op(pf).abs() - op(pg).abs() - tau) ** 2
        if top > 0:
            v = e[(wb > 0).expand_as(e)]
            out[k] = v.topk(max(1, int(top * v.numel()))).values.mean() if v.numel() else e.sum() * 0
        else:
            out[k] = (e * wb).sum() / (wb.sum() * e.shape[0] * e.shape[1]).clamp(min=1)
    return out[1] + out[2], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True); ap.add_argument("--val-tag", default="")
    ap.add_argument("--init", default="/workspace/dub/r16in/lora/lora_pick.pt"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cfg", default="_s8"); ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--rank", type=int, default=64); ap.add_argument("--alpha", type=float, default=64); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--huber", type=float, default=0.01); ap.add_argument("--ckpt-every", type=int, default=0, help="0 = only at the chunk end")
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--tau", type=float, default=0.0); ap.add_argument("--swap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--swap-keys", default="crossattn", help="the audio conditions (KeySync: crossattn = per-frame audio)")
    ap.add_argument("--spike-top", type=float, default=0.0, help="jitter = mean of the top fraction of excesses (CVaR), e.g. 0.05; 0 = plain mean")
    ap.add_argument("--pair", type=float, default=0.0, help="prob. of training 2 adjacent segments jointly (jitter across the shared keyframe)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    st_path = f"{a.out}/state.pt"; st = torch.load(st_path, map_location="cpu", weights_only=False) if a.resume and os.path.exists(st_path) else None
    step0 = st["step"] if st else 0; random.seed(a.seed + step0); torch.manual_seed(a.seed + step0)
    model, _, n_batch = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", dev, 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    guider = model.sampler.guider; sig = model.sampler.discretization(8, device=dev).float(); G = [float(x) for x in sig[:9]]
    for m in list(sys.modules.values()):
        if getattr(m, "__name__", "").startswith("sgm") and getattr(m, "checkpoint", None) is tuc.checkpoint:
            m.checkpoint = functools.partial(tuc.checkpoint, use_reentrant=False)
    dm = model.model.diffusion_model
    for p in model.parameters(): p.requires_grad_(False)
    names = add_lora(dm, a.rank, a.alpha); dm.to(dev)
    sd = st["lora"] if st else torch.load(a.init, map_location="cpu")
    miss = dm.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    print(f"LORA loaded {'state' if st else a.init} step0 {step0} unexpected {len(miss.unexpected_keys)}", flush=True)
    params = [p for n, p in dm.named_parameters() if ".down." in n or ".up." in n]
    for p in params: p.requires_grad_(True)
    json.dump({"rank": a.rank, "alpha": a.alpha, "modules": names, "grid": G, "cfg": a.cfg}, open(f"{a.out}/lora_meta.json", "w"))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    if st and "opt" in st: opt.load_state_dict(st["opt"])
    files = sorted(glob.glob(f"{a.dump}/*.pt")); files = [f for f in files if os.path.basename(f) != f"{a.val_tag}.pt"]
    train = load_segments(files); print(f"SEGMENTS train {len(train)} from {len(files)} clips", flush=True)
    akeys = None

    def D(x, s, c):
        return model.denoiser(model.model, x, s, c, num_overlap_frames=1, num_frames=14, n_skips=n_batch, chunk_size=x.shape[0],
                              image_only_indicator=torch.zeros(n_batch, 14, device=dev), num_video_frames=14)

    def teacher_euler(x, sn, sn1, c, uc):
        LORA_ON[0] = False; sv = x.new_ones([x.shape[0]]) * sn
        dc = guider(D(*guider.prepare_inputs(x, sv, c, uc)), sv); LORA_ON[0] = True
        return x + (sn1 - sn) * (x - dc) / sn

    def f_student(x, s, c):
        cs, co = cm_coef(s); return cs * x + co * D(x, x.new_ones([x.shape[0]]) * s, c)

    def masks_of(c):
        from einops import rearrange
        m = rearrange(c["masks"], "b c t h w -> (b t) c h w") if c["masks"].dim() == 5 else c["masks"]
        g = rearrange(c["gt"], "b c t h w -> (b t) c h w") if c["gt"].dim() == 5 else c["gt"]
        return m, g

    t_start = time.perf_counter(); hist = []; step = step0; tl = time.perf_counter(); nsw = 0; npair = 0
    while step < a.steps and (time.perf_counter() - t_start) / 60 < a.minutes:
        tag, ch, i, n = random.choice(train)
        pair = a.pair > 0 and random.random() < a.pair
        if pair:   # two adjacent segments: seg i frame 13 == seg i+1 frame 0 (the shared keyframe) -> the jitter spans the segment boundary
            while i + 1 >= n: tag, ch, i, n = random.choice(train)
            npair += 1
        swap = None
        if not pair and a.swap > 0 and random.random() < a.swap:
            t2, ch2, i2, n2 = random.choice(train); swap = (seg(ch2["c"], i2, n2, dev), seg(ch2["uc"], i2, n2, dev)); nsw += 1
        k = random.randrange(0, 8); sn, sn1 = G[k], G[k + 1]; lcd = 0; FS, X0, BB = [], [], []
        for ii in ([i, i + 1] if pair else [i]):
            c, uc = seg(ch["c"], ii, n, dev), seg(ch["uc"], ii, n, dev); m, x0 = masks_of(c); w = m.clone(); w[0] = 0; w[13] = 0
            if akeys is None:
                akeys = [kk for kk in a.swap_keys.split(",") if kk in c]
                print("COND keys", {kk: tuple(v.shape) for kk, v in c.items() if isinstance(v, torch.Tensor)}, "swap keys", akeys,
                      "uc zero", {kk: bool((uc[kk] == 0).all()) for kk in akeys if kk in uc}, flush=True)
            if swap:
                c2, uc2 = swap
                for kk in akeys:
                    if kk in c2 and c2[kk].shape == c[kk].shape: c[kk] = c2[kk]
                    if kk in uc and kk in uc2 and uc2[kk].shape == uc[kk].shape: uc[kk] = uc2[kk]
            xn = x0 + sn * torch.randn_like(x0)
            with torch.no_grad():
                with torch.autocast(dev, dtype=torch.float16):
                    xh = teacher_euler(xn, sn, sn1, c, uc).float()
                with torch.autocast(dev, dtype=torch.bfloat16):
                    ft = xh if k + 1 == 8 else f_student(xh, sn1, c).float()
            with torch.autocast(dev, dtype=torch.bfloat16):
                fs = f_student(xn, sn, c).float()
            diff = (fs - ft) * w
            lcd = lcd + ((diff ** 2 + a.huber ** 2).sqrt() - a.huber).sum() / (w.sum() * 4).clamp(min=1)
            fsm = fs * m + x0 * (1 - m)   # as the sampler: outside the mask = the source latents
            fsm = torch.cat([x0[:1], fsm[1:13], x0[13:]])   # the keyframe ends = the keyframes (generated by the keyframe model at inference)
            FS.append(fsm); X0.append(x0); BB.append(band_weights(m))
        lcd = lcd / len(FS)
        if pair: FSc, X0c, B = torch.cat([FS[0], FS[1][1:]]), torch.cat([X0[0], X0[1][1:]]), torch.maximum(BB[0], BB[1])
        else: FSc, X0c, B = FS[0], X0[0], BB[0]
        lj, parts = jitter(FSc, X0c, B, a.tau, top=a.spike_top)
        loss = lcd + a.lam * lj
        loss.backward(); hist.append((k, float(lcd), float(lj), float(parts[1]), float(parts[2])))
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g_ in opt.param_groups: g_["lr"] = a.lr * min(1.0, (step + 1) / 50)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 25 == 0:
            h = hist[-25:]; mean = lambda j: sum(x[j] for x in h) / len(h)
            print(f"STEP {step} cd {mean(1):.5f} jit {mean(2):.6f} (d1 {mean(3):.6f} d2 {mean(4):.6f}) gnorm {float(gn):.3f} "
                  f"{(time.perf_counter() - tl) / 25:.2f}s/it swaps {nsw} pairs {npair} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True); tl = time.perf_counter()
        if a.ckpt_every and step % a.ckpt_every == 0:
            torch.save({k_: v.half() for k_, v in lora_state(dm).items()}, f"{a.out}/lora_step_{step:05d}.pt")
    L = lora_state(dm)
    torch.save({"step": step, "lora": L, "opt": opt.state_dict()}, st_path + ".tmp"); os.replace(st_path + ".tmp", st_path)
    torch.save({k_: v.half() for k_, v in L.items()}, f"{a.out}/lora_step_{step:05d}.pt")
    json.dump({"step": step, "hist_last": hist[-100:]}, open(f"{a.out}/hist_{step:05d}.json", "w"))
    print(f"CHUNK_DONE step {step} minutes {(time.perf_counter() - t_start) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
