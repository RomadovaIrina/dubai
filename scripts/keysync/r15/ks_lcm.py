#!/usr/bin/env python3
"""r15 K: guided consistency distillation (LCM-LoRA, EDM form) of the KeySync INTERPOLATION UNet.
  teacher  = the production interpolation UNet (LoRA off) with the 3-way AudioRef CFG (audio 5 / ref 2), ONE Euler step on the
             production AYS-8 grid sigma_n -> sigma_n+1 (= exactly the production sampler's step, without DeepCache);
  student  = the same UNet + LoRA, ONE row (full condition): f(x, s) = c_skip(s) x + c_out(s) D(x, s) (LCM boundary scaling x10, f = x at s_min);
  target   = the student itself (stop-grad, LCM-LoRA: no EMA) at the teacher's point: f(x_hat, sigma_n+1);
  loss     = pseudo-Huber inside the generated mask, keyframe ends (frames 0 / 13) excluded;
  data     = ks_dump.py conditions of the non-test clips: x0 = the real latents of each 14-frame segment, the clip's own audio.
Validation: per checkpoint, the student's N-step LCM sample vs the production 8-step CFG sample from the SAME noise (latent PSNR in the mask).
    python ks_lcm.py --dump DIR --val-tag TAG --out DIR [--steps 3000 --minutes 120 --rank 64 --lr 1e-4]"""
import argparse, functools, glob, json, math, os, random, sys, time
sys.path.insert(0, os.getcwd())
os.environ.pop("KS_DC", None)
import torch, torch.nn as nn
import torch.utils.checkpoint as tuc
from einops import rearrange
import scripts.sampling.dubbing_pipeline_raw as d

LORA_ON = [True]
SMIN = 0.002


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base = base; self.scale = alpha / r
        self.down = nn.Linear(base.in_features, r, bias=False); self.up = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5)); nn.init.zeros_(self.up.weight)

    def forward(self, x):
        y = self.base(x)
        return y + self.up(self.down(x)) * self.scale if LORA_ON[0] else y


class LoRAConv(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base = base; self.scale = alpha / r
        self.down = nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride, base.padding, bias=False); self.up = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5)); nn.init.zeros_(self.up.weight)

    def forward(self, x):
        y = self.base(x)
        return y + self.up(self.down(x)) * self.scale if LORA_ON[0] else y


def add_lora(dm, r, alpha):
    names = []
    for name, mod in list(dm.named_modules()):
        for cn, ch in list(mod.named_children()):
            full = f"{name}.{cn}" if name else cn
            if isinstance(ch, nn.Linear) and min(ch.in_features, ch.out_features) >= 4:
                setattr(mod, cn, LoRALinear(ch, r, alpha)); names.append(full)
            elif type(ch) is nn.Conv2d and min(ch.in_channels, ch.out_channels) >= 4 and ch.groups == 1:
                setattr(mod, cn, LoRAConv(ch, r, alpha)); names.append(full)
    return names


def lora_state(dm):
    return {n: p.detach().cpu() for n, p in dm.named_parameters() if ".down." in n or ".up." in n}


def cm_coef(s):   # LCM boundary condition, steep (x1000): f = x at s_min, f ~ D (the x0 prediction) for s >= 0.01 -- as LCM (its x10 scaling acts on 0..999 timesteps)
    u = (s - SMIN) * 1000.0
    return 1.0 / (u * u + 1.0), u / (u * u + 1.0) ** 0.5


def seg(dd, i, n, dev):
    out = {}
    for k, v in dd.items():
        if isinstance(v, torch.Tensor):
            v = v[14 * i:14 * i + 14] if v.shape[0] == 14 * n else v[i:i + 1] if v.shape[0] == n else v
            out[k] = v.to(dev, torch.float32) if v.is_floating_point() else v.to(dev)
        else:
            out[k] = v
    return out


def load_segments(files):
    S = []
    for f in files:
        D = torch.load(f, map_location="cpu", weights_only=False)
        for ch in D["chunks"]:
            n = ch["n_frames"] // 14
            for i in range(n): S.append((D["tag"], ch, i, n))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True); ap.add_argument("--val-tag", default="")
    ap.add_argument("--cfg", default="_s8"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--rank", type=int, default=64); ap.add_argument("--alpha", type=float, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=1); ap.add_argument("--huber", type=float, default=0.01); ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--val-n", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lcm-steps", default="4,2")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"; random.seed(a.seed); torch.manual_seed(a.seed)
    model, _, n_batch = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", dev, 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    guider = model.sampler.guider; sig = model.sampler.discretization(8, device=dev).float()   # [s0 .. s8, 0], the production grid
    G = [float(x) for x in sig[:9]]; print("GRID", [round(x, 4) for x in G], flush=True)
    for m in list(sys.modules.values()):   # gradient checkpointing without the reentrant variant (the frozen base + LoRA need use_reentrant=False)
        if getattr(m, "__name__", "").startswith("sgm") and getattr(m, "checkpoint", None) is tuc.checkpoint:
            m.checkpoint = functools.partial(tuc.checkpoint, use_reentrant=False)
    dm = model.model.diffusion_model
    for p in model.parameters(): p.requires_grad_(False)
    names = add_lora(dm, a.rank, a.alpha); dm.to(dev)
    params = [p for n, p in dm.named_parameters() if ".down." in n or ".up." in n]
    for p in params: p.requires_grad_(True)
    print(f"LORA modules {len(names)} params {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    json.dump({"rank": a.rank, "alpha": a.alpha, "modules": names, "grid": G, "cfg": a.cfg}, open(f"{a.out}/lora_meta.json", "w"))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    files = sorted(glob.glob(f"{a.dump}/*.pt")); vfiles = [f for f in files if os.path.basename(f) == f"{a.val_tag}.pt"]
    train = load_segments([f for f in files if f not in vfiles]); val = load_segments(vfiles)
    val = [val[j] for j in torch.linspace(0, len(val) - 1, min(a.val_n, len(val))).long().tolist()] if val else []
    print(f"SEGMENTS train {len(train)} from {len(files) - len(vfiles)} clips, val {len(val)} ({a.val_tag})", flush=True)

    def D(x, s, c):
        return model.denoiser(model.model, x, s, c, num_overlap_frames=1, num_frames=14, n_skips=n_batch, chunk_size=x.shape[0],
                              image_only_indicator=torch.zeros(n_batch, 14, device=dev), num_video_frames=14)

    def teacher_euler(x, sn, sn1, c, uc):
        LORA_ON[0] = False
        sv = x.new_ones([x.shape[0]]) * sn
        dc = guider(D(*guider.prepare_inputs(x, sv, c, uc)), sv)
        LORA_ON[0] = True
        return x + (sn1 - sn) * (x - dc) / sn

    def f_student(x, s, c):
        cs, co = cm_coef(s)
        return cs * x + co * D(x, x.new_ones([x.shape[0]]) * s, c)

    def masks_of(c):
        m = rearrange(c["masks"], "b c t h w -> (b t) c h w") if c["masks"].dim() == 5 else c["masks"]
        g = rearrange(c["gt"], "b c t h w -> (b t) c h w") if c["gt"].dim() == 5 else c["gt"]
        return m, g

    @torch.no_grad()
    def validate(step, ref=None):
        res = {"step": step}; out_ref = ref if ref is not None else []
        for steps in [int(x) for x in a.lcm_steps.split(",")]:
            idx = [round(i * 8 / steps) for i in range(steps)]   # 4 -> s0 s2 s4 s6, 2 -> s0 s4
            ps, pg = [], []
            for j, (tag, ch, i, n) in enumerate(val):
                c, uc = seg(ch["c"], i, n, dev), seg(ch["uc"], i, n, dev); m, g = masks_of(c); w = m.clone(); w[0] = 0; w[13] = 0
                gen = torch.Generator(device=dev).manual_seed(1000 + j); z = torch.randn(g.shape, device=dev, generator=gen)
                with torch.autocast(dev, dtype=torch.float16):
                    if ref is None and steps == int(a.lcm_steps.split(",")[0]):   # the production 8-step CFG Euler sample (no DeepCache) from z
                        LORA_ON[0] = False; x = z * math.sqrt(1 + float(sig[0]) ** 2)
                        for k in range(len(sig) - 1):
                            sv = x.new_ones([x.shape[0]]) * sig[k]; dc = guider(D(*guider.prepare_inputs(x, sv, c, uc)), sv)
                            x = x + (sig[k + 1] - sig[k]) * (x - dc) / sig[k]
                        LORA_ON[0] = True; out_ref.append(x.float())
                    x = z * math.sqrt(1 + G[0] ** 2)
                    for k, gi in enumerate(idx):
                        fx = f_student(x, G[gi], c).float(); fx = fx * m + g * (1 - m)
                        if k == len(idx) - 1: break
                        x = fx + G[idx[k + 1]] * torch.randn(g.shape, device=dev, generator=gen)
                t = out_ref[j]; mse = lambda u, v: float(((u - v) ** 2 * w).sum() / (w.sum() * 4).clamp(min=1))
                ps.append(10 * math.log10(4.0 / max(mse(fx, t), 1e-10))); pg.append(10 * math.log10(4.0 / max(mse(fx, g), 1e-10)))
            res[f"psnr_vs_teacher_{steps}"] = round(sum(ps) / max(len(ps), 1), 2); res[f"psnr_vs_gt_{steps}"] = round(sum(pg) / max(len(pg), 1), 2)
        if ref is None and out_ref:
            tg = []
            for j, (tag, ch, i, n) in enumerate(val):
                c = seg(ch["c"], i, n, dev); m, g = masks_of(c); w = m.clone(); w[0] = 0; w[13] = 0
                tg.append(10 * math.log10(4.0 / max(float(((out_ref[j] - g) ** 2 * w).sum() / (w.sum() * 4).clamp(min=1)), 1e-10)))
            res["teacher_psnr_vs_gt"] = round(sum(tg) / len(tg), 2)
        print("VAL " + json.dumps(res), flush=True)
        return out_ref

    ref = validate(0) if val else None
    t_start = time.perf_counter(); hist = []; step = 0; tl = time.perf_counter()
    while step < a.steps and (time.perf_counter() - t_start) / 60 < a.minutes:
        for _ in range(a.accum):
            tag, ch, i, n = random.choice(train)
            c, uc = seg(ch["c"], i, n, dev), seg(ch["uc"], i, n, dev); m, x0 = masks_of(c); w = m.clone(); w[0] = 0; w[13] = 0
            k = random.randrange(0, 8); sn, sn1 = G[k], G[k + 1]
            xn = x0 + sn * torch.randn_like(x0)
            with torch.no_grad():
                with torch.autocast(dev, dtype=torch.float16):
                    xh = teacher_euler(xn, sn, sn1, c, uc).float()
                with torch.autocast(dev, dtype=torch.bfloat16):
                    ft = xh if k + 1 == 8 else f_student(xh, sn1, c).float()
            with torch.autocast(dev, dtype=torch.bfloat16):
                fs = f_student(xn, sn, c).float()
            diff = (fs - ft) * w
            loss = ((diff ** 2 + a.huber ** 2).sqrt() - a.huber).sum() / (w.sum() * 4).clamp(min=1)
            (loss / a.accum).backward(); hist.append((k, float(loss)))
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g_ in opt.param_groups: g_["lr"] = a.lr * min(1.0, (step + 1) / 50)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 25 == 0:
            h = hist[-25 * a.accum:]; per = {}
            for k_, l_ in h: per.setdefault(k_, []).append(l_)
            print(f"STEP {step} loss {sum(l for _, l in h) / len(h):.5f} gnorm {float(gn):.3f} {(time.perf_counter() - tl) / 25:.2f}s/it "
                  f"per_k {{{', '.join(f'{k_}: {sum(v) / len(v):.4f}' for k_, v in sorted(per.items()))}}} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True); tl = time.perf_counter()
        if step % a.ckpt_every == 0:
            torch.save(lora_state(dm), f"{a.out}/lora_step_{step:05d}.pt")
            if val: validate(step, ref)
    torch.save(lora_state(dm), f"{a.out}/lora_last.pt"); print(f"TRAIN_DONE steps {step} minutes {(time.perf_counter() - t_start) / 60:.1f}", flush=True)
    if val and step % a.ckpt_every: validate(step, ref)


if __name__ == "__main__":
    main()
