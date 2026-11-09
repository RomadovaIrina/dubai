"""r15 K: the LCM-LoRA student of the KeySync interpolation UNet at inference (used by ks_fast.py --interp-lora).
merge_lora: W += alpha / r * up @ down for every LoRA'd Linear / Conv2d (ks_lcm.py's lora_meta.json next to the checkpoint) = zero runtime cost.
LCMSampler: ONE row (the full condition, no CFG), N steps on the production AYS-8 grid (4 -> s0 s2 s4 s6; 2 -> s0 s4), the consistency function
f = c_skip x + c_out D (the training parameterisation), outside the mask = the source latents (as the production denoiser), re-noise fresh / ddim."""
import json, math, os
import torch
from einops import rearrange

SMIN = 0.002


def cm_coef(s):
    u = (s - SMIN) * 1000.0
    return 1.0 / (u * u + 1.0), u / (u * u + 1.0) ** 0.5


@torch.no_grad()
def merge_lora(dm, path):
    meta = json.load(open(os.path.join(os.path.dirname(path), "lora_meta.json"))); sd = torch.load(path, map_location="cpu")
    sc = meta["alpha"] / meta["rank"]; n = 0
    for name in meta["modules"]:
        mod = dm.get_submodule(name); dn, up = sd[f"{name}.down.weight"].float(), sd[f"{name}.up.weight"].float()
        if mod.weight.dim() == 2:
            delta = up @ dn
        else:   # conv: down (r, in, kh, kw), up (out, r, 1, 1)
            delta = torch.einsum("or,rikl->oikl", up[:, :, 0, 0], dn)
        mod.weight.data += (sc * delta).to(mod.weight.device, mod.weight.dtype); n += 1
    print(f"LORA merged {n} modules from {path}", flush=True)



def ks_lsmooth(f):   # r16c K1: temporal smoothing of the latent's spatial HF (texture boil), LF (shape) kept
    w = float(os.environ.get("KS_LSMOOTH", "0") or 0)
    T = int(os.environ.get("KS_LSMOOTH_T", "14")); kk = int(os.environ.get("KS_LSMOOTH_K", "3")); npass = int(os.environ.get("KS_LSMOOTH_N", "1"))
    if w <= 0 or f.shape[0] % T: return f
    lf = torch.nn.functional.avg_pool2d(f, kk, stride=1, padding=kk // 2, count_include_pad=False); hf = f - lf
    B = f.shape[0] // T; h = hf.view(B, T, *hf.shape[1:])
    for _ in range(npass):
        hp = torch.cat([h[:, :1], h, h[:, -1:]], 1); h = w * hp[:, :-2] + (1 - 2 * w) * hp[:, 1:-1] + w * hp[:, 2:]
    return lf + h.reshape(hf.shape)


class LCMSampler:
    def __init__(self, base_sampler, steps=4, noise="fresh"):
        self.guider = base_sampler.guider; self.discretization = base_sampler.discretization; self.noise = noise
        self.grid = [float(x) for x in self.discretization(8)[:9]]; self.idx = [round(i * 8 / steps) for i in range(steps)]
        print("LCM sigmas", [round(self.grid[i], 4) for i in self.idx], "noise", noise, flush=True)

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0):
        G = self.grid; m, g = cond["masks"], cond["gt"]
        m = rearrange(m, "b c t h w -> (b t) c h w") if m.dim() == 5 else m
        g = rearrange(g, "b c t h w -> (b t) c h w") if g.dim() == 5 else g
        n0 = x.clone() if os.environ.get("KS_NOISE") else None   # r16c K1: re-noise with the shared initial field
        x = x * math.sqrt(1.0 + G[0] ** 2)
        for k, gi in enumerate(self.idx):
            s = G[gi]; cs, co = cm_coef(s)
            f = cs * x + co * denoiser(x, x.new_ones([x.shape[0]]) * s, cond)
            f = f * m + g * (1 - m)
            if k == len(self.idx) - 1: return ks_lsmooth(f) * m + g * (1 - m)
            n = n0 if n0 is not None else (torch.randn_like(x) if self.noise == "fresh" else (x - f) / s)
            x = f + G[self.idx[k + 1]] * n
