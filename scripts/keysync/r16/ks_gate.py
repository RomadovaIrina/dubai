"""r16 K2: KeySync fast with MORE interpolation steps only on wide-open segments (Irina 07.11: "b лучше на маме").
KS_GATE="thr=0.12,cap=0.25,hi=8,lo=2" (ks_fast.py installs it after the LCM sampler):
- the gate: the keyframes are generated (and decoded) before the interpolation; per keyframe, the share of dark pixels (luma < 0.18) inside the
  SOURCE mouth box (68-pt landmarks from the KS landmark cache, padded 0.15 width) = an open-mouth / black-cavity proxy (corr 0.83 with the
  generated inner-lip opening on m03). A keyframe is "wide" when the share > thr; per sample_interpolation call at most the darkest `cap` share.
  A segment (14 frames between keyframes j, j+1) is "hi" when either end is wide.
- the sampler: the hi-step AYS grid for everyone, per-row sigma; lo segments are evaluated only at the lo grid points (lo=2 -> s0, s4) and frozen
  in between, so their math equals the plain 2-step student; a 28-row UNet chunk with no active row is SKIPPED (that is the saving).
Prints KS_GATE lines (hi segment share, UNet chunk calls run / skipped)."""
import math, os
import numpy as np, torch
import ks_student

ST = {"lmk": None, "seg_off": 0, "active": None, "out_shape": None, "calls": 0, "skipped": 0, "hi": 0, "seg": 0}
cfg = dict(x.split("=") for x in os.environ.get("KS_GATE", "").split(",") if "=" in x)
THR, CAP, HI, LO = float(cfg.get("thr", 0.12)), float(cfg.get("cap", 0.25)), int(cfg.get("hi", 8)), int(cfg.get("lo", 2))


def gate(samples_x, nf=14):
    """samples_x: decoded keyframes (n_kf, 3, H, W) in [-1, 1]; returns hi flags per segment (n_kf - 1)."""
    L = ST["lmk"]; n = samples_x.shape[0]; dark = np.zeros(n)
    g = ((samples_x.float() + 1) / 2).clamp(0, 1); g = (0.299 * g[:, 0] + 0.587 * g[:, 1] + 0.114 * g[:, 2]).cpu().numpy()
    for j in range(n):
        fi = (ST["seg_off"] + j) * (nf - 1)
        if L is None: continue
        m = np.asarray(L[min(fi, len(L) - 1)])[48:68].astype(np.float32)
        if not np.isfinite(m).all(): continue
        x0, y0 = m.min(0); x1, y1 = m.max(0); w = x1 - x0; y0 -= 0.15 * w; y1 += 0.15 * w
        H, W = g.shape[1:]; sl = g[j, max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))]
        if sl.size: dark[j] = float((sl < 0.18).mean())
    wide = dark > THR
    k = max(1, int(math.ceil(CAP * n)))
    if wide.sum() > k: wide = dark >= np.sort(dark)[-k]
    hi = wide[:-1] | wide[1:]
    ST["seg_off"] += n - 1; ST["hi"] += int(hi.sum()); ST["seg"] += len(hi)
    return hi


class GatedLCM(ks_student.LCMSampler):
    def __init__(self, base):
        self.guider = base.guider; self.discretization = base.discretization; self.noise = "fresh"
        self.grid = [float(x) for x in self.discretization(8)[:9]]
        self.ih = [round(i * 8 / HI) for i in range(HI)]; self.il = [round(i * 8 / LO) for i in range(LO)]
        print("KS_GATE sampler: hi sigmas", [round(self.grid[i], 4) for i in self.ih], "lo", [round(self.grid[i], 4) for i in self.il], flush=True)

    @torch.no_grad()
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0):
        from einops import rearrange
        G = self.grid; m, g = cond["masks"], cond["gt"]
        m = rearrange(m, "b c t h w -> (b t) c h w") if m.dim() == 5 else m
        g = rearrange(g, "b c t h w -> (b t) c h w") if g.dim() == 5 else g
        R = x.shape[0]; hi = ST.pop("plan", None)
        rh = torch.zeros(R, dtype=torch.bool, device=x.device)
        if hi is not None and len(hi) and R % len(hi) == 0:
            rh = torch.as_tensor(np.repeat(hi, R // len(hi)), device=x.device)
        elif hi is not None: print(f"KS_GATE plan mismatch: {R} rows, {len(hi)} segments -> all lo", flush=True)
        n0 = x.clone() if os.environ.get("KS_NOISE") else None   # r16c K1: re-noise with the shared initial field
        steps = sorted(set(self.ih) | set(self.il)); x = x * math.sqrt(1.0 + G[0] ** 2); out = torch.zeros_like(x)
        for gi in steps:
            act = (rh & (gi in self.ih)) | (~rh & (gi in self.il))
            if not act.any(): continue
            ST["active"] = act; s = G[gi]
            cs, co = ks_student.cm_coef(s)
            f = cs * x + co * denoiser(x, x.new_ones([R]) * s, cond)
            f = f * m + g * (1 - m)
            for rows, grid in ((rh, self.ih), (~rh, self.il)):
                r = act & rows
                if not r.any(): continue
                k = grid.index(gi)
                if k == len(grid) - 1: out[r] = f[r]
                else: x[r] = f[r] + G[grid[k + 1]] * (n0[r] if n0 is not None else torch.randn_like(f[r]))
        ST["active"] = None
        return ks_student.ks_lsmooth(out) * m + g * (1 - m)


def install(d, model):
    import sgm.modules.diffusionmodules.denoiser as dn
    xl = d.extract_video_landmarks
    def lmk(*a, **k):
        r = xl(*a, **k); ST["lmk"] = r; ST["seg_off"] = 0; return r
    d.extract_video_landmarks = lmk
    si = d.sample_interpolation
    def sample_interpolation(model_, samples_z, samples_x, *a, **k):
        tr = k.get("to_remove") or []; n_seg = len(a[0])   # a[0] = this chunk's audio_interpolation_list = the segments
        kept = torch.stack([s for i, s in zip(tr, samples_x) if not i]) if len(tr) else samples_x   # the pipeline's own filter (zip truncates)
        hi = gate(kept); ST["plan"] = hi[:n_seg] if len(hi) >= n_seg else np.concatenate([hi, np.zeros(n_seg - len(hi), bool)])
        return si(model_, samples_z, samples_x, *a, **k)
    d.sample_interpolation = sample_interpolation
    cn = dn.chunk_network
    def chunk_network(network, input, c_in, c_noise, cond, ami, chunk_size, num_frames=1):
        act = ST["active"]
        if act is None or act.shape[0] != input.shape[0]: return cn(network, input, c_in, c_noise, cond, ami, chunk_size, num_frames=num_frames)
        outs = []
        for i in range(0, input.shape[0], chunk_size):
            n = min(chunk_size, input.shape[0] - i)
            if not act[i:i + n].any() and ST["out_shape"] is not None:
                outs.append(input.new_zeros((n,) + ST["out_shape"])); ST["skipped"] += 1; continue
            def sl(v):
                if not isinstance(v, torch.Tensor) or v.dim() == 0: return v
                if v.shape[0] == input.shape[0]: return v[i:i + n]
                if v.shape[0] == input.shape[0] // num_frames: return v[i // num_frames:(i + n) // num_frames]
                return v
            # one chunk through the original function (its own cond / extra-input slicing)
            o = cn(network, input[i:i + n], sl(c_in), sl(c_noise), {kk: sl(vv) for kk, vv in cond.items()}, ami, chunk_size, num_frames=num_frames)
            ST["out_shape"] = tuple(o.shape[1:]); ST["calls"] += 1; outs.append(o)
        return torch.cat(outs, 0)
    dn.chunk_network = chunk_network
    model.sampler = GatedLCM(model.sampler)
    import atexit
    atexit.register(lambda: print(f"KS_GATE hi segments {ST['hi']}/{ST['seg']} ({ST['hi'] / max(ST['seg'], 1) * 100:.0f} %), UNet chunks run {ST['calls']} skipped {ST['skipped']}", flush=True))
