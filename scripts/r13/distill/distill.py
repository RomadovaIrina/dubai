#!/usr/bin/env python3
"""r13 distillation PILOT ("does it make sense?"): LatentSync 1.6, teacher 20-step DDIM + CFG 2.0 (no DeepCache) -> a LoRA student that
samples in 4 steps, one pass per step, no CFG. Guided x0 distillation (the LCM / guided-distillation idea, simplest form):
  teacher  per captured window: fixed noise z -> 20-step DDIM, CFG 2.0 (uncond = zero audio, as production / LipsyncPipeline) -> x0_T (cached)
  student  the same UNet + LoRA (rank R) on every attention projection (to_q / to_k / to_v / to_out.0, spatial + temporal); at one of its 4
           timesteps t_k (DDIM set_timesteps(4)): input x = z at k = 0 (what 4-step inference starts from), else add_noise(x0_T, z, t_k);
           one conditional pass -> eps -> x0_S; loss = MSE(x0_S, x0_T) in latent space (+ the SyncNet loss when the stable-syncnet weight exists)
  val      held-out windows: the FULL 4-step student sampler vs x0_T (latent MSE), and the same for the untrained weights at 4 steps (the floor
           the student must beat) and for the teacher-at-10-steps-CFG (production-like) -> how far 4-step is from the teacher.
    python scripts/r13/distill/distill.py teacher --cap DIR --out DIR [--max N]
    python scripts/r13/distill/distill.py train   --out DIR --steps 1500 [--rank 32 --lr 1e-4]
Output: DIR/lora.pt (the LoRA weights; scripts/r13/distill/student_gen.py merges them at load), DIR/train_log.jsonl, DIR/val.json."""
import argparse, glob, json, math, os, pathlib, random, sys, time
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[2]; LS = ROOT / "third_party" / "latentsync"
sys.path.insert(0, str(LS)); sys.path.insert(0, str(ROOT / "scripts" / "pilot"))
import torch, torch.nn as nn, torch.nn.functional as F


def load_unet():
    from omegaconf import OmegaConf
    from latentsync.models.unet import UNet3DConditionModel
    cfg = OmegaConf.load(LS / "configs/unet/stage2_512.yaml")
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model), str(LS / "checkpoints/latentsync_unet.pt"), device="cpu")
    return unet.to("cuda", dtype=torch.float16).eval()


def sched(n):
    from diffusers import DDIMScheduler
    s = DDIMScheduler.from_pretrained(str(LS / "configs")); s.set_timesteps(n, device="cuda"); return s


def load_vae():
    from diffusers import AutoencoderKL
    v = AutoencoderKL.from_pretrained(str(ROOT / "models" / "sd-vae-ft-mse"), torch_dtype=torch.float16).to("cuda").eval(); v.requires_grad_(False); return v


def load_syncnet():
    """LatentSync's training SyncNet (stable_syncnet.pt, configs/syncnet/syncnet_16_pixel_attn.yaml): 16 frames, 256 px, lower half"""
    from omegaconf import OmegaConf
    from latentsync.models.stable_syncnet import StableSyncNet
    c = OmegaConf.load(LS / "configs/syncnet/syncnet_16_pixel_attn.yaml")
    m = StableSyncNet(OmegaConf.to_container(c.model), gradient_checkpointing=True).to("cuda", dtype=torch.float16)
    m.load_state_dict(torch.load(ROOT / "models" / "latentsync" / "stable_syncnet.pt", map_location="cuda", weights_only=True)["state_dict"]); m.requires_grad_(False); m.eval()
    return m, c


def decode(vae, x0, grad):
    """(1, 4, F, 64, 64) latents -> (F, 3, 512, 512) pixels in [-1, 1]; 4-frame chunks, checkpointed when grad (memory)"""
    from torch.utils.checkpoint import checkpoint
    lat = x0[0].permute(1, 0, 2, 3) / 0.18215; outs = []
    for i in range(0, lat.shape[0], 4):
        ch = lat[i:i + 4].half()
        outs.append(checkpoint(lambda z: vae.decode(z).sample, ch, use_reentrant=False) if grad else vae.decode(ch).sample)
    return torch.cat(outs)


def sync_sim(syncnet, sc, vae, x0, mel, grad=False):
    """the stable SyncNet's cosine similarity of the decoded x0 window with its mel (LatentSync stage-2 input: 256 px, lower half)"""
    px = decode(vae, x0, grad); px = F.interpolate(px.float(), size=(sc.data.resolution, sc.data.resolution), mode="bilinear", align_corners=False)
    if sc.data.lower_half: px = px[:, :, px.shape[2] // 2:, :]
    v = px.reshape(1, -1, px.shape[2], px.shape[3]).half()          # (1, F*3, 128, 256) = "b (f c) h w"
    m_ = mel.cuda().half(); m_ = m_[None] if m_.dim() == 3 else m_   # (1, 80, 52) -> (1, 1, 80, 52)
    ve, ae = syncnet(v, m_)
    return F.cosine_similarity(ve.float(), ae.float())


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__(); self.base = base; self.r = r; self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.A), self.B) * self.scale).to(x.dtype)


def add_lora(unet, r, alpha):
    from latentsync.models.attention import Attention
    names = []
    for mname, m in list(unet.named_modules()):
        if isinstance(m, Attention):
            for p in ("to_q", "to_k", "to_v"):
                setattr(m, p, LoRA(getattr(m, p), r, alpha)); names.append(f"{mname}.{p}")
            m.to_out[0] = LoRA(m.to_out[0], r, alpha); names.append(f"{mname}.to_out.0")
    return names


def lora_state(unet):
    return {n: p.detach().cpu() for n, p in unet.named_parameters() if n.endswith(".A") or n.endswith(".B")}


def flat_audio(au):
    return au.reshape(-1, *au.shape[-2:]) if au.dim() == 4 else au


def sample(unet, s, z, cond, au, cfg=1.0):
    """DDIM sampling from z (production order: [uncond, cond] rows with CFG)"""
    au = flat_audio(au)   # LatentSync UNet takes (B*F, 50, 384), frames of each window contiguous (lipsync_pipeline.py); a 4-D (B, F, 50, 384) input
                          # is silently mis-read (the r13 pilot did that: its teacher targets were noise in the mouth)
    x = z * s.init_noise_sigma
    for t in s.timesteps:
        if cfg > 1.0:
            x2 = torch.cat([x] * 2); c2 = torch.cat([cond] * 2); a2 = torch.cat([torch.zeros_like(au), au])
            e = unet(torch.cat([s.scale_model_input(x2, t), c2], 1), t, encoder_hidden_states=a2).sample; eu, ec = e.chunk(2); e = eu + cfg * (ec - eu)
        else:
            e = unet(torch.cat([s.scale_model_input(x, t), cond], 1), t, encoder_hidden_states=au).sample
        x = s.step(e, t, x).prev_sample
    return x


def load_windows(cap, mx):
    fs = sorted(glob.glob(str(pathlib.Path(cap) / "*.pt"))); random.Random(0).shuffle(fs); return fs[:mx] if mx else fs


@torch.no_grad()
def cmd_teacher(a):
    out = pathlib.Path(a.out); (out / "t").mkdir(parents=True, exist_ok=True); unet = load_unet(); s20 = sched(20); t0 = time.time()
    fs = load_windows(a.cap, a.max)
    for i, f in enumerate(fs):
        o = out / "t" / pathlib.Path(f).name
        if o.exists(): continue
        d = torch.load(f); cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]
        g = torch.Generator(device="cuda").manual_seed(1000 + i); z = torch.randn((1, 4) + cond.shape[2:], generator=g, device="cuda", dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            x0 = sample(unet, s20, z, cond, au, cfg=2.0)
        torch.save({"cond": d["cond"], "audio": d["audio"], "mel": d.get("mel"), "z": z[0].cpu(), "x0_teacher": x0[0].to(torch.float16).cpu(), "src": f, "job": d.get("job")}, o)
        if i % 25 == 0: print(f"[teacher] {i + 1}/{len(fs)} {time.time() - t0:.0f} s", flush=True)
    print(f"[teacher] done {len(fs)} windows in {time.time() - t0:.0f} s; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def cmd_train(a):
    out = pathlib.Path(a.out); fs = sorted(glob.glob(str(out / "t" / "*.pt"))); random.Random(1).shuffle(fs)
    full = [f for f in fs if (lambda d: d.get("mel") is not None and d["cond"].shape[1] == 16)(torch.load(f))]   # val: 16-frame windows with a mel (SyncNet-measurable)
    nv = min(a.val_max, max(4, len(fs) // 10)); val = full[:nv]; tr = [f for f in fs if f not in set(val)]; print(f"[train] {len(tr)} train / {len(val)} val windows", flush=True)
    unet = load_unet(); unet.requires_grad_(False)
    if hasattr(unet, "enable_gradient_checkpointing"): unet.enable_gradient_checkpointing()
    names = add_lora(unet, a.rank, a.rank); params = [p for n, p in unet.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    for p in params: p.requires_grad_(True)
    print(f"[train] LoRA on {len(names)} projections, {sum(p.numel() for p in params) / 1e6:.2f} M trainable params", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0); s4 = sched(4); T4 = list(s4.timesteps); ac = s4.alphas_cumprod.to("cuda")
    vae = load_vae(); syncnet, sc = load_syncnet() if a.sync_w > 0 else (None, None); print(f"[train] sync loss weight {a.sync_w}", flush=True)
    cache = {f: torch.load(f) for f in fs}; log = open(out / "train_log.jsonl", "a"); t0 = time.time()

    @torch.no_grad()
    def evaluate(tag, lora_on=True):
        for m in unet.modules():
            if isinstance(m, LoRA): m.scale = (m.r if lora_on else 0) / m.r   # alpha = rank -> scale 1 (on) / 0 (off)
        unet.eval(); errs, sims, tsims = [], [], []
        for f in val:
            d = cache[f]; cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]; z = d["z"].cuda()[None]; x0t = d["x0_teacher"].cuda()[None].float()
            with torch.autocast("cuda", dtype=torch.float16):
                x = sample(unet, s4, z, cond, au, cfg=1.0)
            errs.append(F.mse_loss(x.float(), x0t).item())
            if syncnet is not None and d.get("mel") is not None and x.shape[2] == 16:
                sims.append(sync_sim(syncnet, sc, vae, x, d["mel"]).item())
                if tag == "base4": tsims.append(sync_sim(syncnet, sc, vae, x0t.half(), d["mel"]).item())
        for m in unet.modules():
            if isinstance(m, LoRA): m.scale = 1.0
        unet.train(); r = {"mse": float(sum(errs) / len(errs)), "sync": float(sum(sims) / len(sims)) if sims else None}
        if tsims: r["sync_teacher"] = float(sum(tsims) / len(tsims))
        return r

    b = evaluate("base4", lora_on=False); base4 = b["mse"]
    r = {"step": 0, "val_mse_student4": base4, "val_mse_base4_untrained": base4, "val_sync_student4": b["sync"], "val_sync_base4_untrained": b["sync"], "val_sync_teacher20_cfg2": b.get("sync_teacher")}
    # how far production-like sampling (10 steps, CFG 2) is from the 20-step teacher: the noise floor a 'good' student would sit at
    with torch.no_grad():
        s10 = sched(10); e10 = []
        for m in unet.modules():
            if isinstance(m, LoRA): m.scale = 0.0
        for f in val:
            d = cache[f]; cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]; z = d["z"].cuda()[None]
            with torch.autocast("cuda", dtype=torch.float16):
                e10.append(F.mse_loss(sample(unet, s10, z, cond, au, cfg=2.0).float(), d["x0_teacher"].cuda()[None].float()).item())
        for m in unet.modules():
            if isinstance(m, LoRA): m.scale = 1.0
    r["val_mse_teacher10_cfg2"] = float(sum(e10) / len(e10)); print(f"[val] {r}", flush=True); log.write(json.dumps(r) + "\n"); vals = [r]
    unet.train(); run = 0.0
    for step in range(1, a.steps + 1):
        d = cache[random.choice(tr)]; cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]; z = d["z"].cuda()[None].float(); x0t = d["x0_teacher"].cuda()[None].float()
        k = random.randrange(4); t = T4[k]
        x = z if k == 0 else s4.add_noise(x0t, z, t.reshape(1))
        with torch.autocast("cuda", dtype=torch.float16):
            eps = unet(torch.cat([s4.scale_model_input(x.half(), t), cond], 1), t, encoder_hidden_states=au).sample.float()
        at = ac[t]; x0s = (x - (1 - at).sqrt() * eps) / at.sqrt()
        lmse = F.mse_loss(x0s, x0t); lsync = torch.zeros((), device="cuda")
        if syncnet is not None and d.get("mel") is not None and x0t.shape[2] == 16 and step % a.sync_every == 0:   # LatentSync stage 2: cosine (log) loss of the stable SyncNet on the decoded x0
            sim = sync_sim(syncnet, sc, vae, x0s, d["mel"], grad=True); lsync = F.binary_cross_entropy(sim.clamp(1e-6, 1 - 1e-6), torch.ones_like(sim))
        loss = lmse + a.sync_w * lsync
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 25 == 0:
            log.write(json.dumps({"step": step, "loss": loss.item(), "mse": lmse.item(), "sync_loss": lsync.item(), "loss_ema": run, "k": k, "s": round(time.time() - t0, 1), "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}) + "\n"); log.flush()
        if step % 100 == 0: print(f"[train] step {step} loss_ema {run:.4f} {time.time() - t0:.0f} s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            v = evaluate("student"); r = {"step": step, "val_mse_student4": v["mse"], "val_sync_student4": v["sync"]}; vals.append(r); print(f"[val] {r}", flush=True); log.write(json.dumps(r) + "\n")
            torch.save({"lora": lora_state(unet), "rank": a.rank, "names": names, "step": step}, out / "lora.pt")
    (out / "val.json").write_text(json.dumps({"base4_untrained": base4, "teacher10_cfg2": vals[0]["val_mse_teacher10_cfg2"], "sync_teacher20": vals[0].get("val_sync_teacher20_cfg2"), "curve": vals}, indent=1))
    print(f"[train] done {a.steps} steps in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["teacher", "train"]); ap.add_argument("--cap"); ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--sync-w", type=float, default=0.05); ap.add_argument("--sync-every", type=int, default=1); ap.add_argument("--val-max", type=int, default=24); a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    {"teacher": cmd_teacher, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
