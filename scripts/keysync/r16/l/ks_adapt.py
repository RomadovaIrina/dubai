#!/usr/bin/env python3
"""r16 L (B): per-person quick adaptation of the KeySync KEYFRAME UNet on the SOURCE video itself (self-reconstruction, source audio).
  1. dump: the authors' sample() on (source crop, source audio) with the keyframe sampler replaced by a dumper (c / uc of every keyframe call:
     gt = the real keyframe latents, masks, audio, reference), the interpolation sampler + VAE decodes stubbed (no generation);
  2. train: a small LoRA (rank --rank on every Linear / Conv of the keyframe UNet), EDM denoising loss inside the mask,
     sigma from the production AYS-8 grid (log-uniform between its ends), full condition, for --seconds of GPU;
  3. save lora + lora_meta.json (ks_fast_l.py merges it with KS_KF_LORA=<path>).
Timing (dump / train) is printed as ADAPT_TIME for the GPU-s/vmin budget.
    python ks_adapt.py --video SRC512.mp4 --audio SRC16k.wav --out DIR [--seconds 120 --rank 16 --lr 2e-4]"""
import argparse, functools, json, math, os, random, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "/workspace/dub/keysync")
os.environ.pop("KS_DC", None)
import torch
import torch.utils.checkpoint as tuc
from einops import rearrange
import scripts.sampling.dubbing_pipeline_raw as d
from ks_lcm import add_lora, lora_state


class Dumper:
    def __init__(self, guider): self.guider = guider; self.calls = []

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0):
        g = lambda dd: {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in dd.items()}
        self.calls.append({"c": g(cond), "uc": g(uc), "n": int(x.shape[0])}); return torch.zeros_like(x)


class Zero:
    def __init__(self, s): self.guider = s.guider

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0): return torch.zeros_like(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="_s8"); ap.add_argument("--seconds", type=float, default=120); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-at", default="", help="also save at these train seconds, e.g. 60,120")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"; random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    model, _, n_batch = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", dev, 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    mk, _, n_kf = d.load_model(f"scripts/sampling/configs/keyframe{a.cfg}.yaml", dev, 14, "latents", "pretrained_models/checkpoints/keyframe_dub.pt")
    hubert = d.HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda()
    wavlm = d.WavLM_wrapper(model_size="Base+", feed_as_frames=False, merge_type="None", model_path="pretrained_models/checkpoints/WavLM-Base+.pt").cuda()
    vae = d.VaeWrapper("video"); lmk = d.LandmarksExtractor()
    torch.cuda.synchronize(); t_load = time.perf_counter() - t0
    dump = Dumper(mk.sampler.guider); guider = mk.sampler.guider; disc = mk.sampler.discretization
    mk.sampler = dump; model.sampler = Zero(model.sampler)
    d.save_audio_video = lambda *x, **k: None
    model.decode_first_stage = lambda z: torch.zeros(z.shape[0], 3, 8, 8, device=z.device)
    mk.decode_first_stage = lambda z: torch.zeros(z.shape[0], 3, 512, 512, device=z.device)
    t1 = time.perf_counter()
    d.sample(model, mk, video_path=a.video, audio_path=a.audio, num_frames=14, resize_size=512, video_folder="videos", latent_folder="videos",
             landmark_folder="landmarks", audio_folder="audios", audio_emb_folder="audios", fps_id=24, cond_aug=0.0, decoding_t=1,
             output_folder=os.path.join(a.out, "_null"), force_uc_zero_embeddings=["cond_frames", "audio_emb"], chunk_size=2, add_zero_flag=True,
             n_batch=n_batch, n_batch_keyframes=n_kf, extra_audio=None, compute_until="end", audio_emb_type="hubert", extra_naming="adapt",
             what_mask="box", fix_occlusion=False, position=None, start_frame=0, recompute=True,
             hubert_model=hubert, wavlm_model=wavlm, vae_model=vae, landmarks_model=lmk)
    torch.cuda.synchronize(); t_dump = time.perf_counter() - t1
    cl = lambda dd: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in dd.items()}   # out of inference mode (autograd-safe)
    C = [{"c": cl(x["c"]), "uc": cl(x["uc"]), "n": x["n"]} for x in dump.calls]; print(f"DUMP keyframe calls {len(C)} frames {sum(x['n'] for x in C)} keys",
                          {k: tuple(v.shape) for k, v in C[0]["c"].items() if isinstance(v, torch.Tensor)}, flush=True)
    del hubert, wavlm, model; torch.cuda.empty_cache()
    for m in list(sys.modules.values()):
        if getattr(m, "__name__", "").startswith("sgm") and getattr(m, "checkpoint", None) is tuc.checkpoint:
            m.checkpoint = functools.partial(tuc.checkpoint, use_reentrant=False)
    dm = mk.model.diffusion_model
    for p in mk.parameters(): p.requires_grad_(False)
    names = add_lora(dm, a.rank, a.alpha); dm.to(dev)
    params = [p for n, p in dm.named_parameters() if ".down." in n or ".up." in n]
    for p in params: p.requires_grad_(True)
    json.dump({"rank": a.rank, "alpha": a.alpha, "modules": names, "cfg": a.cfg, "model": "keyframe"}, open(f"{a.out}/lora_meta.json", "w"))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sig = disc(8, device=dev).float(); smax, smin = float(sig[0]), float(sig[7])
    print(f"LORA kf modules {len(names)} params {sum(p.numel() for p in params) / 1e6:.1f}M sigma {smin:.3f}..{smax:.2f}", flush=True)

    def D(x, s, c, n):
        return mk.denoiser(mk.model, x, s, c, image_only_indicator=torch.zeros(1, n, device=dev), num_video_frames=n)

    saves = sorted(float(x) for x in a.save_at.split(",") if x); torch.cuda.synchronize(); t2 = time.perf_counter(); step = 0; hist = []
    while time.perf_counter() - t2 < a.seconds:
        call = random.choice(C); c = call["c"]; n = call["n"]
        m = rearrange(c["masks"], "b c t h w -> (b t) c h w") if c["masks"].dim() == 5 else c["masks"]
        x0 = (rearrange(c["gt"], "b c t h w -> (b t) c h w") if c["gt"].dim() == 5 else c["gt"]).float()
        s = math.exp(random.uniform(math.log(smin), math.log(smax)))
        xn = x0 + s * torch.randn_like(x0)
        with torch.autocast(dev, dtype=torch.bfloat16):
            dx = D(xn, xn.new_ones([n]) * s, c, n).float()
        wgt = (s ** 2 + 1) / s ** 2 if s > 0.5 else (0.25 + 1) / 0.25   # EDM weighting (sigma_data 1), capped for small sigma
        loss = (((dx - x0) ** 2) * m).sum() / (m.sum() * 4).clamp(min=1) * min(wgt, 5.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g_ in opt.param_groups: g_["lr"] = a.lr * min(1.0, (step + 1) / 10)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1; hist.append(float(loss))
        el = time.perf_counter() - t2
        if step % 20 == 0: print(f"STEP {step} loss {sum(hist[-20:]) / 20:.5f} {el:.0f}s", flush=True)
        if saves and el >= saves[0]:
            torch.save({k: v.half() for k, v in lora_state(dm).items()}, f"{a.out}/kf_lora_{int(saves[0])}s.pt"); saves.pop(0)
    torch.cuda.synchronize(); t_train = time.perf_counter() - t2
    torch.save({k: v.half() for k, v in lora_state(dm).items()}, f"{a.out}/kf_lora.pt")
    info = {"load_s": round(t_load, 1), "dump_s": round(t_dump, 1), "train_s": round(t_train, 1), "steps": step, "kf_frames": sum(x["n"] for x in C)}
    json.dump(info, open(f"{a.out}/adapt_time.json", "w")); print("ADAPT_TIME", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
