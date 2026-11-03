#!/usr/bin/env python3
"""r13 distillation, student ROUND 2 (after the pilot, DISTILL_PILOT.md): the same teacher / LoRA student as distill.py, plus
  data      every captured window of the 11 non-test clips (old03 / m03 / m18 never captured) + its HORIZONTAL FLIP as a separate window
            (the teacher is run on the flipped conditioning, so the target is the teacher's own answer, not a flipped guess)
  teacher   batched: up to --tbatch windows of the same length per UNet call (CFG rows of all of them together), same math as distill.py
  train     cosine lr (warmup 100) + EMA of the LoRA weights (validation and the saved weights use the EMA);
            loss = x0 MSE + sync_w x stable-SyncNet + pix_w x L1 + lpips_w x LPIPS(squeeze) between the decoded student and teacher x0 on the
            LOWER HALF of the face crop (LatentSync's mask = the mouth half): against the pilot's "dull mouth" (small openings, flat teeth)
  student   --student-steps 4 or 6
    python scripts/r13/distill/distill2.py teacher --cap DIR --out DIR [--flip] [--max N] [--tbatch 4]
    python scripts/r13/distill/distill2.py train --out DIR --steps 3000 [--student-steps 4] [--sync-w 0.15 --pix-w 0.5 --lpips-w 0.2]"""
import argparse, glob, json, math, pathlib, random, sys, time
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import torch, torch.nn.functional as F
import distill as D   # load_unet, sched, LoRA, add_lora, lora_state, sample, load_vae, load_syncnet, decode, sync_sim


@torch.no_grad()
def cmd_teacher(a):
    out = pathlib.Path(a.out); (out / "t").mkdir(parents=True, exist_ok=True); unet = D.load_unet(); s20 = D.sched(20); t0 = time.time()
    fs = sorted(glob.glob(str(pathlib.Path(a.cap) / "*.pt"))); random.Random(0).shuffle(fs); fs = fs[:a.max] if a.max else fs
    items = [(f, False) for f in fs] + ([(f, True) for f in fs] if a.flip else [])
    todo = [(i, f, fl) for i, (f, fl) in enumerate(items) if not (out / "t" / (pathlib.Path(f).stem + ("_f" if fl else "") + ".pt")).exists()]
    by_len = {}
    for it in todo: by_len.setdefault(torch.load(it[1])["cond"].shape[1], []).append(it)
    done = 0
    for L, its in by_len.items():
        for k in range(0, len(its), a.tbatch):
            grp = its[k:k + a.tbatch]; ds = [torch.load(f) for _, f, _ in grp]
            cond = torch.stack([(d["cond"].flip(-1) if fl else d["cond"]) for d, (_, _, fl) in zip(ds, grp)]).cuda()
            au = torch.stack([d["audio"] for d in ds]).cuda()
            z = torch.stack([torch.randn((4,) + cond.shape[2:], generator=torch.Generator(device="cuda").manual_seed(1000 + i), device="cuda", dtype=torch.float16) for i, _, _ in grp])
            with torch.autocast("cuda", dtype=torch.float16):
                x0 = D.sample(unet, s20, z, cond, au, cfg=2.0)
            for j, (d, (i, f, fl)) in enumerate(zip(ds, grp)):
                torch.save({"cond": cond[j].cpu(), "audio": d["audio"], "mel": d.get("mel"), "z": z[j].cpu(), "x0_teacher": x0[j].to(torch.float16).cpu(), "src": f, "flip": fl, "job": d.get("job")},
                           out / "t" / (pathlib.Path(f).stem + ("_f" if fl else "") + ".pt"))
            done += len(grp)
            if done % 40 < a.tbatch: print(f"[teacher] {done}/{len(todo)} {time.time() - t0:.0f} s ({(time.time() - t0) / done:.2f} s / window)", flush=True)
    print(f"[teacher] done {done} windows in {time.time() - t0:.0f} s; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def cmd_train(a):
    import lpips
    out = pathlib.Path(a.out); fs = sorted(glob.glob(str(out / "t" / "*.pt"))); random.Random(1).shuffle(fs)
    cache = {f: torch.load(f) for f in fs}
    # val: whole SOURCE windows (both orientations of a window go to the same side), 16 frames with a mel
    srcs = sorted({cache[f]["src"] for f in fs}); random.Random(2).shuffle(srcs)
    vsrc = [s for s in srcs if (lambda d: d.get("mel") is not None and d["cond"].shape[1] == 16)(torch.load(s))][:a.val_max]
    val = [f for f in fs if cache[f]["src"] in set(vsrc) and not cache[f].get("flip")]; tr = [f for f in fs if cache[f]["src"] not in set(vsrc)]
    print(f"[train] {len(tr)} train / {len(val)} val windows ({sum(cache[f].get('flip', False) for f in tr)} flipped in train), student {a.student_steps} steps", flush=True)
    unet = D.load_unet(); unet.requires_grad_(False)
    if hasattr(unet, "enable_gradient_checkpointing"): unet.enable_gradient_checkpointing()
    names = D.add_lora(unet, a.rank, a.rank); params = [p for n, p in unet.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    for p in params: p.requires_grad_(True)
    ema = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    lr_at = lambda s: a.lr * (s / 100 if s < 100 else 0.5 * (1 + math.cos(math.pi * (s - 100) / max(1, a.steps - 100))) * 0.95 + 0.05)
    sS = D.sched(a.student_steps); TS = list(sS.timesteps); ac = sS.alphas_cumprod.to("cuda")
    vae = D.load_vae(); syncnet, sc = D.load_syncnet(); lp = lpips.LPIPS(net="squeeze", verbose=False).to("cuda").eval(); lp.requires_grad_(False)
    log = open(out / "train_log.jsonl", "a"); t0 = time.time()

    def swap_ema():   # exchange the live and the EMA weights (evaluation / saving with the EMA)
        for p, e in zip(params, ema): t = p.data.clone(); p.data.copy_(e); e.copy_(t)

    @torch.no_grad()
    def evaluate(lora_on=True):
        for m in unet.modules():
            if isinstance(m, D.LoRA): m.scale = 1.0 if lora_on else 0.0
        unet.eval(); errs, sims = [], []
        for f in val:
            d = cache[f]; cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]; z = d["z"].cuda()[None]
            with torch.autocast("cuda", dtype=torch.float16):
                x = D.sample(unet, sS, z, cond, au, cfg=1.0)
            errs.append(F.mse_loss(x.float(), d["x0_teacher"].cuda()[None].float()).item())
            if d.get("mel") is not None and x.shape[2] == 16: sims.append(D.sync_sim(syncnet, sc, vae, x, d["mel"]).item())
        for m in unet.modules():
            if isinstance(m, D.LoRA): m.scale = 1.0
        unet.train(); return {"mse": sum(errs) / len(errs), "sync": sum(sims) / len(sims) if sims else None}

    with torch.no_grad():
        tsims = [D.sync_sim(syncnet, sc, vae, cache[f]["x0_teacher"].cuda()[None], cache[f]["mel"]).item() for f in val if cache[f].get("mel") is not None and cache[f]["x0_teacher"].shape[1] == 16]
    b = evaluate(lora_on=False); r = {"step": 0, "val_mse_untrained": b["mse"], "val_sync_untrained": b["sync"], "val_sync_teacher20": sum(tsims) / max(1, len(tsims))}
    print(f"[val] {r}", flush=True); log.write(json.dumps(r) + "\n"); curve = [r]
    unet.train(); run = None
    for step in range(1, a.steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        d = cache[random.choice(tr)]; cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]; z = d["z"].cuda()[None].float(); x0t = d["x0_teacher"].cuda()[None].float()
        k = random.randrange(len(TS)); t = TS[k]; x = z if k == 0 else sS.add_noise(x0t, z, t.reshape(1))
        with torch.autocast("cuda", dtype=torch.float16):
            eps = unet(torch.cat([sS.scale_model_input(x.half(), t), cond], 1), t, encoder_hidden_states=au).sample.float()
        at = ac[t]; x0s = (x - (1 - at).sqrt() * eps) / at.sqrt()
        lmse = F.mse_loss(x0s, x0t); lsync = lpix = llp = torch.zeros((), device="cuda")
        if a.sync_w > 0 or a.pix_w > 0 or a.lpips_w > 0:
            ps = D.decode(vae, x0s, grad=True).float()
            with torch.no_grad(): pt = D.decode(vae, x0t.half(), grad=False).float()
            h = ps.shape[2] // 2; ms, mt = ps[:, :, h:], pt[:, :, h:]          # the mouth half of the face crop
            lpix = F.l1_loss(ms, mt)
            if a.lpips_w > 0: llp = lp(F.interpolate(ms, scale_factor=0.5, mode="bilinear"), F.interpolate(mt, scale_factor=0.5, mode="bilinear")).mean()
            if a.sync_w > 0 and d.get("mel") is not None and x0s.shape[2] == 16:
                px = F.interpolate(ps, size=(sc.data.resolution, sc.data.resolution), mode="bilinear", align_corners=False)[:, :, sc.data.resolution // 2:, :]
                m_ = d["mel"].cuda().half(); m_ = m_[None] if m_.dim() == 3 else m_
                ve, ae = syncnet(px.reshape(1, -1, px.shape[2], px.shape[3]).half(), m_)
                sim = F.cosine_similarity(ve.float(), ae.float()); lsync = F.binary_cross_entropy(sim.clamp(1e-6, 1 - 1e-6), torch.ones_like(sim))
        loss = lmse + a.sync_w * lsync + a.pix_w * lpix + a.lpips_w * llp
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        with torch.no_grad():
            dec = min(a.ema, (1 + step) / (10 + step))   # EMA warm-up: the early average is not dominated by the zero-initialised LoRA
            for p, e in zip(params, ema): e.mul_(dec).add_(p.detach(), alpha=1 - dec)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 25 == 0:
            log.write(json.dumps({"step": step, "loss": loss.item(), "mse": lmse.item(), "sync": lsync.item(), "pix": lpix.item(), "lpips": llp.item(), "lr": lr_at(step),
                                  "s": round(time.time() - t0, 1), "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}) + "\n"); log.flush()
        if step % 100 == 0: print(f"[train] step {step} loss_ema {run:.4f} lr {lr_at(step):.2e} {time.time() - t0:.0f} s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            # checkpoints (LoRA only, fp16, ~30 MB each): ckpt/<run>/step_XXXXX.pt (live), step_XXXXX_ema.pt, step_XXXXX.json; best.pt = the EMA with the lowest val MSE
            ck = pathlib.Path(a.ckpt_root) / a.run; ck.mkdir(parents=True, exist_ok=True)
            meta = lambda: {"lora": {k: v.half() for k, v in D.lora_state(unet).items()}, "rank": a.rank, "names": names, "step": step, "ema": a.ema, "student_steps": a.student_steps}
            torch.save(meta(), ck / f"step_{step:05d}.pt")
            swap_ema(); v = evaluate(); torch.save(meta(), ck / f"step_{step:05d}_ema.pt"); torch.save(meta(), out / "lora.pt")
            r = {"step": step, "val_mse_ema": v["mse"], "val_sync_ema": v["sync"]}; curve.append(r)
            if v["mse"] <= min(c.get("val_mse_ema", 9e9) for c in curve): torch.save(meta(), ck / "best.pt"); r["best"] = True
            swap_ema(); (ck / f"step_{step:05d}.json").write_text(json.dumps({**r, "config": vars(a)}, indent=1))
            print(f"[val] {r} -> {ck}", flush=True); log.write(json.dumps(r) + "\n")
    (out / "val.json").write_text(json.dumps({"curve": curve, "args": vars(a)}, indent=1)); print(f"[train] done {a.steps} steps in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["teacher", "train"]); ap.add_argument("--cap"); ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--flip", action="store_true"); ap.add_argument("--tbatch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--student-steps", type=int, default=4); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-max", type=int, default=24); ap.add_argument("--sync-w", type=float, default=0.15); ap.add_argument("--pix-w", type=float, default=0.5)
    ap.add_argument("--lpips-w", type=float, default=0.2); ap.add_argument("--run", default="r2_s4"); ap.add_argument("--ckpt-root", default="/workspace/dub/r13s/distill/ckpt")
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    {"teacher": cmd_teacher, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
