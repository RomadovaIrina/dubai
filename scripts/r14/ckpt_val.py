#!/usr/bin/env python3
"""r14: offline validation of every saved LoRA checkpoint on the training run's own val windows (the same selection as distill2.py train):
full student sampling -> latent MSE to the teacher, decoded mouth-half pixel L1 to the teacher ("pix"), mouth jitter, stable-SyncNet sim,
mouth opening / the teacher's (distill2.opening). Picks best-by-pix next to the run's best-by-MSE, and the eval "pick" (opening >= 1.03, then MSE).
    python scripts/r14/ckpt_val.py --out DIR(train dir with t/) --ckpt DIR(ckpt/<run>) [--val-max 24] [--student-steps 4]"""
import argparse, glob, json, pathlib, random, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "r13" / "distill"))
import torch, torch.nn.functional as F
import distill as D, distill2 as D2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--val-max", type=int, default=24)
    ap.add_argument("--student-steps", type=int, default=4); ap.add_argument("--rank", type=int, default=32)
    # --open-min: the proxy reads high; 1.03 reproduces the E2E pick on r2 (step 1500, x0.93 vs q2; 0.95 would pick the damped 3000)
    ap.add_argument("--open-box", default="0.35,0.56,0.65,0.72"); ap.add_argument("--open-min", type=float, default=1.03)
    # r15 (r5): --mse-tol T = among the checkpoints that wide, those within (1 + T) x the lowest MSE go to the highest val sync (r4_s6: its lowest-MSE
    # pick had the run's weakest val sync and lost E2E to the final step); 0 = the r14 rule (lowest MSE)
    ap.add_argument("--mse-tol", type=float, default=0.0); a = ap.parse_args()
    fs = sorted(glob.glob(str(pathlib.Path(a.out) / "t" / "*.pt"))); random.Random(1).shuffle(fs); cache = {f: torch.load(f) for f in fs}
    srcs = sorted({cache[f]["src"] for f in fs}); random.Random(2).shuffle(srcs)
    vsrc = set([s for s in srcs if (lambda d: d.get("mel") is not None and d["cond"].shape[1] == 16)(torch.load(s))][:a.val_max])
    val = [f for f in fs if cache[f]["src"] in vsrc and not cache[f].get("flip")]
    unet = D.load_unet(); D.add_lora(unet, a.rank, a.rank); vae = D.load_vae(); sn, sc = D.load_syncnet(); sS = D.sched(a.student_steps)
    mods = dict(unet.named_modules()); res = []; obox = [float(v) for v in a.open_box.split(",")]
    with torch.no_grad():   # the teacher's mouth opening per val window (the r3 proxy, distill2.opening): the reference of the opening ratio
        ot = {f: D2.opening(D.decode(vae, cache[f]["x0_teacher"].cuda()[None].half(), grad=False).float(), obox).sum().item() for f in val}
    cks = sorted(glob.glob(str(pathlib.Path(a.ckpt) / "step_*_ema.pt")))
    for ck in cks:
        sd = torch.load(ck, map_location="cpu")["lora"]
        with torch.no_grad():
            for n, p in unet.named_parameters():
                if n in sd: p.copy_(sd[n].to(p.dtype))
        m = {"mse": [], "pix": [], "jitter": [], "sync": []}; osum = 0.0
        with torch.no_grad():
            for f in val:
                d = cache[f]; cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]; z = d["z"].cuda()[None]
                with torch.autocast("cuda", dtype=torch.float16):
                    x = D.sample(unet, sS, z, cond, au, cfg=1.0)
                x0t = d["x0_teacher"].cuda()[None]; m["mse"].append(F.mse_loss(x.float(), x0t.float()).item())
                ps = D.decode(vae, x.half(), grad=False).float(); pt = D.decode(vae, x0t.half(), grad=False).float(); h = ps.shape[2] // 2
                m["pix"].append(F.l1_loss(ps[:, :, h:], pt[:, :, h:]).item()); m["jitter"].append(D2.jitter(ps[:, :, h:]).mean().item()); osum += D2.opening(ps, obox).sum().item()
                if d.get("mel") is not None and x.shape[2] == 16: m["sync"].append(D.sync_sim(sn, sc, vae, x, d["mel"]).item())
        r = {"ckpt": pathlib.Path(ck).name, **{k: sum(v) / len(v) for k, v in m.items() if v}, "open_ratio": osum / max(1e-6, sum(ot.values()))}
        res.append(r); print(json.dumps(r), flush=True)
    best_pix = min(res, key=lambda r: r["pix"]); best_mse = min(res, key=lambda r: r["mse"])
    # the eval pick (coordinator, r3: the mouth damping grows with the steps): opening ratio >= --open-min first, then the lowest MSE;
    # none that wide -> the widest
    ok = [r for r in res if r["open_ratio"] >= a.open_min]
    pick = min(ok, key=lambda r: r["mse"]) if ok else max(res, key=lambda r: r["open_ratio"])
    near = [r for r in ok if r["mse"] <= pick["mse"] * (1 + a.mse_tol)] if ok and a.mse_tol > 0 else []
    if len(near) > 1:
        lo = pick; pick = max(near, key=lambda r: r.get("sync", 0.0))
        why = (f"opening ratio {pick['open_ratio']:.3f} >= {a.open_min}; of the {len(near)} within {a.mse_tol:.0%} of the lowest MSE ({lo['mse']:.4f}, {lo['ckpt']}) "
               f"the highest val sync ({pick.get('sync', 0):.3f}, MSE {pick['mse']:.4f})")
    else: why = None
    why = why or (f"opening ratio {pick['open_ratio']:.3f} >= {a.open_min}, lowest MSE ({pick['mse']:.4f}) of the {len(ok)} that wide" if ok else
           f"no checkpoint reached opening ratio {a.open_min}; the widest ({pick['open_ratio']:.3f})")
    rep = {"val_windows": len(val), "by_ckpt": res, "best_by_pix": best_pix["ckpt"], "best_by_mse": best_mse["ckpt"], "pick": pick["ckpt"], "pick_reason": why}
    (pathlib.Path(a.ckpt) / "ckpt_val.json").write_text(json.dumps(rep, indent=1))
    print("best_by_pix", best_pix["ckpt"], "best_by_mse", best_mse["ckpt"], "pick", pick["ckpt"], "-", why)


if __name__ == "__main__":
    main()
