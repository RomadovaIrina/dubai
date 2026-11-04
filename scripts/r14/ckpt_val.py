#!/usr/bin/env python3
"""r14: offline validation of every saved LoRA checkpoint on the training run's own val windows (the same selection as distill2.py train):
full student sampling -> latent MSE to the teacher, decoded mouth-half pixel L1 to the teacher ("pix"), mouth jitter, stable-SyncNet sim.
Picks best-by-pix next to the run's best-by-MSE.
    python scripts/r14/ckpt_val.py --out DIR(train dir with t/) --ckpt DIR(ckpt/<run>) [--val-max 24] [--student-steps 4]"""
import argparse, glob, json, pathlib, random, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "r13" / "distill"))
import torch, torch.nn.functional as F
import distill as D, distill2 as D2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--val-max", type=int, default=24)
    ap.add_argument("--student-steps", type=int, default=4); ap.add_argument("--rank", type=int, default=32); a = ap.parse_args()
    fs = sorted(glob.glob(str(pathlib.Path(a.out) / "t" / "*.pt"))); random.Random(1).shuffle(fs); cache = {f: torch.load(f) for f in fs}
    srcs = sorted({cache[f]["src"] for f in fs}); random.Random(2).shuffle(srcs)
    vsrc = set([s for s in srcs if (lambda d: d.get("mel") is not None and d["cond"].shape[1] == 16)(torch.load(s))][:a.val_max])
    val = [f for f in fs if cache[f]["src"] in vsrc and not cache[f].get("flip")]
    unet = D.load_unet(); D.add_lora(unet, a.rank, a.rank); vae = D.load_vae(); sn, sc = D.load_syncnet(); sS = D.sched(a.student_steps)
    mods = dict(unet.named_modules()); res = []
    cks = sorted(glob.glob(str(pathlib.Path(a.ckpt) / "step_*_ema.pt")))
    for ck in cks:
        sd = torch.load(ck, map_location="cpu")["lora"]
        with torch.no_grad():
            for n, p in unet.named_parameters():
                if n in sd: p.copy_(sd[n].to(p.dtype))
        m = {"mse": [], "pix": [], "jitter": [], "sync": []}
        with torch.no_grad():
            for f in val:
                d = cache[f]; cond = d["cond"].cuda()[None]; au = d["audio"].cuda()[None]; z = d["z"].cuda()[None]
                with torch.autocast("cuda", dtype=torch.float16):
                    x = D.sample(unet, sS, z, cond, au, cfg=1.0)
                x0t = d["x0_teacher"].cuda()[None]; m["mse"].append(F.mse_loss(x.float(), x0t.float()).item())
                ps = D.decode(vae, x.half(), grad=False).float(); pt = D.decode(vae, x0t.half(), grad=False).float(); h = ps.shape[2] // 2
                m["pix"].append(F.l1_loss(ps[:, :, h:], pt[:, :, h:]).item()); m["jitter"].append(D2.jitter(ps[:, :, h:]).mean().item())
                if d.get("mel") is not None and x.shape[2] == 16: m["sync"].append(D.sync_sim(sn, sc, vae, x, d["mel"]).item())
        r = {"ckpt": pathlib.Path(ck).name, **{k: sum(v) / len(v) for k, v in m.items() if v}}; res.append(r); print(json.dumps(r), flush=True)
    best_pix = min(res, key=lambda r: r["pix"]); best_mse = min(res, key=lambda r: r["mse"])
    rep = {"val_windows": len(val), "by_ckpt": res, "best_by_pix": best_pix["ckpt"], "best_by_mse": best_mse["ckpt"]}
    (pathlib.Path(a.ckpt) / "ckpt_val.json").write_text(json.dumps(rep, indent=1)); print("best_by_pix", best_pix["ckpt"], "best_by_mse", best_mse["ckpt"])


if __name__ == "__main__":
    main()
