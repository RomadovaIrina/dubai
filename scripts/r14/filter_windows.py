#!/usr/bin/env python3
"""r14 student data: the LatentSync data-pipeline SYNC filter, per captured window, for windows with their ORIGINAL audio (external clips).
The window's real frames are its reference latents (production inference uses the same frames as the reference; checked here: the upper,
unmasked half of the masked latents must match the reference) -> decoded -> LatentSync's stable SyncNet similarity with the window's mel.
A window is kept when the visible face is really the one speaking (interviewer's voice over a silent face, b-roll, off-sync dubbing: dropped).
Threshold: from the data itself, against the NEGATIVES = our own clips' windows (real mouth vs the DUB audio = out of sync by construction):
keep sim >= max(--min-sim, p95 of the negatives).
    python scripts/r14/filter_windows.py --cap DIR [--neg DIR] --out keep.json [--min-sim 0.5]"""
import argparse, glob, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "r13" / "distill"))
import numpy as np, torch, torch.nn.functional as F
import distill as D


@torch.no_grad()
def score(fs, vae, sn, sc):
    out = {}
    for f in fs:
        d = torch.load(f); c = d["cond"]
        if d.get("mel") is None or c.shape[1] != 16: continue
        ref = c[5:9].cuda()[None].half(); msk = c[1:5].float(); h = c.shape[2] // 2
        same = F.mse_loss(msk[:, :, :h // 2], c[5:9].float()[:, :, :h // 2]).item()   # top quarter of the latent: never masked
        out[f] = {"sim": D.sync_sim(sn, sc, vae, ref, d["mel"]).item(), "ref_vs_masked_top_mse": same, "job": d.get("job")}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", required=True); ap.add_argument("--neg"); ap.add_argument("--out", required=True)
    ap.add_argument("--min-sim", type=float, default=0.5); ap.add_argument("--neg-max", type=int, default=300); a = ap.parse_args()
    vae = D.load_vae(); sn, sc = D.load_syncnet()
    pos = score(sorted(glob.glob(str(pathlib.Path(a.cap) / "*.pt"))), vae, sn, sc)
    neg = score(sorted(glob.glob(str(pathlib.Path(a.neg) / "*.pt")))[:a.neg_max], vae, sn, sc) if a.neg else {}
    ns = np.array([v["sim"] for v in neg.values()]) if neg else np.array([])
    thr = max(a.min_sim, float(np.percentile(ns, 95)) if len(ns) else 0.0)
    keep = [f for f, v in pos.items() if v["sim"] >= thr]
    ps = np.array([v["sim"] for v in pos.values()])
    by_job = {}
    for f, v in pos.items(): j = by_job.setdefault(v["job"], [0, 0]); j[0] += 1; j[1] += v["sim"] >= thr
    rep = {"threshold": thr, "neg_n": len(ns), "neg_p50": float(np.median(ns)) if len(ns) else None, "neg_p95": float(np.percentile(ns, 95)) if len(ns) else None,
           "pos_n": len(ps), "pos_p50": float(np.median(ps)) if len(ps) else None, "kept": len(keep), "by_job": by_job,
           "ref_is_same_frame_mse_p95": float(np.percentile([v["ref_vs_masked_top_mse"] for v in pos.values()], 95)) if pos else None,
           "keep": keep, "scores": {pathlib.Path(f).name: v for f, v in {**pos, **neg}.items()}}
    pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("keep", "scores")}, indent=1))


if __name__ == "__main__":
    main()
