#!/usr/bin/env python3
"""r14 round 3: where the lips sit in LatentSync's aligned 512 face crop, and a differentiable mouth-opening proxy checked against MediaPipe.
  dump  (/venv/dabai):     decode teacher x0 + real (reference) frames of N windows on CPU -> DIR/*.png + DIR/frames.npy
  lm    (/venv/mediapipe): MediaPipe IMAGE mode on the pngs -> DIR/lm.npz (inner-lip gap / mouth width, lip box)
  check (/venv/dabai):     the proxy (dark-cavity share inside the lip box) vs the MediaPipe opening: correlation -> DIR/lipbox.json
    python scripts/r14/lipbox.py dump|lm|check --t DIR/t --dir OUT [--n 48]"""
import argparse, glob, json, pathlib, random, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def opening_proxy(m, box, xp=None):
    """m: (F, 3, H, W) in [-1, 1] (a full crop); box (x0, y0, x1, y1) in fractions. Share of 'dark cavity' pixels inside the lip box:
    luma below 0.55 x the lower-face median luma (skin), soft (sigmoid, 0.04). Works on torch (differentiable) or numpy."""
    import torch
    H, W = m.shape[-2:]; y = (0.299 * m[:, 0] + 0.587 * m[:, 1] + 0.114 * m[:, 2] + 1) / 2
    skin = y[:, H // 2:].flatten(1).median(1).values if isinstance(y, torch.Tensor) else None
    b = y[:, int(box[1] * H):int(box[3] * H), int(box[0] * W):int(box[2] * W)]
    return torch.sigmoid((0.55 * skin[:, None, None] - b) / 0.04).mean((1, 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--t"); ap.add_argument("--dir", required=True); ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args(); O = pathlib.Path(a.dir); O.mkdir(parents=True, exist_ok=True)
    if a.cmd == "dump":
        sys.path.insert(0, str(HERE.parent / "r13" / "distill")); import torch, distill as D
        from PIL import Image
        fs = sorted(glob.glob(str(pathlib.Path(a.t) / "*.pt"))); random.Random(5).shuffle(fs); seen, pick = {}, []
        for f in fs:   # spread over identities: at most 4 windows per job
            j = str(torch.load(f).get("job", pathlib.Path(f).stem.split("_")[0]))
            if seen.get(j, 0) < 4: seen[j] = seen.get(j, 0) + 1; pick.append(f)
            if len(pick) >= a.n: break
        vae = D.load_vae().float().cpu(); arr = []
        with torch.no_grad():
            for i, f in enumerate(pick):
                d = torch.load(f)
                for kind, lat in (("t", d["x0_teacher"][None]), ("r", d["cond"][None, 5:9])):
                    l = lat[0].permute(1, 0, 2, 3)[::5].float() / 0.18215; px = vae.decode(l).sample.clamp(-1, 1)
                    for k, p in enumerate(px):
                        im = ((p.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8); Image.fromarray(im).save(O / f"{i:03d}_{kind}_{k}.png"); arr.append(im)
                print(i, f, flush=True)
        np.save(O / "frames.npy", np.stack(arr)); (O / "names.json").write_text(json.dumps(sorted(p.name for p in O.glob("*.png"))))
    elif a.cmd == "lm":
        import mediapipe as mp
        M = HERE.parents[1] / "models" / "mediapipe" / "face_landmarker.task"
        lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(M))))
        names = sorted(p.name for p in O.glob("*.png")); out = {}
        for n in names:
            r = lm.detect(mp.Image.create_from_file(str(O / n)))
            if r.face_landmarks: out[n] = np.array([[p.x, p.y] for p in r.face_landmarks[0]], np.float32)
        np.savez(O / "lm.npz", names=np.array(list(out)), lm=np.stack(list(out.values()))); print(f"{len(out)}/{len(names)} faces")
    else:
        import torch
        from PIL import Image
        z = np.load(O / "lm.npz"); L = z["lm"]; names = list(z["names"])
        LIPS = [61, 291, 0, 17, 13, 14, 78, 308, 37, 267, 84, 314]
        P = L[:, LIPS]; lo, hi = np.percentile(P.reshape(-1, 2), 1, 0), np.percentile(P.reshape(-1, 2), 99, 0)
        box = [float(lo[0] - .03), float(lo[1] - .03), float(hi[0] + .03), float(hi[1] + .04)]
        gap = np.linalg.norm(L[:, 13] - L[:, 14], axis=1) / np.linalg.norm(L[:, 61] - L[:, 291], axis=1)
        m = torch.tensor(np.stack([np.asarray(Image.open(O / n)) for n in names])).permute(0, 3, 1, 2).float() / 127.5 - 1
        pr = opening_proxy(m, box).numpy(); c = float(np.corrcoef(pr, gap)[0, 1])
        kinds = np.array([n.split("_")[1] for n in names])
        rep = {"faces": len(names), "lip_box_frac": [round(v, 3) for v in box], "corr_proxy_vs_mediapipe_gap": round(c, 3),
               "corr_teacher": round(float(np.corrcoef(pr[kinds == "t"], gap[kinds == "t"])[0, 1]), 3), "corr_real": round(float(np.corrcoef(pr[kinds == "r"], gap[kinds == "r"])[0, 1]), 3),
               "spearman": round(float(np.corrcoef(np.argsort(np.argsort(pr)), np.argsort(np.argsort(gap)))[0, 1]), 3),
               "gap_mean_teacher": float(gap[kinds == "t"].mean()), "gap_mean_real": float(gap[kinds == "r"].mean()),
               "lip_centre_y_frac": float(np.median(P[..., 1])), "lip_x_frac": [float(lo[0]), float(hi[0])]}
        (O / "lipbox.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))


if __name__ == "__main__":
    main()
