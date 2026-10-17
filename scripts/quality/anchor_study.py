#!/usr/bin/env python3
"""Round 2 (track A): rigid head motion, shaky windows and crop-anchor statistics per landmark filter, on the 25 fps master only (CPU).

    python scripts/quality/anchor_study.py --master M/03/master_25fps.mp4 --npz M/03/master_mediapipe.npz --id 03 --out DIR \
        [--manifest RUN/03.manifest.json]   # LS segments -> score only frames LatentSync would see
Writes DIR/<id>_rigid.npz (H per frame, cached), DIR/<id>_anchor_study.json (shaky seconds + per-filter anchor stats).
Shake = per second, high-pass energy of the rigid upper-face translation (DIS flow, RANSAC similarity; no detector in the motion):
hp = per-frame head velocity at the face centre minus its Gaussian(sigma 2 frames) low-pass, energy = mean |hp|^2 (px^2 / frame^2).
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
import anchor_filters as af

FILTERS = ["raw", "box:2", "box:1", "flow:2", "flow:4", "flow:8", "kalman:0.05:1.0", "kalman:0.2:1.0", "kalman:1.0:0.5", "kalman:5.0:0.5", "euro:1.0:0.05", "euro:2.0:0.1",
           "sub:2:2", "sub:3:2", "sub:3:4"]


def read_all(p):
    import cv2
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--master", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--id", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", default=None); ap.add_argument("--filters", nargs="*", default=FILTERS)
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import clean_face_stack as cfs
    lm = np.load(a.npz)["lm"].astype(np.float64); T = len(lm); ok = np.isfinite(lm[:, 0, 0])
    rp = out / f"{a.id}_rigid.npz"
    if rp.exists():
        z = np.load(rp); H, info = z["H"], z["info"]
    else:
        frames = read_all(a.master); assert len(frames) == T, (len(frames), T)
        H, info = af.rigid_motion(frames, lm, return_info=True); np.savez_compressed(rp, H=H, info=info)
        W_, H_ = frames[0].shape[1], frames[0].shape[0]
    import cv2
    cap = cv2.VideoCapture(a.master); W_, H_ = int(cap.get(3)), int(cap.get(4)); cap.release()
    # head velocity at the face centre (eye midpoint) and its high-pass
    cen = np.where(ok[:, None], 0.5 * (lm[:, 33] + lm[:, 263]), np.nan)
    hv = np.full((T, 2), np.nan)
    for t in range(1, T):
        if ok[t] and ok[t - 1]: hv[t] = af._apply(H[t], cen[t - 1][None])[0] - cen[t - 1]
    okv = np.isfinite(hv[:, 0]); hvz = np.nan_to_num(hv); hp = hvz - af.gauss1d(hvz, 2.0); hp[~okv] = 0
    iod = np.nanmedian(np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1))
    sec = []
    for s in range(int(np.ceil(T / 25))):
        sl = slice(s * 25, min(T, s * 25 + 25)); m = okv[sl]
        if m.sum() < 15: continue
        e = float((hp[sl][m] ** 2).sum(1).mean()); ex = float((hp[sl][m, 0] ** 2).mean()); sp = float(np.linalg.norm(hv[sl][m], axis=1).mean())
        sec.append({"s": s, "hp_energy": round(e, 4), "hp_energy_x": round(ex, 4), "hp_rms_px": round(e ** 0.5, 3), "head_speed_px": round(sp, 3)})
    sel = ok.copy()
    if a.manifest:
        man = json.loads(pathlib.Path(a.manifest).read_text()); sel[:] = False
        for sg in man["lipsync"]["segments"]:
            if sg["action"] == "LATENT_SYNC": sel[sg["start_frame"]:sg["end_frame"]] = True
        sel &= ok
    top = sorted([x for x in sec], key=lambda x: -x["hp_energy"])
    res = {"id": a.id, "frames": T, "face_frames": int(ok.sum()), "iod_median_px": round(float(iod), 1), "resolution": [W_, H_],
           "rigid_fit": {"median_samples": float(np.median(info[info[:, 0] > 0, 0])) if (info[:, 0] > 0).any() else 0,
                         "median_inlier_share": round(float(np.median(info[info[:, 0] > 0, 1])), 3) if (info[:, 0] > 0).any() else 0},
           "shake_per_second": sec, "shaky_top": top[:8], "filters": {}}
    shaky_frames = np.zeros(T, bool)
    for x in top[:5]: shaky_frames[x["s"] * 25:(x["s"] + 1) * 25] = True
    for spec in a.filters:
        f = af.filter_landmarks(lm, H, spec)
        dets = cfs.ls_detections(f.astype(np.float32), W_, H_, calib=True)
        l106 = np.stack([d[1] if d[1] is not None else np.full((106, 2), np.nan, np.float32) for d in dets]).astype(np.float64)
        anc = af.ls_anchors(l106, rounded=True); ancf = af.ls_anchors(l106, rounded=False)
        bb = np.array([d[0] if d[0] is not None else (np.nan,) * 4 for d in dets], np.float64)
        bacc = np.abs(bb[2:] - 2 * bb[1:-1] + bb[:-2]).mean(1); bacc = bacc[np.isfinite(bacc)]
        res["filters"][spec] = {"all": af.anchor_stats(anc, H, sel), "shaky_top5s": af.anchor_stats(anc, H, sel & shaky_frames),
                                "all_unrounded": af.anchor_stats(ancf, H, sel), "shaky_unrounded": af.anchor_stats(ancf, H, sel & shaky_frames),
                                "bbox_acc_px": round(float(bacc.mean()), 4) if len(bacc) else None}
        r = res["filters"][spec]
        print(f"{a.id} {spec:18s} resid {r['all']['resid_vs_head_px']} shaky {r['shaky_top5s']['resid_vs_head_px']} gain {r['shaky_top5s']['hp_gain_x']} "
              f"lag {r['shaky_top5s']['lag_frames_x']} acc {r['all']['acc_px']} | unrounded shaky {r['shaky_unrounded']['resid_vs_head_px']}", flush=True)
    (out / f"{a.id}_anchor_study.json").write_text(json.dumps(res, indent=1))
    print("top shaky seconds:", [(x["s"], x["hp_rms_px"]) for x in top[:8]])


if __name__ == "__main__":
    main()
