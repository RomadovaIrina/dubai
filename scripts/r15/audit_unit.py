#!/usr/bin/env python3
"""r15 Z audit fixes: unit checks on real data (old03 / m18 q2d work dirs)"""
import glob, json, os, pathlib, sys, time
import numpy as np
R = pathlib.Path(__file__).resolve().parents[2]
for p in ("scripts/pilot", "scripts/optim"): sys.path.insert(0, str(R / p))
os.environ["DUB_MASTER_TIMING"] = "fps"
import frame_map as fmm, clean_face_stack as cfs, codeformer_accel as ca, anchor_filters as af
print("LM106_TO_5 eyes:", ca.LM106_TO_5[0], ca.LM106_TO_5[1], "| MP:", [cfs.MAP106[i] for i in ca.LM106_TO_5[0]], [cfs.MAP106[i] for i in ca.LM106_TO_5[1]])
print("af: MIN_INLIER", af.MIN_INLIER, "CACHE_TAG", af.CACHE_TAG, "UPPER", len(af.UPPER))
for v in ("old03", "m18"):
    W = glob.glob(f"/workspace/dub/artifacts/r4m/r9g_q2_{v}_q2d/work_*")[0]; T = json.load(open(glob.glob(f"/workspace/dub/artifacts/r4m/r9g_q2_{v}_q2d/timing/*_master_timing_F.json")[0]))
    t0 = time.time(); ex = fmm.exact_master_to_source(T["src"]); j = np.array(T["j"]); n = min(len(ex), len(j))
    th = np.array(json.load(open(glob.glob(W + "/*_25fps.manifest.json")[0]))["frame_map"]["master_to_source"])
    print(f"{v}: exact map {len(ex)} frames in {time.time() - t0:.1f}s; vs r3 j disagree {int((ex[:n] != j[:n]).sum())}; thumb vs exact disagree {int((th[:n] != ex[:n]).sum())}")
    z = np.load(W + "/video/master_mediapipe.npz"); lm = z["lm"].astype(np.float64); lm[~z["presence"]] = np.nan
    ln = cfs.nose_rigid(lm); ok = np.isfinite(lm[:, 0, 0]); iod = np.nanmedian(np.linalg.norm(lm[ok][:, 33] - lm[ok][:, 263], axis=1))
    A0 = lm[:, [5, 64, 294, 4]].mean(1); A1 = ln[:, [5, 64, 294, 4]].mean(1).astype(np.float64); R_ = lm[:, [5, 4, 1, 195]].mean(1)
    d0 = A0 - R_; d1 = A1 - R_
    print(f"   nose anchor - rigid nose: prod std {np.nanstd(d0[ok], 0).mean() / iod * 1000:.2f}  nose_rigid std {np.nanstd(d1[ok], 0).mean() / iod * 1000:.2f} (IOD x1000);"
          f" mean shift {np.linalg.norm(np.nanmean(A1[ok] - A0[ok], 0)) / iod * 1000:.2f}; other points unchanged: {np.allclose(np.delete(ln, [64, 294], 1)[ok], np.delete(lm, [64, 294], 1)[ok])}")
    l106 = lm[:, cfs.MAP106]; lms = [ca.lm5_from_insightface106(l106[k]) if ok[k] else None for k in range(min(64, len(lm)))]
    rm = ca.rigid_mouth_lm5(lms); k0 = next(k for k in range(len(lms)) if lms[k] is not None)
    print(f"   rigid mouth: eyes/nose kept {np.allclose(rm[k0][:3], lms[k0][:3])}; mouth moved {np.linalg.norm(rm[k0][3:] - lms[k0][3:], axis=1).round(2)} px;"
          f" lsq vs lmeds affine diff {np.abs(ca.lsq_similarity(lms[k0], ca.FFHQ_TEMPLATE_512) - ca.CodeFormerAccel.affine_for(lms[k0])).max():.3f}")
