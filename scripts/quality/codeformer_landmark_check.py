#!/usr/bin/env python3
"""Which landmarks can the CodeFormer stage reuse? Compare, on sampled master frames of the test videos:
  R  facexlib RetinaFace 5 landmarks from the router (scripts/optim/retinaface_router.py, the same detector/weights CodeFormer's
     FaceRestoreHelper uses by default) — the reference alignment input;
  U  the upstream CodeFormer detection path (FaceRestoreHelper.get_face_landmarks_5 with resize=640 on the read_image-upscaled frame,
     mapped back to master pixels) — what inference_codeformer.py actually aligns with;
  I  5 points derived from LatentSync's insightface landmark_2d_106 (already computed for every LATENT_SYNC frame by the verify pass).
For I, every 106 index is scored against each R point (median distance over frames) and the best single index / best contour
centroid is reported, so the mapping in codeformer_accel.LM106_TO_5 is measured, not assumed. Also reports the crop-level
consequence: PSNR between the 512x512 aligned crops produced from R vs I vs U landmarks.
    python scripts/quality/codeformer_landmark_check.py --videos test_videos/04.mp4 test_videos/01.MP4 --every 40 --json /tmp/x.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True); ap.add_argument("--every", type=int, default=40); ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--json", default=None); ap.add_argument("--work", default="/tmp/dabai_quality/lmcheck")
    a = ap.parse_args()
    import face_aware_latentsync as fa
    if str(fa.LS) not in sys.path: sys.path.insert(0, str(fa.LS))   # latentsync.* imports (FaceDetector)
    from retinaface_router import RetinaFaceRouter
    from codeformer_accel import CodeFormerAccel, FFHQ_TEMPLATE_512, CF_ROOT
    sys.path.insert(0, str(CF_ROOT))
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    router = RetinaFaceRouter(50, 80, 0.8); det = fa.face_detector()
    helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50", save_ext="png", use_parse=False, device="cuda")
    assert np.allclose(helper.face_template, FFHQ_TEMPLATE_512, atol=1e-4), "template mismatch"
    work = pathlib.Path(a.work); work.mkdir(parents=True, exist_ok=True)
    R, U, I106, crops_stats = [], [], [], []
    for v in a.videos:
        src = pathlib.Path(v).resolve(); w = work / src.stem; w.mkdir(exist_ok=True)
        master, _ = fa.make_master(src, w)
        cap = cv2.VideoCapture(str(master)); i = 0; taken = 0
        while taken < a.max_frames:
            ok, fr = cap.read()
            if not ok: break
            if i % a.every == 0:
                r = router.classify_frame(fr)
                bbox, lmk = det(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                helper.clean_all(); helper.read_image(fr); n = helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                if r["cls"] == "VALID_FACE" and r.get("landmarks_5") and lmk is not None and n > 0:
                    scale = helper.input_img.shape[0] / fr.shape[0]      # read_image upscale factor (short side -> 512)
                    # upstream picks all faces; take the one closest to the router bbox
                    rb = r["bbox"]; rc = np.array([(rb["x1"] + rb["x2"]) / 2, (rb["y1"] + rb["y2"]) / 2])
                    j = int(np.argmin([np.linalg.norm(np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2]) / scale - rc) for d in helper.det_faces]))
                    u = np.asarray(helper.all_landmarks_5[j], dtype=np.float32) / scale
                    R.append(np.asarray(r["landmarks_5"], dtype=np.float32)); U.append(u); I106.append(np.asarray(lmk, dtype=np.float32))
                    crops_stats.append((src.stem, i, fr))
                taken += 1
            i += 1
        cap.release()
    R = np.stack(R); U = np.stack(U); I = np.stack(I106); n = len(R)
    print(f"{n} frames with all three detections")
    names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    res = {"frames": n, "retina_vs_upstream_px": {}, "best_single_index": {}, "best_group": {}, "chosen": {}}
    d = np.linalg.norm(R - U, axis=2)
    for k, nm in enumerate(names):
        res["retina_vs_upstream_px"][nm] = {"median": round(float(np.median(d[:, k])), 2), "p95": round(float(np.percentile(d[:, k], 95)), 2)}
    # single best 106 index per point
    for k, nm in enumerate(names):
        dist = np.linalg.norm(I - R[:, k:k + 1, :], axis=2)         # (n, 106)
        med = np.median(dist, axis=0); j = int(np.argmin(med))
        res["best_single_index"][nm] = {"index": j, "median_px": round(float(med[j]), 2), "p95_px": round(float(np.percentile(dist[:, j], 95)), 2)}
    # contour-group centroids (contiguous index ranges of length 4..12) per point
    for k, nm in enumerate(names):
        best = None
        for L in range(4, 13):
            for s in range(0, 106 - L + 1):
                c = I[:, s:s + L, :].mean(axis=1); dist = np.linalg.norm(c - R[:, k, :], axis=1); med = float(np.median(dist))
                if best is None or med < best[0]:
                    best = (med, s, L, float(np.percentile(dist, 95)))
        res["best_group"][nm] = {"range": [best[1], best[1] + best[2]], "median_px": round(best[0], 2), "p95_px": round(best[3], 2)}
    # choose per point: group if it beats the single index by > 0.1 px else single
    mapping = []
    for nm in names:
        g, s_ = res["best_group"][nm], res["best_single_index"][nm]
        if g["median_px"] + 0.1 < s_["median_px"]:
            mapping.append(list(range(g["range"][0], g["range"][1]))); res["chosen"][nm] = {"group": g["range"], "median_px": g["median_px"], "p95_px": g["p95_px"]}
        else:
            mapping.append(s_["index"]); res["chosen"][nm] = {"index": s_["index"], "median_px": s_["median_px"], "p95_px": s_["p95_px"]}
    res["LM106_TO_5"] = mapping
    # crop-level consequence
    from codeformer_accel import lm5_from_insightface106
    ps_ri, ps_ru, cent = [], [], []
    for (stem, fi, fr), r5, u5, l106 in zip(crops_stats, R, U, I):
        i5 = lm5_from_insightface106(l106, mapping)
        cr = CodeFormerAccel.align_crop(fr, CodeFormerAccel.affine_for(r5)); ci = CodeFormerAccel.align_crop(fr, CodeFormerAccel.affine_for(i5)); cu = CodeFormerAccel.align_crop(fr, CodeFormerAccel.affine_for(u5))
        def psnr(x, y):
            m = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2)); return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)
        ps_ri.append(psnr(cr, ci)); ps_ru.append(psnr(cr, cu)); cent.append(float(np.linalg.norm(i5 - r5, axis=1).mean()))
    res["crop_psnr_retina_vs_insight"] = {"median": round(float(np.median(ps_ri)), 2), "min": round(float(np.min(ps_ri)), 2)}
    res["crop_psnr_retina_vs_upstream"] = {"median": round(float(np.median(ps_ru)), 2), "min": round(float(np.min(ps_ru)), 2)}
    res["insight5_vs_retina_mean_px"] = {"median": round(float(np.median(cent)), 2), "p95": round(float(np.percentile(cent, 95)), 2)}
    print(json.dumps(res, indent=1))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
