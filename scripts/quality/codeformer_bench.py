#!/usr/bin/env python3
"""CodeFormer stage micro-benchmarks on real LatentSync output frames (week 2, steps 6/7/14).

Frames = the LATENT_SYNC frames of a baseline run (<run-dir>/<id>_baseline.mp4 + manifest); landmarks = LatentSync's insightface
verify pass on the 25 fps master (the same call the pipeline makes) mapped to 5 points, or fresh RetinaFace per frame.
Variants restore the SAME frames with the same weights; each reports faces/s, wall, peak VRAM and the numeric difference of the
restored frames against the reference variant (PSNR over the whole frame and over the pasted region, max |diff|).
  --check-reference   also runs the upstream FaceRestoreHelper functions (align_warp_face / paste_faces_to_input_image with the
                      SAME injected landmarks, upscale 1, no read_image upscale) frame by frame -> proves the runner reproduces
                      the upstream arithmetic (expected PSNR > 50 dB / max diff <= 1 from float ordering).
    python scripts/quality/codeformer_bench.py --id 04 --frames 200 --variants b1 b2 b4 b8 b16 b1_empty b8_fp16 b8_retina --check-reference
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
Q = pathlib.Path("/tmp/dabai_quality")

VARIANTS = {"b1": {"batch_size": 1}, "b2": {"batch_size": 2}, "b4": {"batch_size": 4}, "b8": {"batch_size": 8}, "b16": {"batch_size": 16},
            "b1_empty": {"batch_size": 1, "empty_cache_per_face": True}, "b8_empty": {"batch_size": 8, "empty_cache_per_face": True},
            "b8_fp16": {"batch_size": 8, "autocast": True}, "b8_pfp16": {"batch_size": 8, "parse_autocast": True}, "b8_allfp16": {"batch_size": 8, "autocast": True, "parse_autocast": True},
            "b8_w1": {"batch_size": 8, "workers": 1}, "b8_w8": {"batch_size": 8, "workers": 8}, "b8_nopin": {"batch_size": 8, "pinned": False},
            "b8_retina": {"batch_size": 8, "landmark_source": "retinaface"}, "b8_noparse": {"batch_size": 8, "use_parse": False},
            "b8_cv2mask": {"batch_size": 8, "mask_blur": "cv2"}, "b8_fullpaste": {"batch_size": 8, "roi_paste": False}, "b8_nooverlap": {"batch_size": 8, "overlap": False},
            "b8_legacyexec": {"batch_size": 8, "mask_blur": "cv2", "roi_paste": False, "overlap": False}, "b1_legacyexec": {"batch_size": 1, "mask_blur": "cv2", "roi_paste": False, "overlap": False}}


def psnr(a, b):
    m = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)); return 99.0 if m == 0 else round(10 * np.log10(255 ** 2 / m), 2)


def main() -> int:
    import cv2, torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="04"); ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work"))
    ap.add_argument("--frames", type=int, default=96, help="LATENT_SYNC frames to use (taken from the first LS segments)")
    ap.add_argument("--variants", nargs="+", default=["b1", "b2", "b4", "b8", "b16", "b1_empty", "b8_fp16"]); ap.add_argument("--reference", default="b1")
    ap.add_argument("--w", type=float, default=0.5); ap.add_argument("--repeat", type=int, default=2); ap.add_argument("--check-reference", action="store_true")
    ap.add_argument("--json", default=str(Q / "codeformer" / "bench.json")); ap.add_argument("--sheet", default=str(Q / "frames" / "codeformer_bench.png"))
    a = ap.parse_args()
    import face_aware_latentsync as fa
    if str(fa.LS) not in sys.path: sys.path.insert(0, str(fa.LS))
    from codeformer_accel import CodeFormerAccel, lm5_from_insightface106, FFHQ_TEMPLATE_512, CF_ROOT
    from retinaface_router import RetinaFaceRouter
    man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_baseline.manifest.json").read_text())
    segs = [s for s in sorted(man["lipsync"]["segments"], key=lambda s: s["start_frame"]) if s["action"] == "LATENT_SYNC"]
    want = []
    for s in segs:
        for f in range(s["start_frame"], s["end_frame"]):
            want.append(f)
            if len(want) >= a.frames: break
        if len(want) >= a.frames: break
    wanted = set(want); out_mp4 = pathlib.Path(a.run_dir) / f"{a.id}_baseline.mp4"
    cap = cv2.VideoCapture(str(out_mp4)); frames = {}; i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i in wanted: frames[i] = fr
        i += 1
    cap.release(); idx = sorted(frames); F = [frames[k] for k in idx]
    print(f"{len(F)} LATENT_SYNC frames from {out_mp4.name} ({F[0].shape[1]}x{F[0].shape[0]})")
    det = fa.face_detector(); t0 = time.perf_counter()
    dets = [det(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in F]; t_det = time.perf_counter() - t0
    lms = [lm5_from_insightface106(d[1]) for d in dets]; n_lm = sum(l is not None for l in lms)
    print(f"insightface verify-pass detections: {n_lm}/{len(F)} in {t_det:.2f}s ({t_det / len(F) * 1000:.1f} ms/frame, this cost is already paid by the pipeline)")
    router = RetinaFaceRouter(50, 80, 0.8)
    def retina_lm(fr):
        r = router.classify_frame(fr); return np.asarray(r["landmarks_5"], dtype=np.float32) if r.get("landmarks_5") and r["cls"] != "NO_FACE" else None
    t0 = time.perf_counter(); _ = [retina_lm(f) for f in F]; t_ret = time.perf_counter() - t0
    print(f"retinaface router per frame: {t_ret / len(F) * 1000:.1f} ms/frame")
    results = {"id": a.id, "frames": len(F), "shape": list(F[0].shape), "w": a.w, "insightface_detect_ms_per_frame": round(t_det / len(F) * 1000, 2),
               "retinaface_detect_ms_per_frame": round(t_ret / len(F) * 1000, 2), "landmarks_available": n_lm, "variants": {}}
    outputs = {}
    for name in a.variants:
        cfg = VARIANTS[name]; kw = {k: v for k, v in cfg.items()}
        runner = CodeFormerAccel(fidelity_weight=a.w, fallback_detector=retina_lm, **kw)
        lm_in = [None] * len(F) if runner.landmark_source == "retinaface" else lms
        runs = []
        for r in range(a.repeat):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out, st = runner.restore_frames(F, lm_in); torch.cuda.synchronize(); wall = time.perf_counter() - t0
            runs.append({"wall": round(wall, 3), "faces_per_s": round(st["faces_restored"] / wall, 2), "peak_vram_mib": st["peak_vram_mib"], "stages": st["seconds"],
                         "faces": st["faces_restored"], "fallback": st["fallback_detections"], "no_landmarks": st["frames_without_landmarks"]})
        runner.close(); del runner; torch.cuda.empty_cache()
        outputs[name] = out
        best = min(runs, key=lambda r: r["wall"])
        results["variants"][name] = {"config": cfg, "runs": runs, "best_wall": best["wall"], "faces_per_s": best["faces_per_s"], "peak_vram_mib": max(r["peak_vram_mib"] for r in runs),
                                     "s_per_video_min_at_25fps": round(best["wall"] / (len(F) / 25) * 60, 2)}
        print(f"{name:10s} {cfg} -> {best['faces_per_s']:7.2f} faces/s  wall {best['wall']:.2f}s  peak {results['variants'][name]['peak_vram_mib']} MiB  stages {best['stages']}")
    ref = outputs.get(a.reference)
    if ref is not None:
        for name, out in outputs.items():
            if name == a.reference: continue
            ps = [psnr(x, y) for x, y in zip(out, ref)]; md = max(int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()) for x, y in zip(out, ref))
            diff_px = [int((np.abs(x.astype(np.int16) - y.astype(np.int16)).max(axis=2) > 2).sum()) for x, y in zip(out, ref)]
            results["variants"][name]["vs_reference"] = {"psnr_mean": round(float(np.mean(ps)), 2), "psnr_min": round(float(np.min(ps)), 2), "max_abs_diff": md,
                                                         "pixels_diff_gt2_mean": round(float(np.mean(diff_px)), 1)}
            print(f"  {name} vs {a.reference}: PSNR mean {np.mean(ps):.2f} min {np.min(ps):.2f} dB, max|diff| {md}, px>2 per frame {np.mean(diff_px):.1f}")
    if a.check_reference and ref is not None:
        sys.path.insert(0, str(CF_ROOT))
        from torchvision.transforms.functional import normalize
        from basicsr.utils import img2tensor, tensor2img
        from basicsr.archs import codeformer_arch  # noqa
        from basicsr.utils.registry import ARCH_REGISTRY
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50", save_ext="png", use_parse=True, device="cuda")
        assert np.allclose(helper.face_template, FFHQ_TEMPLATE_512, atol=1e-4)
        net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to("cuda")
        net.load_state_dict(torch.load(str(CF_ROOT / "weights/CodeFormer/codeformer.pth"))["params_ema"]); net.eval()
        ps = []; md = 0; t0 = time.perf_counter(); n = 0
        for fr, lm, ours in zip(F, lms, ref):
            if lm is None: continue
            helper.clean_all(); helper.input_img = fr; helper.is_gray = False; helper.all_landmarks_5 = [np.asarray(lm, dtype=np.float32)]
            helper.align_warp_face()
            t = img2tensor(helper.cropped_faces[0] / 255.0, bgr2rgb=True, float32=True); normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            with torch.no_grad(): o = net(t.unsqueeze(0).to("cuda"), w=a.w, adain=True)[0]
            helper.add_restored_face(tensor2img(o, rgb2bgr=True, min_max=(-1, 1)).astype("uint8"), helper.cropped_faces[0]); helper.get_inverse_affine(None)
            up = helper.paste_faces_to_input_image(upsample_img=None, draw_box=False); n += 1
            ps.append(psnr(up, ours)); md = max(md, int(np.abs(up.astype(np.int16) - ours.astype(np.int16)).max()))
        wall = time.perf_counter() - t0
        results["upstream_helper_check"] = {"frames": n, "psnr_mean": round(float(np.mean(ps)), 2), "psnr_min": round(float(np.min(ps)), 2), "max_abs_diff": md,
                                            "faces_per_s_unbatched_upstream_functions": round(n / wall, 2)}
        print(f"upstream FaceRestoreHelper functions (same landmarks, batch 1, per-face empty_cache absent) vs {a.reference}: PSNR mean {np.mean(ps):.2f} min {np.min(ps):.2f}, max|diff| {md}; {n / wall:.2f} faces/s")
    # contact sheet: input / reference / other variants for 3 frames (face crop)
    try:
        import cv2
        picks = [0, len(F) // 2, len(F) - 1]; tiles = []
        for k in picks:
            lm = lms[k]
            if lm is None: continue
            x1, y1 = int(lm[:, 0].min() - 40), int(lm[:, 1].min() - 60); x2, y2 = int(lm[:, 0].max() + 40), int(lm[:, 1].max() + 50)
            row = [F[k][max(0, y1):y2, max(0, x1):x2]] + [outputs[nm][k][max(0, y1):y2, max(0, x1):x2] for nm in a.variants]
            tiles.append(np.concatenate([cv2.resize(t, (200, int(200 * t.shape[0] / t.shape[1]))) for t in row], axis=1))
        h = max(t.shape[0] for t in tiles); canvas = np.zeros((sum(t.shape[0] + 4 for t in tiles) + 24, tiles[0].shape[1], 3), np.uint8); y = 24
        cv2.putText(canvas, "input | " + " | ".join(a.variants), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        for t in tiles: canvas[y:y + t.shape[0], :t.shape[1]] = t; y += t.shape[0] + 4
        pathlib.Path(a.sheet).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(a.sheet, canvas); results["sheet"] = a.sheet
    except Exception as e:
        results["sheet_error"] = str(e)
    pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.json).write_text(json.dumps(results, indent=1)); print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
