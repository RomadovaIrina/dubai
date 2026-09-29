#!/usr/bin/env python3
"""CodeFormer candidate vs LatentSync-only output: where did the pixels change, and are the seams visible? (week 2, steps 10/11/15)

Compares <run-dir>/<id>_<tag>.mp4 (candidate_video_stage.py --codeformer optimized) with the baseline <id>_baseline.mp4 frame by frame:
  geometry            width/height/fps/frame count/duration of both files (must be identical);
  untouched frames    frames outside LATENT_SYNC segments must be bit-identical (max |diff| = 0);
  change locality     per LS frame: bounding box of changed pixels (|diff| > 2) vs the RetinaFace face bbox dilated by the
                      feather margin -> fraction of changed pixels outside the dilated face box (expected 0);
  seam metric         mean gradient magnitude on a ring just outside / just inside the change boundary, candidate vs baseline
                      (ratio ~1 = no visible edge added), and the temporal jitter of the change-region centroid relative to the
                      face bbox centroid (mask moving against the head);
  eyes / background   Laplacian sharpness of the upper-face crop and of a background strip, candidate / baseline (extra blur < 1).
Contact sheets: <frames>/<id>_<tag>/cf_face_<frame>.png (baseline | candidate | |diff| x8) for a few LS frames and the 2 frames
around every LS<->pass-through transition. JSON -> <out>/<id>_<tag>.codeformer_quality.json
    python scripts/quality/codeformer_quality.py --id 04 --tag cf_w05 --run-dir /tmp/dabai_quality/candidates
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
Q = pathlib.Path("/tmp/dabai_quality")


def probe(p: pathlib.Path) -> dict:
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-count_frames", "-of", "json", str(p)], check=True, capture_output=True, text=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video"); num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    return {"duration_s": round(float(j["format"]["duration"]), 3), "width": v["width"], "height": v["height"], "fps": round(int(num) / max(int(den), 1), 4), "frames": int(v.get("nb_read_frames") or 0),
            "audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def sharp(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) if g.size else 0.0


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--run-dir", default=str(Q / "candidates")); ap.add_argument("--baseline-dir", default=str(Q / "baseline"))
    ap.add_argument("--out", default=str(Q / "manifests")); ap.add_argument("--frames", default=str(Q / "frames")); ap.add_argument("--sample-every", type=int, default=1)
    a = ap.parse_args()
    from retinaface_router import RetinaFaceRouter
    cand = pathlib.Path(a.run_dir) / f"{a.id}_{a.tag}.mp4"; base = pathlib.Path(a.baseline_dir) / f"{a.id}_baseline.mp4"
    man = json.loads(cand.with_suffix(".manifest.json").read_text())
    segs = sorted(man["lipsync"]["segments"], key=lambda s: s["start_frame"]); ls = [s for s in segs if s["action"] == "LATENT_SYNC"]
    in_ls = np.zeros(man["lipsync"]["master_frames"], dtype=bool)
    for s in ls: in_ls[s["start_frame"]:s["end_frame"]] = True
    trs = [b["start_frame"] for a_, b in zip(segs, segs[1:]) if (a_["action"] == "LATENT_SYNC") != (b["action"] == "LATENT_SYNC")]
    pc, pb = probe(cand), probe(base)
    res = {"id": a.id, "tag": a.tag, "candidate": str(cand), "baseline": str(base), "geometry": {"candidate": pc, "baseline": pb,
           "identical": all(pc[k] == pb[k] for k in ("width", "height", "fps", "frames")) and abs(pc["duration_s"] - pb["duration_s"]) < 0.02}}
    router = RetinaFaceRouter(50, 80, 0.8)
    cc, cb = cv2.VideoCapture(str(cand)), cv2.VideoCapture(str(base)); i = 0
    untouched_max = 0; untouched_n = 0; ls_rows = []; sheets = []; want = set()
    for s in ls[:6]:
        n_ = s["end_frame"] - s["start_frame"]; want.update(s["start_frame"] + int(n_ * f) for f in (0.1, 0.5, 0.9))
    for t in trs: want.update((t - 1, t))
    fd = pathlib.Path(a.frames) / f"{a.id}_{a.tag}"; fd.mkdir(parents=True, exist_ok=True)
    prev_c = None; prev_face = None; bb = None
    while True:
        okc, fc = cc.read(); okb, fb = cb.read()
        if not (okc and okb): break
        d = np.abs(fc.astype(np.int16) - fb.astype(np.int16)).max(axis=2)
        if not in_ls[i]:
            untouched_max = max(untouched_max, int(d.max())); untouched_n += 1
        elif i % a.sample_every == 0:
            r = router.classify_frame(fb)
            if r["bbox"] and r["cls"] != "NO_FACE": bb = (r["bbox"]["x1"], r["bbox"]["y1"], r["bbox"]["x2"], r["bbox"]["y2"])
            ch = d > 2; ys, xs = np.nonzero(ch); row = {"frame": i, "changed_px": int(ch.sum())}
            if bb is not None and len(xs):
                x1, y1, x2, y2 = bb; w, h = x2 - x1, y2 - y1; m = int(0.35 * max(w, h))         # feather / template margin
                X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(d.shape[1], x2 + m), min(d.shape[0], y2 + m)
                outside = ch.copy(); outside[Y1:Y2, X1:X2] = False
                row.update(change_bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], face_bbox=list(bb), outside_dilated_face_px=int(outside.sum()),
                           outside_fraction=round(float(outside.sum() / max(ch.sum(), 1)), 5), change_centroid=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                           face_centroid=[(x1 + x2) / 2, (y1 + y2) / 2])
                # seam: gradient energy on a 3 px ring around the changed region boundary, candidate vs baseline
                k = np.ones((3, 3), np.uint8); mask = ch.astype(np.uint8); ring = cv2.dilate(mask, k, iterations=3) - cv2.erode(mask, k, iterations=3)
                if ring.sum() > 50:
                    gc = cv2.Laplacian(cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY), cv2.CV_32F); gb = cv2.Laplacian(cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
                    row["seam_grad_cand"] = round(float(np.abs(gc[ring > 0]).mean()), 3); row["seam_grad_base"] = round(float(np.abs(gb[ring > 0]).mean()), 3)
                    row["seam_ratio"] = round(row["seam_grad_cand"] / max(row["seam_grad_base"], 1e-6), 3)
                # sharpness of eyes (upper half of the face bbox) and a background strip (left of the face box)
                up_c, up_b = fc[y1:y1 + h // 2, x1:x2], fb[y1:y1 + h // 2, x1:x2]
                row["upper_sharp_ratio"] = round(sharp(up_c) / max(sharp(up_b), 1e-6), 3)
                mo_c, mo_b = fc[y1 + h // 2:y2 + int(0.1 * h), x1:x2], fb[y1 + h // 2:y2 + int(0.1 * h), x1:x2]
                row["mouth_sharp_ratio"] = round(sharp(mo_c) / max(sharp(mo_b), 1e-6), 3)
                bx1, bx2 = max(0, X1 - 60), X1
                if bx2 - bx1 > 20:
                    row["background_max_diff"] = int(d[Y1:Y2, bx1:bx2].max())
                if prev_c is not None and prev_face is not None and in_ls[i - 1]:
                    row["centroid_rel_jitter_px"] = round(float(np.hypot(*(np.array(row["change_centroid"]) - np.array(row["face_centroid"]) - prev_face))), 2)
                prev_face = np.array(row["change_centroid"]) - np.array(row["face_centroid"])
            ls_rows.append(row)
            if i in want:
                x1, y1, x2, y2 = bb if bb else (0, 0, fc.shape[1], fc.shape[0]); m = int(0.4 * (y2 - y1))
                cy1, cy2, cx1, cx2 = max(0, y1 - m), min(fc.shape[0], y2 + m), max(0, x1 - m), min(fc.shape[1], x2 + m)
                tiles = [fb[cy1:cy2, cx1:cx2], fc[cy1:cy2, cx1:cx2], cv2.cvtColor(np.clip(d[cy1:cy2, cx1:cx2] * 8, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]
                tiles = [cv2.resize(t, (260, int(260 * t.shape[0] / t.shape[1]))) for t in tiles]; sheet = np.concatenate(tiles, axis=1)
                cv2.putText(sheet, f"{a.id} {a.tag} f{i} baseline | candidate | diff x8", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
                p = fd / f"cf_face_{i:05d}.png"; cv2.imwrite(str(p), sheet); sheets.append(str(p))
        prev_c = fc; i += 1
    cc.release(); cb.release()
    rows = [r for r in ls_rows if "outside_fraction" in r]
    def agg(k, fn=np.mean):
        v = [r[k] for r in rows if k in r]; return round(float(fn(v)), 4) if v else None
    res.update(frames_compared=i, untouched_frames={"n": untouched_n, "max_abs_diff": untouched_max, "bit_identical": untouched_max == 0},
               ls_frames_measured=len(rows), ls_frames_changed=sum(1 for r in ls_rows if r["changed_px"] > 0),
               locality={"outside_dilated_face_px_max": agg("outside_dilated_face_px", np.max), "outside_fraction_mean": agg("outside_fraction"), "background_max_diff": agg("background_max_diff", np.max)},
               seam={"ratio_mean": agg("seam_ratio"), "ratio_p95": agg("seam_ratio", lambda v: np.percentile(v, 95)), "grad_cand_mean": agg("seam_grad_cand"), "grad_base_mean": agg("seam_grad_base")},
               sharpness={"upper_face_ratio_mean": agg("upper_sharp_ratio"), "mouth_ratio_mean": agg("mouth_sharp_ratio")},
               mask_motion={"centroid_rel_jitter_px_mean": agg("centroid_rel_jitter_px"), "p95": agg("centroid_rel_jitter_px", lambda v: np.percentile(v, 95))},
               transitions=len(trs), sheets=sheets)
    op = pathlib.Path(a.out) / f"{a.id}_{a.tag}.codeformer_quality.json"; op.parent.mkdir(parents=True, exist_ok=True); op.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("geometry", "untouched_frames", "ls_frames_measured", "ls_frames_changed", "locality", "seam", "sharpness", "mask_motion")}, indent=1))
    print(f"-> {op} ({len(sheets)} sheets in {fd})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
