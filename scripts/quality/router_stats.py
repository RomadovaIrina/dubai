#!/usr/bin/env python3
"""Borderline-face statistics for the RetinaFace router (Phase 3.3 evidence, no pipeline change).

Runs facexlib RetinaFace on EVERY master frame (conf_threshold 0.02, like the router) and records the best detection
(score, w, h). Reports how many frames flip class under alternative thresholds vs the baseline (min_w 50, min_h 80, conf 0.8)
and writes a contact sheet of borderline frames (score in [0.5, 0.8) or height in [60, 80)).
    python scripts/quality/router_stats.py --ids 01 02 03 04
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
Q = pathlib.Path("/tmp/dabai_quality"); FPS = 25
ALTS = [(50, 80, 0.8), (50, 80, 0.7), (50, 80, 0.6), (50, 80, 0.5), (50, 70, 0.8), (50, 60, 0.8), (40, 60, 0.6)]


def classify(best, min_w, min_h, conf):
    if best is None: return "NO_FACE"
    s, w, h = best
    if s < conf: return "NO_FACE"
    if w < min_w or h < min_h or not (0.2 < w / max(h, 1) < 1.5): return "SMALL_FACE"
    return "VALID_FACE"


def main() -> int:
    import cv2
    from retinaface_router import RetinaFaceRouter
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--work-dir", default=str(Q / "work"))
    ap.add_argument("--out", default=str(Q / "manifests")); ap.add_argument("--frames", default=str(Q / "frames")); a = ap.parse_args()
    router = RetinaFaceRouter(50, 80, 0.8)
    for vid in a.ids:
        master = pathlib.Path(a.work_dir) / f"work_{vid}" / "master_25fps.mp4"; cap = cv2.VideoCapture(str(master)); rows = []; keep = {}; i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            import torch
            with torch.no_grad(): dets = router.det.detect_faces(fr, conf_threshold=0.02)
            best = None
            if len(dets):
                d = max(dets, key=lambda d: float(d[4])); best = (round(float(d[4]), 3), int(d[2] - d[0]), int(d[3] - d[1]))
            rows.append(best)
            if best and ((0.5 <= best[0] < 0.8) or (best[0] >= 0.8 and 60 <= best[2] < 80)) and len(keep) < 200 and i % 5 == 0: keep[i] = fr
            i += 1
        cap.release()
        n = len(rows); base = [classify(b, 50, 80, 0.8) for b in rows]
        rep = {"video": vid, "frames": n, "baseline_classes": {c: base.count(c) for c in ("VALID_FACE", "SMALL_FACE", "NO_FACE")},
               "score_hist": {}, "height_hist": {}, "alternatives": []}
        scores = np.array([b[0] if b else 0.0 for b in rows]); hs = np.array([b[2] if b else 0 for b in rows])
        for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)): rep["score_hist"][f"{lo}-{hi}"] = int(((scores >= lo) & (scores < hi)).sum())
        for lo, hi in ((0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100), (100, 150), (150, 10000)): rep["height_hist"][f"{lo}-{hi}"] = int(((hs >= lo) & (hs < hi) & (scores >= 0.5)).sum())
        for mw, mh, cf in ALTS:
            alt = [classify(b, mw, mh, cf) for b in rows]
            rep["alternatives"].append({"min_w": mw, "min_h": mh, "conf": cf, "classes": {c: alt.count(c) for c in ("VALID_FACE", "SMALL_FACE", "NO_FACE")},
                                        "valid_gained_vs_baseline": sum(1 for x, y in zip(base, alt) if y == "VALID_FACE" and x != "VALID_FACE")})
        borderline = [i for i, b in enumerate(rows) if b and ((0.5 <= b[0] < 0.8) or (b[0] >= 0.8 and 60 <= b[2] < 80))]
        runs = []
        for f in borderline:
            if runs and f - runs[-1][1] <= 2: runs[-1][1] = f
            else: runs.append([f, f])
        rep["borderline_frames"] = len(borderline); rep["borderline_runs_s"] = [[round(s / FPS, 2), round((e + 1) / FPS, 2)] for s, e in runs if e - s >= 5][:40]
        if keep:
            ks = sorted(keep)[:24]; tiles = []
            for f in ks:
                t = cv2.resize(keep[f], None, fx=0.3, fy=0.3); b = rows[f]
                cv2.rectangle(t, (0, 0), (t.shape[1], 18), (0, 0, 0), -1); cv2.putText(t, f"f{f} s{b[0]} h{b[2]}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1); tiles.append(t)
            h, w = tiles[0].shape[:2]; cols = 8; rws = (len(tiles) + cols - 1) // cols; canvas = np.zeros((rws * h, cols * w, 3), np.uint8)
            for k, t in enumerate(tiles): r, c = divmod(k, cols); canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
            p = pathlib.Path(a.frames) / f"{vid}_baseline" / "router_borderline.png"; p.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(p), canvas); rep["sheet"] = str(p)
        (pathlib.Path(a.out) / f"{vid}_router_stats.json").write_text(json.dumps(rep, indent=1))
        print(f"== {vid}: {n} frames baseline {rep['baseline_classes']} | scores {rep['score_hist']} | heights(score>=.5) {rep['height_hist']}")
        for alt in rep["alternatives"]: print(f"   min_w {alt['min_w']} min_h {alt['min_h']} conf {alt['conf']}: {alt['classes']} (+{alt['valid_gained_vs_baseline']} VALID vs baseline)")
        print(f"   borderline frames {rep['borderline_frames']}, runs>=0.2s: {rep['borderline_runs_s'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
