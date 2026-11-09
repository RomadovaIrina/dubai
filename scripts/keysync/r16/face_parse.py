#!/usr/bin/env python3
"""r16c K1: ONE face-parsing module for the paste mask (K1) and the occlusion guard (T).
Model: MediaPipe selfie multiclass segmenter (selfie_multiclass_256x256.tflite, Apache-2.0, Google): per pixel
  0 background, 1 hair, 2 body skin (neck, HANDS, arms), 3 face skin, 4 clothes, 5 others (accessories, cups, objects).
No CelebAMask-HQ / NC weights.

CLI (needs mediapipe; run with the mediapipe venv):
    python face_parse.py --video master_25fps.mp4 --box box.json --out parse.npz [--model selfie_multiclass_256x256.tflite] [--res 128]
  -> parse.npz: prob (n, 6, res, res) uint8 (0..255) on the paste box (static x / y / side, or the tracked xs / ys), box, res, classes

Library (numpy + cv2 only; ks_plain.py / the e2e guard import it from the same dir):
    P = load("parse.npz")
    face_prob(P, k, s, beard_y=None) -> float32 (s, s): P(face) on the box at master frame k; hair BELOW beard_y (box px, e.g. the
        nose tip) counts as face (a beard is "hair" to the model, but it is part of the lower face the generator must own)
    occlusion(P, k, s, poly) -> share of the polygon (box px, e.g. the mouth / lower-face hull) covered by hands (body skin),
        objects (others) or clothes: the per-frame occlusion score for T's guard (0 = clear, 1 = fully covered)"""
import json, os, subprocess, sys
import numpy as np, cv2

CLASSES = ["background", "hair", "body_skin", "face_skin", "clothes", "others"]
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"


def load(path):
    d = np.load(path); return {"prob": d["prob"], "res": int(d["res"])}


def _probs(P, k, half=None):   # r16c K1: temporally smoothed (triangular +-half frames; env FACE_PARSE_T, default 2) -> no frame-to-frame flicker of the parse mask
    pr = P["prob"]
    if k < 0 or k >= len(pr): return None
    h = int(os.environ.get("FACE_PARSE_T", "2")) if half is None else half
    if h <= 0: return pr[k].astype(np.float32) / 255.0
    js = range(max(0, k - h), min(len(pr), k + h + 1)); ws = np.array([h + 1 - abs(j - k) for j in js], np.float32)
    return np.tensordot(ws / ws.sum(), pr[list(js)].astype(np.float32), 1) / 255.0


def face_prob(P, k, s, beard_y=None):
    p = _probs(P, k)
    if p is None: return None
    r = P["res"]; f = p[3].copy()
    if beard_y is not None:
        yy = (np.arange(r, dtype=np.float32)[:, None] + 0.5) * s / r
        f = f + p[1] * (yy > beard_y)
    return cv2.resize(np.clip(f, 0, 1), (s, s), interpolation=cv2.INTER_LINEAR)


def occlusion(P, k, s, poly):
    p = _probs(P, k)
    if p is None: return float("nan")
    m = np.zeros((s, s), np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    if m.sum() == 0: return float("nan")
    occ = cv2.resize(np.clip(p[2] + p[4] + p[5], 0, 1), (s, s), interpolation=cv2.INTER_LINEAR)
    return float((occ * m).sum() / m.sum())


def _cli():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--box", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfie_multiclass_256x256.tflite")); ap.add_argument("--res", type=int, default=128)
    a = ap.parse_args()
    if not os.path.exists(a.model):
        print(f"face_parse: model missing ({a.model}); get it from {MODEL_URL} (Apache-2.0)", file=sys.stderr); sys.exit(2)
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=a.model),
                                                    output_confidence_masks=True, output_category_mask=False, running_mode=vision.RunningMode.IMAGE))
    b = json.load(open(a.box)); W, H, s = b["W"], b["H"], b["side"]; xs = b.get("xs"); ys = b.get("ys")
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out = []; k = 0
    while True:
        buf = pr.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        x0 = int(xs[min(k, len(xs) - 1)]) if xs else b["x"]; y0 = int(ys[min(k, len(ys) - 1)]) if ys else b["y"]
        c = np.ascontiguousarray(f[y0:y0 + s, x0:x0 + s])
        r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=c))
        pm = np.stack([cv2.resize(m.numpy_view(), (a.res, a.res), interpolation=cv2.INTER_AREA) for m in r.confidence_masks], 0)
        out.append(np.clip(pm * 255 + 0.5, 0, 255).astype(np.uint8)); k += 1
    P = np.stack(out, 0)
    np.savez_compressed(a.out, prob=P, res=a.res, box=json.dumps(b), classes=np.array(CLASSES))
    mean = P.reshape(len(P), 6, -1).mean((0, 2)) / 255
    print(f"face_parse: {len(P)} frames, box {s} px -> {a.res}; mean class share " + " ".join(f"{c} {v:.2f}" for c, v in zip(CLASSES, mean)), flush=True)


if __name__ == "__main__":
    _cli()
