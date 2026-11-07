#!/usr/bin/env python3
"""r16 Q: per-face beard / stubble score -> face_engine = keysync | latentsync (DUB_FACE=auto routing; KeySync is weak on beards).

Each face is measured on ~N sampled source frames (MediaPipe FaceLandmarker, image mode, up to 4 faces). Each face is rescaled
so the eye distance (IOD) is 80 px. Regions:
  beard zone = the jaw oval below the lower lip, minus the lips ring; moustache = the strip between the nose base and the upper lip
  reference skin = both upper cheeks (under the eyes) + the forehead
  (beard zone = inside the jaw outline eroded off its edge, from the lower lip to 3/4 of the way to the chin: the shadowed
   underside of the chin made smooth chins look dark)
Features (medians over frames):
  speck = share of beard-zone pixels darker than their own neighbourhood (L* < blur_sigma4 - 3: hair strands / stubble dots,
          not smooth shading) / the same share on the reference skin   <- the main feature
  dark  = share of beard-zone pixels with L* < ref_L* - 18 (a full dark beard)
  tex, must, speck_abs: logged only (tex and must confuse wrinkles / lipstick shadow with hair)
beard = (speck >= 2.0 and native IOD >= 50 px) or dark >= 0.5 -> face_engine latentsync, else keysync.
Calibration 07.11: stubble old03 2.9, beards 2.5-16.6 vs. none m03 0.0 / m18 0.7 / q02 1.6 (wrinkles) / raw04 0.3;
q08 (480p, IOD < 50) 3.2 = compression noise -> the resolution guard.
CPU only. Usage: /venv/mediapipe/bin/python r16q_beard.py VIDEO [--n 40] [--out beard.json]"""
import argparse, json, os, pathlib, sys
import cv2, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPECK_THR = float(os.environ.get("BEARD_SPECK", "2.0")); DARK_THR = float(os.environ.get("BEARD_DARK", "0.5"))
IOD_MIN = float(os.environ.get("BEARD_IOD_MIN", "50"))
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109]
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]   # ear to ear along the jaw
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
MOUST = [98, 97, 2, 326, 327, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
CHEEK_L, CHEEK_R, FORE = [117, 118, 101, 36, 205, 187, 123], [346, 347, 330, 266, 425, 411, 352], [109, 10, 338, 337, 151, 108]
EYE_L, EYE_R = [33, 133], [362, 263]

ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", default="")
ap.add_argument("--model", default=str(ROOT / "models/mediapipe/face_landmarker.task")); ap.add_argument("--debug", default="")
a = ap.parse_args()

import mediapipe as mp
lmk = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=a.model), running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=4,
    min_face_detection_confidence=0.5))
cap = cv2.VideoCapture(a.video); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(3)); H = int(cap.get(4))


def poly_mask(shape, pts):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1); return m.astype(bool)


def measure(img, lm):
    eL, eR = lm[EYE_L].mean(0), lm[EYE_R].mean(0); iod = np.linalg.norm(eR - eL)
    if iod < 20: return None   # face too small to see hair texture
    yaw = abs((np.linalg.norm(lm[1] - lm[234]) - np.linalg.norm(lm[1] - lm[454])) / (np.linalg.norm(lm[234] - lm[454]) + 1e-6))
    if yaw > 0.35: return None   # strong profile: one side unmeasurable
    s = 80.0 / iod; x0, y0 = lm[OVAL].min(0) - 10 / s; x1, y1 = lm[OVAL].max(0) + 10 / s
    x0, y0 = max(int(x0), 0), max(int(y0), 0); x1, y1 = min(int(x1), img.shape[1]), min(int(y1), img.shape[0])
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    p = (lm - [x0, y0]) * s; L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * 100 / 255
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(L, (3, 3), 0), cv2.CV_32F, ksize=3))
    sh = L.shape
    hull = cv2.erode(poly_mask(sh, p[JAW]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)   # off the jaw edge
    u, c = float(p[LOWER_LIP][:, 1].max()), float(p[152, 1]); yy = np.arange(sh[0])[:, None]
    band = (yy > u + 0.08 * (c - u)) & (yy < u + 0.75 * (c - u))   # lip to 3/4 of the chin: the shadowed chin underside excluded
    lips = cv2.dilate(poly_mask(sh, p[LIPS]).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    beard = hull & band & ~lips
    must = poly_mask(sh, p[MOUST]) & ~lips
    ref = poly_mask(sh, p[CHEEK_L]) | poly_mask(sh, p[CHEEK_R]) | poly_mask(sh, p[FORE])
    if beard.sum() < 200 or ref.sum() < 200: return None
    refL = float(np.median(L[ref]))
    if refL < 12: return None   # too dark to judge
    spk = L < cv2.GaussianBlur(L, (0, 0), 4) - 3.0   # darker than its neighbourhood: hair strands / stubble dots, not smooth shading
    dark = float((L[beard] < refL - 18).mean()); mdark = float((L[must] < refL - 18).mean()) if must.sum() > 50 else 0.0
    tex = float(lap[beard].mean() / max(lap[ref].mean(), 0.3))
    speck = float(spk[beard].mean() / max(spk[ref].mean(), 0.01)); speck_abs = float(spk[beard].mean())
    if a.debug:
        dbg = crop.copy(); dbg[beard] = (dbg[beard] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8); dbg[beard & spk] = (0, 0, 255)
        dbg[ref] = (dbg[ref] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        measure.dbg = dbg
    return {"dark": dark, "must": mdark, "tex": tex, "speck": speck, "speck_abs": speck_abs, "refL": refL, "c": lm[OVAL].mean(0), "iod": float(iod)}


faces = []   # [{c, rows}]
idx = np.linspace(0, max(N - 1, 0), a.n).astype(int)
for fi in idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, img = cap.read()
    if not ok: continue
    r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    for fl in r.face_landmarks:
        lm = np.array([[q.x * img.shape[1], q.y * img.shape[0]] for q in fl], np.float32)
        m = measure(img, lm)
        if m is None: continue
        if a.debug and len(faces) == 0 and not os.path.exists(a.debug): cv2.imwrite(a.debug, measure.dbg)
        best = min(faces, key=lambda f: np.linalg.norm(f["c"] - m["c"]), default=None)
        if best is not None and np.linalg.norm(best["c"] - m["c"]) < 1.5 * m["iod"] * 2.2: best["rows"].append(m); best["c"] = 0.7 * best["c"] + 0.3 * m["c"]
        else: faces.append({"c": m["c"], "rows": [m]})
out = {"video": a.video, "size": [W, H], "sampled": int(len(idx)), "thr": {"speck": SPECK_THR, "dark": DARK_THR, "iod_min": IOD_MIN}, "faces": []}
for k, f in enumerate(sorted(faces, key=lambda f: -len(f["rows"]))):
    if len(f["rows"]) < max(3, a.n // 10): continue
    med = {q: float(np.median([r[q] for r in f["rows"]])) for q in ("dark", "must", "tex", "speck", "speck_abs", "refL", "iod")}
    sharp = med["iod"] >= IOD_MIN   # below this the upscaled texture is compression noise (q08 480p)
    sc = med["speck"] / SPECK_THR if sharp else 0.0   # >= 1 = hair texture
    beard = (sharp and med["speck"] >= SPECK_THR) or med["dark"] >= DARK_THR
    out["faces"].append({"face": k, "center_xy": [round(float(x), 1) for x in f["c"]], "frames": len(f["rows"]),
                         **{q: round(v, 3) for q, v in med.items()}, "texture_reliable": bool(sharp), "beard_score": round(sc, 3),
                         "beard": bool(beard), "face_engine": "latentsync" if beard else "keysync"})
if a.out: json.dump(out, open(a.out, "w"), indent=1)
print("BEARD " + pathlib.Path(a.video).stem + " " + json.dumps([{k: f[k] for k in ("face", "frames", "iod", "dark", "speck", "texture_reliable", "beard", "face_engine")} for f in out["faces"]]))
