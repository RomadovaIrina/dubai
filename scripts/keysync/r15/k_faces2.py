#!/usr/bin/env python3
"""r15 K: the OTHER faces of a multi-face source (m03: the child next to the mother) for the paste-side face clamp.
MediaPipe Face Landmarker (same model / VIDEO mode as scripts/quality/mp_landmarks.py) with num_faces=3. The speaker = the detected face whose
landmark centroid is nearest the speaker track (--speaker-lm, the 1-face npz the KeySync crop was made from); the rest = 'others'.
npz: others (T, 2, 478, 2) float32 pixels, NaN where absent; speaker (T, 478, 2) from the multi-face pass (NaN if not matched).
    /venv/mediapipe/bin/python k_faces2.py --video SRC --speaker-lm src.mediapipe.npz --out faces2.npz"""
import argparse, pathlib, time
import numpy as np
import cv2, mediapipe as mp

MODEL = pathlib.Path("/workspace/dub/dubai/models/mediapipe/face_landmarker.task")
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--speaker-lm", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
SP = np.load(a.speaker_lm)["lm"][..., :2].astype(np.float32)
opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)), running_mode=mp.tasks.vision.RunningMode.VIDEO,
                                             num_faces=3, min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4)
lmk = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
O, S = [], []; i = 0; t0 = time.perf_counter(); nfo = 0
while True:
    ok, f = cap.read()
    if not ok: break
    res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(round(i * 1000 / fps)))
    faces = [np.array([[q.x * w, q.y * h] for q in p], np.float32) for p in (res.face_landmarks or [])]
    o = np.full((2, 478, 2), np.nan, np.float32); s = np.full((478, 2), np.nan, np.float32)
    if faces:
        ref = SP[min(i, len(SP) - 1)]
        if np.isfinite(ref).all():
            dd = [float(np.linalg.norm(fc.mean(0) - ref.mean(0))) for fc in faces]; k = int(np.argmin(dd)); s = faces[k]
            rest = [fc for j, fc in enumerate(faces) if j != k]
        else:
            rest = faces
        for j, fc in enumerate(rest[:2]): o[j] = fc
        nfo += bool(rest)
    O.append(o); S.append(s); i += 1
cap.release(); lmk.close()
np.savez_compressed(a.out, others=np.stack(O), speaker=np.stack(S), size=np.array([w, h]))
print(f"{a.video}: {i} frames, another face in {nfo}, {time.perf_counter() - t0:.0f} s -> {a.out}")
