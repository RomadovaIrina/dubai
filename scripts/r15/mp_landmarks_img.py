#!/usr/bin/env python3
"""r15 Z: MediaPipe Face Landmarker in IMAGE running mode = an INDEPENDENT detection per frame (no temporal tracking / smoothing), for
measuring frame-to-frame lip-contour jitter (the VIDEO-mode pass of scripts/quality/mp_landmarks.py smooths it away). Same output as
mp_landmarks.py. Runs in /venv/mediapipe.
    /venv/mediapipe/bin/python scripts/r15/mp_landmarks_img.py --video v.mp4 --out v.mp_lm_img.npz"""
import argparse, pathlib, time
import numpy as np

MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "face_landmarker.task"


def main():
    import cv2, mediapipe as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True); ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args()
    opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)), running_mode=mp.tasks.vision.RunningMode.IMAGE,
                                                 num_faces=1, min_face_detection_confidence=a.min_conf, min_face_presence_confidence=a.min_conf)
    lmk = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(a.video); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); LM, OK = [], []; t0 = time.perf_counter()
    while True:
        r, f = cap.read()
        if not r: break
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if res.face_landmarks:
            p = res.face_landmarks[0]; LM.append(np.array([[q.x * w, q.y * h, q.z * w] for q in p], np.float32)); OK.append(True)
        else:
            LM.append(np.full((478, 3), np.nan, np.float32)); OK.append(False)
    cap.release(); lmk.close()
    np.savez_compressed(a.out, lm=np.stack(LM), ok=np.array(OK), frames=np.arange(len(LM)), size=np.array([w, h]))
    print(f"{a.video}: {len(LM)} frames, face in {int(np.sum(OK))}, {time.perf_counter() - t0:.1f} s (IMAGE mode) -> {a.out}")


if __name__ == "__main__":
    main()
