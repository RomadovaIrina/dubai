#!/usr/bin/env python3
"""Clean face landmarks for the paste-back / mask / texture-transfer stages: MediaPipe Face Landmarker (478 points, Apache-2.0 code and
model `face_landmarker.task`), VIDEO running mode (temporal tracking). Runs in its OWN venv (/venv/mediapipe: mediapipe pulls
opencv-contrib, which must not shadow the pinned cv2 of /venv/dabai); the product path reads the .npz it writes.

    /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video master_25fps.mp4 --out lm.npz [--frames a:b]
npz: lm (T, 478, 3) float32 in pixels (x, y) and MediaPipe's relative depth z * width; ok (T,) bool; frames (T,) int; size (w, h).
"""
from __future__ import annotations
import argparse, pathlib, time
import numpy as np

MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "face_landmarker.task"
URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def main() -> int:
    import cv2, mediapipe as mp
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True); ap.add_argument("--frames", default=None)
    ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args()
    if not MODEL.exists():
        import urllib.request; MODEL.parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(URL, MODEL)
    opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)), running_mode=mp.tasks.vision.RunningMode.VIDEO,
                                                 num_faces=1, min_face_detection_confidence=a.min_conf, min_face_presence_confidence=a.min_conf, min_tracking_confidence=a.min_conf)
    lmk = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    A, B = (0, 10 ** 9) if not a.frames else [int(x) for x in a.frames.split(":")]
    LM, OK, FR = [], [], []; i = 0; t0 = time.perf_counter()
    while True:
        ok, f = cap.read()
        if not ok or i >= B: break
        if i >= A:
            res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(round(i * 1000 / fps)))
            if res.face_landmarks:
                p = res.face_landmarks[0]; LM.append(np.array([[q.x * w, q.y * h, q.z * w] for q in p], np.float32)); OK.append(True)
            else:
                LM.append(np.full((478, 3), np.nan, np.float32)); OK.append(False)
            FR.append(i)
        i += 1
    cap.release(); lmk.close()
    np.savez_compressed(a.out, lm=np.stack(LM), ok=np.array(OK), frames=np.array(FR), size=np.array([w, h]))
    print(f"{a.video}: {len(FR)} frames, face in {int(np.sum(OK))}, {time.perf_counter() - t0:.1f} s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
