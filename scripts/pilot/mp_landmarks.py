#!/usr/bin/env python3
"""MediaPipe Face Landmarker on every frame of a video -> npz (clean face stack, track A 2026-10-16).

Licence: mediapipe (Apache-2.0) + face_landmarker.task float16 v1 (BlazeFace short-range detector + Face Mesh V2, model cards
"LICENSED UNDER Apache License, Version 2.0"), sha256 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff.
Runs in its own env (/venv/mediapipe: mediapipe 0.10.21 needs protobuf 4.25 / opencv-contrib, which the pinned /venv/dabai must not get);
the project env only reads the npz.  VIDEO running mode = MediaPipe's own frame-to-frame tracking (default); --mode image = per frame.

    /venv/mediapipe/bin/python scripts/pilot/mp_landmarks.py VIDEO OUT.npz [--model models/mediapipe/face_landmarker.task]
npz: lm (T, 478, 2) pixel x, y (NaN = no face), z (T, 478) relative depth (pixel units), presence (T,) bool, wh (2,) frame size.
"""
from __future__ import annotations
import argparse, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--model", default=str(ROOT / "models/mediapipe/face_landmarker.task"))
    ap.add_argument("--mode", choices=["video", "image"], default="video")
    ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args()
    mode = vision.RunningMode.VIDEO if a.mode == "video" else vision.RunningMode.IMAGE
    opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=a.model), running_mode=mode, num_faces=1,
                                        min_face_detection_confidence=a.min_conf, min_face_presence_confidence=a.min_conf, min_tracking_confidence=a.min_conf)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lms, zs, pres = [], [], []; t0 = time.perf_counter(); i = 0; H = W = 0
    with vision.FaceLandmarker.create_from_options(opts) as det:
        while True:
            ok, f = cap.read()
            if not ok: break
            H, W = f.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            r = det.detect_for_video(img, int(round(i * 1000 / fps))) if a.mode == "video" else det.detect(img)
            if r.face_landmarks:
                p = r.face_landmarks[0]
                lms.append(np.array([[q.x * W, q.y * H] for q in p], np.float32)); zs.append(np.array([q.z * W for q in p], np.float32)); pres.append(True)
            else:
                lms.append(np.full((478, 2), np.nan, np.float32)); zs.append(np.full(478, np.nan, np.float32)); pres.append(False)
            i += 1
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, lm=np.stack(lms), z=np.stack(zs), presence=np.array(pres), wh=np.array([W, H]), mode=a.mode, video=a.video,
                        model="face_landmarker.task float16 v1 (MediaPipe, Apache-2.0)")
    tmp.replace(out)
    print(f"{a.video}: {i} frames, face in {int(np.sum(pres))}, {time.perf_counter() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
