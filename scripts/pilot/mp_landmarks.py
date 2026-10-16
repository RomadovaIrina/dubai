#!/usr/bin/env python3
"""MediaPipe Face Landmarker on every frame of a video -> npz (clean face stack, track A 2026-10-16).

Licence: mediapipe (Apache-2.0) + face_landmarker.task float16 v1 (BlazeFace short-range detector + Face Mesh V2, model cards
"LICENSED UNDER Apache License, Version 2.0"), sha256 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff.
Runs in its own env (/venv/mediapipe: mediapipe 0.10.21 needs protobuf 4.25 / opencv-contrib, which the pinned /venv/dabai must not get);
the project env only reads the npz.  VIDEO running mode = MediaPipe's own frame-to-frame tracking (default); --mode image = per frame.

    /venv/mediapipe/bin/python scripts/pilot/mp_landmarks.py VIDEO OUT.npz [--model models/mediapipe/face_landmarker.task]
npz: lm (T, 478, 2) pixel x, y (NaN = no face), z (T, 478) relative depth (pixel units), presence (T,) bool, wh (2,) frame size,
     matrix (T, 4, 4) facial transformation matrix (canonical face -> camera; head pose for spec 5.4 / 5.5).
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
    ap.add_argument("--num-faces", type=int, default=1, help="> 1: also store every face per frame (lm_all / matrix_all, unordered; spec 8.2 two faces)")
    a = ap.parse_args()
    mode = vision.RunningMode.VIDEO if a.mode == "video" else vision.RunningMode.IMAGE
    opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=a.model), running_mode=mode, num_faces=a.num_faces,
                                        min_face_detection_confidence=a.min_conf, min_face_presence_confidence=a.min_conf, min_tracking_confidence=a.min_conf,
                                        output_facial_transformation_matrixes=True)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lms, zs, pres, mats, alls, amats = [], [], [], [], [], []; t0 = time.perf_counter(); i = 0; H = W = 0; F = max(1, a.num_faces)
    with vision.FaceLandmarker.create_from_options(opts) as det:
        while True:
            ok, f = cap.read()
            if not ok: break
            H, W = f.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            r = det.detect_for_video(img, int(round(i * 1000 / fps))) if a.mode == "video" else det.detect(img)
            fa_ = np.full((F, 478, 2), np.nan, np.float32); fm_ = np.full((F, 4, 4), np.nan, np.float32)
            for q_, pts_ in enumerate(r.face_landmarks[:F]):
                fa_[q_] = np.array([[q.x * W, q.y * H] for q in pts_], np.float32)
                if r.facial_transformation_matrixes and q_ < len(r.facial_transformation_matrixes): fm_[q_] = np.asarray(r.facial_transformation_matrixes[q_], np.float32).reshape(4, 4)
            alls.append(fa_); amats.append(fm_)
            if r.face_landmarks:   # `lm` = the LARGEST face (single-face consumers)
                big = max(range(len(r.face_landmarks)), key=lambda q_: np.ptp(fa_[q_, :, 0]) * np.ptp(fa_[q_, :, 1]) if q_ < F else -1)
                p = r.face_landmarks[big]
                lms.append(np.array([[q.x * W, q.y * H] for q in p], np.float32)); zs.append(np.array([q.z * W for q in p], np.float32)); pres.append(True)
                m = r.facial_transformation_matrixes[big] if r.facial_transformation_matrixes and big < len(r.facial_transformation_matrixes) else None
                mats.append(np.asarray(m, np.float32).reshape(4, 4) if m is not None else np.full((4, 4), np.nan, np.float32))
            else:
                lms.append(np.full((478, 2), np.nan, np.float32)); zs.append(np.full(478, np.nan, np.float32)); pres.append(False)
                mats.append(np.full((4, 4), np.nan, np.float32))
            i += 1
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, lm=np.stack(lms), z=np.stack(zs), presence=np.array(pres), wh=np.array([W, H]), mode=a.mode, video=a.video, matrix=np.stack(mats), **({"lm_all": np.stack(alls), "matrix_all": np.stack(amats)} if F > 1 else {}),
                        model="face_landmarker.task float16 v1 (MediaPipe, Apache-2.0)")
    tmp.replace(out)
    print(f"{a.video}: {i} frames, face in {int(np.sum(pres))}, {time.perf_counter() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
