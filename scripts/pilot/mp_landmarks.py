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
    ap.add_argument("--every", type=int, default=1, help="process only every N-th frame (others NaN); a cheap probe, use with --mode image")
    ap.add_argument("--num-faces", type=int, default=1, help="> 1: also store every face per frame (lm_all / matrix_all, unordered; spec 8.2 two faces)")
    ap.add_argument("--tiles", type=int, default=0, help="r19 (01 closing shot, 05 inset: a 54-70 px face in a tall 480x848 frame is missed, the detector letterboxes the frame to a square): 1 = also run the detector on square tiles (side = the short frame side, ~50 %% overlap along the long side), each with its own tracker; a tile face is added only where the whole-frame pass has none")
    a = ap.parse_args()
    mode = vision.RunningMode.VIDEO if a.mode == "video" else vision.RunningMode.IMAGE
    opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=a.model), running_mode=mode, num_faces=a.num_faces,
                                        min_face_detection_confidence=a.min_conf, min_face_presence_confidence=a.min_conf, min_tracking_confidence=a.min_conf,
                                        output_facial_transformation_matrixes=True)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    NTILE = [None]; lms, zs, pres, mats, alls, amats = [], [], [], [], [], []; t0 = time.perf_counter(); i = 0; H = W = 0; F = max(1, a.num_faces)
    TD = []   # r19 --tiles: [(detector, x0, y0, side)], made at the first frame
    with vision.FaceLandmarker.create_from_options(opts) as det:
        while True:
            ok, f = cap.read()
            if not ok: break
            H, W = f.shape[:2]
            if a.every > 1 and i % a.every:
                alls.append(np.full((F, 478, 2), np.nan, np.float32)); amats.append(np.full((F, 4, 4), np.nan, np.float32))
                lms.append(np.full((478, 2), np.nan, np.float32)); zs.append(np.full(478, np.nan, np.float32)); pres.append(False)
                mats.append(np.full((4, 4), np.nan, np.float32)); i += 1; continue
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            ts_ = int(round(i * 1000 / fps)); r = det.detect_for_video(img, ts_) if a.mode == "video" else det.detect(img)
            FC = []   # every face of this frame: (xy px (478, 2), z px (478,), matrix or None)
            for q_, pts_ in enumerate(r.face_landmarks):
                m_ = r.facial_transformation_matrixes[q_] if r.facial_transformation_matrixes and q_ < len(r.facial_transformation_matrixes) else None
                FC.append((np.array([[q.x * W, q.y * H] for q in pts_], np.float32), np.array([q.z * W for q in pts_], np.float32), m_))
            if a.tiles and max(W, H) >= 1.3 * min(W, H):
                if not TD:
                    S_ = min(W, H); L_ = max(W, H); nt = int(np.ceil((L_ - S_) / (S_ / 2))) + 1
                    for o_ in np.round(np.linspace(0, L_ - S_, nt)).astype(int): TD.append((vision.FaceLandmarker.create_from_options(opts), 0 if H > W else int(o_), int(o_) if H > W else 0, S_))
                    NTILE[0] = 0
                rgb_ = img.numpy_view()
                for dt, x0, y0, S_ in TD:
                    ti = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_[y0:y0 + S_, x0:x0 + S_]))
                    rt = dt.detect_for_video(ti, ts_) if a.mode == "video" else dt.detect(ti)
                    for q_, pts_ in enumerate(rt.face_landmarks):
                        xy = np.array([[q.x * S_ + x0, q.y * S_ + y0] for q in pts_], np.float32); lo, hi = xy.min(0), xy.max(0); c_ = (lo + hi) / 2; ax = 1 if H > W else 0; o0 = y0 if H > W else x0
                        if (lo[ax] < o0 + 0.03 * S_ and o0 > 0) or (hi[ax] > o0 + 0.97 * S_ and o0 + S_ < max(W, H)): continue   # cut by an inner tile border: the next tile has it whole
                        m_ = rt.facial_transformation_matrixes[q_] if rt.facial_transformation_matrixes and q_ < len(rt.facial_transformation_matrixes) else None
                        new = (xy, np.array([q.z * S_ for q in pts_], np.float32), m_, "tile")
                        dup = [j_ for j_, g in enumerate(FC) if np.linalg.norm(c_ - (g[0].min(0) + g[0].max(0)) / 2) < 0.5 * max(np.ptp(g[0][:, 0]), np.ptp(xy[:, 0]))]
                        if not dup: FC.append(new); NTILE[0] += 1   # a face the whole-frame pass missed
                        elif len(FC[dup[0]]) == 3 and np.ptp(xy[:, 0]) < 0.25 * S_: FC[dup[0]] = new; NTILE[0] += 1   # a SMALL face found by both: the tile's landmarks (more pixels, and one source over time, no jumps between the two)
            fa_ = np.full((F, 478, 2), np.nan, np.float32); fm_ = np.full((F, 4, 4), np.nan, np.float32)
            for q_, (xy, _, m_, *_r) in enumerate(FC[:F]):
                fa_[q_] = xy
                if m_ is not None: fm_[q_] = np.asarray(m_, np.float32).reshape(4, 4)
            alls.append(fa_); amats.append(fm_)
            if FC:   # `lm` = the LARGEST face (single-face consumers)
                big = max(range(min(len(FC), F)), key=lambda q_: np.ptp(fa_[q_, :, 0]) * np.ptp(fa_[q_, :, 1]))
                lms.append(FC[big][0]); zs.append(FC[big][1]); pres.append(True); m = FC[big][2]
                mats.append(np.asarray(m, np.float32).reshape(4, 4) if m is not None else np.full((4, 4), np.nan, np.float32))
            else:
                lms.append(np.full((478, 2), np.nan, np.float32)); zs.append(np.full(478, np.nan, np.float32)); pres.append(False)
                mats.append(np.full((4, 4), np.nan, np.float32))
            i += 1
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, lm=np.stack(lms), z=np.stack(zs), presence=np.array(pres), wh=np.array([W, H]), mode=a.mode, video=a.video, matrix=np.stack(mats), **({"lm_all": np.stack(alls), "matrix_all": np.stack(amats)} if F > 1 else {}),
                        model="face_landmarker.task float16 v1 (MediaPipe, Apache-2.0)")
    tmp.replace(out)
    print(f"{a.video}: {i} frames, face in {int(np.sum(pres))}, {time.perf_counter() - t0:.1f}s -> {out}" + (f" (tiles {len(TD)}: {NTILE[0]} faces added)" if NTILE[0] is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
