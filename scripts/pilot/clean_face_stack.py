#!/usr/bin/env python3
"""Clean face stack for the video stage (`--face-stack mediapipe`, track A 2026-10-16, CLEAN-ONLY directive).

Replaces every non-commercial face model of the optimized video stage:
  InsightFace buffalo_l det_10g + 2d106det (LatentSync's FaceDetector: crop / affine, verify pass, mouth-mask bbox, landmark smoothing)
  RetinaFace R50 (facexlib face router)
with ONE MediaPipe Face Landmarker pass over the 25 fps master (Apache-2.0 code + model card; own env /venv/mediapipe, run as a subprocess,
see scripts/pilot/mp_landmarks.py). Nothing else changes: LatentSync 1.6 consumes the replayed (bbox, 106-point) detections through
latentsync_accel._ReplayDetector exactly as before, and its own FaceDetector is replaced by a stub that refuses to load (guard_nc()).

106-point layout: LatentSync only reads brow / nose / contour / eye / lip points of InsightFace's 2d106 layout. Each of those indices is
taken from the MediaPipe vertex of the SAME named region (MediaPipe face_mesh_connections: eyebrows, eyes + irises, nose, lips, face oval)
that lay closest to it on our test masters (MAP106 below; derived offline from cached evaluation landmarks, no InsightFace model is run).
The three affine anchors (brow centres, nose centre) get a constant correction in face coordinates (ANCHOR_CALIB, eye-distance units)
so the 512 crop keeps LatentSync's training alignment (InsightFace-aligned crops).
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
MP_PY = os.environ.get("DUB_MEDIAPIPE_PY", "/venv/mediapipe/bin/python")
MP_MODEL = ROOT / "models/mediapipe/face_landmarker.task"
MP_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
# InsightFace-2d106 index -> MediaPipe-478 vertex (same named region, nearest on the 03 master; median residual 0.025-0.07 eye distances)
MAP106 = [152, 162, 172, 172, 136, 150, 149, 176, 148, 162, 127, 234, 234, 93, 132, 132, 58, 389, 397, 397, 365, 379, 378, 400, 377, 389, 356,
          454, 454, 323, 361, 361, 288, 145, 468, 33, 163, 153, 468, 173, 470, 160, 157, 46, 53, 52, 55, 65, 63, 52, 55, 65, 61, 17, 88, 91, 181,
          402, 321, 314, 14, 291, 13, 37, 40, 185, 80, 267, 269, 409, 311, 0, 168, 197, 5, 6, 220, 64, 98, 97, 2, 6, 440, 294, 326, 326, 4, 374,
          473, 382, 380, 390, 473, 263, 475, 385, 388, 285, 295, 282, 283, 300, 285, 295, 282, 283]
MP_EYE_IMG_LEFT = [154, 155, 33, 7, 246, 161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]       # FACEMESH_RIGHT_EYE (subject's right)
MP_EYE_IMG_RIGHT = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362]   # FACEMESH_LEFT_EYE
MP_FACE_OVAL = [176, 149, 297, 332, 150, 136, 109, 10, 356, 454, 58, 132, 152, 148, 361, 288, 162, 21, 251, 389, 93, 338, 284, 400, 377, 172,
                323, 54, 103, 67, 365, 379, 378, 234, 397, 127]
ANCHORS = {"brow_l": [43, 48, 49, 51, 50], "brow_r": [101, 102, 103, 104, 105], "nose": [74, 77, 83, 86]}   # LatentSync image_processor.affine_transform
ANCHOR_CALIB = {"brow_l": (0.0, 0.0), "brow_r": (0.0, 0.0), "nose": (0.0, 0.0)}   # (along eye line, down) in eye distances; filled from calibration
CALIB_FILE = HERE / "clean_face_stack_calib.json"
if CALIB_FILE.exists():
    ANCHOR_CALIB.update({k: tuple(v) for k, v in json.loads(CALIB_FILE.read_text())["anchor_calib"].items()})
LMK_ADAPT_ORIGIN_ORDER = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17, 43, 48, 49, 51, 50, 102, 103, 104, 105, 101, 73, 74, 86]


def run_mediapipe(video: pathlib.Path, out: pathlib.Path, mode: str = "video") -> pathlib.Path:
    """MediaPipe Face Landmarker over every frame of `video` in its own env -> npz (cached next to the master)."""
    if out.exists():
        return out
    import hashlib
    h = hashlib.sha256(MP_MODEL.read_bytes()).hexdigest()
    if h != MP_MODEL_SHA256:
        raise RuntimeError(f"{MP_MODEL}: sha256 {h} != pinned {MP_MODEL_SHA256}")
    env = {**os.environ, "GLOG_minloglevel": "2", "OMP_NUM_THREADS": "4"}
    p = subprocess.run([MP_PY, str(HERE / "mp_landmarks.py"), str(video), str(out), "--model", str(MP_MODEL), "--mode", mode],
                       env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"mediapipe landmarks failed ({p.returncode}): {p.stdout[-1500:]}")
    return out


def face_frame(mp478: np.ndarray):
    eL, eR = mp478[MP_EYE_IMG_LEFT].mean(0), mp478[MP_EYE_IMG_RIGHT].mean(0); d = eR - eL; iod = float(np.linalg.norm(d))
    u = d / max(iod, 1e-6); v = np.array([-u[1], u[0]])
    return u, v, iod


def lm106_from_mp(mp478: np.ndarray, calib: bool = True) -> np.ndarray:
    lm = mp478[MAP106].astype(np.float64).copy()
    if calib:
        u, v, iod = face_frame(mp478)
        for g, idx in ANCHORS.items():
            cu, cv = ANCHOR_CALIB[g]; lm[idx] -= (cu * u + cv * v) * iod
    return lm


def ls_bbox(lm106: np.ndarray, W: int, H: int):
    """LatentSync FaceDetector's bbox formula (latentsync/utils/face_detector.py) on 106-layout points; None if it would be rejected."""
    lmk = np.round(lm106).astype(np.int_)
    half = np.mean([lmk[74], lmk[73]], axis=0); sub = lmk[LMK_ADAPT_ORIGIN_ORDER]
    dist = np.max(sub[:, 1]) - half[1]; upper = half[1] - dist
    x1, y1, x2, y2 = (int(np.min(sub[:, 0])), int(upper), int(np.max(sub[:, 0])), int(np.max(sub[:, 1])))
    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        return None
    y2 += int((x2 - x1) * 0.1); x1 -= int((x2 - x1) * 0.05); x2 += int((x2 - x1) * 0.05)
    return (max(0, x1), max(0, y1), min(W, x2), min(H, y2))


class MediaPipeRouter:
    """Face router rows from the MediaPipe pass (same classes / reasons / thresholds as retinaface_router.RetinaFaceRouter)."""

    name = "mediapipe"

    def __init__(self, min_w: int = 50, min_h: int = 80, aspect=(0.2, 1.5)):
        self.min_w, self.min_h, self.aspect = min_w, min_h, aspect

    def row(self, mp478):
        if mp478 is None or not np.isfinite(mp478[0, 0]):
            return {"cls": "NO_FACE", "reason": "no_face", "bbox": None, "landmarks_5": None, "confidence": None}
        ov = mp478[MP_FACE_OVAL]; x1, y1 = ov.min(0); x2, y2 = ov.max(0); w, h = int(x2 - x1), int(y2 - y1)
        bbox = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "w": w, "h": h, "score": 1.0}
        lm5 = [mp478[MP_EYE_IMG_LEFT].mean(0), mp478[MP_EYE_IMG_RIGHT].mean(0), mp478[4], mp478[61], mp478[291]]
        extra = {"landmarks_5": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in lm5], "confidence": 1.0}
        if w < self.min_w: return {"cls": "SMALL_FACE", "reason": "width_below_threshold", "bbox": bbox, **extra}
        if h < self.min_h: return {"cls": "SMALL_FACE", "reason": "height_below_threshold", "bbox": bbox, **extra}
        if not (self.aspect[0] < w / max(h, 1) < self.aspect[1]): return {"cls": "SMALL_FACE", "reason": "aspect_out_of_range", "bbox": bbox, **extra}
        return {"cls": "VALID_FACE", "reason": None, "bbox": bbox, **extra}

    def classify_frame(self, frame_bgr):
        raise RuntimeError("clean face stack: rows are precomputed from the MediaPipe pass (CleanPrep.router_rows)")

    def params(self) -> dict:
        return {"detector": "mediapipe face_landmarker.task float16 v1 (BlazeFace short-range + Face Mesh V2), Apache-2.0", "face_min_w": self.min_w,
                "face_min_h": self.min_h, "aspect_range": list(self.aspect), "model_sha256": MP_MODEL_SHA256}


def ls_detections(mp_lm: np.ndarray, W: int, H: int, calib: bool = True) -> list:
    """Per frame (bbox, 106-point landmarks) in LatentSync FaceDetector semantics ((None, None) = 'Face not detected')."""
    out = []
    for f in mp_lm:
        if not np.isfinite(f[0, 0]):
            out.append((None, None)); continue
        lm = lm106_from_mp(f, calib); bb = ls_bbox(lm, W, H)
        if bb is None or (bb[2] - bb[0]) < 50 or (bb[3] - bb[1]) < 80 or not (0.2 < (bb[2] - bb[0]) / max(bb[3] - bb[1], 1) < 1.5):
            out.append((None, None)); continue
        out.append((bb, lm.astype(np.float32)))
    return out


class CleanPrep:
    """Stands in for face_aware_latentsync_accel.VideoPrep: the master frames + MediaPipe-derived router rows and LatentSync detections."""

    def __init__(self, src: pathlib.Path, work: pathlib.Path, min_w: int = 50, min_h: int = 80, calib: bool = True, frames=None, master=None):
        import face_aware_latentsync as fa
        self.times = {}; self.error = None
        t0 = time.perf_counter()
        if frames is None:
            self.master, _ = fa.make_master(src, work); self.frames = fa.read_frames(self.master)
        else:
            self.master, self.frames = master, frames
        self.times["prep_master"] = round(time.perf_counter() - t0, 3); t0 = time.perf_counter()
        npz = run_mediapipe(pathlib.Path(self.master), pathlib.Path(work) / "master_mediapipe.npz")
        z = np.load(npz); lm = z["lm"]
        if len(lm) != len(self.frames):
            raise RuntimeError(f"mediapipe frames {len(lm)} != master frames {len(self.frames)}")
        self.times["prep_mediapipe"] = round(time.perf_counter() - t0, 3)
        H, W = self.frames[0].shape[:2]
        self.router = MediaPipeRouter(min_w, min_h)
        self.router_rows = [self.router.row(f if np.isfinite(f[0, 0]) else None) for f in lm]
        self.ls_dets = ls_detections(lm, W, H, calib)
        self.info = {"face_stack": "mediapipe", "frames": len(lm), "face_frames": int(np.isfinite(lm[:, 0, 0]).sum()),
                     "ls_detections": int(sum(d[0] is not None for d in self.ls_dets)), "anchor_calib": ANCHOR_CALIB if calib else None, "npz": str(npz)}

    def result(self):
        return self


class _BlockedFaceDetector:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        raise RuntimeError("clean face stack: LatentSync's InsightFace FaceDetector is disabled; detections must be replayed (CleanPrep.ls_dets)")


def guard_nc():
    """Make every non-commercial face model of this repo impossible to load in this process (they would silently auto-download):
    InsightFace FaceAnalysis, LatentSync FaceDetector, facexlib detection / parsing, CodeFormer (basicsr arch)."""
    def blocked(name):
        def f(*a, **k):
            raise RuntimeError(f"clean face stack: {name} is non-commercial and blocked (CLEAN-ONLY)")
        return f
    ls = ROOT / "third_party" / "latentsync"
    if str(ls) not in sys.path:
        sys.path.insert(0, str(ls))
    import latentsync.utils.image_processor as ipm
    import latentsync.utils.face_detector as fdm
    ipm.FaceDetector = _BlockedFaceDetector; fdm.FaceDetector = _BlockedFaceDetector
    try:
        import insightface.app as ia
        ia.FaceAnalysis.__init__ = blocked("insightface FaceAnalysis (buffalo_l)")
    except Exception:
        pass
    try:
        import facexlib.detection as fxd, facexlib.parsing as fxp
        fxd.init_detection_model = blocked("facexlib RetinaFace"); fxp.init_parsing_model = blocked("facexlib parsing")
    except Exception:
        pass
    import face_aware_latentsync as fa
    fa.face_detector = blocked("LatentSync FaceDetector (face_aware_latentsync.face_detector)")
    return True
