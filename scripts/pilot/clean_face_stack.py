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


def run_mediapipe(video: pathlib.Path, out: pathlib.Path, mode: str = "video", num_faces: int = 1) -> pathlib.Path:
    """MediaPipe Face Landmarker over every frame of `video` in its own env -> npz (cached next to the master)."""
    if out.exists():
        try:
            f_ = np.load(out).files
            if "matrix" in f_ and (num_faces <= 1 or "lm_all" in f_): return out   # older caches lack the pose matrices / extra faces -> recompute
        except Exception:
            pass
    import hashlib
    h = hashlib.sha256(MP_MODEL.read_bytes()).hexdigest()
    if h != MP_MODEL_SHA256:
        raise RuntimeError(f"{MP_MODEL}: sha256 {h} != pinned {MP_MODEL_SHA256}")
    env = {**os.environ, "GLOG_minloglevel": "2", "OMP_NUM_THREADS": "4"}
    p = subprocess.run([MP_PY, str(HERE / "mp_landmarks.py"), str(video), str(out), "--model", str(MP_MODEL), "--mode", mode, "--num-faces", str(num_faces)],
                       env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"mediapipe landmarks failed ({p.returncode}): {p.stdout[-1500:]}")
    return out


def pose_from_matrix(M: np.ndarray) -> np.ndarray:
    """MediaPipe facial transformation matrices (T, 4, 4) -> (T, 3) yaw, pitch, roll in degrees (NaN where no face). Validated on 03:
    yaw follows the nose-vs-eye-midpoint offset (r 0.69), roll the eye-line angle (r -0.99, opposite sign convention)."""
    out = np.full((len(M), 3), np.nan)
    ok = np.isfinite(M[:, 0, 0])
    if ok.any():
        R = M[ok, :3, :3] / np.linalg.norm(M[ok, :3, 0], axis=1)[:, None, None]
        out[ok, 0] = np.degrees(np.arctan2(R[:, 0, 2], R[:, 2, 2]))
        out[ok, 1] = np.degrees(np.arctan2(-R[:, 1, 2], np.sqrt(R[:, 0, 2] ** 2 + R[:, 2, 2] ** 2)))
        out[ok, 2] = np.degrees(np.arctan2(R[:, 1, 0], R[:, 1, 1]))
    return out


POSE_PRECHECK = {"yaw_deg": 30.0, "max_fraction": 0.30}
HOLD_GAP_FRAMES = 5   # spec 11: missing landmarks for <= 5 frames (0.2 s) inside a face run -> previous valid frame   # spec 5.4: > 30 % of face frames with |yaw| > 30 deg -> voice-only result, flagged


def track_faces(lm_all: np.ndarray, mats_all: np.ndarray | None = None):
    """Unordered faces per frame (T, F, 478, 2) -> identity-consistent tracks (same shape): greedy nearest-centroid continuity, a face
    that moved more than 1.5 face widths starts / joins the track that was free longest."""
    T, F = lm_all.shape[:2]; out = np.full_like(lm_all, np.nan); mo = None if mats_all is None else np.full_like(mats_all, np.nan)
    last = [None] * F; seen = [-10 ** 9] * F
    for t in range(T):
        cur = [k for k in range(F) if np.isfinite(lm_all[t, k, 0, 0])]
        cen = {k: lm_all[t, k].mean(0) for k in cur}; wid = {k: float(np.ptp(lm_all[t, k, :, 0])) for k in cur}
        pairs = sorted((float(np.linalg.norm(cen[k] - last[j])), k, j) for k in cur for j in range(F) if last[j] is not None)
        uk, uj, asg = set(), set(), {}
        for d, k, j in pairs:
            if k in uk or j in uj or d > 1.5 * wid[k]: continue
            asg[k] = j; uk.add(k); uj.add(j)
        for k in cur:
            if k not in asg:
                j = min((j for j in range(F) if j not in uj), key=lambda j: seen[j]); asg[k] = j; uj.add(j)
        for k, j in asg.items():
            out[t, j] = lm_all[t, k]; last[j] = cen[k]; seen[j] = t
            if mo is not None: mo[t, j] = mats_all[t, k]
    return out, mo


def mouth_open(lm478: np.ndarray) -> np.ndarray:
    """(T, 478, 2) -> inner-lip gap (MediaPipe 13 / 14) in eye distances, NaN where no face."""
    eL, eR = lm478[:, MP_EYE_IMG_LEFT].mean(1), lm478[:, MP_EYE_IMG_RIGHT].mean(1)
    return np.linalg.norm(lm478[:, 13] - lm478[:, 14], axis=1) / np.maximum(np.linalg.norm(eR - eL, axis=1), 1e-6)


def assign_speakers(tracks: np.ndarray, turns: list, fps: float = 25.0, min_presence: float = 0.1) -> dict:
    """Diarization speakers -> face tracks (spec 8.2): Pearson correlation of each face's mouth-opening speed with each speaker's
    speech indicator (source diarization turns); each face is used at most once, best scores first."""
    T, F = tracks.shape[:2]; vel = {}
    for j in range(F):
        pres = np.isfinite(tracks[:, j, 0, 0])
        if pres.mean() < min_presence: continue
        o = mouth_open(np.where(pres[:, None, None], tracks[:, j], np.nan)); idx = np.arange(T)
        o = np.interp(idx, idx[pres], o[pres]) if pres.sum() > 1 else np.zeros(T)
        v = np.abs(np.gradient(o)); k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2); vel[j] = (np.convolve(v, k / k.sum(), "same"), pres)
    spk = sorted({t["speaker"] for t in turns}); scores = {}
    for sp in spk:
        ind = np.zeros(T)
        for t in turns:
            if t["speaker"] == sp: ind[int(t["start"] * fps):int(np.ceil(t["end"] * fps))] = 1.0
        for j, (v, pres) in vel.items():
            m = pres & np.isfinite(v)
            scores[(sp, j)] = float(np.corrcoef(v[m], ind[m])[0, 1]) if m.sum() > 10 and ind[m].std() > 0 and v[m].std() > 0 else 0.0
    mapping, used = {}, set()
    for (sp, j), sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if sp in mapping or j in used: continue
        mapping[sp] = j; used.add(j)
    return {"mapping": mapping, "scores": {f"{sp}->face{j}": round(v, 3) for (sp, j), v in scores.items()}, "faces": sorted(vel)}


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

    def __init__(self, src: pathlib.Path, work: pathlib.Path, min_w: int = 50, min_h: int = 80, calib: bool = True, frames=None, master=None,
                 pose_precheck: bool = True, pose_gate_deg: float = 0.0, pose_gate_hold: int = 3, max_faces: int = 2, turns: list | None = None):
        import face_aware_latentsync as fa
        self.times = {}; self.error = None
        t0 = time.perf_counter()
        if frames is None:
            self.master, _ = fa.make_master(src, work); self.frames = fa.read_frames(self.master)
        else:
            self.master, self.frames = master, frames
        self.times["prep_master"] = round(time.perf_counter() - t0, 3); t0 = time.perf_counter()
        try:   # spec 9.4 support: exact source-frame <-> master-frame mapping (scripts/pilot/frame_map.py), recorded in the manifest
            import frame_map as fmap
            self.frame_map = fmap.frame_map(pathlib.Path(src), self.frames)
        except Exception as e:   # never fail the video stage for a bookkeeping table
            self.frame_map = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        self.times["prep_frame_map"] = round(time.perf_counter() - t0, 3); t0 = time.perf_counter()
        # spec 11.1-11.3: a MediaPipe failure never crashes the worker: one retry, then a voice-only result (no lip-sync), flagged
        self.failure = None; npz = pathlib.Path(work) / "master_mediapipe.npz"; lm = None; z = None
        for attempt in (1, 2):
            try:
                run_mediapipe(pathlib.Path(self.master), npz, num_faces=max_faces); z = np.load(npz); lm = z["lm"]
                if len(lm) != len(self.frames):
                    raise RuntimeError(f"mediapipe frames {len(lm)} != master frames {len(self.frames)}")
                break
            except Exception as e:
                self.failure = f"attempt {attempt}: {type(e).__name__}: {str(e)[:300]}"; lm = None
                try: npz.unlink(missing_ok=True)
                except Exception: pass
        # spec 8.2: several faces -> lip-sync only the active speaker's face (diarization <-> face by mouth motion); the others stay untouched
        self.faces = {"max_faces": max_faces, "tracks": 1}
        if lm is not None and "lm_all" in z.files:
            tr, trm = track_faces(z["lm_all"], z["matrix_all"] if "matrix_all" in z.files else None)
            present = [j for j in range(tr.shape[1]) if np.isfinite(tr[:, j, 0, 0]).mean() >= 0.1]
            if len(present) >= 2 and turns:
                asg = assign_speakers(tr, turns); mp_ = asg["mapping"]
                if mp_:
                    T_ = len(lm); sel = np.full(T_, next(iter(mp_.values())), int); cur = sel[0]
                    order = sorted(turns, key=lambda t: t["start"])
                    spk_at = np.full(T_, "", object)
                    for t in order: spk_at[int(t["start"] * 25):int(np.ceil(t["end"] * 25))] = t["speaker"]
                    for i in range(T_):   # between turns keep the last active speaker's face
                        if spk_at[i] in mp_: cur = mp_[spk_at[i]]
                        sel[i] = cur
                    lm = tr[np.arange(T_), sel].astype(np.float32)
                    z = {"matrix": trm[np.arange(T_), sel] if trm is not None else np.full((T_, 4, 4), np.nan, np.float32), "files": ["matrix"]}
                    self.faces = {"max_faces": max_faces, "tracks": len(present), "assignment": asg, "selected_face_frames": {int(j): int((sel == j).sum()) for j in np.unique(sel)}}
            else:
                self.faces = {"max_faces": max_faces, "tracks": len(present), "note": "single face (or no diarization turns): largest face"}
        self.face_stack_failed = lm is None
        if lm is None:
            lm = np.full((len(self.frames), 478, 2), np.nan, np.float32)
        zf = (z["files"] if isinstance(z, dict) else z.files) if z is not None else []
        matrix = z["matrix"] if "matrix" in zf else np.full((len(self.frames), 4, 4), np.nan, np.float32)
        # short landmark gaps inside a face run: hold the previous valid frame (a split would add two LS <-> original switches)
        lm = lm.copy(); held = 0; ok = np.isfinite(lm[:, 0, 0]); i = 0; n_ = len(lm)
        while i < n_:
            if not ok[i]:
                j = i
                while j < n_ and not ok[j]: j += 1
                if 0 < i and j < n_ and (j - i) <= HOLD_GAP_FRAMES:
                    lm[i:j] = lm[i - 1]; held += j - i
                i = j
            else:
                i += 1
        self.held_frames = held
        self.times["prep_mediapipe"] = round(time.perf_counter() - t0, 3)
        H, W = self.frames[0].shape[:2]
        self.router = MediaPipeRouter(min_w, min_h)
        self.router_rows = [self.router.row(f if np.isfinite(f[0, 0]) else None) for f in lm]
        self.ls_dets = ls_detections(lm, W, H, calib)
        # ---- head pose (spec 5.4 / 5.5): pre-check -> voice-only; optional per-frame gate -> pass-through (eased by the LS<->original crossfade)
        pose = pose_from_matrix(matrix)
        face = np.isfinite(pose[:, 0]); big = face & (np.abs(pose[:, 0]) > POSE_PRECHECK["yaw_deg"])
        frac = float(big.sum() / max(face.sum(), 1)); self.voice_only = bool(pose_precheck and face.any() and frac > POSE_PRECHECK["max_fraction"])
        gated = 0
        if self.face_stack_failed:   # router rows are already NO_FACE everywhere -> the whole video passes through with the dub (voice-only)
            self.voice_only = True
            for r in self.router_rows: r.update({"cls": "NO_FACE", "reason": "face_stack_failed"})
        elif self.voice_only:
            for r in self.router_rows: r.update({"cls": "NO_FACE", "reason": "pose_precheck_voice_only"})
        elif pose_gate_deg and pose_gate_deg > 0:
            out_of = face & (np.abs(pose[:, 0]) > pose_gate_deg)
            if pose_gate_hold > 0 and out_of.any():   # hold the decision +-hold frames: no 1-frame flicker between LatentSync and the original
                k = np.ones(2 * pose_gate_hold + 1); out_of = np.convolve(out_of.astype(float), k, "same") > 0
            for i in np.where(out_of)[0]:
                if self.router_rows[i]["cls"] == "VALID_FACE":
                    self.router_rows[i] = {**self.router_rows[i], "cls": "SMALL_FACE", "reason": "pose_out_of_range"}; gated += 1
        self.pose = pose
        self.pose_info = {"available": bool(face.any()), "yaw_p1_p50_p99": np.round(np.nanpercentile(pose[:, 0], [1, 50, 99]), 1).tolist() if face.any() else None,
                          "abs_yaw_max": round(float(np.nanmax(np.abs(pose[:, 0]))), 1) if face.any() else None,
                          "precheck": {**POSE_PRECHECK, "fraction_over": round(frac, 4), "enabled": bool(pose_precheck), "voice_only": self.voice_only},
                          "gate": {"yaw_deg": pose_gate_deg, "hold_frames": pose_gate_hold, "gated_frames": int(gated)}}
        self.info = {"face_stack": "mediapipe", "frames": len(lm), "face_frames": int(np.isfinite(lm[:, 0, 0]).sum()),
                     "ls_detections": int(sum(d[0] is not None for d in self.ls_dets)), "anchor_calib": ANCHOR_CALIB if calib else None, "npz": str(npz),
                     "frame_map_summary": self.frame_map.get("summary", self.frame_map), "pose": self.pose_info,
                     "voice_only": self.voice_only, "face_stack_failed": self.face_stack_failed, "faces": self.faces,
                     "held_landmark_frames": int(self.held_frames), "failure": self.failure}

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
    try:   # background bed separator: torchaudio HDEMUCS_HIGH_MUSDB_PLUS (no weights licence, MUSDB18-HQ NC); it would auto-download to the torch hub cache
        import torchaudio
        torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model = blocked("torchaudio HDEMUCS_HIGH_MUSDB_PLUS (background bed)")
    except Exception:
        pass
    try:
        import background_bed
        background_bed.load_demucs = blocked("background_bed.load_demucs (HDemucs MUSDB18-HQ)")
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------------------------------------------------------------
# CLEAN-ONLY policy (2026-10-16, user decision): clean is the default everywhere; non-commercial components only with DUB_ALLOW_NC=1
# ---------------------------------------------------------------------------------------------------------------------------------
def nc_allowed() -> bool:
    return os.environ.get("DUB_ALLOW_NC", "0") == "1"


TRIPWIRES = [ROOT / "third_party/latentsync/checkpoints/auxiliary/models/buffalo_l", pathlib.Path(os.path.expanduser("~/.insightface/models/buffalo_l"))]


def install_tripwires() -> list:
    """insightface.utils.storage.download() fetches a model pack whenever its directory is missing. An EMPTY directory makes FaceAnalysis
    fail (no detection model) instead of silently downloading the non-commercial buffalo_l pack again."""
    made = []
    for d in TRIPWIRES:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            (d / "NOTICE_NC_QUARANTINED.txt").write_text("InsightFace buffalo_l is non-commercial and quarantined (CLEAN-ONLY, 2026-10-16). This empty "
                                                          "directory blocks insightface's auto-download. Remove it only together with DUB_ALLOW_NC=1.\n")
            made.append(str(d))
    return made


def enforce_clean(face_stack: str | None = "mediapipe", codeformer: str | None = "off", video_backend: str | None = "optimized", what: str = "run",
                  background: str | None = "off"):
    """Refuse non-commercial settings unless DUB_ALLOW_NC=1, then block every NC loader in this process (guard_nc) and arm the tripwires."""
    if nc_allowed():
        print(f"[clean] DUB_ALLOW_NC=1: non-commercial components allowed for this {what} (NOT a product-path run)", flush=True)
        return False
    bad = []
    if face_stack not in (None, "mediapipe"): bad.append(f"--face-stack {face_stack} (InsightFace buffalo_l / RetinaFace R50)")
    if codeformer not in (None, "off"): bad.append(f"--codeformer {codeformer} (CodeFormer S-Lab NC)")
    if video_backend not in (None, "optimized"): bad.append(f"--video-backend {video_backend} (InsightFace classification)")
    if background not in (None, "off"): bad.append(f"--background {background} (HDemucs MUSDB18-HQ separator, non-commercial)")
    if bad:
        raise SystemExit("CLEAN-ONLY: non-commercial component requested: " + "; ".join(bad) + ". Use the clean defaults, or set DUB_ALLOW_NC=1 "
                         "for a research-only run.")
    install_tripwires(); guard_nc()
    return True
