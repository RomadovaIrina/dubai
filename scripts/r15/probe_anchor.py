#!/usr/bin/env python3
"""r15 F: a SHORT generator probe of the LatentSync crop geometry: master frames [t0, t1) of a q2 run with its dub audio, the PRODUCTION
detections (anchor filter flow:4 over the whole video + head-locked mask boxes, as clean_face_stack.CleanPrep) from the original rigid track
vs the validated one (anchor_filters.validate_motion, LS_RIGID_VALIDATE). Panels: source | original | validated (lower-face crop), 0.5x.
    flock /workspace/dub/gpu.lock python scripts/r15/probe_anchor.py q2_m03 16 22 OUT.mp4 [--thr 0.02]"""
import os, sys, pathlib, subprocess, tempfile, time, argparse
import numpy as np, cv2
R = pathlib.Path(__file__).resolve().parents[1]
for d in ("quality", "pilot", "optim", "r12"): sys.path.insert(0, str(R / d))
import latentsync_accel  # noqa: F401
from ls_stable import StableRunner
import face_aware_latentsync as fa
import clean_face_stack as cfs; cfs.guard_nc()
import anchor_filters as af
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float); ap.add_argument("out"); ap.add_argument("--thr", type=float, default=0.02)
a = ap.parse_args()
B = "/workspace/dub/artifacts/r4m"; W = f"{B}/r9g_{a.run}/work_{a.run}_r9g_{a.run}"; idn = a.run.split("_", 1)[1]; k0, k1 = int(a.t0 * 25), int(a.t1 * 25)
frames = fa.read_frames(pathlib.Path(W + "/video/master_25fps.mp4"))[k0:k1]; frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
lm = np.load(W + "/video/master_mediapipe.npz")["lm"][:, :, :2].astype(np.float32); Hrig = np.load(W + "/video/master_rigid.npz")["H"]; n = min(len(lm), len(Hrig)); lm, Hrig = lm[:n], Hrig[:n]
H0, W0 = frames[0].shape[:2]
def dets_for(H):
    lm_ls = af.filter_landmarks(lm, H, "flow:4").astype(np.float32); d = cfs.ls_detections(lm_ls, W0, H0, True)
    l106 = np.stack([x[1] if x[1] is not None else np.full((106, 2), np.nan, np.float32) for x in d]).astype(np.float64); hb = af.head_locked_boxes(l106, H)
    d = [(tuple(float(v) for v in hb[i]) if x[0] is not None and np.isfinite(hb[i, 0]) else x[0], x[1]) for i, x in enumerate(d)]
    return d[k0:k1]
Hv, info = af.validate_motion(Hrig, lm, a.thr, return_info=True); print("   [rigid] validate", info, flush=True)
cfgs = [("original", dets_for(Hrig)), ("validated", dets_for(Hv))]
q = np.nanmedian(lm[k0:k1], 0); iod = np.linalg.norm(q[33] - q[263]); cx, cy = q[[61, 291, 13, 14]].mean(0); c = int(2.6 * iod) // 2 * 2; x0, y0 = max(0, int(cx - c / 2)), max(0, int(cy - 0.5 * c))
td = pathlib.Path(tempfile.mkdtemp()); wav = td / "a.wav"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(a.t0), "-t", str(a.t1 - a.t0), "-i", f"/workspace/dub/artifacts/r4a/r9_{a.run}_htd/{a.run}/dubbed_16k.wav", str(wav)], check=True)
os.chdir(R.parent / "third_party" / "latentsync"); panels = []
import torch
for name, dets in cfgs:
    runner = StableRunner(2, True, 10, 2.0, 1247, 5, pipe=None, bridge_len=8, bridge_steps=4)
    torch.cuda.synchronize(); ts = time.perf_counter(); gen = runner.run_frames(np.asarray(frames), str(wav), dets); torch.cuda.synchronize(); dt = time.perf_counter() - ts
    if isinstance(gen, tuple): gen = gen[0]
    print(f"== probe {name}: {len(frames)} frames in {dt:.1f} s", flush=True)
    panels.append([cv2.putText(cv2.resize(cv2.cvtColor(np.ascontiguousarray(g)[y0:y0 + c, x0:x0 + c], cv2.COLOR_RGB2BGR), (420, 420)), name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) for g in gen])
    del runner; torch.cuda.empty_cache()
src = [cv2.putText(cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)[y0:y0 + c, x0:x0 + c], (420, 420)), f"master {a.t0 + i / 25:.2f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) for i, f in enumerate(frames)]
vw = cv2.VideoWriter(str(td / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (420 * 3, 420))
for i in range(len(frames)): vw.write(np.concatenate([src[i]] + [p[i] for p in panels], 1))
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(td / "v.mp4"), "-i", str(wav), "-filter_complex", "[1:a]atempo=0.5[a]", "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", a.out], check=True)
k = 476 - k0
if 0 <= k < len(frames): cv2.imwrite(str(pathlib.Path(a.out).with_suffix(".f476.jpg")), np.concatenate([src[k]] + [p[k] for p in panels], 1))
print("== probe clip", a.out, flush=True)
