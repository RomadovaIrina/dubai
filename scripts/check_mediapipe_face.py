#!/usr/bin/env python3
"""Clean face stack smoke (2026-10-16): MediaPipe Face Landmarker (own env /venv/mediapipe, Apache-2.0) on 2 s of LatentSync's demo clip."""
import pathlib, subprocess, sys, tempfile
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts/pilot"))
import clean_face_stack as cfs
tmp = pathlib.Path(tempfile.mkdtemp(prefix="mp_smoke_")); clip = tmp / "demo_25fps.mp4"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ROOT / "third_party/latentsync/assets/demo1_video.mp4"), "-t", "2", "-r", "25", "-an", str(clip)], check=True)
z = np.load(cfs.run_mediapipe(clip, tmp / "mp.npz")); lm, pres = z["lm"], z["presence"]
W, H = (int(v) for v in z["wh"]); dets = cfs.ls_detections(lm, W, H)
print(f"frames {len(lm)}, face {int(pres.sum())}, LatentSync-usable detections {sum(d[0] is not None for d in dets)}, model sha256 {cfs.MP_MODEL_SHA256[:12]}")
ok = len(lm) >= 45 and pres.all() and all(d[0] is not None for d in dets)
print("PASS mediapipe face stack" if ok else "FAIL"); sys.exit(0 if ok else 1)
