#!/usr/bin/env python3
"""Clean LatentSync smoke (2026-10-16): LatentSync 1.6 on 2 s of its demo clip + demo audio, with the clean face stack's MediaPipe
detections replayed (guard_nc(): InsightFace / LatentSync FaceDetector / facexlib loaders raise). Replaces check_latentsync.sh, whose
upstream inference loads InsightFace buffalo_l (non-commercial)."""
import pathlib, subprocess, sys, tempfile
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
for d in ("scripts/pilot", "scripts/optim"): sys.path.insert(0, str(ROOT / d))
import clean_face_stack as cfs
cfs.enforce_clean(what="check_latentsync_clean")
import cv2
import face_aware_latentsync as fa, face_aware_latentsync_accel as faa
tmp = pathlib.Path(tempfile.mkdtemp(prefix="ls_clean_")); clip = tmp / "demo_25fps.mp4"; wav = tmp / "demo_16k.wav"
LS = ROOT / "third_party/latentsync/assets"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(LS / "demo1_video.mp4"), "-t", "2", "-r", "25", "-an", str(clip)], check=True)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(LS / "demo1_audio.wav"), "-t", "2", "-ar", "16000", "-ac", "1", str(wav)], check=True)
frames = fa.read_frames(clip); H, W = frames[0].shape[:2]
dets = cfs.ls_detections(np.load(cfs.run_mediapipe(clip, tmp / "mp.npz"))["lm"], W, H)
runner = faa.AccelRunner(window_batch_size=2, deepcache=True, deepcache_interval=5)
out, dt = runner.run_frames(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]), wav, dets)
dst = ROOT / "reports/latentsync_clean_demo.mp4"
w = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
for f in out[:len(frames)]: w.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
w.release()
ok = len(out) >= len(frames) and dst.stat().st_size > 0 and float(np.abs(np.asarray(out[len(frames) // 2], np.float32) - cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2RGB)).mean()) > 0.1
print(f"{len(frames)} frames in, {len(out)} out, LatentSync {dt}s, replay fallbacks {runner.acc.last_stats.get('detector_fallback_calls')} -> {dst.name}")
print("PASS latentsync (clean face stack)" if ok else "FAIL"); sys.exit(0 if ok else 1)
