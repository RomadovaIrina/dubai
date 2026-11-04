#!/usr/bin/env python3
"""r14 F: a SHORT generator probe (user rule: no full-video ceiling runs). LatentSync on master frames [t0, t1) of a finished run with its dub audio,
in several runner configs; the raw generated frames side by side (mouth / lower-face crop) at 0.5x with the audio slowed, + seconds per config.
    flock /workspace/dub/gpu.lock python scripts/r12/probe_seg.py <id> <t0> <t1> OUT.mp4 "q2=dc,stride0" "nd=nodc,stride0" "fuse8=nodc,stride8,fuse" ..."""
import os, sys, pathlib, subprocess, tempfile, time
import numpy as np, cv2
R = pathlib.Path(__file__).resolve().parents[1]
for d in ("quality", "pilot", "optim"): sys.path.insert(0, str(R / d))
import latentsync_accel  # noqa: F401  (third_party/latentsync on sys.path)
from ls_stable import StableRunner
import face_aware_latentsync as fa
import clean_face_stack as cfs; cfs.guard_nc()   # production face stack: InsightFace is quarantined, detections are replayed from MediaPipe

idn, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]; cfgs = sys.argv[5:]
B = "/workspace/dub/artifacts/r4m"; W = f"{B}/r9g_{idn}/work_{idn}_r9g_{idn}"; k0, k1 = int(t0 * 25), int(t1 * 25)
frames = fa.read_frames(pathlib.Path(W + "/video/master_25fps.mp4"))[k0:k1]; frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
lm = np.load(W + "/video/master_mediapipe.npz")["lm"][k0:k1]; q = np.nanmedian(lm, 0); iod = np.linalg.norm(q[33, :2] - q[263, :2])
cx, cy = q[[61, 291, 13, 14], :2].mean(0); c = int(1.8 * iod) // 2 * 2; x0, y0 = max(0, int(cx - c / 2)), max(0, int(cy - 0.45 * c))
td = pathlib.Path(tempfile.mkdtemp()); wav = td / "a.wav"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t0), "-t", str(t1 - t0), "-i", f"/workspace/dub/artifacts/r4a/r9_{idn}_htd/{idn}/dubbed_16k.wav", str(wav)], check=True)
panels = []; runner = None
dets = cfs.ls_detections(lm, frames[0].shape[1], frames[0].shape[0], True)   # raw master landmarks (no anchor filter): the same for every config
os.chdir(R.parent / "third_party" / "latentsync")   # insightface / LatentSync checkpoints are cwd-relative (checkpoints/auxiliary)
for cfg in cfgs:
    name, spec = cfg.split("="); opts = spec.split(",")
    stride = next((int(o[6:]) for o in opts if o.startswith("stride") and o[6:]), 0) or None
    ref = next((o for o in opts if o.startswith("shift")), "same"); g = next((float(o[1:]) for o in opts if o[:1] == "g" and o[1:].replace(".", "").isdigit()), 2.0)   # item 3: reference S frames away / CFG
    runner = StableRunner(2, "nodc" not in opts, 10, g, 1247, 5, stride=stride, ref_mode=ref, overlap="fuse" if "fuse" in opts else "continue", pipe=None,   # a fresh pipe per config (DeepCache wraps the UNet)
                          bridge_len=0 if stride else 8, bridge_steps=0 if stride else 4)
    import torch; torch.cuda.synchronize(); ts = time.perf_counter()
    gen = runner.run_frames(np.asarray(frames), str(wav), dets); torch.cuda.synchronize(); dt = time.perf_counter() - ts
    if isinstance(gen, tuple): gen, dt_in = gen; print(f"   (runner-reported {dt_in})", flush=True)   # StableRunner returns (frames, seconds)
    print(f"== probe {name}: {spec}: {len(frames)} frames in {dt:.1f} s ({1000 * dt / len(frames):.0f} ms / frame)", flush=True)
    if os.environ.get("PROBE_SAVE"):   # full generated frames at 25 fps (for MediaPipe measures: scripts/r12/sticky.py label=path@t0)
        sw = cv2.VideoWriter(f"{os.environ['PROBE_SAVE']}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (frames[0].shape[1], frames[0].shape[0]))
        for g_ in gen: sw.write(cv2.cvtColor(np.ascontiguousarray(g_), cv2.COLOR_RGB2BGR))
        sw.release()
    crop = [cv2.cvtColor(np.ascontiguousarray(g)[y0:y0 + c, x0:x0 + c], cv2.COLOR_RGB2BGR) for g in gen]
    crop = [cv2.putText(cv2.resize(f, (400, 400)), name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) for f in crop]; panels.append(crop)
src = [cv2.putText(cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)[y0:y0 + c, x0:x0 + c], (400, 400)), "source", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) for f in frames]
vw = cv2.VideoWriter(str(td / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (400 * (len(panels) + 1), 400))
for i in range(len(frames)): vw.write(np.concatenate([src[i]] + [p[i] for p in panels], 1))
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(td / "v.mp4"), "-i", str(wav), "-filter_complex", "[1:a]atempo=0.5[a]", "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out], check=True)
print("== probe clip", out, flush=True)
