#!/usr/bin/env python3
"""r14 F: a SHORT generator probe (user rule: no full-video ceiling runs). LatentSync on master frames [t0, t1) of a finished run with its dub audio,
in several runner configs; the raw generated frames side by side (mouth / lower-face crop) at 0.5x with the audio slowed, + seconds per config.
    flock /workspace/dub/gpu.lock python scripts/r12/probe_seg.py <id> <t0> <t1> OUT.mp4 "q2=dc" "nd=nodc" "c8=dc,stride8" ...
config options (comma list): dc | nodc, strideN (overlapping windows; + fuse = per-step blending, else continue), shiftN (reference N frames away),
gX (CFG X), rf (night P2: fixed reference = 16 smart source frames, scripts/r12/ref_smart.py), jaw (night P3: LS_MASK=mask_jaw.png +
LS_MASK_FEATHER=6), cfge (night P4: LS_CFG_EARLY=3.0:0.35). PROBE_SAVE=dir also writes each config's full generated frames (25 fps)."""
import os, sys, pathlib, subprocess, tempfile, time
import numpy as np, cv2
R = pathlib.Path(__file__).resolve().parents[1]
for d in ("quality", "pilot", "optim", "r12"): sys.path.insert(0, str(R / d))
import latentsync_accel  # noqa: F401  (third_party/latentsync on sys.path)
from ls_stable import StableRunner
import face_aware_latentsync as fa
import clean_face_stack as cfs; cfs.guard_nc()   # production face stack: InsightFace is quarantined, detections are replayed from MediaPipe
import ref_smart

idn, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]; cfgs = sys.argv[5:]
B = "/workspace/dub/artifacts/r4m"; W = f"{B}/r9g_{idn}/work_{idn}_r9g_{idn}"; k0, k1 = int(t0 * 25), int(t1 * 25)
master = pathlib.Path(W + "/video/master_25fps.mp4")
frames = fa.read_frames(master)[k0:k1]; frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
lm_all = np.load(W + "/video/master_mediapipe.npz")["lm"]; lm = lm_all[k0:k1]; q = np.nanmedian(lm, 0); iod = np.linalg.norm(q[33, :2] - q[263, :2])
cx, cy = q[[61, 291, 13, 14], :2].mean(0); c = int(1.8 * iod) // 2 * 2; x0, y0 = max(0, int(cx - c / 2)), max(0, int(cy - 0.45 * c))
td = pathlib.Path(tempfile.mkdtemp()); wav = td / "a.wav"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t0), "-t", str(t1 - t0), "-i", f"/workspace/dub/artifacts/r4a/r9_{idn}_htd/{idn}/dubbed_16k.wav", str(wav)], check=True)
os.chdir(R.parent / "third_party" / "latentsync")   # insightface / LatentSync checkpoints are cwd-relative (checkpoints/auxiliary)
H0, W0 = frames[0].shape[:2]
dets = cfs.ls_detections(lm, W0, H0, True)   # raw master landmarks (no anchor filter): the same for every config
smart = None
panels = []; runner = None
ENV = {"jaw": {"LS_MASK": "mask_jaw.png", "LS_MASK_FEATHER": "6"}, "cfge": {"LS_CFG_EARLY": "3.0:0.35"}}
for cfg in cfgs:
    name, spec = cfg.split("="); opts = spec.split(",")
    stride = next((int(o[6:]) for o in opts if o.startswith("stride") and o[6:]), 0) or None
    ref = "fixed" if "rf" in opts else next((o for o in opts if o.startswith("shift")), "same")
    g = next((float(o[1:]) for o in opts if o[:1] == "g" and o[1:].replace(".", "").isdigit()), 2.0)   # item 3: reference S frames away / CFG
    env = {k: v for o in opts for k, v in ENV.get(o, {}).items()}; os.environ.update(env)
    runner = StableRunner(2, "nodc" not in opts, 10, g, 1247, 5, stride=stride, ref_mode=ref, overlap="fuse" if "fuse" in opts else "continue", pipe=None,   # a fresh pipe per config (DeepCache wraps the UNet)
                          bridge_len=0 if stride else 8, bridge_steps=0 if stride else 4)
    if ref == "fixed":
        if smart is None:
            lm_ok = lm_all.copy(); dd = cfs.ls_detections(lm_all, W0, H0, True); lm_ok[[i for i, d in enumerate(dd) if d[0] is None]] = np.nan
            idx, info, ref_bgr = ref_smart.pick_from_video(master, lm_ok); print(f"   [rf] {info}", flush=True)
            smart = (np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in ref_bgr]), [dd[i] for i in idx]) if idx else False
        if smart: runner.acc.set_reference(*smart)
        else: print("   [rf] fallback: the probe's first 16 frames", flush=True); runner.acc.set_reference(np.asarray(frames[:16]), dets[:16])
    import torch; torch.cuda.synchronize(); ts = time.perf_counter()
    gen = runner.run_frames(np.asarray(frames), str(wav), dets); torch.cuda.synchronize(); dt = time.perf_counter() - ts
    if isinstance(gen, tuple): gen, dt_in = gen; print(f"   (runner-reported {dt_in})", flush=True)   # StableRunner returns (frames, seconds)
    for k in env: os.environ.pop(k, None)
    print(f"== probe {name}: {spec}: {len(frames)} frames in {dt:.1f} s ({1000 * dt / len(frames):.0f} ms / frame)", flush=True)
    if os.environ.get("PROBE_SAVE"):   # full generated frames at 25 fps (for MediaPipe measures: scripts/r12/sticky.py label=path@t0)
        sw = cv2.VideoWriter(f"{os.environ['PROBE_SAVE']}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (W0, H0))
        for g_ in gen: sw.write(cv2.cvtColor(np.ascontiguousarray(g_), cv2.COLOR_RGB2BGR))
        sw.release()
    crop = [cv2.cvtColor(np.ascontiguousarray(g)[y0:y0 + c, x0:x0 + c], cv2.COLOR_RGB2BGR) for g in gen]
    crop = [cv2.putText(cv2.resize(f, (400, 400)), name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) for f in crop]; panels.append(crop)
    del runner; torch.cuda.empty_cache()
src = [cv2.putText(cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)[y0:y0 + c, x0:x0 + c], (400, 400)), "source", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) for f in frames]
vw = cv2.VideoWriter(str(td / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (400 * (len(panels) + 1), 400))
for i in range(len(frames)): vw.write(np.concatenate([src[i]] + [p[i] for p in panels], 1))
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(td / "v.mp4"), "-i", str(wav), "-filter_complex", "[1:a]atempo=0.5[a]", "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out], check=True)
print("== probe clip", out, flush=True)
