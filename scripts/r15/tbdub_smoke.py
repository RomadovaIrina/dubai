#!/usr/bin/env python3
"""r15 W0 smoke: the TBDub worker on N frames of a master + the matching slice of a dub wav (under gpu.lock, run by the caller).
    python scripts/r15/tbdub_smoke.py <master_25fps.mp4> <dub_16k.wav> <start> <n> <out.mp4>"""
import json, os, subprocess, sys, time, pathlib
import numpy as np, soundfile as sf, cv2
m, wav, s, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cap = cv2.VideoCapture(m); fr = []; i = 0
while len(fr) < n:
    ok, f = cap.read()
    if not ok: break
    if i >= s: fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    i += 1
fr = np.stack(fr); n = len(fr); a, sr = sf.read(wav, dtype="float32"); assert sr == 16000
seg = a[int(round(s / 25 * sr)):int(round(s / 25 * sr)) + int(round(n / 25 * sr))]; sw = f"/dev/shm/smoke_{os.getpid()}.wav"; sf.write(sw, seg, sr, subtype="PCM_16")
np.save("/dev/shm/smoke_in.npy", fr)
env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}; t0 = time.perf_counter()
p = subprocess.Popen(["/workspace/dub/tbdub/venv/bin/python", str(pathlib.Path(__file__).with_name("tbdub_worker.py"))], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, bufsize=1, cwd="/workspace/dub/tbdub")
print("ready", p.stdout.readline().strip(), round(time.perf_counter() - t0, 1), "s", flush=True)
for k in range(2):   # twice: the 2nd call shows the warm speed
    t1 = time.perf_counter()
    p.stdin.write(json.dumps({"master": m, "start": s, "n": n, "wav": sw, "in": "/dev/shm/smoke_in.npy", "out": "/dev/shm/smoke_out.npy"}) + "\n"); p.stdin.flush()
    print("call", k, p.stdout.readline().strip(), round(time.perf_counter() - t1, 1), "s", flush=True)
p.stdin.close(); p.wait()
o = np.load("/dev/shm/smoke_out.npy"); h, w = o.shape[1:3]
vw = cv2.VideoWriter(out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (w * 2, h))
for x, y in zip(fr, o): vw.write(cv2.cvtColor(np.concatenate([x, y], 1), cv2.COLOR_RGB2BGR))
vw.release(); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out + ".tmp.mp4", "-i", sw, "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-shortest", out]); os.unlink(out + ".tmp.mp4")
print("diff vs input (mean abs):", float(np.abs(o.astype(np.int16) - fr.astype(np.int16)).mean()), "->", out)
