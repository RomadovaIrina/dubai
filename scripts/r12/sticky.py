#!/usr/bin/env python3
"""r14 F item 3 (sticky lips): over [t0, t1) of a run, per frame lip opening (MediaPipe 13-14 / face height 10-152) of the source and of each
final, next to the dub audio energy; flags frames where the dub is voiced (energy over the speech level) but the generated lips are closed
(opening < CLOSED) while the speech is ongoing = generator glue, vs frames where the dub itself has a closure (energy dip).
    python scripts/r12/sticky.py old03 4.5 6.5 label=final.mp4 ...  -> table + summary"""
import sys, pathlib, subprocess, tempfile, json
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot")); import clean_face_stack as cfs
idn, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); arms = [a.split("=", 1) for a in sys.argv[4:]]
src = next(pathlib.Path(f"/workspace/dub/r10/runs/{idn}/work/in").glob(f"{idn}.*"))
wav = f"/workspace/dub/artifacts/r4a/r9_{idn}_htd/{idn}/dubbed_16k.wav"; CLOSED = 0.02
td = pathlib.Path(tempfile.mkdtemp())
def opening(video, tag):
    video, _, off = str(video).partition("@"); off = float(off or 0)   # path@t0: a clip that starts at source time t0 (probe_seg PROBE_SAVE)
    cut = td / f"{tag}.mp4"; subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t0 - off), "-t", str(t1 - t0), "-i", str(video), "-an", "-r", "30", "-c:v", "libx264", "-crf", "12", str(cut)], check=True)
    lm = np.load(cfs.run_mediapipe(cut, td / f"{tag}.npz"))["lm"]
    return np.linalg.norm(lm[:, 13, :2] - lm[:, 14, :2], axis=1) / np.maximum(np.linalg.norm(lm[:, 10, :2] - lm[:, 152, :2], axis=1), 1e-6)
ops = {"source": opening(src, "source")} | {k: opening(v, k) for k, v in arms}
a = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0), "-i", wav, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
a = np.frombuffer(a, np.float32); n = min(len(o) for o in ops.values()); hop = 16000 / 30
db = np.array([20 * np.log10(np.sqrt(np.mean(a[int(i * hop):int((i + 1) * hop)] ** 2)) + 1e-6) for i in range(n)])
speech = db > np.percentile(db, 80) - 15
print("t      dub_dB  " + "  ".join(f"{k:>8}" for k in ops))
for i in range(n): print(f"{t0 + i / 30:6.2f} {db[i]:7.1f} " + "  ".join(f"{ops[k][i]:8.3f}" + ("*" if k != "source" and speech[i] and ops[k][i] < CLOSED else " ") for k in ops))
summ = {k: {"closed_while_voiced": int(np.sum(speech[:n] & (o[:n] < CLOSED))), "mean_open_voiced": round(float(np.nanmean(o[:n][speech[:n]])), 4)} for k, o in ops.items()}
summ["voiced_frames"] = int(speech[:n].sum()); summ["dub_dips"] = int(np.sum(speech[1:n - 1] & (db[1:n - 1] < np.minimum(db[:n - 2], db[2:n]) - 6)))
print(json.dumps(summ))
