#!/usr/bin/env python3
"""r16d T: the silent tail of a shot -> the source picture (Irina 10.11 / Q: m03 37.9-39.9 s "scary face at the end": after the last dub phrase the
face only smiles, and KeySync's silent render widens the face and draws its own jaw ~1 cm under the real one).
From the end of the LAST STRONG dub speech of the shot (20 ms RMS > --end-thr 0.25 x p99, i.e. -12 dB; the weak phrase tail does not count) + --hold s (default 0; Irina 10.11 "starts earlier"), the paste fades to the source over --ramp s (0.25) and stays the source to the end:
  out_k = P_k + w_k (S_k - P_k), w = 0 before t0, linear 0 -> 1 over the ramp. Only when the silent tail is >= --min-tail s long.
Dub speech = 20 ms RMS > --thr x the clip's p99 (gaps < 0.3 s merged). Video copied when nothing to do. Colour as ks_plain (KS_CMS).
    python ks_tailsrc.py --master S.mp4 --paste P.mp4 --dub dub.wav --out OUT.mp4 [--hold -0.15 --ramp 0.25 --min-tail 1.0]"""
import argparse, json, shutil, subprocess, sys, os, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("master", "paste", "dub", "out")]
ap.add_argument("--hold", type=float, default=0.0); ap.add_argument("--ramp", type=float, default=0.25); ap.add_argument("--min-tail", type=float, default=1.0, dest="min_tail")
ap.add_argument("--thr", type=float, default=0.03); ap.add_argument("--end-thr", type=float, default=0.25, dest="end_thr"); a = ap.parse_args()
w = wave.open(a.dub); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32); hop = sr // 50
nf = len(x) // hop; e = np.sqrt((x[:nf * hop].reshape(nf, hop) ** 2).mean(1)) if nf else np.zeros(0); on = e > a.thr * max(float(np.percentile(e, 99)) if nf else 1.0, 1.0)
strong = e > a.end_thr * max(float(np.percentile(e, 99)) if nf else 1.0, 1.0)   # the end = the last STRONG speech (-12 dB rel p99): the weak phrase tail (m03 37.9-38.3 s) already shows a frozen generated mouth while the source talks
last = (np.flatnonzero(strong)[-1] + 1) / 50 if strong.any() else ((np.flatnonzero(on)[-1] + 1) / 50 if on.any() else 0.0)
pr = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_packets", "-of", "csv=p=0", a.paste], capture_output=True, text=True).stdout.strip().split(",")
W, H, N = int(pr[0]), int(pr[1]), int(pr[2]); dur = N / 25; t0 = last + a.hold
info = {"last_speech_s": round(last, 2), "t0_s": round(t0, 2), "frames": N, "tail_s": round(dur - t0, 2)}
if dur - t0 < a.min_tail or not on.any():
    shutil.copy(a.paste, a.out); info["applied"] = False; print("KS_TAILSRC " + json.dumps(info)); sys.exit(0)
import ks_cms
k0 = int(np.floor(t0 * 25)); FB = W * H * 3
rd = lambda p: subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", p] + ks_cms.dec(None, "rgb24") + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=FB * 2)
P, S = rd(a.paste), rd(a.master); tmp = a.out + ".v.mp4"
enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "medium"] + ks_cms.enc() + [tmp], stdin=subprocess.PIPE)
for k in range(N):
    pb, sb = P.stdout.read(FB), S.stdout.read(FB)
    if len(pb) < FB: break
    wk = min(1.0, max(0.0, (k / 25 - t0) / a.ramp))
    if wk > 0 and len(sb) == FB:
        pf = np.frombuffer(pb, np.uint8).astype(np.float32); sf_ = np.frombuffer(sb, np.uint8).astype(np.float32); pb = np.clip(pf + wk * (sf_ - pf) + 0.5, 0, 255).astype(np.uint8).tobytes()
    enc.stdin.write(pb)
enc.stdin.close(); enc.wait(); P.kill(); S.kill()
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-i", a.paste, "-map", "0:v", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart", a.out], check=True); os.remove(tmp)
info["applied"] = True; print("KS_TAILSRC " + json.dumps(info))
