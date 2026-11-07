#!/usr/bin/env python3
"""r16 T: KeySync on long videos in ~KS_CHUNK_S pieces (ks_fast keeps every frame of a job in RAM: ~12 GB per video minute, 61 GB at 5 min).
split: the padded crop + the (delayed, padded) dub are cut at PAUSES of the dub (the lowest-energy 0.2 s near every chunk length: the generated
       mouth is closed on both sides of a cut) into chunks [a, b). Each chunk input = the SAME reference frame (the global KS_REF_IDX frame) as its
       frame 0 + frames a .. b + TAIL (KeySync drops a few tail frames that do not fill its windows); its audio = 40 ms of silence + the dub from a.
       ks_fast then runs every chunk with KS_REF_IDX 0, in ONE resident process (one model load).
join:  each chunk's render: frame 0 (the reference) dropped, frames [0, b - a) kept (the tail discarded), the chunks concatenated (x264 crf 10).
    python ks_chunk.py split --video pad.mp4 --audio dubp.wav --ref K --chunk-s 60 --out DIR [--tag f0]   -> DIR/chunks.json + jobs lines on stdout
    python ks_chunk.py join --dir DIR --renders OUTDIR --out raw.mp4"""
import argparse, json, os, subprocess, sys, wave
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["split", "join"]); ap.add_argument("--video"); ap.add_argument("--audio"); ap.add_argument("--ref", type=int, default=0)
ap.add_argument("--chunk-s", type=float, default=60.0); ap.add_argument("--tail", type=int, default=20); ap.add_argument("--out"); ap.add_argument("--tag", default="f0")
ap.add_argument("--dir"); ap.add_argument("--renders"); a = ap.parse_args()
FPS = 25


def writer(path, n_hint=0):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "10",
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


if a.cmd == "split":
    os.makedirs(a.out, exist_ok=True)
    c = cv2.VideoCapture(a.video); F = []
    while True:
        ok, f = c.read()
        if not ok: break
        F.append(f)
    n = len(F); w = wave.open(a.audio); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    spf = sr // FPS; e = np.array([np.sqrt(np.mean(x[k * spf:(k + 5) * spf].astype(np.float64) ** 2)) if k * spf < len(x) else 0.0 for k in range(n)])   # 0.2 s windows
    L = int(a.chunk_s * FPS); cuts = [0]
    while n - cuts[-1] > 1.5 * L:   # the quietest 0.2 s within +-25 % of the chunk length (a dub pause)
        lo, hi = cuts[-1] + int(0.75 * L), min(n - L // 2, cuts[-1] + int(1.25 * L)); cuts.append(lo + int(np.argmin(e[lo:hi])) + 2)
    cuts.append(n); ref = F[min(a.ref, n - 1)]; CH = []
    for j, (s0, s1) in enumerate(zip(cuts[:-1], cuts[1:])):
        e1 = min(n, s1 + a.tail); v = f"{a.out}/c{j:02d}.mp4"; p = writer(v)
        for f in [ref] + F[s0:e1]: p.stdin.write(f.tobytes())
        p.stdin.close(); p.wait()
        seg = np.concatenate([np.zeros(spf, np.int16), x[s0 * spf:e1 * spf]])
        with wave.open(f"{a.out}/c{j:02d}.wav", "wb") as o: o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes(seg.tobytes())
        CH.append({"j": j, "a": s0, "b": s1, "in_frames": 1 + e1 - s0, "cut_rms": round(float(e[s0]) if s0 else 0.0, 1)})
        print(f"{a.tag}c{j:02d} {v} {a.out}/c{j:02d}.wav 0")   # a ks_fast job line: the reference = frame 0
    json.dump({"n": n, "chunks": CH, "ref": a.ref}, open(f"{a.out}/chunks.json", "w"), indent=1)
    print(f"KS_CHUNKS {a.tag}: {n} frames -> {len(CH)} chunks at {cuts[1:-1]} (dub rms there {[c['cut_rms'] for c in CH[1:]]})", file=sys.stderr)
else:
    meta = json.load(open(f"{a.dir}/chunks.json")); p = writer(a.out); tot = 0
    for ch in meta["chunks"]:
        tag = os.path.basename(a.dir.rstrip("/"))   # renders: <renders>/<tag>c<j>/*.mp4
        r = [os.path.join(dp, f) for dp, _, fs in os.walk(a.renders) for f in fs if f.endswith(".mp4") and os.path.basename(dp) == f"{a.tag}c{ch['j']:02d}"]
        assert r, f"no render for chunk {ch['j']}"
        c = cv2.VideoCapture(sorted(r)[0]); G = []
        while True:
            ok, f = c.read()
            if not ok: break
            G.append(f)
        need = ch["b"] - ch["a"]; G = G[1:1 + need]
        if len(G) < need: print(f"WARN chunk {ch['j']}: {len(G)} < {need} frames, the last one held", file=sys.stderr); G += [G[-1]] * (need - len(G))
        for f in G: p.stdin.write(f.tobytes())
        tot += len(G)
    p.stdin.close(); p.wait(); print(f"KS_JOIN {len(meta['chunks'])} chunks -> {a.out} ({tot} frames of {meta['n']})")
