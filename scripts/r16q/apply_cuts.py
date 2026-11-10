#!/usr/bin/env python3
"""Apply a selector cut list to a final: base video everywhere, other variants on the listed segments, 0.32 s crossfades centred on the
segment edges (a segment touching the start / end runs past it, so no switch-back at the edges). Audio is copied from the base.
  python3 apply_cuts.py --base FINAL.mp4 --var A8b=A8B.mp4 --var source=SRC_25fps.mp4 --cuts CUTS.json --out OUT.mp4
  CUTS.json: [[t0, t1, "A8b"], [-1.0, 2.0, "source"], ...]   (seconds; must use names given by --var)
Variants must be the same render / timing as the base (same dub, same 25 fps grid)."""
import argparse, json, subprocess
import numpy as np

FPS = 25.0; XF = 8   # 0.32 s


def reader(v, W, H):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "2", "-i", v, "-vf", f"fps={FPS},scale={W}:{H}:flags=bicubic", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=W * H * 6); nb = W * H * 3
    while True:
        b = p.stdout.read(nb)
        if len(b) < nb: break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.stdout.close(); p.wait()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--var", action="append", default=[]); ap.add_argument("--cuts", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    V = dict(s.split("=", 1) for s in a.var); cuts = json.load(open(a.cuts))
    import os, sys
    bad = [p for p in [a.base] + list(V.values()) if not (os.path.isfile(p) and os.path.getsize(p) > 0)]
    bad += [f"cut name {c[2]!r} has no --var" for c in cuts if c[2] not in V]
    if bad: sys.exit("apply_cuts: missing input(s): " + "; ".join(bad))
    def nfr(p): return int(json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", p], capture_output=True, text=True).stdout)["streams"][0]["nb_read_frames"])
    nb_ = nfr(a.base)
    for nm, p in V.items():   # a variant must be the same render length (+-2 frames), else the cut times do not line up
        if abs(nfr(p) - nb_) > 2: sys.exit(f"apply_cuts: {nm} has {nfr(p)} frames vs base {nb_}: not the same render")
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", a.base], capture_output=True, text=True).stdout)["streams"][0]
    W, H, n = pr["width"], pr["height"], int(pr["nb_read_frames"])
    k = np.arange(n); w = {nm: np.zeros(n) for nm in V}
    for t0, t1, nm in cuts:
        a0, b0 = int(round(t0 * FPS)), int(round(t1 * FPS)) - 1
        up = np.clip((k - (a0 - XF / 2)) / XF, 0, 1); dn = np.clip(((b0 + XF / 2) - k) / XF, 0, 1)
        w[nm] = np.fmax(w[nm], np.minimum(up, dn))
    tot = sum(w.values()); s = np.where(tot > 1, tot, 1.0)
    for nm in w: w[nm] = w[nm] / s
    wb = 1 - sum(w.values()); used = [nm for nm in V if w[nm].max() > 0]
    rb = reader(a.base, W, H); rv = {nm: reader(V[nm], W, H) for nm in used}
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", a.base,
                            "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", a.out], stdin=subprocess.PIPE)
    for i, fb in enumerate(rb):
        fr = {}
        for nm in used:
            try: fr[nm] = next(rv[nm])
            except StopIteration: fr[nm] = fb
        if i < n and any(w[nm][i] > 0 for nm in used):
            acc = fb.astype(np.float32) * wb[i]
            for nm in used: acc += fr[nm].astype(np.float32) * w[nm][i]
            fb = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        try: enc.stdin.write(fb.tobytes())
        except BrokenPipeError: break
    try: enc.stdin.close()
    except BrokenPipeError: pass
    enc.wait(); print("WROTE", a.out, {nm: round(float((w[nm] > 0.5).sum() / FPS), 2) for nm in used})


if __name__ == "__main__":
    main()
