#!/usr/bin/env python3
"""r17 quality guard for speedups: two outputs of the same input (+ the same dub) compared.
  video: PSNR over the whole frame and inside the box where the two outputs differ by > 12 levels over all frames (= the generated region), per-frame min / mean; identical = every frame bit-equal
  audio: md5 of the decoded PCM (s16le) of both
    python3 out_compare.py --a OUT1.mp4 --b OUT2.mp4 --src SRC.mp4 [--json out.json]"""
import argparse, hashlib, json, subprocess, sys
import numpy as np


def frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-vf", f"scale={w}:{h}:flags=neighbor", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def size(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    w, h = o.strip().split(","); return int(w), int(h)


def pcm_md5(path):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-f", "s16le", "-"], capture_output=True).stdout
    return hashlib.md5(b).hexdigest(), len(b)


def psnr(x, y):
    m = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2)); return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--json", default=""); a = ap.parse_args()
    w, h = size(a.a); box = None
    for fa, fs in zip(frames(a.a, w, h), frames(a.b, w, h)):   # pass 1: the box where the two outputs differ by > 12 levels (= the generated mouth / face;
        m = np.abs(fa.astype(np.int16) - fs.astype(np.int16)).max(-1) > 12   # the source is at its own fps, so it cannot be paired frame by frame)
        ys, xs = np.nonzero(m)
        if len(ys):
            b = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    full, inbox, n, same = [], [], 0, 0
    for fa, fb in zip(frames(a.a, w, h), frames(a.b, w, h)):
        n += 1; same += int(np.array_equal(fa, fb)); full.append(psnr(fa, fb))
        if box: x0, y0, x1, y1 = box; inbox.append(psnr(fa[y0:y1, x0:x1], fb[y0:y1, x0:x1]))
    ma, mb = pcm_md5(a.a), pcm_md5(a.b)
    r = {"frames": n, "identical_frames": same, "all_identical": same == n and n > 0, "box": [int(v) for v in box] if box else None,
         "psnr_full_mean": round(float(np.mean(full)), 2) if full else None, "psnr_full_min": round(float(np.min(full)), 2) if full else None,
         "psnr_box_mean": round(float(np.mean(inbox)), 2) if inbox else None, "psnr_box_min": round(float(np.min(inbox)), 2) if inbox else None,
         "audio_md5_a": ma[0], "audio_md5_b": mb[0], "audio_identical": ma == mb}
    print("OUT_COMPARE " + json.dumps(r))
    if a.json: json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
