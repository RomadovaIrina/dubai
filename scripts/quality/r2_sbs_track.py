#!/usr/bin/env python3
"""Round 2 track B: side-by-side review clip with a HEAD-TRACKED 2x nose / mouth crop (same crop for every panel, centred on the SOURCE
nose / mouth landmarks at each frame's true time), so head and camera motion cancel and any slide of the pasted lower face against the
source nose is directly visible. Panels are read by frame index (same frame rate for every panel), audio from --audio (time window cut).
    python scripts/quality/r2_sbs_track.py --lm timing/04_source_lm.npz --frames 751:943 --fps 30 --out sbs/x.mp4 --audio F0.mp4 \
        --panel ORIGINAL=src.mp4 --panel "CLEAN0-30fps=F0.mp4" --panel FIX=T3.mp4 [--half]"""
import argparse, subprocess, pathlib
import numpy as np, cv2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--frames", required=True); ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--audio", required=True); ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--half", action="store_true"); ap.add_argument("--zoom", type=float, default=2.0); ap.add_argument("--grid", action="store_true", help="thin lines at the source nose tip / subnasale")
    a = ap.parse_args()
    A, B = (int(x) for x in a.frames.split(":")); lm = np.load(a.lm)["lm"].astype(np.float64)
    iod = float(np.nanmedian(np.linalg.norm(lm[A:B, 33] - lm[A:B, 263], axis=1))); w, h = int(1.3 * iod) // 2 * 2, int(1.05 * iod) // 2 * 2
    caps = []
    for p in a.panel:
        lab, path = p.split("=", 1); c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, A); caps.append((lab, c))
    W, H = int(w * a.zoom), int(h * a.zoom); out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(".v_" + out.name)
    rate = a.fps / (2 if a.half else 1)
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W * len(caps)}x{H}", "-r", str(rate), "-i", "-",
                            "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    for t in range(A, B):
        L = lm[t]; c = 0.5 * (L[1] + 0.5 * (L[61] + L[291])) if np.isfinite(L[0, 0]) else c
        x0, y0 = int(round(c[0] - w / 2)), int(round(c[1] - h / 2)); row = []
        for lab, cap in caps:
            ok, f = cap.read()
            if not ok: f = np.zeros((2000, 2000, 3), np.uint8)
            f = cv2.copyMakeBorder(f, h, h, w, w, cv2.BORDER_CONSTANT); crop = f[y0 + h:y0 + 2 * h, x0 + w:x0 + 2 * w]
            z = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)
            if a.grid and np.isfinite(L[0, 0]):
                nx = int((L[1, 0] - x0) * a.zoom); ny = int((L[2, 1] - y0) * a.zoom); cv2.line(z, (nx, 0), (nx, H // 6), (0, 255, 255), 1); cv2.line(z, (nx, H - H // 6), (nx, H), (0, 255, 255), 1)
            cv2.putText(z, f"{lab}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3); cv2.putText(z, f"{lab}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            row.append(z)
        enc.stdin.write(np.hstack(row).tobytes())
    enc.stdin.close(); enc.wait()
    t0, dur = A / a.fps, (B - A) / a.fps
    af = ["-af", "atempo=0.5"] if a.half else []
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", a.audio, "-map", "0:v", "-map", "1:a:0", *af,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    tmp.unlink(); print(out)


if __name__ == "__main__":
    main()
