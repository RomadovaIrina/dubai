#!/usr/bin/env python3
"""Round 2 (track A): head-stabilise a window clip with the SOURCE's rigid upper-face motion (same warp for every arm), keep its audio.
Used to test whether a SyncNet difference between arms on shaky footage comes from SyncNet's own 13-frame median-filtered face crop
(which cannot follow camera shake) rather than from the lip-sync itself.
    python scripts/quality/stabilise_clip.py --clip win/04_w1_clean0_dubonly.mp4 --id 04 --t0 25 --out x.mp4 [--start 0 --dur 6.6]
"""
from __future__ import annotations
import argparse, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
R2 = pathlib.Path("/workspace/dub/artifacts/lipsync-temporal/r2")


def main():
    import cv2
    import anchor_filters as af
    ap = argparse.ArgumentParser(); ap.add_argument("--clip", required=True); ap.add_argument("--id", required=True); ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip"); ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--no-stab", action="store_true", help="only cut (control)")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.clip); n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); k0 = int(round(a.start * 25)); n = n_all - k0 if a.dur is None else int(round(a.dur * 25))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k0)
    f0 = int(round(a.t0 * 25)) + k0
    H = np.load(R2 / f"anchor/{a.id}_rigid.npz")["H"]; C = af.cumulative(H, f0, f0 + n)
    tmp = pathlib.Path(a.out).with_name(".tmp_" + pathlib.Path(a.out).name); vw = None
    for k in range(n):
        ok, fr = cap.read()
        if not ok: break
        if vw is None: vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (fr.shape[1], fr.shape[0]))
        if not a.no_stab:
            fr = cv2.warpAffine(fr, af._inv(C[k]), (fr.shape[1], fr.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        vw.write(fr)
    vw.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(tmp), "-ss", f"{a.start:.3f}", "-t", f"{n / 25:.3f}", "-i", a.clip,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", a.out], check=True)
    tmp.unlink()


if __name__ == "__main__":
    main()
