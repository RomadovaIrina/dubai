#!/usr/bin/env python3
"""Round 2 (track A): HEAD-STABILISED side-by-side. Every panel is warped by the SAME transform (the inverse cumulative rigid upper-face
motion of the 25 fps master, DIS flow, anchor_study rigid npz) so the source head stands still; anything that still moves in a panel moves
RELATIVE TO THE HEAD (a sliding generated lower face / nose base, a sliding mask edge). 2x crop of the nose / mouth area, optional
|panel - ORIGINAL| x4 difference row (shows where the composite changed the frame and whether that region slides).

    python scripts/quality/r2_stab_sbs.py --id 04 --t0 25 --t1 31.6 --panel CLEAN0=win/04_w1_clean0.mp4 --panel FLOW=win/04_w1_flow4nr32.mp4 \
        --out sbs/r2/04_w1_stab.mp4 [--diff] [--half]
"""
from __future__ import annotations
import argparse, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
R2 = pathlib.Path("/workspace/dub/artifacts/lipsync-temporal/r2")


def read(p, a, n):
    import cv2
    cap = cv2.VideoCapture(str(p)); cap.set(cv2.CAP_PROP_POS_FRAMES, a); out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


def main():
    import cv2
    import anchor_filters as af
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--t0", type=float, required=True); ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--panel", action="append", required=True, help="LABEL=window clip starting at t0"); ap.add_argument("--out", required=True)
    ap.add_argument("--diff", action="store_true"); ap.add_argument("--half", action="store_true"); ap.add_argument("--audio-from", default=None)
    a = ap.parse_args()
    f0, f1 = int(round(a.t0 * 25)), int(round(a.t1 * 25)); n = f1 - f0
    lm = np.load(R2 / f"masters/{a.id}/master_mediapipe.npz")["lm"].astype(np.float64); H = np.load(R2 / f"anchor/{a.id}_rigid.npz")["H"]
    C = af.cumulative(H, f0, f1)                     # frame-f0 coords -> frame t coords
    L = np.nanmedian(lm[f0:f1], 0); iod = float(np.linalg.norm(L[33] - L[263])); cx, cy = L[[4, 13]].mean(0); s = 1.4 * iod
    box = (int(cx - s / 2), int(cy - 0.45 * s), int(cx + s / 2), int(cy + 0.55 * s))
    panels = [("ORIGINAL", read(R2 / f"masters/{a.id}/master_25fps.mp4", f0, n))]
    for p in a.panel:
        lab, path = p.split("=", 1); panels.append((lab, read(path, 0, n)))
    n = min(len(fr) for _, fr in panels)
    W, Hh = panels[0][1][0].shape[1], panels[0][1][0].shape[0]
    tmp = pathlib.Path(a.out).with_name(".tmp_" + pathlib.Path(a.out).name); pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    zw, zh = (box[2] - box[0]) * 2, (box[3] - box[1]) * 2
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (zw * len(panels), zh * (2 if a.diff else 1)))
    for k in range(n):
        Ci = af._inv(C[k])                            # frame t -> frame-f0 coords (head held still)
        row, drow = [], []
        ref = None
        for lab, fr in panels:
            st = cv2.warpAffine(fr[k], Ci, (W, Hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            crop = cv2.resize(st[box[1]:box[3], box[0]:box[2]], (zw, zh), interpolation=cv2.INTER_CUBIC)
            if ref is None: ref = crop
            lab_img = crop.copy(); cv2.putText(lab_img, f"{lab} {a.t0 + k / 25:.2f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            row.append(lab_img)
            if a.diff:
                d = np.clip(np.abs(crop.astype(np.int16) - ref.astype(np.int16)).max(2) * 4, 0, 255).astype(np.uint8)
                drow.append(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
        img = np.hstack(row)
        if a.diff: img = np.vstack([img, np.hstack(drow)])
        vw.write(img)
    vw.release()
    outs = [(a.out, "1.0")] + ([(a.out.replace(".mp4", "_half.mp4"), "2.0")] if a.half else [])
    for o, sp in outs:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(tmp), "-filter:v", f"setpts={sp}*PTS", "-c:v", "libx264", "-crf", "16",
                        "-pix_fmt", "yuv420p", o], check=True)
    tmp.unlink()
    print("wrote", [o for o, _ in outs], "box", box, "frames", n)


if __name__ == "__main__":
    main()
