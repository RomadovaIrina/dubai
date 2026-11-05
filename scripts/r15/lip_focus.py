#!/usr/bin/env python3
"""r15 T: a stage-by-stage still sheet of chosen instants (the eye's complaints): for each time, one row
   current: raw | cfd (CodeFormer) | sr | rep (layer repair) | FINAL  ||  clean: FINAL  ||  source
plus the numbers of lip_vanish.py at those frames (ra = lip-vs-skin redness, rg = lip edge, as a ratio to the source; fb = fallback weight).
    lip_focus.py --cur t15_ab_m03_cur --cln t15_ab_m03_cln --src <src.mp4> --t 15.0 15.2 ... --lipvis <dir> --out <jpg>
"""
import argparse, json, pathlib, sys
import cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lip_vanish import A, frames, lm_of, measure, stages_of


def grab(path, idx):
    want, got = set(idx), {}
    for k, f in enumerate(frames(path)):
        if k in want: got[k] = f
        if len(got) == len(want): break
    return got


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cur", required=True); ap.add_argument("--cln", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--t", type=float, nargs="+", required=True); ap.add_argument("--lipvis", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); K = [int(round(t * 25)) for t in a.t]
    fc = json.load(open(A / f"r9g_{a.cur}" / "timing" / f"{a.cur}_master_timing_F.json")); J = np.array(fc["j"])
    W, p, st, _ = stages_of(a.cur); Lrep = lm_of(W / f"{st[-1][1].stem}.mediapipe.npz")
    cols = [(f"cur {n}", v, k) for n, v in st for k in [None]]   # 25 fps stages: index k
    imgs = {name: grab(v, K) for name, v, _ in cols}
    fj = {k: J[k] for k in K}
    imgs["cur FINAL"] = grab(pathlib.Path(f"/workspace/dub/r15t/jobs/{a.cur}/{a.cur}.mp4"), list(fj.values()))
    imgs["clean FINAL"] = grab(pathlib.Path(f"/workspace/dub/r15t/jobs/{a.cln}/{a.cln}.mp4"), list(fj.values()))
    imgs["source"] = grab(pathlib.Path(a.src), list(fj.values()))
    vc = np.load(pathlib.Path(a.lipvis) / f"{a.cur}_lipvis.npz"); vk = np.load(pathlib.Path(a.lipvis) / f"{a.cln}_lipvis.npz")
    order = [c[0] for c in cols] + ["cur FINAL", "clean FINAL", "source"]; rows, lines = [], []
    for t, k in zip(a.t, K):
        _, _, _, box = measure(imgs["cur FINAL"][fj[k]], Lrep[k]); cx, cy, mw = box; s = int(2.6 * mw)
        x0 = int(np.clip(cx - s / 2, 0, 720 - s)); y0 = int(np.clip(cy - s / 2, 0, 1280 - s)); row = []
        for name in order:
            f = imgs[name][k] if name.startswith("cur ") and name != "cur FINAL" else imgs[name][fj[k]]
            c = cv2.resize(f[y0:y0 + s, x0:x0 + s], (220, 220), interpolation=cv2.INTER_LANCZOS4)
            key = {"cur raw": "raw", "cur cfd": "cfd", "cur sr": "sr", "cur rep": "rep", "cur FINAL": "final"}.get(name)
            if key: lab = f"{name} {vc['w_' + key][k]:.2f}/{vc['rg_' + key][k]:.2f}"
            elif name == "clean FINAL": lab = f"clean {vk['w_final'][k]:.2f}/{vk['rg_final'][k]:.2f}"
            else: lab = f"src t={t:.2f}s"
            cv2.rectangle(c, (0, 0), (220, 20), (0, 0, 0), -1); cv2.putText(c, lab, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1); row.append(c)
        rows.append(np.hstack(row))
        lines.append(f"t={t:.2f}s k={k} src_frame={fj[k]} fb_cur={vc['fb'][k]:.2f} fb_cln={vk['fb'][k]:.2f} " +
                     " ".join(f"{n}={vc['w_' + n][k]:.2f}/{vc['rg_' + n][k]:.2f}" for n in ("raw", "cfd", "sr", "rep", "final") if 'ra_' + n in vc.files) +
                     f" | clean raw={vk['w_raw'][k]:.2f}/{vk['rg_raw'][k]:.2f} final={vk['w_final'][k]:.2f}/{vk['rg_final'][k]:.2f}")
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("\n".join(lines)); print("sheet", a.out, "columns:", " | ".join(order), "(lip contrast vs source / lip edge vs source)")


if __name__ == "__main__":
    sys.exit(main())
